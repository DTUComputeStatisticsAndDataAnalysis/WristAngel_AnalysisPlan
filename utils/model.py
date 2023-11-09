import pandas as pd
import numpy as np
import pickle
import os
import time
import shap

from tqdm import tqdm

from Config import config

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, precision_score, recall_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import binarize, StandardScaler
from sklearn.model_selection import PredefinedSplit, GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from merf import MERF
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )

class MERF_handler(BaseEstimator):
    def __init__(self):
        self.MERF = MERF()
    
    def fit(self, X, y):
        (X, Z, clusters) = X
        self.MERF.fit(X, Z, clusters, y)
    
    def predict(self, X):
        (X, Z, clusters) = X
        return self.MERF.predict(X, Z, clusters)

class model_with_paramSelection(BaseEstimator):
    def __init__(self, CVmethod, NParticipants, verbose = 0, upsample=False):

        self.CVmethod = CVmethod #'Participants' or 'Temporal'
        self.NParticipants = NParticipants
        self.verbose = verbose
        
        self.NN = MLPClassifier(max_iter=500)
        self.LR = LogisticRegression(max_iter=500)
        self.RF = RandomForestClassifier()
        self.MERF = MERF()
        
        self.standadizer = StandardScaler()
        self.classifier = None
        self.classifierParams = {}

        self.upsample =  np.random.randint(low = 0, high=100000) if upsample else None #Store a random integer to be used as random_state for overSampling so all models have same rng.
                
    def fit(self, X, y, participantLabel, temporalLabel, evaluationId=None):
        
        #Downsample X perparticipan level
        #if self.CVmethod=='TemporalDownSample':
        #    X, y, participantLabel, temporalLabel = underSampleParticipants(evaluationId, X, y, participantLabel, temporalLabel)

        if self.CVmethod=='Temporal':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel)
        elif self.CVmethod=='TemporalDownSample':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel, evaluationId)
        elif self.CVmethod=='PersonalDownSample':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel, evaluationId)
        elif self.CVmethod=='Participant':      
            #Reorder CVlabel to count from 0 to k without jumps. Example [0,1,2,4,5,6] -> [0,1,2,3,4,5]. This has to be done for PredefinedSplit
            splitLabels = mapParticipantLabelstoCVlabel(participantLabel)
            test_splits = PredefinedSplit(splitLabels) #Make CV splits
        elif self.CVmethod=='Random':
            test_splits = KFold(n_splits=10, shuffle=True)
            

        bestAvgValidationAccPerModelType = self._fitCVModel(X, y, test_splits, participantLabel)  
        
        #upsample X for final training
        if self.upsample is not None:
            overSampler = RandomOverSampler(random_state=self.upsample)
            X_samp, y_samp = overSampler.fit_resample(X,y)
        else:
            X_samp, y_samp = X, y

        X_samp = self.standadizer.fit_transform(X_samp)
        #Train the final classifier on the upsampled data
        if type(self.classifier).__name__ =='MERF': 
            Z = np.expand_dims(np.ones_like(y_samp),1)
            clusters = pd.Series(participantLabel[overSampler.sample_indices_]) if self.upsample is not None else pd.Series(participantLabel)
            self.classifier.fit(X_samp,Z,clusters,y_samp)  
        else:
            self.classifier.fit(X_samp,y_samp)

        #use the final classifier to get training prediction on the non-sampled data
        y_pred = self.predict(X,participantLabel)
        y_pred = binarize(y_pred.reshape(-1, 1), threshold=0.5).reshape(-1)
        
        return y_pred, bestAvgValidationAccPerModelType


    
    @ignore_warnings(category=ConvergenceWarning)      
    def _fitCVModel(self, X, y, CVfolds, participantLabel):
    
        grid_NN = self._crossvalNN(X, y, CVfolds)
        grid_LR = self._crossvalLR(X, y, CVfolds)
        grid_RF = self._crossvalRF(X, y, CVfolds)
        
        #Fit MERF with RF best hyperparams
        accMERF = self._crossvalMERF(X, y, CVfolds, participantLabel, grid_RF)
        
        grids = [grid_NN, grid_LR, grid_RF]
        models = [self.NN, self.LR, self.RF]
        
        best_idx = np.argmax([grid.best_score_ for grid in grids])
        
        if accMERF>grids[best_idx].best_score_:
            idx = np.where(grid_RF.cv_results_['mean_test_score']==grid_RF.best_score_)[0][0]
            param_dict = grid_RF.cv_results_['params'][idx]
            new_param_dict = {}
            for key in param_dict.keys():
                new_param_dict[key.split('__')[-1]] = param_dict[key]
            
            self.classifier = MERF()
            self.classifier.fe_model.set_params(**new_param_dict)
            self.classifierParams = param_dict
        else:
            self.classifier = models[best_idx]    
            best_grid = grids[best_idx]
            idx = np.where(best_grid.cv_results_['mean_test_score']==best_grid.best_score_)[0][0]    
            param_dict = best_grid.cv_results_['params'][idx]
            new_param_dict = {}
            for key in param_dict.keys():
                new_param_dict[key.split('__')[-1]] = param_dict[key]
            
            self.classifier.set_params(**new_param_dict)
            self.classifierParams = best_grid.cv_results_['params'][idx]
            
        if type(self.classifier).__name__ in ['MLPClassifier', 'LogisticRegression']: 
            self.classifier.set_params(**{'max_iter': 5000})

        return np.hstack([grid.best_score_ for grid in grids] + [accMERF])
        
    def _crossvalNN(self, X, y, CVfolds):
        if self.upsample is not None:
            pipeline = Pipeline(steps=[['overSampler', RandomOverSampler(random_state=self.upsample)],['standadizer', StandardScaler()],['classifier', self.NN]])
        else:
            pipeline = Pipeline(steps=[['standadizer', StandardScaler()],['classifier', self.NN]])
        
        return GridSearchCV(pipeline, config.param_grid_NN, cv=CVfolds, n_jobs=config.n_CV_jobs, verbose=self.verbose).fit(X,y)

    def _crossvalLR(self, X, y, CVfolds):
        if self.upsample is not None:
            pipeline = Pipeline(steps=[['overSampler', RandomOverSampler(random_state=self.upsample)],['standadizer', StandardScaler()],['classifier', self.LR]])
        else:
            pipeline = Pipeline(steps=[['standadizer', StandardScaler()],['classifier', self.LR]])
        
        return GridSearchCV(pipeline, config.param_grid_LR, cv=CVfolds, n_jobs=config.n_CV_jobs, verbose=self.verbose).fit(X,y)
        
    def _crossvalRF(self, X, y, CVfolds):        
        if self.upsample is not None:
            pipeline = Pipeline(steps=[['overSampler', RandomOverSampler(random_state=self.upsample)],['standadizer', StandardScaler()],['classifier', self.RF]])
        else:
            pipeline = Pipeline(steps=[['standadizer', StandardScaler()],['classifier', self.RF]])

        return GridSearchCV(pipeline, config.param_grid_RF, cv=CVfolds, n_jobs=config.n_CV_jobs, verbose=self.verbose).fit(X,y)
        
    def _crossvalMERF(self, X, y, CVfolds, participantLabel, grid_RF):
        idx = np.where(grid_RF.cv_results_['mean_test_score']==grid_RF.best_score_)[0][0]
        param_dict = grid_RF.cv_results_['params'][idx]
        self.MERF = MERF()
        
        new_param_dict = {}
        for key in param_dict.keys():
            new_param_dict[key.split('__')[-1]] = param_dict[key]
        
        self.MERF.fe_model.set_params(**new_param_dict)

        K = CVfolds.get_n_splits()
        acc = np.zeros(K)
        for i, (train_index, test_index) in enumerate(CVfolds.split(X)): #X is ignored for PredefinedSplit

            Xtrain = X[train_index]
            ytrain = y[train_index]
            Xtest = X[test_index]
            ytest = y[test_index]
        
            #upsample Xtrain for training
            if self.upsample is not None:
                overSampler = RandomOverSampler(random_state=self.upsample)
                X_samp, y_samp = overSampler.fit_resample(Xtrain,ytrain)
            else:
                X_samp, y_samp = Xtrain, ytrain

            standadizer = StandardScaler()
            X_samp = standadizer.fit_transform(X_samp)

            Z = np.expand_dims(np.ones_like(y_samp),1)
            clusters = pd.Series(participantLabel[overSampler.sample_indices_]) if self.upsample is not None else pd.Series(participantLabel)
            self.MERF.fit(X_samp, Z, clusters, y_samp)
            
            Xtest = standadizer.transform(Xtest)
            Ztest = np.expand_dims(np.ones_like(ytest),1)
            clusters_test = pd.Series(participantLabel[test_index])    
            y_pred = self.MERF.predict(Xtest,Ztest,clusters_test) #When the test data has a cluster which is not seen in the trainig data it just makes a RF prediction
            y_pred = binarize(y_pred.reshape(-1, 1), threshold=0.5).reshape(-1)
            acc[i] = accuracy_score(ytest, y_pred)
        
        return np.mean(acc)
            
    @available_if(_classifier_has("predict"))
    def predict(self, X, participantLabel):                

        X = self.standadizer.transform(X)

        if type(self.classifier).__name__ =='MERF':
            Z = np.ones((X.shape[0],1))
            clusters = pd.Series(participantLabel)
            return self.classifier.predict(X, Z, clusters)
        else: 
            return self.classifier.predict(X)
            
    @available_if(_classifier_has("predict"))
    def predict_proba(self, X, participantLabel):                

        X = self.standadizer.transform(X)

        if type(self.classifier).__name__ =='MERF':
            Z = np.ones((X.shape[0],1))
            clusters = pd.Series(participantLabel)
            return self.classifier.predict(X, Z, clusters)
        else: 
            return self.classifier.predict_proba(X)

    
class two_layer_CV():
    def __init__(self, CVmethod, NParticipants, verbose=0, upsample=False):
    
        self.CVmethod = CVmethod #'Participants' or 'Temporal'
        self.NParticipants = NParticipants
        self.verbose = verbose
        self.upsample = upsample
        
    def estimateTestACC(self, X, y, participantLabel, temporalLabel, evaluationId=None, percentage=1.0):
    
        #Find CV indexes. This need to take Participants/Temporal into account. Will be similar to in inner loop.
        if self.CVmethod=='Temporal':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel)
        elif self.CVmethod=='TemporalDownSample':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel, evaluationId)
        elif self.CVmethod=='PersonalDownSample':
            test_splits = _findTemporalSplits(participantLabel, temporalLabel, evaluationId)
        elif self.CVmethod=='Participant':
            splitLabels = mapParticipantLabelstoCVlabel(participantLabel) #This has to happen to account for missing participants
            test_splits = PredefinedSplit(splitLabels) #Make CV splits
        elif self.CVmethod=='Random':
            test_splits = KFold(n_splits=10, shuffle=True)

        K = test_splits.get_n_splits()
        acc = np.zeros(K)
        fScore = np.zeros(K)
        auc = np.zeros(K)
        precision = np.zeros(K)
        recall = np.zeros(K)
        ix_training, ix_test = [], []
        fpr_tot = []
        tpr_tot = []
        par_fprtpr = {}
        selected_models = []
        selected_params = []
        y_preds = []
        train_preds = np.ones((K,len(y)))*-1
        bestAvgValidationAccPerModelType = np.zeros((K,4))
        participantResults = np.zeros((5, self.NParticipants, K))
        SHAP_values_per_fold = []
        for i, (train_index, test_index) in enumerate(tqdm(test_splits.split(X))): #X is ignored for PredefinedSplit
            if self.CVmethod=='PersonalDownSample':
                sample_idxs, _ = train_test_split(np.arange(train_index.shape[0]), test_size=1-percentage, stratify=y[train_index])
                sample_idxs = sorted(sample_idxs)
                train_index = train_index[sample_idxs]
                print(train_index.shape)
                
            ix_training.append(train_index), ix_test.append(test_index)
            Xtrain = X[train_index]
            ytrain = y[train_index]
            Xtest = X[test_index]
            ytest = y[test_index]
            
            model = model_with_paramSelection(self.CVmethod, self.NParticipants, self.verbose,upsample=self.upsample)
            y_predTrain = model.fit(Xtrain,ytrain, participantLabel[train_index], temporalLabel[train_index], evaluationId=evaluationId)
            train_preds[i,train_index], bestAvgValidationAccPerModelType[i,:] = y_predTrain

            selected_models.append(type(model.classifier).__name__) #Cant clone MERF so instead just save the name and param_dict
            selected_params.append(model.classifierParams)
            
            y_pred = model.predict(Xtest,participantLabel[test_index]) #When the test data has a cluster which is not seen in the trainig data it just makes a RF prediction
            y_score = model.predict_proba(Xtest,participantLabel[test_index]) #When the test data has a cluster which is not seen in the trainig data it just makes a RF prediction
            y_score = y_score[:,-1] if y_score.ndim==2 else y_score #take the proba for class 1. For MERF we only have one dimension hence the if clause 
            y_pred = binarize(y_pred.reshape(-1, 1), threshold=0.5).reshape(-1)
            y_preds.append(y_pred)

            acc[i] = accuracy_score(ytest, y_pred)
            fScore[i] = f1_score(ytest, y_pred)
            try:
                auc[i] = roc_auc_score(ytest, y_score) #For Personal models/controls we might not have positive observations in test set due to low number of tags
            except:
                print(f'Error calculating the ROCAUC in fold {i}. Unique classes in yTrue are {np.unique(ytest)}.')
            precision[i] =  precision_score(ytest, y_pred)
            recall[i] =  recall_score(ytest, y_pred)

            parLabels = [evaluationId] if self.CVmethod=='TemporalDownSample' else np.unique(participantLabel)
            for j, par in enumerate(parLabels):
                filter_ = participantLabel[test_index]==par
                idx = j if self.CVmethod=='TemporalDownSample' else par
                participantResults[0,idx,i] = accuracy_score(ytest[filter_], y_pred[filter_])
                participantResults[1,idx,i] = f1_score(ytest[filter_], y_pred[filter_])
                participantResults[2,idx,i] = precision_score(ytest[filter_], y_pred[filter_])
                participantResults[3,idx,i] = recall_score(ytest[filter_], y_pred[filter_])
                try:
                    participantResults[4,idx,i] = roc_auc_score(ytest[filter_], y_score[filter_])
                    if self.CVmethod=='Temporal':
                        fpr, tpr, _ = roc_curve(ytest[filter_], y_score[filter_])
                        par_fprtpr[str(par)] = (fpr, tpr)
                except:
                    print(f'Error calculating the ROCAUC for participant {par} in fold {i}. Unique classes in yTrue are {np.unique(ytest[filter_])}.')

            fpr, tpr, _ = roc_curve(ytest, y_score)
            fpr_tot.append(fpr)
            tpr_tot.append(tpr)

            if type(model.classifier).__name__ == 'MERF':
                explainer = shap.TreeExplainer(model.classifier.trained_fe_model)
                shap_values = explainer.shap_values(Xtest)
            elif type(model.classifier).__name__ == 'RandomForestClassifier':
                explainer = shap.TreeExplainer(model.classifier)
                shap_values = explainer.shap_values(Xtest)
            elif type(model.classifier).__name__ == 'MLPClassifier':
                explainer = shap.KernelExplainer(model.classifier.predict, shap.sample(Xtrain,50))
                shap_values = explainer.shap_values(Xtest, nsamples=100)
            elif type(model.classifier).__name__ == 'LogisticRegression':
                explainer = shap.KernelExplainer(model.classifier.predict, shap.sample(Xtrain,50))
                shap_values = explainer.shap_values(Xtest, nsamples=100)
            
            for SHAPs in shap_values:
                SHAP_values_per_fold.append(SHAPs)            

        return np.mean(acc), np.mean(fScore), np.mean(auc), acc, fScore, (auc, fpr_tot, tpr_tot), (precision, recall), (selected_models, selected_params, y_preds, train_preds, bestAvgValidationAccPerModelType), (participantResults, par_fprtpr), (ix_training, ix_test, SHAP_values_per_fold)
        
def _findTemporalSplits(participantLabel, temporalLabel, participantId=None):
    #for each participant take out data from the last day/week
    tmp = []
    parLabels = np.unique(participantLabel) if participantId is None else [participantId]
    for subject in parLabels:
        filter_ = participantLabel==subject #participant filter
        
        uniqueTemporalLabels = np.unique(temporalLabel[filter_], return_counts=False) #get unique temporal labels for participnt
        temporalThreshold = np.quantile(uniqueTemporalLabels, 0.875) #determine temporal threshold as 87.5% percentile

        tmp.append(np.where((temporalLabel>=temporalThreshold) & filter_)[0]) #find indicies where participant=participantLabel and temporalLabel is higher than 87.5 percentile

    tmp = np.hstack(tmp)

    #Create testset filter
    test_idx = np.ones((len(participantLabel)),dtype=np.int32)*-1
    test_idx[tmp] = 0
    
    return PredefinedSplit(test_idx)

def _findTemporalSplits_dtu(participantLabel, temporalLabel):

    #for each participant take out data from the last day/week
    tmp = []
    for subject in np.unique(participantLabel):
        filter_ = participantLabel==subject
        lastday_ = np.max(temporalLabel[filter_])
        tmp.append(np.where((temporalLabel==lastday_) & filter_)[0])

    tmp = np.hstack(tmp)

    #Create testset filter
    test_idx = np.ones((len(participantLabel)),dtype=np.int32)*-1
    test_idx[tmp] = 0
    
    return PredefinedSplit(test_idx)

def mapParticipantLabelstoCVlabel(participantLabel):

    # Create a dictionary that maps each unique value in participantLabel to its index in the output array
    value_to_index = {}
    count = 0
    for value in sorted(set(participantLabel)):
        value_to_index[value] = count
        count += 1

    # Use the dictionary to populate the output array
    output = [value_to_index[value] for value in participantLabel]

    return output

#Old way of doing above
#missing_ps = [ps not in np.unique(participantLabel) for ps in np.arange(self.NParticipants)] #Find missing partipicant if any
#missing_ps = np.arange(self.NParticipants)[missing_ps] #Convert to int

#if len(missing_ps) > 0: #If we have a missing participant. That is a jump (missing value) in participantLabel
#    splitLabels = np.copy(participantLabel)
#    splitLabels[participantLabel > missing_ps] = splitLabels[participantLabel > missing_ps]-1 #Subtract one from all values higher than the missing value

def underSampleParticipants(evaluationId, X, y, participantLabel, temporalLabel):

    n = np.sum(participantLabel==evaluationId)
    X_sample = []
    y_sample = []
    participant_sample = []
    temporal_sample = []
    for subject in np.unique(participantLabel):
        filter_ = participantLabel==subject
        n_sample = min(n, np.sum(filter_))

        idxs = np.where(filter_)[0]
        sample_idxs = sorted(np.random.choice(idxs, size=n_sample, replace=False))

        X_sample.append(X[sample_idxs,:])
        y_sample.append(y[sample_idxs])
        participant_sample.append(participantLabel[sample_idxs])
        temporal_sample.append(temporalLabel[sample_idxs])

    X_sample = np.concatenate(X_sample,axis=0)
    y_sample = np.concatenate(y_sample,axis=0)
    participant_sample = np.concatenate(participant_sample,axis=0)
    temporal_sample = np.concatenate(temporal_sample,axis=0)

    return X_sample, y_sample, participant_sample, temporal_sample