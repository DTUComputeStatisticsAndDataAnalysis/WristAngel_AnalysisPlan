import numpy as np
import pandas as pd
import csv
import datetime
import sys
import argparse
import pickle
import shap

from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split

from utils.readData import readData
from utils.model import two_layer_CV, model_with_paramSelection, underSampleParticipants
from Config import config

import warnings
warnings.filterwarnings('ignore')

def main(args):
    #argName = args.argName
    dataset = args.dataset
    samplingMethod = args.samplingMethod
    windowLength = args.windowLength
    modelType = args.modelType
    group = args.group
    upsample = args.upsample

    dataFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + group if len(args.dataFileName)==0 else dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + group + '_' + args.dataFileName 
    
    try:    
        with open('./data/' + dataFileName + '_anonymous.pkl', 'rb') as fp:
            (X, y, days, participants, timeOfDay, removedData) = pickle.load(fp)        
    except:
        with open('./data/creationOfFeatureMatrix' + dataFileName + '_anonymous.log', 'wb') as f:
            with redirect_stdout(f):
                X, y, days, participants, timeOfDay, removedData = readData(samplingMethod, windowLength, group=group)
                
        out = (X, y, days, participants, timeOfDay, removedData)
        with open('./data/' + dataFileName + '_anonymous.pkl', 'wb') as fp:
            pickle.dump(out, fp)


    timeOfDay = np.array(timeOfDay)
    participants = np.array(participants)
    
    n,m = X.shape
    N = len(np.unique(participants))

    print(f'Number of observations: {n}')
    print(f'Number of features: {m}')
    print(f'Number of unique participants: {N}')
        
    assert n == len(y)
    assert n == len(days)
    assert n == len(participants)
    assert n == len(timeOfDay)
    assert m == len(config.featuresNames)

    upsampleStr = '_upsampled' if upsample else ''
    if modelType=='Personal':
        for i in range(1,11):
            model = model_with_paramSelection('Temporal', 1, upsample=upsample)
            CV = two_layer_CV('Temporal', 1, verbose=1, upsample=upsample)

            for id_ in np.unique(participants):
                filter_ = participants==id_
                avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X[filter_,:], y[filter_], np.zeros((len(participants[filter_])),dtype=int), days[filter_])
                
                print(f'The average accuracy on the outer layer is {avg_acc}')
                print(f'The average F1 score on the outer layer is {avg_f1score}')
                print(f'The average ROC AUC on the outer layer is {avg_auc}')

                resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
                out = {
                    'avgAcc': avg_acc,
                    'avgF1Score': avg_f1score,
                    'avgAUC': avg_auc,
                    'accuracies': all_accuracies,
                    'f1Scores': all_f1scores,
                    'rocLists': rocInfo,
                    'precisionRecall': precisionRecall,
                    'selectedModels': selected_models,
                    'participantResults': participantResults,
                    'shapInfo': shapInfo
                }
                with open('./results/' + resultsFileName + '_ID' + str(id_) + '_' + str(i) + '.pkl', 'wb') as fp:
                    pickle.dump(out, fp)
    if modelType=='PersonalDownSample':
        for i in range(1,11):
            model = model_with_paramSelection('PersonalDownSample', 1, upsample=upsample)
            CV = two_layer_CV('PersonalDownSample', 1, verbose=1, upsample=upsample)

            filter_ = participants==7
            for percentage in [0.1, 0.25, 0.5, 0.75, 0.9]:
                #No statefied sampling
                #n_sample = int(percentage*(X[filter_,:].shape[0]))
                #sample_idxs = sorted(np.random.choice(np.arange(X[filter_,:].shape[0]), size=n_sample, replace=False))
                
                #stratified sampling
                #sample_idxs, _ = train_test_split(np.arange(X[filter_,:].shape[0]), test_size=1-percentage, stratify=y[filter_])
                #sample_idxs = sorted(sample_idxs)

                X_sample = X[filter_,:]#[sample_idxs,:]
                y_sample = y[filter_]#[sample_idxs]
                participant_sample = np.zeros((len(y[filter_])),dtype=int)
                days_sample = days[filter_]#[sample_idxs]

                avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X_sample, y_sample, participant_sample, days_sample, percentage=percentage)
                
                print(f'The average accuracy on the outer layer is {avg_acc}')
                print(f'The average F1 score on the outer layer is {avg_f1score}')
                print(f'The average ROC AUC on the outer layer is {avg_auc}')

                ytest = y_sample[shapInfo[1]] #test indicies are found in shapInfo[1]

                resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
                out = {
                    'avgAcc': avg_acc,
                    'avgF1Score': avg_f1score,
                    'avgAUC': avg_auc,
                    'accuracies': all_accuracies,
                    'f1Scores': all_f1scores,
                    'rocLists': rocInfo,
                    'precisionRecall': precisionRecall,
                    'selectedModels': selected_models,
                    'participantResults': participantResults,
                    'shapInfo': shapInfo,
                    'ytest': ytest
                }
                with open('./results/' + resultsFileName + '_ID7_percentageData' + str(percentage) + '_' + str(i) + '.pkl', 'wb') as fp:
                    pickle.dump(out, fp)
    if modelType=='PersonalDownSampleRandomCV':
        for i in range(1,11):
            model = model_with_paramSelection('Random', 1, upsample=upsample)
            CV = two_layer_CV('Random', 1, verbose=1, upsample=upsample)

            filter_ = participants==7
            for percentage in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
                #No statefied sampling
                #n_sample = int(percentage*(X[filter_,:].shape[0]))
                #sample_idxs = sorted(np.random.choice(np.arange(X[filter_,:].shape[0]), size=n_sample, replace=False))
                
                #stratified sampling
                if percentage<1.0:
                    sample_idxs, _ = train_test_split(np.arange(X[filter_,:].shape[0]), test_size=1-percentage, stratify=y[filter_])
                    sample_idxs = sorted(sample_idxs)
                else:
                    sample_idxs = np.arange(X[filter_,:].shape[0])
                    
                X_sample = X[filter_,:][sample_idxs,:]
                y_sample = y[filter_][sample_idxs]
                participant_sample = np.zeros((len(y_sample)),dtype=int)
                days_sample = days[filter_][sample_idxs]

                print(X_sample.shape)

                avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X_sample, y_sample, participant_sample, days_sample)
                
                print(f'The average accuracy on the outer layer is {avg_acc}')
                print(f'The average F1 score on the outer layer is {avg_f1score}')
                print(f'The average ROC AUC on the outer layer is {avg_auc}')

                ytest = [y_sample[test_idx] for test_idx in shapInfo[1]] #test indicies are found in shapInfo[1]

                resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
                out = {
                    'avgAcc': avg_acc,
                    'avgF1Score': avg_f1score,
                    'avgAUC': avg_auc,
                    'accuracies': all_accuracies,
                    'f1Scores': all_f1scores,
                    'rocLists': rocInfo,
                    'precisionRecall': precisionRecall,
                    'selectedModels': selected_models,
                    'participantResults': participantResults,
                    'shapInfo': shapInfo,
                    'ytest': ytest
                }
                with open('./results/' + resultsFileName + '_ID7_percentageData' + str(percentage) + '_' + str(i) + '.pkl', 'wb') as fp:
                    pickle.dump(out, fp)
    elif modelType=='TemporalDownSample':
        for i in range(1,11):
            
            model = model_with_paramSelection('TemporalDownSample', 1, upsample=upsample)
            CV = two_layer_CV('TemporalDownSample', 1, verbose=1, upsample=upsample)
            for id_ in np.unique(participants):
            
                #Downsample participants with more data
                X_sample, y_sample, participant_sample, days_sample = underSampleParticipants(id_, X, y, participants, days)

                avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X_sample, y_sample, participant_sample, days_sample, evaluationId=id_)
                
                print(f'The average accuracy on the outer layer is {avg_acc}')
                print(f'The average F1 score on the outer layer is {avg_f1score}')
                print(f'The average ROC AUC on the outer layer is {avg_auc}')

                resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
                out = {
                    'avgAcc': avg_acc,
                    'avgF1Score': avg_f1score,
                    'avgAUC': avg_auc,
                    'accuracies': all_accuracies,
                    'f1Scores': all_f1scores,
                    'rocLists': rocInfo,
                    'precisionRecall': precisionRecall,
                    'selectedModels': selected_models,
                    'participantResults': participantResults,
                    'shapInfo': shapInfo
                }
                with open('./results/' + resultsFileName + '_ID' + str(id_) + '_' + str(i) + '.pkl', 'wb') as fp:
                    pickle.dump(out, fp)
    elif modelType=='RandomDownSample':
        for i in range(1,2):
            model = model_with_paramSelection('Random', N, upsample=upsample)
            CV = two_layer_CV('Random', N, verbose=1, upsample=upsample)
            
            #Downsample participants with more data
            X_sample, y_sample, participant_sample, days_sample = underSampleParticipants(1, X, y, participants, days) #participant 3 has the lowest data amount with 30, 0 is the median with 213 #Random DownSample. Does not account for class distribution

            avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X_sample, y_sample, participant_sample, days_sample)
            
            print(f'The average accuracy on the outer layer is {avg_acc}')
            print(f'The average F1 score on the outer layer is {avg_f1score}')
            print(f'The average ROC AUC on the outer layer is {avg_auc}')

            resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
            out = {
                'avgAcc': avg_acc,
                'avgF1Score': avg_f1score,
                'avgAUC': avg_auc,
                'accuracies': all_accuracies,
                'f1Scores': all_f1scores,
                'rocLists': rocInfo,
                'precisionRecall': precisionRecall,
                'selectedModels': selected_models,
                'participantResults': participantResults,
                'shapInfo': shapInfo
            }
            with open('./results/' + resultsFileName + str(i) + '.pkl', 'wb') as fp:
                pickle.dump(out, fp)
    elif modelType=='topShapFeatures':
        for i in range(1,2):
            model = model_with_paramSelection('Random', N, upsample=upsample)
            CV = two_layer_CV('Random', N, verbose=1, upsample=upsample)
            
            #Filter the top shap values
            fnames = np.array(config.featuresNames)
            cols = np.hstack([
            np.where(fnames=='BVP_max_slope')[0],
            np.where(fnames=='BVP_min_slope')[0],
            np.where(fnames=='BVP_freq_mean_real')[0],
            np.where(fnames=='BVP_time_min')[0],
            np.where(fnames=='BVP_freq_min_real')[0],
            np.where(fnames=='BVP_time_max')[0],
            np.where(fnames=='BVP_freq_sma_real')[0],
            np.where(fnames=='BVP_freq_max_real')[0],
            np.where(fnames=='BVP_time_std')[0],
            np.where(fnames=='BVP_freq_mean_in')[0],
            np.where(fnames=='Phasic_BandPower_0.4_1.0')[0],
            np.where(fnames=='HR_std')[0],
            np.where(fnames=='BVP_freq_iqr_real')[0],
            np.where(fnames=='BVP_freq_iqr_im')[0],
            np.where(fnames=='BVP_avg_slopq')[0],
            np.where(fnames=='TEMP_max')[0],
            np.where(fnames=='HR_25quantile')[0],
            np.where(fnames=='BVP_freq_sma_im')[0],
            np.where(fnames=='HR_max')[0],
            np.where(fnames=='Phasic_BandPower_0.04_0.15')[0],
            ])

            cols = sorted(cols)
            X_sample = X[:,cols]
            avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X_sample, y, participants, days)
            
            print(f'The average accuracy on the outer layer is {avg_acc}')
            print(f'The average F1 score on the outer layer is {avg_f1score}')
            print(f'The average ROC AUC on the outer layer is {avg_auc}')

            resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
            out = {
                'avgAcc': avg_acc,
                'avgF1Score': avg_f1score,
                'avgAUC': avg_auc,
                'accuracies': all_accuracies,
                'f1Scores': all_f1scores,
                'rocLists': rocInfo,
                'precisionRecall': precisionRecall,
                'selectedModels': selected_models,
                'participantResults': participantResults,
                'shapInfo': shapInfo
            }
            with open('./results/' + resultsFileName + str(i) + '.pkl', 'wb') as fp:
                pickle.dump(out, fp)    
    else:
        n_repeats = 10 if modelType=='Temporal' else 1
        for i in range(1,n_repeats+1):
            model = model_with_paramSelection(modelType, N, upsample=upsample)
            CV = two_layer_CV(modelType, N, verbose=1, upsample=upsample)
            
            avg_acc, avg_f1score, avg_auc, all_accuracies, all_f1scores, rocInfo, precisionRecall, selected_models, participantResults, shapInfo = CV.estimateTestACC(X, y, participants, days)
            
            print(f'The average accuracy on the outer layer is {avg_acc}')
            print(f'The average F1 score on the outer layer is {avg_f1score}')
            print(f'The average ROC AUC on the outer layer is {avg_auc}')

            resultsFileName = dataset + '_' +  samplingMethod + '_' + str(windowLength) + '_' + modelType + '_' + group + upsampleStr
            out = {
                'avgAcc': avg_acc,
                'avgF1Score': avg_f1score,
                'avgAUC': avg_auc,
                'accuracies': all_accuracies,
                'f1Scores': all_f1scores,
                'rocLists': rocInfo,
                'precisionRecall': precisionRecall,
                'selectedModels': selected_models,
                'participantResults': participantResults,
                'shapInfo': shapInfo
            }
            with open('./results/' + resultsFileName + '_' + str(i) + '.pkl', 'wb') as fp:
                pickle.dump(out, fp)
    
    #model.fit(X, y, participants, days)
    
    
                    
def parse_arguments(argv):
    """Command line parser.
    Use like:
    python main.py --arg1 string --arg2 value --arg4
    
    For help:
    python main.py -h
    """
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--arg1', type=str, default='String', help='String value. Default = "String"')
    #parser.add_argument('--arg2', type=int, default=50, choices=[50, 100, 200], help='Integer value with limited choices. Default = 50')
    #parser.add_argument('--arg3', type=float, default=0.001, help='Float value. Default = 0.001')
    #parser.add_argument('--arg4', type=bool, default=False, help='Bool value. Default = False')
    #parser.add_argument("--optional", action="store_true", help="Optional argument")

    parser.add_argument("--upsample", action="store_true", help="Whether to upsample the positive class")

    parser.add_argument('--dataset', type=str, default='wristAngel', choices=['wristAngel'], help='The data set to use. Choices are DTU ("DTU") or Wrist Angel ("wristAngel"). Default = "DTU"')    
    parser.add_argument('--samplingMethod', type=str, default='extractTags', choices=['rollingWindows', 'extractTags'], help='The method to extract observations. Choices are rolling windows ("rollingWindows") or windows sampled around tags ("extractTags"). Default = "extractTags"')
    parser.add_argument('--windowLength', type=int, default=300, help='Window length in seconds to use for feature extraction. Default=120')     
    parser.add_argument('--modelType', type=str, choices=['Temporal', 'TemporalDownSample', 'Participant', 'Personal','PersonalDownSample', 'Random','RandomDownSample','topShapFeatures','PersonalDownSampleRandomCV'], help='The model type to train. Choices are Temporal ("Temporal") Participant ("Participant") or Personal ("Personal"). Required')        
    parser.add_argument('--group', type=str, default='patients', choices=['all', 'patients','controls'], help='The group which data to use. Choices are all ("all") or patients ("patients") or controls ("controls"). Default = "all"')    
    
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
