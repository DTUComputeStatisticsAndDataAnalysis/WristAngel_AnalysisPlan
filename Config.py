import datetime
import numpy as np
import pickle

class config(object):

    BUFFER_TIME = 300 #300 for WristAngel

    wristAngelRootFolder = '' #path of base datafolder

    patientIDs = [] #Integer Ids of OCD patients.
    controlIDs = [] #Integer Ids of controls.
    IDs = patientIDs+controlIDs

    minimumNumberOfnegativeCasesPerDay = 3
    minSignalLengthForPreproc = 25 #this is decided based on the padding length of butterworth filter. Only expected to be used in rare cases where wristband turned off within a few seconds. The feature extraction will ignore these segments anyway.

    BVPWindowSize = 5
    BVPWindowStepSize = 1
    BVPMinWindowSize = 1
    BVPSkewnessBound = 1
    BVPKurtosisThreshold = -0.5

    HRFeatures = ['HR_mean','HR_std','HR_25quantile','HR_median','HR_75quantile','HR_quantiledeviation','HR_min','HR_max','HR_slope']
    EDAPhasicFeatures = ['EDA_Phasic_mean', 'EDA_Phasic_std', 'EDA_Phasic_numPeaks', 'EDA_Phasic_avgPeakAmplitude', 'EDA_Phasic_avgPeakResponseTime', 'Phasic_BandPower_0.01_0.04', 'Phasic_BandPower_0.04_0.15', 'Phasic_BandPower_0.15_0.4', 'Phasic_BandPower_0.4_1.0']
    EDATonicFeatures = ['EDA_Tonic_25quantile','EDA_Tonic_median','EDA_Tonic_75quantile','EDA_Tonic_quantiledeviation','EDA_Tonic_min','EDA_Tonic_max','EDA_Tonic_slope']
    EDAFeatures = ['EDA_mean', 'EDA_std', 'EDA_25quantile', 'EDA_median', 'EDA_75quantile', 'EDA_quantiledeviation', 'EDA_min', 'EDA_max', 'BandPower_0.01_0.04', 'BandPower_0.04_0.15', 'BandPower_0.15_0.4', 'BandPower_0.4_1.0']
    TEMPFeatures = ['TEMP_mean','TEMP_std','TEMP_min','TEMP_max','TEMP_slope']
    BVPFeatures = ['BVP_time_mean', 'BVP_time_median', 'BVP_time_std', 'BVP_time_min', 'BVP_time_max', 'BVP_freq_mean_real', 'BVP_freq_median_real', 'BVP_freq_std_real', 'BVP_freq_min_real', 'BVP_freq_max_real', 'BVP_freq_sma_real', 'BVP_freq_iqr_real', 'BVP_freq_mean_im', 'BVP_freq_median_im', 'BVP_freq_std_im', 'BVP_freq_min_im', 'BVP_freq_max_im', 'BVP_freq_sma_im', 'BVP_freq_iqr_im', 'BVP_avg_slope', 'BVP_max_slope', 'BVP_min_slope', 'Average of RR intervals', 'RMS of Sucessive RR Differences']
        
    featuresNames = HRFeatures + EDAPhasicFeatures + EDATonicFeatures + EDAFeatures + TEMPFeatures + BVPFeatures

    n_CV_jobs = 4 #How many processors to use for SKLearn CV loops. -1 for all avaliable, None for just main thread
        
    param_grid_NN = {
        'classifier__hidden_layer_sizes':[[200],[100],[50],[100,50],[200,100],[200,100,50]], 
        'classifier__alpha':np.logspace(-6.0, -2.0, num=5), 
        'classifier__activation':['logistic', 'tanh', 'relu'], # 
        'classifier__learning_rate_init': [0.01, 0.1]
    }
    
    param_grid_LR = {
        'classifier__penalty':['l2'], #'l1', 'elasticnet' 
        'classifier__C':np.logspace(-2.0, 3, num=10), 
        #'l1_ratio': np.logspace(-10, 0, num=10) #Only used for elasticNet
    }
    
    param_grid_RF = {
        'classifier__n_estimators':[100,200,500], 
        'classifier__max_features':np.linspace(0, 0.5, num=5, endpoint=False)[1:], 
        'classifier__min_samples_leaf': [1,2,3,4,5]
    }

    pathToPckleWithKnownSleepPeriods = ''
    with open(pathToPckleWithKnownSleepPeriods, 'rb') as fp:
        knownSleepPeriods = pickle.load(fp)
