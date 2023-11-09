import numpy as np
import pandas as pd
import neurokit2 as nk
import csv
import datetime
import scipy 

from scipy.fft import rfft
from scipy.stats import stats

from Config import config

def _gradient_resampler(y):
    
    if y.name=='time':
        return None
    else:
        try:
            y = y.dropna()        
            x = np.arange(len(y))*config.samplePeriods[y.name]
            slope = np.polyfit(x, y, 1)[0]
            
            return slope
        except:
            return None

def bandpower(x, fmin, fmax, fs):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def _extractHRfeatures(values):
    
    x = np.arange(len(values))*config.samplePeriods['HR']
    try:
        slope = np.polyfit(x, values, 1)[0]
    except:
        slope = np.nan

    features = np.array([
        np.mean(values), #'HR_mean'
        np.std(values), #'HR_std'
        np.quantile(values,0.25), #'HR_25quantile'
        np.median(values), #'HR_median'
        np.quantile(values,0.75), #'HR_75quantile'
        np.quantile(values,0.75)-np.quantile(values,0.25), #'HR_quantiledeviation'
        np.min(values), # 'HR_min'
        np.max(values), # 'HR_max'
        slope # 'HR_slope'
    ])
       
    return features

def _extractEDAPhasicfeatures(values):
    
    
    try: #I encounter a wierd problem with eda_peaks "ValueError: zero-size array to reduction operation maximum which has no identity" but values are populated and this github issue suggest it is something with the storage https://github.com/neuropsychology/NeuroKit/issues/603
        fs = int(1/config.samplePeriods['EDA_Phasic'])
        _, info = nk.eda_peaks(values, sampling_rate=fs)
        
        features = np.array([
            np.mean(values), #'EDA_Phasic_mean'
            np.std(values), #'EDA_Phasic_std'
            len(info['SCR_Peaks']), #'EDA_Phasic_numPeaks'
            np.mean(info['SCR_Amplitude'][np.isfinite(info['SCR_Amplitude'])]), #'EDA_Phasic_avgPeakAmplitude'
            np.mean(info['SCR_RecoveryTime'][np.isfinite(info['SCR_RecoveryTime'])]), #'EDA_Phasic_avgPeakResponseTime'
            bandpower(values, 0.01, 0.04, fs=fs), # 'Phasic_BandPower_0.01_0.04'
            bandpower(values, 0.04, 0.15, fs=fs), # 'Phasic_BandPower_0.04_0.15'
            bandpower(values, 0.15, 0.4, fs=fs), # 'Phasic_BandPower_0.15_0.4'
            bandpower(values, 0.4, 1, fs=fs) # 'Phasic_BandPower_0.4_1.0' 
        ])
    except:
        features = np.empty((len(config.EDAPhasicFeatures)))
        features.fill(np.nan)

    return features

def _extractEDATonicfeatures(values):
    
    x = np.arange(len(values))*config.samplePeriods['EDA_Tonic']
    try:
        slope = np.polyfit(x, values, 1)[0]
    except:
        slope = np.nan
    
    features = np.array([
        np.quantile(values,0.25), #'EDA_Tonic_25quantile'
        np.median(values), #'EDA_Tonic_median'
        np.quantile(values,0.75), #'EDA_Tonic_75quantile'
        np.quantile(values,0.75)-np.quantile(values,0.25), #'EDA_Tonic_quantiledeviation'
        np.min(values), #'EDA_Tonic_min'
        np.max(values), #'EDA_Tonic_max'
        slope #'EDA_Tonic_slope'
    ])
    
    return features

def _extractTEMPfeatures(values):
    
    x = np.arange(len(values))*config.samplePeriods['TEMP']
    try:
        slope = np.polyfit(x, values, 1)[0]
    except:
        slope = np.nan

    features = np.array([
        np.mean(values), #'TEMP_mean'
        np.std(values), #'TEMP_std'
        np.min(values), #'TEMP_min'
        np.max(values), #'TEMP_max'
        slope #'TEMP_slope'
    ])
    
    return features

def _extractEDAFeatures(values):

    fs = int(1/config.samplePeriods['EDA'])
    
    features = np.array([
        np.mean(values), #'EDA_mean'
        np.std(values), #'EDA_std'
        np.quantile(values,0.25), #'EDA_25quantile'
        np.median(values), #'EDA_median'
        np.quantile(values,0.75), #'EDA_75quantile'
        np.quantile(values,0.75)-np.quantile(values,0.25), #'EDA_quantiledeviation'
        np.min(values), #'EDA_min'
        np.max(values), #'EDA_max'
        bandpower(values, 0.01, 0.04, fs=fs), #'EDA_BandPower_0.01_0.04'
        bandpower(values, 0.04, 0.15, fs=fs), #'EDA_BandPower_0.04_0.15'
        bandpower(values, 0.15, 0.4, fs=fs), #'EDA_BandPower_0.15_0.4'
        bandpower(values, 0.4, 1, fs=fs) #'EDA_BandPower_0.4_1.0' 
    ])
    
    return features

def _extractBVPfeaturesFromSegment(values):

    #RR Intervals
    try:
        info = nk.ppg_findpeaks(values, sampling_rate=int(1/config.samplePeriods['BVP']), method='elgendi', show=False)
        RR = np.diff(info['PPG_Peaks'])*config.samplePeriods['BVP']
        RR_SSD = np.diff(RR)
    except:
        RR = np.array([])
        RR_SSD = np.array([])

    x = np.arange(len(values))*config.samplePeriods['BVP']
    slope = np.polyfit(x, values, 1)[0]

    #Time features
    time_feat_i_o = pd.DataFrame()
    time_feat_i_o.at[0,"{}_time_mean".format('BVP')] = np.mean(values)
    time_feat_i_o.at[0,"{}_time_median".format('BVP')] = np.median(values)
    time_feat_i_o.at[0,"{}_time_std".format('BVP')] = np.std(values)
    time_feat_i_o.at[0,"{}_time_min".format('BVP')] = np.min(values)
    time_feat_i_o.at[0,"{}_time_max".format('BVP')] = np.max(values)
    #time_feat_i_o.at[0,"{}_time_abs_int".format('BVP')] = np.trapz(values)/len(values)
 
    #Frequency features
    freq_feat_i_o = pd.DataFrame()
    freq_item = rfft(values) # transform to frequency domain using the Fourier transform
        
    freq_feat_i_o.at[0,"{}_freq_mean".format('BVP')] = np.mean(freq_item)
    freq_feat_i_o.at[0,"{}_freq_median".format('BVP')] = np.median(freq_item)
    freq_feat_i_o.at[0,"{}_freq_std".format('BVP')] = np.std(freq_item)
    freq_feat_i_o.at[0,"{}_freq_min".format('BVP')] = np.min(freq_item)
    freq_feat_i_o.at[0,"{}_freq_max".format('BVP')] = np.max(freq_item)
    freq_feat_i_o.at[0,"{}_freq_sma".format('BVP')] = np.sum(freq_item)
    freq_feat_i_o.at[0,"{}_freq_iqr".format('BVP')] = stats.iqr(freq_item)
        
    # divide into real and imaginary parts respectively
    col_names = freq_feat_i_o.columns
    real_col_names = [x + "_real" for x in col_names]
    im_col_names = [x + "_im" for x in col_names]
    real_val = freq_feat_i_o.values.real
    im_val = freq_feat_i_o.values.imag

    #slope of time signal
    #slope_dict = _slidingWindowSlopeFeatures(values,50,25,int(1/config.samplePeriods['BVP']))
    time_feat_i_o.at[0,"{}_slope".format('BVP')] = slope
    #time_feat_i_o.at[0,"{}_max_slope".format('BVP')] = slope_dict["max"]
    #time_feat_i_o.at[0,"{}_min_slope".format('BVP')] = slope_dict["min"]
    #time_feat_i_o.at[0,"{}_avg_slope".format('BVP')] = slope_dict["avg"]
    #time_feat_i_o.at[0,"{}_overall_slope".format('BVP')] = slope_dict["overall"]

    freq_feat_i_o = pd.concat([pd.DataFrame(real_val,columns = real_col_names),pd.DataFrame(im_val, columns = im_col_names)], axis = 1)

    return pd.concat([time_feat_i_o,freq_feat_i_o], axis=1), RR, RR_SSD # collect features


def _getBVPSegmentsFromIBIData(df_bvp):

    #Extract segments with IBI data
    df_bvp['group'] = df_bvp['IBIAvaliable'].ne(df_bvp['IBIAvaliable'].shift()).cumsum() #Make an indicator column which increases when IBIAvaliable change
    df_bvp_groups = df_bvp.groupby('group') #Group and split by the indicator column
    dfs = []
    nondfs = []
    for name, data in df_bvp_groups: #Loop over all groups        
        if data['IBIAvaliable'].values[0] == 1: #If IBI is avaliable
            dfs.append(data) #Save the group
        else:
            nondfs.append(data) # Segments with no IBI data
            
    return dfs, nondfs
    
def _getBVPSegments(df_bvp, kurtosisThreshold=None, skewThreshold=None):

    if kurtosisThreshold is None:
        kurtosisThreshold = config.BVPKurtosisThreshold
    if skewThreshold is None:
        skewThreshold = config.BVPSkewnessBound
    
    windowSize = int(config.BVPWindowSize / config.samplePeriods['BVP'])
    stepSize = int(config.BVPWindowStepSize / config.samplePeriods['BVP'])
    min_observations = int(config.BVPMinWindowSize / config.samplePeriods['BVP'])

    dfs = []
    nondfs = []
    
    # compute all indices 
    indices = np.arange(len(df_bvp))
    
    # compute all windows
    start = 0 #initiate start to 0
    end = stepSize #initiate end to 0
    windows = []
    while start < len(df_bvp): #While start is smaller than sequence length
        windows.append(indices[start:end]) #Append the current window
        end = end+stepSize if (end+stepSize) < len(df_bvp) else len(df_bvp) #increase end index. Upper limit of sequence length
        start = start+stepSize if (end > windowSize or end == len(df_bvp)) else 0 #increase start index if end is larger than windowsize or reached end of sequence
        
    # compute measure of fit for each window and save relevant windows
    savedSegments = []
    savedNonSegments = []
    for i in range(len(windows)):
        df = df_bvp.iloc[windows[i],:]
        if np.abs(stats.skew(df['BVP'].values)) < skewThreshold and stats.kurtosis(df['BVP'].values) < kurtosisThreshold and len(df) > stepSize: #Disregard the last window if it is shorter than stepsize
            savedSegments.append(df)
        else:
            savedNonSegments.append(df)
        
    return savedSegments, savedNonSegments

def _extractBVPfeaturesFromWindow(df_bvp, segmentation='SkewKurtosis', kurtosisThreshold=None, skewThreshold=None):

    if segmentation=='IBI':
        dfs, _ = _getBVPSegmentsFromIBIData(df_bvp) #Column 'IBIAvaliable' required in df_bvp for this function
    elif segmentation=='SkewKurtosis':
        dfs, _ = _getBVPSegments(df_bvp, kurtosisThreshold, skewThreshold)
    
    if len(dfs)==0: #No IBI data is avaliable in window. I.e., No BVP features can be extracted, return array of nan
        feat_i_o = np.empty((len(config.BVPFeatures)))
        feat_i_o.fill(np.nan)
        
        return feat_i_o
    
    #Compute BVP features on each segment and use average
    feat_i_o = pd.DataFrame()
    RR = []
    RRSucessiveDifferences = []    
    for df in dfs:
        feat_i_o_temp, RR_tmp, RRSucessiveDifferences_tmp = _extractBVPfeaturesFromSegment(df['BVP_filtered'].values)
        feat_i_o = pd.concat([feat_i_o,feat_i_o_temp])
        RR.extend(RR_tmp)
        RRSucessiveDifferences.extend(RRSucessiveDifferences_tmp)
    
    minSlope = np.min(feat_i_o['BVP_slope'].values)
    maxSlope = np.max(feat_i_o['BVP_slope'].values)
    RR = np.mean(RR)
    RRSucessiveDifferences = np.array(RRSucessiveDifferences)
    RMSSD = np.sqrt(np.mean(RRSucessiveDifferences*RRSucessiveDifferences))

    feat_i_o = feat_i_o.mean(axis=0) # mean

    features = np.hstack([feat_i_o.values, maxSlope, minSlope, RR, RMSSD])

    return features

def _extractFeaturesFromWindow(window):
    
    try:
        HR_features = _extractHRfeatures(window['HR'].dropna().values)
        EDAPhasic_features = _extractEDAPhasicfeatures(window['EDA_Phasic'].dropna().values)
        EDATonic_features = _extractEDATonicfeatures(window['EDA_Tonic'].dropna().values)
        EDA_features = _extractEDAFeatures(window['EDA'].dropna().values)
        TEMP_features = _extractTEMPfeatures(window['TEMP'].dropna().values)
        BVP_features = _extractBVPfeaturesFromWindow(window[['BVP','BVP_filtered']].dropna()) #Column 'IBIAvaliable' required to be passed as well for segmentation='IBI'
    except:
        nanFeatures = np.empty((len(config.featuresNames)))
        nanFeatures.fill(np.nan)
        
        return nanFeatures

    assert len(HR_features) == len(config.HRFeatures)
    assert len(EDAPhasic_features) == len(config.EDAPhasicFeatures)
    assert len(EDATonic_features) == len(config.EDATonicFeatures)
    assert len(EDA_features) == len(config.EDAFeatures)
    assert len(TEMP_features) == len(config.TEMPFeatures)
    assert len(BVP_features) == len(config.BVPFeatures)
     
    return np.hstack([HR_features, EDAPhasic_features, EDATonic_features, EDA_features, TEMP_features, BVP_features]) 
    
def _extractFeaturesFromResampledDataFrames(dataframes):
    
    (df_max, df_mean, df_std, df_min, df_slope) = dataframes
    
    X = []
    for feature in config.signals:      
        X.extend([df_max[feature].values, df_mean[feature].values, df_std[feature].values, df_min[feature].values, df_slope[feature].values])

    X = [np.expand_dims(mat, axis=1) for mat in X]
    X = np.hstack(X)
    y = df_max['tags'].values
    
    return X, y
    
def extractRollingFeatures(df, windowLength): #This function is old and not expected to work
    
    windowLength = str(windowLength)
    
    #Compute features using resampling
    df_mean = df.resample(windowLength+'s',on='time').mean().drop('tags', axis=1)
    df_std = df.resample(windowLength+'s',on='time').std().drop('tags', axis=1)
    df_min = df.resample(windowLength+'s',on='time').min().drop('tags', axis=1)
    df_max = df.resample(windowLength+'s',on='time').max().fillna(0)
    df_slope = df.resample(windowLength+'s', on='time').apply(_gradient_resampler)
    df_slope = df_slope.drop(['time','tags'], axis=1)
    
    #Convert features to numpy matrices
    X, y = _extractFeaturesFromResampledDataFrames((df_max, df_mean, df_std, df_min, df_slope))

    return X, y, df_mean.index

def extractWindowsAroundTags(df, windowLength, extractSegments=False):
    
    secsBefore = windowLength
    secsAfter = 0 #Feature Extraction ends at tag
    bufferTime = config.BUFFER_TIME #buffer after tags where nothing can be sampled
    
    minEventLength = windowLength*0.6

    #Extract windows around each tag
    starts = df[df['tags']==1]['time']-datetime.timedelta(seconds=secsBefore) #get list of starts 
    ends = df[df['tags']==1]['time']+datetime.timedelta(seconds=secsAfter) #get list of ends
    buffers = df[df['tags']==1]['time']+datetime.timedelta(seconds=bufferTime) #get list of buffers
    tags = []
    for start, end, buffer_ in zip(starts, ends, buffers): #for each start, end:
        segment = df[(df['time']>=start) & (df['time']<=end)]
        dur = (segment['time'].iloc[-1]-segment['time'].iloc[0]) #get duration
        if dur.total_seconds() >= minEventLength: #if duration is not too short.
            tags.append(segment) #Take out segment
        df.loc[(df['time']>=start) & (df['time']<=buffer_),'tags'] = 1.0 #mark period as unavaliable
    
    #Extract slices with no tags
    df['group'] = df['tags'].ne(df['tags'].shift()).cumsum() #Make an indicator column which increases when tags change
    df = df.groupby('group') #Group and split by the indicator column
    dfs = []
    for name, data in df: #Loop over all groups        
        if data['tags'].values[0] == 0: #If no tag
            dur = (data['time'].iloc[-1]-data['time'].iloc[0]) #get duration
            if dur.total_seconds() >= (secsBefore+secsAfter): #if duration longer than timewindow
                dfs.append(data) #Save the group

    #Sample windows with no tags
    n = max(len(tags),config.minimumNumberOfnegativeCasesPerDay) #number of samples
    if len(dfs) > 0:
        groups = np.random.choice(np.arange(len(dfs)),n, replace=True) #choose a random group
        maxIdxs = [len(dfs[g][dfs[g]['time'] <= dfs[g]['time'].iloc[-1]-datetime.timedelta(seconds=secsBefore+secsAfter)]) for g in groups] #max possible start index in each group
        no_tags = []
        for g, idx  in zip(groups, maxIdxs):    
            #k = np.random.randint(0, idx) #Get random start index
            #startTime = dfs[g]['time'].iloc[k] #Get the startTime
            #endTime = startTime+datetime.timedelta(seconds=(secsBefore+secsAfter)) #Get the endtime    
            
            #segment = dfs[g][(dfs[g]['time']>=startTime) & (dfs[g]['time']<=endTime)] #Get the segment
            #no_tags.append(segment) #Take out segment
            iter_ = 0
            segmentAdded = False
            while (not segmentAdded) and (iter_<10):
                k = np.random.randint(0, idx) #Get random start index
                startTime = dfs[g]['time'].iloc[k] #Get the startTime
                endTime = startTime+datetime.timedelta(seconds=(secsBefore+secsAfter)) #Get the endtime    
                
                segment = dfs[g][(dfs[g]['time']>=startTime) & (dfs[g]['time']<=endTime)] #Get the segment
                goodSegment = len(segment['HR'].dropna().values)>25 and len(segment['EDA_Phasic'].dropna().values)>25 and len(segment['EDA_Tonic'].dropna().values)>25 and len(segment['EDA'].dropna().values)>25 and len(segment[['BVP','BVP_filtered']].dropna())>25 and len(segment['TEMP'].dropna().values)>25
                if goodSegment:
                    no_tags.append(segment) #Take out segment
                    segmentAdded = True
                else:
                    iter_ += 1
    else:
        no_tags = []

    if extractSegments:
        return tags, no_tags, None

    #Feature extract from the windows generated    
    if len(tags)==0 and len(no_tags)==0:
        return None, None, None
    
    positiveObservations = [_extractFeaturesFromWindow(window) for window in tags]
    negativeObservations = [_extractFeaturesFromWindow(window) for window in no_tags]
    X = np.vstack(positiveObservations+negativeObservations)
    y = np.hstack([np.ones(len(tags)), np.zeros(len(no_tags))])
    startTimes = [window['time'].iloc[0] for window in tags]+[window['time'].iloc[0] for window in no_tags]
    
    return X, y, startTimes 
