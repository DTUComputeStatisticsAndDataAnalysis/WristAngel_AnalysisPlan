import numpy as np
import pandas as pd
import neurokit2 as nk
import csv
import datetime
import os

from tqdm import tqdm
from functools import reduce

from utils import extractFeatures
from Config import config

def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.datetime.strptime(d2, "%Y%m%d")
    return abs((d2 - d1).days)

def _read_rawRecords(folder, signal):
    
    signalFile = signal + '.csv'
    fileName = os.path.join(folder, signalFile)

    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        epoch = int(float(next(reader)[0]))
        sampleRate = int(float(next(reader)[0]))

    df = pd.read_csv(fileName, header=1)

    begin_time = datetime.datetime.utcfromtimestamp(epoch)
    try:
        timestep = 1/sampleRate
    except: #This catch might be needed for some files where samplerate is corrupted and set to 0.
        if signal=='BVP':
            timestep = 1/64.0
        elif signal=='EDA':
            timestep = 1/4.0
        elif signal=='HR':
            timestep = 1/1.0
        elif signal=='HR':
            timestep = 1/4.0
        elif signal=='ACC':
            timestep = 1/32.0

    timestep = np.arange(len(df))*timestep

    timestamps = [begin_time + datetime.timedelta(seconds=step) for step in timestep]
    if signal=='ACC':
        df = pd.DataFrame({
            'time': timestamps, 
            'ACC_X': list(df.iloc[:,0]),
            'ACC_Y': list(df.iloc[:,1]),
            'ACC_Z': list(df.iloc[:,2])
        })
    else:
        df = pd.DataFrame({
            'time': timestamps, 
            signal: list(df.iloc[:,0])
        })
    
    return df

def _preprocSignal(df, signal):

    if signal=='TEMP':
        df[signal] = nk.signal_filter(df[signal], sampling_rate=4, highcut=1, method="butterworth", order=6)
    elif signal=='BVP':
        df['BVP_filtered'] = nk.signal_filter(df[signal], sampling_rate=64, highcut=8, lowcut=0.5, method="butterworth", order=3)
    elif signal=='EDA':
        df[signal] = nk.signal_filter(df[signal], sampling_rate=4, highcut=1, method="butterworth", order=6)
                
        #Perform phasic/tonic decomposition on normalized signal
        min_ = np.min(df['EDA'])
        max_ = np.max(df['EDA'])
    
        eda_decomp = nk.eda_phasic((df['EDA']-min_)/(max_-min_), sampling_rate=4, method="cvxEDA")  
        df["EDA_Tonic"] = eda_decomp["EDA_Tonic"].to_numpy()
        df["EDA_Phasic"] = eda_decomp["EDA_Phasic"].to_numpy()
            
    return df

def _read_WristAngelTags(folders):

    dataframes = []
    for folder in folders:
        df = _read_rawRecords_Tags(folder) 

        dataframes.append(df)

    #Merge dataframes into df
    df = reduce(lambda df1,df2: pd.merge(df1,df2, how='outer', on=['time','tags']), dataframes)
    df = df.sort_values(by='time').reset_index(drop=True)

    return df

def _read_WristAngelRecords(folders, signal):

    dataframes = []
    for folder in folders:
        df = _read_rawRecords(folder, signal)

        if len(df)<config.minSignalLengthForPreproc:
            continue

        #Preproces the signal
        df = _preprocSignal(df, signal) 

        dataframes.append(df)

    if signal=='EDA' and len(dataframes)>0:
        #df = dataframes[0].merge(right=dataframes[1], how='outer', on=['time', signal,'EDA_Tonic', 'EDA_Phasic']).merge(right=dataframes[2], how='outer', on=['time', signal,'EDA_Tonic', 'EDA_Phasic']).sort_values(by='time').reset_index(drop=True)
        df = reduce(lambda df1,df2: pd.merge(df1,df2, how='outer', on=['time', signal,'EDA_Tonic', 'EDA_Phasic']), dataframes)
        df = df.sort_values(by='time').reset_index(drop=True)
    elif signal=='BVP' and len(dataframes)>0:
        #df = dataframes[0].merge(right=dataframes[1], how='outer', on=['time', signal,'BVP_filtered']).merge(right=dataframes[2], how='outer', on=['time', signal,'BVP_filtered']).sort_values(by='time').reset_index(drop=True)
        df = reduce(lambda df1,df2: pd.merge(df1,df2, how='outer', on=['time', signal, 'BVP_filtered']), dataframes)
        df = df.sort_values(by='time').reset_index(drop=True)
    elif len(dataframes)>0:
        #df = dataframes[0].merge(right=dataframes[1], how='outer', on=['time', signal]).merge(right=dataframes[2], how='outer', on=['time', signal]).sort_values(by='time').reset_index(drop=True)
        df = reduce(lambda df1,df2: pd.merge(df1,df2, how='outer', on=['time', signal]), dataframes)
        df = df.sort_values(by='time').reset_index(drop=True)

    return df    

def _filterWristAngelData(df, folder):

    #Remove known sleep periods
    if folder in config.knownSleepPeriods.keys():
        filtertimes = config.knownSleepPeriods[folder]
        for times in filtertimes:
            filter_start = times[0]
            filter_end = times[1]
            df = df[(df['time'] <= filter_start) | (df['time'] >= filter_end)]
    
    #Remove periods where wristband is not worn. For this we set a threshold of 25 degrees on the temperature and 0.01 on EDA
    #First we have to extract only temp and eda and get a df_filter
    try:
        df_filter = df[(df['TEMP']>25) or (df['EDA']>0.01)]
    except:
        df_filter = df[df['TEMP']>25]

    timeDiff = df_filter['time'].diff() #Split df when there is 5 minutes between consecutive timestamps.
    df_filter['group'] = (timeDiff > pd.Timedelta(minutes=5)).cumsum() #Compute group index based on df_filter

    df = df.merge(right=df_filter[['time', 'group']], how='outer', on=['time'])#Merge group index back onto original df
    df['group'] = df['group'].fillna(method="ffill") #remove group index nans by forward fill

    dfs = [group for _, group in df.groupby('group')]

    return dfs
    
def readData(samplingMethod, windowLength, group='all'):

    if group=='patients':
        ids = config.patientIDs
    elif group=='controls':
        ids = config.controlIDs
    elif group=='all':
        ids = config.IDs

    folders = [os.path.join(config.wristAngelRootFolder, 'ID'+str(id_)) for id_ in ids]

    X = []
    y = []
    days = []
    participants = []
    startTimes = []
    removedData = []
    for i, subjectfolder in enumerate(tqdm(folders)):
        id_ = ids[i]

        X_tmp, y_tmp, days_tmp, startTimes_tmp, removedData_tmp = readWristAngelParticipant(subjectfolder, samplingMethod, windowLength, id_)
        
        if X_tmp is not None:
            X.append(X_tmp)
            y.append(y_tmp)
            days.append(days_tmp)
            participants.append(np.ones(len(y_tmp))*id_)
            startTimes.extend(startTimes_tmp)
            removedData.extend(removedData_tmp)
    
    #Merge subjects
    X = np.vstack(X)
    y = np.hstack(y)
    days = np.hstack(days)   
    participants = np.hstack(participants)  
    
    if True: #remove observation which overlap
        df = pd.DataFrame(data={'ID': participants, 'startTime': startTimes, 'y': y}).sort_values(['ID','startTime'])
        save_idx = []
        for index, row in df.iterrows():
            tmp_par = df['ID'][save_idx[-1]] if len(save_idx)>0 else None
            tmp_start = df['startTime'][save_idx[-1]] if len(save_idx)>0 else None
                
            if len(save_idx)==0 and row['y']==1:
                save_idx.append(index)
            
            if tmp_start is not None:
                if row['y']==1 and row['ID']!=tmp_par:
                    save_idx.append(index)
                elif row['y']==1 and ((row['startTime']-tmp_start).total_seconds()>300):
                    save_idx.append(index)
        
        #Add negative cases
        save_idx.extend(df.index[df['y']==0].tolist())

        save_idx = sorted(save_idx)
        
        X = X[save_idx,:]
        y = y[save_idx]
        days = days[save_idx]
        participants = participants[save_idx]
        startTimes = startTimes[save_idx]

    #Anonymize participants
    participants = np.concatenate([np.where(np.array(ids)==p)[0] for p in participants])

    #Anonymize dates
    startTimes = [s.time() for s in startTimes]

    return X, y, days, participants, startTimes, removedData

def readWristAngelParticipant(subjectfolder, samplingMethod, windowLength, id_, extractSegments=False):
    
    #make get datefolders
    datastorePath = ''
    participantFolder = os.path.join(subjectfolder, datastorePath)
    dateFolders = os.listdir(participantFolder)
    dateFolders = [os.path.join(participantFolder,d) for d in dateFolders]
    dateFolders = [d for d in dateFolders if os.path.isdir(d)]  
    dateFolders = sorted([d.split('\\')[-1] for d in dateFolders])
    
    if len(dateFolders)==0: #If participant has no data return
        return None, None, None, None, None, None

    dateDict = _makeDateDict(participantFolder)
    dateKeys = sorted(dateDict.keys())
    startDate = dateKeys[0]

    X = []
    y = []
    days = []
    startTimes = []
    for i, folder in enumerate(tqdm(dateFolders)):

        #first continue if it is a folder with less than 2 splits. some participants has extra folders not following the naming scheme
        if len(folder.split('_'))<2:
            continue
        
        date = folderNameToDate(folder, participantFolder)

        try:
            folders = [os.path.join(participantFolder, folder)]
        
            df_temp = _read_WristAngelRecords(folders, 'TEMP')
            df_hr = _read_WristAngelRecords(folders, 'HR')        
            df_bvp = _read_WristAngelRecords(folders, 'BVP')
            df_eda = _read_WristAngelRecords(folders, 'EDA')
            df_tags = _read_WristAngelTags(folders)
        except:
            print(f'Error occured for id {id_} for {date} in {folder} while reading the data. Continue to next day')
            continue
        
        try:
            df = df_temp.merge(right=df_hr, how='outer', on='time').merge(right=df_eda, how='outer', on='time').merge(right=df_bvp, how='outer', on='time').merge(right=df_tags, how='outer', on='time').sort_values(by='time').reset_index(drop=True)
            df['tags'] = df['tags'].fillna(value=0)
        except:
            print(f'Error occured for id {id_} for {date} in {folder} while merging the data. Continue to next day.')
            continue

        #Filter dataframe and return list of dataframes
        dfs = _filterWristAngelData(df, folder)

        for df in dfs:
            if samplingMethod=='rollingWindows':
                X_tmp, y_tmp, = extractFeatures.extractRollingFeatures(df, windowLength)  #This function is old and not expected to work
            elif samplingMethod=='extractTags':
                X_tmp, y_tmp, startTimes_tmp = extractFeatures.extractWindowsAroundTags(df, windowLength, extractSegments=extractSegments)         

            if extractSegments:
                d = days_between(startDate, date)
                X.extend(X_tmp)
                y.extend(y_tmp)
                days.extend([d]*len(y_tmp))
            elif startTimes_tmp is not None:
                d = days_between(startDate, date)
                X.append(X_tmp)
                y.append(y_tmp)
                days.extend([d]*len(y_tmp))
                startTimes.extend(startTimes_tmp)

    if extractSegments:
        return X, y, days, None, None

    # Merge X and y for each folder.
    X = np.vstack(X)
    y = np.hstack(y)
    days = np.array(days)    
    startTimes = np.array(startTimes)
        
    #Drop rows where X contains nans. This may happen when bracelet is turned off between phases or if window only contains one second or if no BVP features can be extracted
    nanFilter_ = np.isfinite(X).all(axis=1)
    
    removedData = []
    for idx in np.where(~nanFilter_)[0]:
        removedData.append( (X[idx,:],y[idx],id_,startTimes[idx]) )

    X = X[nanFilter_, :]
    y = y[nanFilter_]
    days = days[nanFilter_]
    startTimes = startTimes[nanFilter_]
        
    return X, y, days, startTimes, removedData

def folderNameToDate(dateFolder, participantFolder):

    folderNameContents = dateFolder.split('_')
    
    date = None
    try:
        if len(folderNameContents[-1])==8:
            date=folderNameContents[-1]
        if len(folderNameContents[-2])==8:
            if len(folderNameContents[-2].split('.'))==3: #format DD.MM.YY
                dateContent = folderNameContents[-2].split('.')
                date = '20' + dateContent[2] + dateContent[1] + dateContent[0]
            else: #format YYYYMMDD
                date = folderNameContents[-2]
        
        if date is None:
            print(f'Error for folder {dateFolder} in {participantFolder}')
            assert False
    except:
        print(folderNameContents)
        assert False

    return date

def _makeDateDict(participantFolder):
    #make a dict with keys=dates and values=fileNames of that date
    dateFolders = os.listdir(participantFolder)
    dateFolders = [os.path.join(participantFolder,d) for d in dateFolders]
    dateFolders = [d for d in dateFolders if os.path.isdir(d)]  
    dateFolders = [d.split('\\')[-1] for d in dateFolders]

    dateDict = {}
    for dateFolder in dateFolders:
        
        folderNameContents = dateFolder.split('_')

        #first continue if it is a .csv or .txt file or a folder with less than 2 splits
        if len(folderNameContents)<2:
            continue
        
        isZip = folderNameContents[-1][-4:] == '.zip'
        date = None

        try:
            if isZip and (len(folderNameContents[-1])==12):
                date=folderNameContents[-1][:8]
            if (not isZip) and (len(folderNameContents[-1])==8):
                date=folderNameContents[-1]
            if (len(folderNameContents[-2])==8):
                if len(folderNameContents[-2].split('.'))==3: #format DD.MM.YY
                    dateContent = folderNameContents[-2].split('.')
                    date = '20' + dateContent[2] + dateContent[1] + dateContent[0]
                else: #format YYYYMMDD
                    date = folderNameContents[-2]
            
            if date is None:
                print(f'Error for folder {dateFolder} in {participantFolder}')
                assert False
        except:
            print(folderNameContents)
            assert False

        if date in dateDict:
            dateDict[date].append(dateFolder)
        else:
            dateDict[date] = [dateFolder]

    return dateDict

def _read_rawRecords_Tags(folder):
    
    try:
        df = pd.read_csv(os.path.join(folder,'tags.csv'), header=None)
        df['time'] = df.apply(lambda row: datetime.datetime.utcfromtimestamp(float(row)), axis=1)
        df = df.drop(0, axis=1)
        df['tags'] = 1
    except:
        df = pd.DataFrame(columns=['time','tags'])
    
    return df