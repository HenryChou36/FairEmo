import joblib
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd

class MSP_Dataset(Dataset):
    ''' MSP datset with vec or hubert as feature and either one of A, V, D or all of them as target.
        Remove speakers with less than 10 utterances per speaker from dataset.
    '''
    def __init__(self, csv_file, target_name = 'A', feature_name = 'hubert', protect = 'gender', 
                 min_sample=None, ignore_unknown=False, pseudo_label=None):
        ''' initiailze dataset with csv_file and specified target, features and protected group.'''
        
        features = pd.read_csv(csv_file, index_col=0)

        if min_sample is not None:
            val, cnt = np.unique(features['spk_id'], return_counts=True)
            val = val[cnt >= min_sample]
            features = features[features['spk_id'].isin(val)]
        if ignore_unknown:
            features = features[features['spk_id'] != 'Unknown']
        
        if feature_name == 'hubert':
            self._X = features['hubert_path'].copy().to_list()
            self._length = features['hubert_len'].copy().to_list()
        else:
            self._X = features['vec_path'].copy().to_list()
            self._length = features['vec_len'].copy().to_list()

        assert target_name in ['A', 'V', 'D', 'All'], "Unrecognized target name."
        targets = features[['A', 'V', 'D']]
        if target_name == 'All':
            targets = targets.values
        else:
            targets = np.expand_dims(targets[target_name].values, -1)
        # rescale to [-1, 1]
        self._Y = ((targets-4) / 3).tolist()
        assert len(self._X) == len(self._Y)
        self._seq_id = list(range(len(self._X)))

        self._gender = []
        for gender in features['gender'].copy().to_list():
            if gender == 'Unknown':
                self._gender.append(0)
            elif gender == 'Male':
                self._gender.append(1)
            elif gender == 'Female':
                self._gender.append(2)

        self._spk_id = []
        for spk_id in features['spk_id'].copy().to_list():
            if spk_id == 'Unknown':
                self._spk_id.append(0)
            else:
                self._spk_id.append(int(spk_id))

        if pseudo_label is not None:
            self._cluster_id = features[pseudo_label].astype('int64').copy().to_list()
        else:
            self._cluster_id = self._spk_id
        # protected attributes for fairness evaluation
        if protect == 'gender':
            self._protect = self._gender
        elif protect == 'spk_id':
            self._protect = self._spk_id
        elif protect == 'cluster_id':
            self._protect = self._cluster_id
            
    def __len__(self):
        return len(self._X)
     
    def __getitem__(self, idx):
        if self._pre_fetch:
            X = self._features[idx]
        else:
            X = joblib.load(self._X[idx])
        Y = self._Y[idx]
        length = self._length[idx]
        seq_id = self._seq_id[idx]
        protect = self._protect[idx]
        
        return X, Y, length, seq_id, protect

    def getSPK_ID(self):
        return self._spk_id
    def getGender(self):
        return self._gender
    def getCLU_ID(self):
        return self._cluster_id
    def getLength(self):
        return self._length
    def getValues(self):
        return np.array(self._Y)
    def getProtect(self):
        return self._protect
    
class IEMOCAP_Dataset(Dataset):
    ''' MSP datset with vec or hubert as feature and either one of A, V, D or all of them as target.
        Remove speakers with less than 10 utterances per speaker from dataset.
    '''
    def __init__(self, csv_file, target_name = 'A', feature_name = 'hubert', protect = 'gender', 
                 min_sample=None, ignore_unknown=False, pseudo_label=None):
        ''' initiailze dataset with csv_file and specified target, features and protected group.'''
        
        features = pd.read_csv(csv_file, index_col=0)

        if min_sample is not None:
            val, cnt = np.unique(features['spk_id'], return_counts=True)
            val = val[cnt >= min_sample]
            features = features[features['spk_id'].isin(val)]
        if ignore_unknown:
            features = features[features['spk_id'] != 'Unknown']
        
        if feature_name == 'hubert':
            self._X = features['hubert_path'].copy().to_list()
            self._length = features['hubert_len'].copy().to_list()
        else:
            self._X = features['vec_path'].copy().to_list()
            self._length = features['vec_len'].copy().to_list()

        assert target_name in ['A', 'V', 'D', 'All'], "Unrecognized target name."
        targets = features[['A', 'V', 'D']]
        if target_name == 'All':
            targets = targets.values
        else:
            targets = np.expand_dims(targets[target_name].values, -1)
        # rescale to [-1, 1]
        self._Y = ((targets-3) / 2).tolist()
        assert len(self._X) == len(self._Y)
        self._seq_id = list(range(len(self._X)))

        self._gender = []
        for gender in features['gender'].copy().to_list():
            if gender == 'Unknown':
                self._gender.append(0)
            elif gender == 'Male':
                self._gender.append(1)
            elif gender == 'Female':
                self._gender.append(2)

        self._spk_id = []
        for spk_id in features['spk_id'].copy().to_list():
            if spk_id == 'Unknown':
                self._spk_id.append(0)
            else:
                self._spk_id.append(int(spk_id))

        if pseudo_label is not None:
            self._cluster_id = features[pseudo_label].astype('int64').copy().to_list()
        else:
            self._cluster_id = self._spk_id
        # protected attributes for fairness evaluation
        if protect == 'gender':
            self._protect = self._gender
        elif protect == 'spk_id':
            self._protect = self._spk_id
        elif protect == 'cluster_id':
            self._protect = self._cluster_id
            
    def __len__(self):
        return len(self._X)
     
    def __getitem__(self, idx):
        if self._pre_fetch:
            X = self._features[idx]
        else:
            X = joblib.load(self._X[idx])
        Y = self._Y[idx]
        length = self._length[idx]
        seq_id = self._seq_id[idx]
        protect = self._protect[idx]
        
        return X, Y, length, seq_id, protect

    def getSPK_ID(self):
        return self._spk_id
    def getGender(self):
        return self._gender
    def getCLU_ID(self):
        return self._cluster_id
    def getLength(self):
        return self._length
    def getValues(self):
        return np.array(self._Y)
    def getProtect(self):
        return self._protect

def seq_collate_pad_zero(batch):
    X_list = [torch.FloatTensor(item[0]) for item in batch]
    X_list_pad = pad_sequence(X_list, batch_first = True)
    Y_list = torch.FloatTensor([item[1] for item in batch])
    seq_lengths = torch.LongTensor([item[2] for item in batch])
    utt_ids = torch.LongTensor([item[3] for item in batch])
    # Assume protected attributed is discrete for now
    protect = torch.LongTensor([item[4] for item in batch])
    
    return X_list_pad, Y_list, seq_lengths, utt_ids, protect
