import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None, pretrain=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.pretrain:
            # border1s = [0, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            # border2s = [12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1s = [0, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None, pretrain=True):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.pretrain:
            # border1s = [0, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            #             12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            # border2s = [12 * 30 * 24 * 4 + 8 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            #             12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1s = [0, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
                        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                        12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DatasetPerIndividual(Dataset):
    def __init__(self, df, individual_id, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_percent=100, val_percent = 0,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.individual_id = individual_id
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_percent = train_percent
        self.val_percent = val_percent

        self.df = df
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.df

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # drop the individual ID column
        df_raw = df_raw.drop('USUBJID', axis=1)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        df_raw = df_raw[cols + [self.target]]
        num_train = int(len(df_raw) * self.train_percent // 100)
        num_test = int(len(df_raw) * self.val_percent // 100)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len)+ self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['LBDTC']][border1:border2]
        df_stamp['LBDTC'] = pd.to_datetime(df_stamp['LBDTC'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['LBDTC'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['LBDTC'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['LBDTC'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['LBDTC'].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['LBDTC'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['LBDTC'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Combined(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Glucose.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_percent=70, val_percent = 20, partition = 'chronological', normalization = 'global',
                 seasonal_patterns=None, gap_tolerance = '1 hour'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.gap_tolerance = gap_tolerance
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.size = size
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.datasets = []
        self.root_path = root_path
        self.data_path = data_path
        self.partition = partition
        self.normalization = normalization
        if partition == 'individual':
        # Partition the individual IDs
            train_ids, val_ids, test_ids = self.__partition_individuals(train_percent, val_percent)
            self.ids = train_ids if flag == 'train' else val_ids if flag == 'val' else test_ids
        elif partition == 'chronological':
            self.ids = pd.read_csv(os.path.join(self.root_path, self.data_path))['USUBJID'].unique()
        else:
            raise ValueError('Invalid partition type: {}'.format(partition))
        

        self.__read_data__()
    

    def __partition_individuals(self, train_percent, val_percent):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        individual_ids = df_raw['USUBJID'].unique()
        
        # Calculate validation and test set percentages
        val_percent_adj = val_percent / (100 - train_percent) * 100 if train_percent < 100 else val_percent
        test_percent_adj = 100 - val_percent_adj
        
        # Split the unique IDs into train and temp sets
        train_ids, temp_ids = train_test_split(individual_ids, train_size=train_percent/100, random_state=42)
        
        # Split the temp set into validation and test sets
        val_ids, test_ids = train_test_split(temp_ids, test_size=test_percent_adj/100, random_state=42)
        
        return train_ids, val_ids, test_ids
    
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # reorder columns
        df_raw = df_raw[['LBDTC', 'USUBJID'] + [col for col in df_raw.columns if col not in ['LBDTC','USUBJID']]]
        if self.normalization == 'global' and self.scale: #TODO: Global normalization is not correct here as it is taking into account the test set
            self.scaler = StandardScaler()
            to_be_scaled = df_raw[[self.target]] if self.features == 'S' else df_raw.iloc[:,2:]
            new_data = self.scaler.fit_transform(to_be_scaled)
            df_raw.iloc[:, 2:] = new_data
        

        for individual_id in self.ids:
            df_per_indiv = df_raw[df_raw['USUBJID'] == individual_id]
            # turn the time column into a datetime object
            df_per_indiv['LBDTC'] = pd.to_datetime(df_per_indiv['LBDTC'])
            time_diff = df_per_indiv['LBDTC'].diff() > pd.Timedelta(self.gap_tolerance)
            groups = time_diff.cumsum()
            list_of_dfs = [group_df for _, group_df in df_per_indiv.groupby(groups)]
            # filter out the groups that are too short
            list_of_dfs = [group_df for group_df in list_of_dfs if len(group_df) > 2 * (self.seq_len + self.pred_len)]


            # Create a dataset for each split DataFrame
            for df_split in list_of_dfs:
                if self.partition == 'individual':
                    self.datasets.append(DatasetPerIndividual(df_split, individual_id, 'train', self.size, self.features, self.data_path,
                                                             self.target, self.scale if self.normalization =='individual' else False, self.timeenc,
                                                             self.freq, 100, 0))
                elif self.partition == 'chronological':
                    self.datasets.append(DatasetPerIndividual(df_split, individual_id, self.flag, self.size, self.features, self.data_path,
                                                             self.target, self.scale if self.normalization =='individual' else False, self.timeenc,
                                                             self.freq, self.train_percent, self.val_percent))
                else:
                    raise ValueError('Invalid partition type: {}'.format(self.partition))
        
    def __len__(self):
        # sum the lengths of all datasets
        return sum([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, index):
        # cumsum the lengths of all datasets
        dataset_cumsum = np.cumsum([0] + [len(dataset) for dataset in self.datasets])
        # find the dataset index
        dataset_idx = np.where(index >= dataset_cumsum)[0][-1]
        # find the index within the dataset
        dataset_index = index - dataset_cumsum[dataset_idx]
        # get the item from the dataset
        return self.datasets[dataset_idx].__getitem__(dataset_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
