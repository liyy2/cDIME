import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
import torch_frame
from torch_frame.data import Dataset
warnings.filterwarnings('ignore')
import tqdm



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
    def __init__(self, df, individual_id, flag='train', size=None, stride=1,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=1, freq='t', train_percent=100, val_percent=0,
                 seasonal_patterns=None, time_column='DateTime', covariates=None, cov_index=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Initialize other parameters
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

        self.data_path = data_path
        self.time_column = time_column
        self.stride = stride

        self.__read_data__(df)
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1
        self.covariates = covariates
        self.cov_index = cov_index

    def __read_data__(self, df):
        self.scaler = StandardScaler()
        df_raw = df

        # Drop the individual ID column
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
            border2 = (border2 - self.seq_len) + self.seq_len

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

        df_stamp = df_raw[[self.time_column]][border1:border2]
        df_stamp[self.time_column] = pd.to_datetime(df_stamp[self.time_column])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_column].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_column].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_column].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_column].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_column], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_column].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Combined(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Glucose.csv',
                 target='OT', scale=True, timeenc=1, freq='t', train_percent=70, val_percent = 20, 
                 partition = 'chronological', normalization = 'global',
                 seasonal_patterns=None, 
                 gap_tolerance = '5 minute', 
                 time_column = 'DateTime', 
                 enable_covariates = False, 
                 cov_path = 'final_dm.csv', num_individuals = -1, stride = 1, cov_type = 'tensor'):
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
        self.time_column = time_column
        self.enable_covariates = enable_covariates
        self.cov_path = cov_path

        self.num_individuals = num_individuals
        self.stride = stride
        self.cov_type = cov_type

        if self.enable_covariates:
            self.covariates = pd.read_csv(os.path.join(self.root_path, self.cov_path))
            self.covariates_preprocess()
        if partition == 'individual':
        # Partition the individual IDs
            train_ids, val_ids, test_ids = self.__partition_individuals(train_percent, val_percent)
            self.ids = train_ids if flag == 'train' else val_ids if flag == 'val' else test_ids
        elif partition == 'chronological':
            self.ids = pd.read_csv(os.path.join(self.root_path, self.data_path))['USUBJID'].unique()
        else:
            raise ValueError('Invalid partition type: {}'.format(partition))
        

        self.__read_data__()
    
    def covariates_preprocess(self):
        assert self.enable_covariates, 'Covariates are not enabled'
        # Specify the stype of each column with a dictionary.
        col_to_stype = {
            "SEX": torch_frame.categorical, 
            "RACE": torch_frame.categorical,
            "ETHNIC": torch_frame.categorical, 
            "ARMCD": torch_frame.categorical,
            "insulin modality": torch_frame.categorical,
            "AGE": torch_frame.numerical, 
            "WEIGHT": torch_frame.numerical, 
            "HEIGHT": torch_frame.numerical,
            "HbA1c": torch_frame.numerical, 
            "DIABETES_ONSET": torch_frame.numerical,
        }
        # Set "target" as the target column.
        dataset = Dataset(self.covariates, col_to_stype=col_to_stype)
        dataset.materialize() # tensorize the data
        self.processed_covariates = dataset


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
        # sanity checks
        assert 'USUBJID' in df_raw.columns, 'USUBJID column not found in the dataset'
        assert self.target in df_raw.columns, 'Target column not found in the dataset'
        assert self.time_column in df_raw.columns, 'Time column not found in the dataset'
        # reorder columns into Time, USUBJID, features_column, target_column
        df_raw = df_raw[[self.time_column, 'USUBJID'] + [col for col in df_raw.columns if col not in [self.time_column,'USUBJID', self.target]] + [self.target]]
        if self.normalization == 'global' and self.scale: #TODO: Global normalization is not correct here as it is taking into account the test set
            self.scaler = StandardScaler()
            to_be_scaled = df_raw[[self.target]] if self.features == 'S' else df_raw.iloc[:,2:]
            new_data = self.scaler.fit_transform(to_be_scaled)
            # print the mean and std of the target column
            print(f'Target column mean: {self.scaler.mean_[0]}, std: {self.scaler.scale_[0]}')
            if self.features == 'S':
                df_raw[self.target] = new_data
            else:
                df_raw.iloc[:, 2:] = new_data
        
        if self.num_individuals >= 0: # -1 means all individuals
            self.ids = self.ids[:self.num_individuals]

        print('Loading data into memory...')
        for individual_id in tqdm.tqdm(self.ids):
            df_per_indiv = df_raw[df_raw['USUBJID'] == individual_id]
            covariates = self.covariates[self.covariates['USUBJID'] == individual_id] if self.enable_covariates else None
            # turn covariates into a dictionary
            covariates = covariates.to_dict(orient='records')[0] if covariates is not None else None
            # get the exact index
            idx = self.covariates[self.covariates['USUBJID'] == individual_id].index[0] if self.enable_covariates else None
            cov_tensor_frame = self.processed_covariates[idx].tensor_frame if self.enable_covariates else None
            # covarites prompt
            if covariates is not None:
                covariates['cov_str'] = f"This is an individual with type I diabetes. Here is the individual's basic information:\n" + str(covariates)

            # turn the time column into a datetime object
            df_per_indiv[self.time_column] = pd.to_datetime(df_per_indiv[self.time_column])
            time_diff = df_per_indiv[self.time_column].diff() > pd.Timedelta(self.gap_tolerance)
            groups = time_diff.cumsum()
            list_of_dfs = [group_df for _, group_df in df_per_indiv.groupby(groups)]
            # filter out the groups that are too short
            list_of_dfs = [group_df for group_df in list_of_dfs if len(group_df) > 2 * (self.seq_len + self.pred_len)]


            # Create a dataset for each split DataFrame
            for df_split in list_of_dfs:
                if self.partition == 'individual':
                    self.datasets.append(DatasetPerIndividual(df_split, individual_id, 'train', self.size, self.stride, self.features, self.data_path,
                                                            self.target, self.scale if self.normalization =='individual' else False, self.timeenc,
                                                            self.freq, 100, 0, time_column=self.time_column, covariates = covariates, cov_index = idx))
                elif self.partition == 'chronological':
                    self.datasets.append(DatasetPerIndividual(df_split, individual_id, self.flag, self.size, self.stride, self.features, self.data_path,
                                                            self.target, self.scale if self.normalization =='individual' else False, self.timeenc,
                                                            self.freq, self.train_percent, self.val_percent, time_column=self.time_column, covariates = covariates, cov_index = idx))
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
        if self.enable_covariates and self.cov_type == 'text':
            return self.datasets[dataset_idx].__getitem__(dataset_index), self.datasets[dataset_idx].covariates
        elif self.enable_covariates and self.cov_type == 'tensor':
            return self.datasets[dataset_idx].__getitem__(dataset_index), self.datasets[dataset_idx].cov_index
        else:
            # get the item from the dataset
            return self.datasets[dataset_idx].__getitem__(dataset_index)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
