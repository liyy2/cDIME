from torch.utils.data import DataLoader

from data_provider_pretrain.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Combined
import torch
from torch_frame.utils import cat
from torch.utils.data.dataloader import default_collate
import numpy as np
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Glucose': Dataset_Combined,
}

def __build_collate_fn__(cov_frame):
    def collate_fn(batch):
        time_series, idx = default_collate(batch)
        covariates = cov_frame[idx]
        return time_series, covariates
    return collate_fn

def data_provider(args, data, data_path, pretrain=True, flag='train'):
    Data = data_dict[data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    if flag == 'test':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if data == 'Glucose':
        data_set = Dataset_Combined(root_path=args.root_path, 
                            flag=flag, 
                            data_path= args.data_path,
                            target='Glucose', 
                            size=[args.seq_len, args.label_len, args.pred_len], 
                            normalization='global', 
                            freq=args.freq,
                            features=args.features, 
                            enable_covariates=args.enable_covariates,
                            num_individuals=args.num_individuals, stride=args.stride)

    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns,
            pretrain=pretrain
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size // args.num_nodes,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last, 
        collate_fn=__build_collate_fn__(data_set.processed_covariates.tensor_frame) if (args.enable_covariates and args.cov_type =='tensor') else None)
    if args.enable_covariates and args.cov_type == 'tensor':
        args.col_names_dict = data_set.processed_covariates.tensor_frame.col_names_dict
        args.col_stats = data_set.processed_covariates.col_stats
    return data_set, data_loader, args
