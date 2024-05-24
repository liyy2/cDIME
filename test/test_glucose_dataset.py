
import sys
sys.path.append('/home/yl2428/Time-LLM/')
from data_provider_pretrain.data_loader import Dataset_Combined    


data_set = Dataset_Combined(root_path='/home/yl2428/Time-LLM/dataset/glucose', 
                            flag='train', 
                            data_path= 'combined_data.csv',
                            target='Glucose', 
                            size=[48, 12, 12], normalization='global', features='M', enable_covariates=True, stride=24)

print(data_set[0])
print(len(data_set))

# test data_loader
from torch.utils.data import DataLoader
data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True)

next(iter(data_loader))
for _ in data_loader:
        print('next batch')