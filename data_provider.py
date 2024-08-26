from .data_loader import Synthetic, Gas, Breast, Breast2, PBMC
from torch.utils.data import DataLoader, Dataset
import torch

data_dict = {
    # regression
    'syn': Synthetic, 
    'gas': Gas,
    # classification
    'breast': Breast,
    'breast2': Breast2,
    'pbmc': PBMC
}

class CustomDataset(Dataset):
    def __init__(self, x, y, g, n, c):
        super(CustomDataset, self).__init__()
        self.x = x
        self.y = y
        self.g = g
        self.n = n
        self.c = c

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return x, y
    
def data_provider(args):
    Data = data_dict[args.data]
    if args.data == 'syn':
        d = Data(
            dtype=args.dtype
        )
    elif args.data == 'pbmc':
        d = Data(
            group_type=args.group_type
        )
    else:
        d = Data()

    train_dataset = CustomDataset(d.tr_data, d.tr_y, d.group_sizes, d.num_coeffs, d.coeff)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataset = CustomDataset(d.va_data, d.va_y, d.group_sizes, d.num_coeffs, d.coeff)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = CustomDataset(d.te_data, d.te_y, d.group_sizes, d.num_coeffs, d.coeff)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
