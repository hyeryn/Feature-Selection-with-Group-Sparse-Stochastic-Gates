import os
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import scvi
from itertools import groupby
import anndata


# --- for reproduction --- #
fix_seed = 2024
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
np.random.seed(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(fix_seed)


def generate(dtype):
    if dtype == 'A':
        '''data generation 1'''
        # set dataset parameters
        n_group = 50
        group_sizes = [np.random.randint(10, 20) for i in range(n_group)]

        # --- custom group --- #
        active_groups = [np.random.randint(2) for _ in group_sizes]     # binary group dependency
        groups = np.concatenate([size * [i] for i, size in enumerate(group_sizes)])
        num_coeffs = sum(group_sizes)
        num_datapoints = 10000
        noise_std = 2

        # generate data matrix
        X = np.random.standard_normal((num_datapoints, num_coeffs))     # (num_datapoints, num_coeffs)

        # generate coefficients
        w = np.concatenate(
            [
                np.random.standard_normal(group_size) * is_active
                for group_size, is_active in zip(group_sizes, active_groups)
            ]
        )
        w = w.reshape(-1, 1)
        intercept = 2

        # generate regression targets
        y_true = X @ w + intercept
        y = y_true + np.random.randn(*y_true.shape) * noise_std

        # for coefficient visualization
        coeff = []
        for i in range(n_group):
            coeff.append([active_groups[i]] * group_sizes[i])  # group 
        c = np.array(sum(coeff, []))

        return group_sizes, num_coeffs, X, y, c
    
    elif dtype == 'B':
        '''data generation 2'''
        '''hard sparsity'''
        num_datapoints = 10000          # number of data
        num_coeffs = 100                # feature
        n_group = 10                    # group
        group_sizes = [n_group for i in range(n_group)]

        X = np.random.standard_normal((num_datapoints, num_coeffs))
        active_group = [1] * 4 + [0] * 6
        random.shuffle(active_group)
        active_groups = []
        for n in range(len(active_group)):
            active_groups.append([active_group[n]]*10)

        '''more sparsity option'''
        t = np.round(np.random.uniform(1,5,(4)))
        k = 0
        for i in range(len(active_group)):
            if active_groups[i][0] == 1:
                num = 10-int(t[k])
                active_groups[i][0:num] = [0] * num  # nonzero
                random.shuffle(active_groups[i])
                k += 1

        '''y = wx + z'''
        w = np.concatenate(
            [
                np.random.standard_normal(group_size) * is_active
                for group_size, is_active in zip(group_sizes, active_groups)
            ]
        )
        z = np.random.normal(0,0.5,1)
        y = X@w + z
        y = y.reshape(y.shape[0], 1)

        return group_sizes, num_coeffs, X, y, np.array(active_groups)
    
    elif dtype == 'C':
        '''data generation 3'''
        '''low individual coefficient'''
        num_datapoints = 10000 # 5000
        num_coeffs = 120
        n_group = 6
        group_sizes = [20 for _ in range(n_group)]

        x = np.random.standard_normal((num_datapoints, num_coeffs))
        active_group = [1] * 1 + [0.2] * 2 + [0] * 3
        random.shuffle(active_group)
        active_groups = []
        for n in range(len(active_group)):
            active_groups.append([active_group[n]]*20)

        '''y = wx + z'''
        w = np.concatenate(
            [
                np.random.standard_normal(group_size) * is_active
                for group_size, is_active in zip(group_sizes, active_groups)
            ]
        )
        z = np.random.normal(0,0.5,1)
        y = x@w + z
        y = y.reshape(y.shape[0], 1)

        return group_sizes, num_coeffs, x, y, np.array(active_groups)

    elif dtype == 'D':
        '''data generation 4'''
        '''group-in coefficient different'''
        num_datapoints = 10000 # 5000
        num_coeffs = 120
        n_group = 6
        group_sizes = [20 for _ in range(n_group)]

        x = np.random.standard_normal((num_datapoints, num_coeffs))
        active_group = [1] * 2 + [0.5] * 1 + [0] * 3
        random.shuffle(active_group)
        active_groups = []
        for n in range(len(active_group)):
            active_groups.append([active_group[n]]*20)

        '''more sparsity option'''
        t = np.round(np.random.uniform(1,5,(4)))
        k = 0
        for i in range(len(active_group)):
            if active_groups[i][0] == 1:
                num = 10-int(t[k])
                active_groups[i][0:num] = [0.2] * num  # nonzero
                random.shuffle(active_groups[i])
                k += 1
                
        '''y = wx + z'''
        w = np.concatenate(
            [
                np.random.standard_normal(group_size) * is_active
                for group_size, is_active in zip(group_sizes, active_groups)
            ]
        )

        '''random noise generation'''
        noise_idx = np.random.permutation(np.arange(num_coeffs))[:10]
        w[noise_idx[:3]] = 0.6
        w[noise_idx[3:6]] = 0.4
        w[noise_idx[6:9]] = 0.2
        w[noise_idx[9:10]] = 0.8

        z = np.random.normal(0,0.5,1)
        y = x@w + z
        y = y.reshape(y.shape[0], 1)
        y_true = [0] * num_datapoints

        return group_sizes, num_coeffs, x, y, np.array(active_groups)

    elif dtype == 'E':
        '''data generation 5'''
        '''diff coeff with indiv'''
        # set dataset parameters
        n_group = 50
        group_sizes = [np.random.randint(10, 20) for i in range(n_group)]

        # --- custom group --- #
        active_groups = [np.random.randint(4) for _ in group_sizes]     # binary group dependency
        groups = np.concatenate([size * [i] for i, size in enumerate(group_sizes)])
        num_coeffs = sum(group_sizes)
        num_datapoints = 10000
        noise_std = 2

        # generate data matrix
        X = np.random.standard_normal((num_datapoints, num_coeffs))     # (num_datapoints, num_coeffs)

        # generate coefficients
        w = np.concatenate(
            [
                np.random.standard_normal(group_size) * is_active
                for group_size, is_active in zip(group_sizes, active_groups)
            ]
        )
        w = w.reshape(-1, 1)
        intercept = 2

        # generate regression targets
        y_true = X @ w + intercept
        y = y_true + np.random.randn(*y_true.shape) * noise_std

        # for coefficient visualization
        coeff = []
        for i in range(n_group):
            coeff.append([active_groups[i]] * group_sizes[i])  # group 
        c = np.array(sum(coeff, []))

        return group_sizes, num_coeffs, X, y, c

    else:
        return 


class Synthetic:
    def __init__(self, dtype='A'):
        super(Synthetic, self).__init__()
        self.group_sizes, self.num_coeffs, self.x, self.y, self.coeff = generate(dtype)

        '''data split'''
        (tr_data, te_data, tr_y, te_y) = train_test_split(self.x, self.y, test_size=0.2)
        (tr_data, va_data, tr_y, va_y) = train_test_split(tr_data, tr_y, test_size=0.2)
        
        self.tr_data = tr_data
        self.tr_y = tr_y
        self.va_data = va_data
        self.va_y = va_y
        self.te_data = te_data
        self.te_y = te_y

class Gas:
    def __init__(self):
        super(Gas, self).__init__()

        df = pd.read_csv('/home/hrpark/Personal/GroupSparsity/python/dataset/gas.csv', index_col=False)
        df[-3600:]['gas class'].value_counts()  # batch10은 모든 class 수가 동일한 경우
        sample_df = df[-3600:]
        X = np.array(sample_df[sample_df.columns[0:-2]])
        y = np.array(sample_df['gas concentration level']).reshape(-1,1)

        # set dataset parameters
        n_group = 16
        group_sizes = [8 for i in range(n_group)]
        num_coeffs = sum(group_sizes)

        (tr_data, te_data, tr_y, te_y) = train_test_split(X, y, test_size=0.2)
        (tr_data, va_data, tr_y, va_y) = train_test_split(tr_data, tr_y, test_size=0.2)

        std_scaler = StandardScaler()

        std_scaler.fit(tr_data)
        tr_data = std_scaler.transform(tr_data)
        va_data = std_scaler.transform(va_data)
        te_data = std_scaler.transform(te_data)

        self.tr_data = tr_data
        self.tr_y = tr_y
        self.va_data = va_data
        self.va_y = va_y
        self.te_data = te_data
        self.te_y = te_y
        self.group_sizes = group_sizes
        self.num_coeffs = num_coeffs
        self.coeff = 0

class Breast:
    def __init__(self):
        super(Breast, self).__init__()

        df = pd.read_csv('/home/hrpark/Personal/GroupSparsity/python/dataset/breast.csv', index_col=False)
        X = np.array(df[df.columns[0:-1]])
        y = np.array(df['recurrence-events']).reshape(-1,1)

        # set dataset parameters
        n_group = 9
        group_sizes = [6, 3, 11, 7, 1, 1, 1, 5, 1]
        num_coeffs = X.shape[-1]

        (tr_data, te_data, tr_y, te_y) = train_test_split(X, y, test_size=0.2)
        (tr_data, va_data, tr_y, va_y) = train_test_split(tr_data, tr_y, test_size=0.2)

        std_scaler = StandardScaler()

        std_scaler.fit(tr_data)
        tr_data = std_scaler.transform(tr_data)
        va_data = std_scaler.transform(va_data)
        te_data = std_scaler.transform(te_data)

        self.tr_data = tr_data
        self.tr_y = tr_y
        self.va_data = va_data
        self.va_y = va_y
        self.te_data = te_data
        self.te_y = te_y
        self.group_sizes = group_sizes
        self.num_coeffs = num_coeffs
        self.coeff = 0

class Breast2:
    def __init__(self):
        super(Breast2, self).__init__()

        df = pd.read_csv('/home/hrpark/Personal/GroupSparsity/python/dataset/breast_wis.csv', index_col=False)
        X = np.array(df[df.columns[0:-1]])
        y = np.array(df['M']).reshape(-1,1)

        # set dataset parameters
        n_group = 10
        group_sizes = [3 for i in range(n_group)]
        num_coeffs = sum(group_sizes)

        (tr_data, te_data, tr_y, te_y) = train_test_split(X, y, test_size=0.2)
        (tr_data, va_data, tr_y, va_y) = train_test_split(tr_data, tr_y, test_size=0.2)

        std_scaler = StandardScaler()

        std_scaler.fit(tr_data)
        tr_data = std_scaler.transform(tr_data)
        va_data = std_scaler.transform(va_data)
        te_data = std_scaler.transform(te_data)

        self.tr_data = tr_data
        self.tr_y = tr_y
        self.va_data = va_data
        self.va_y = va_y
        self.te_data = te_data
        self.te_y = te_y
        self.group_sizes = group_sizes
        self.num_coeffs = num_coeffs
        self.coeff = 0

class PBMC:
    def __init__(self, group_type='chrom'):
        super(PBMC, self).__init__()

        # dataset load
        #pbmc = scvi.data.pbmc_dataset(save_path='/home/hrpark/Personal/GroupSparsity/python/dataset/pbmc/')
        pbmc = anndata.read_h5ad('/home/hrpark/Personal/GroupSparsity/python/dataset/pbmc/pbmc_dataset.h5ad')
        # chrom inform for making group
        pbmc_chrom = pd.read_csv('/home/hrpark/Personal/GroupSparsity/python/dataset/pbmc/pbmc_chrom.xls', sep='\t')
        pbmc_chrom = pbmc_chrom.iloc[:,:-1]
        loc = pbmc_chrom.iloc[:,1].str.split(' ', expand=True).iloc[:,0]
        chrom = pbmc_chrom.iloc[:,1].str.split(' ', expand=True).iloc[:,2].str.split(']').str[0]

        # delete if not have chrom 
        idx = np.where(chrom.isnull() == False)[0]
        pbmc_ = pbmc[:, idx]
        loc = loc[idx]
        chrom = chrom[idx]
        num_coeffs = pbmc_.n_vars

        if group_type == 'chrom':
            # chrom to int
            chrom.replace(to_replace='X', value=23, inplace=True)
            chrom.replace(to_replace='Y', value=24, inplace=True)
            chrom = chrom.astype(int)

            counts = [1, 2, 3, 4, 5, 6, 7, 23, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 20, 19, 24, 22, 21]
            group_sizes = []
            for i in range(len(counts)):
                gs = chrom.value_counts()[counts[i]]
                group_sizes.append(gs)
            group_sizes = np.array(group_sizes)
            start = 0
            for i in range(len(counts)):
                end = group_sizes[i]
                start = start+end

            n_group = len(group_sizes)
            
        elif group_type == 'loc':
            # group making for location
            def group_consecutive(data):
                result = [list(group) for key, group in groupby(data)]
                return result

            a = np.array(loc)
            groups = group_consecutive(a)
            group_sizes = np.array([len(sublist) for sublist in groups])
            n_group = len(group_sizes)
            
        # label making
        sample_idx = np.where((pbmc_.obs.str_labels == 'CD4 T cells') | (pbmc_.obs.str_labels == 'CD8 T cells'))[0]
        def convert_labels(label):
            if label == 2:
                return 0
            elif label == 3:
                return 1
            else:
                return label

        labels = np.array(pbmc_[sample_idx].obs.labels)
        converted_labels = [convert_labels(label) for label in labels]

        X = pbmc_[sample_idx].X.toarray()
        y = np.array(converted_labels).reshape(-1,1)

        (tr_data, te_data, tr_y, te_y) = train_test_split(X, y, test_size=0.2)
        (tr_data, va_data, tr_y, va_y) = train_test_split(tr_data, tr_y, test_size=0.2)

        std_scaler = StandardScaler()

        std_scaler.fit(tr_data)
        tr_data = std_scaler.transform(tr_data)
        va_data = std_scaler.transform(va_data)
        te_data = std_scaler.transform(te_data)

        self.tr_data = tr_data
        self.tr_y = tr_y
        self.va_data = va_data
        self.va_y = va_y
        self.te_data = te_data
        self.te_y = te_y
        self.group_sizes = group_sizes
        self.num_coeffs = num_coeffs
        self.coeff = 0