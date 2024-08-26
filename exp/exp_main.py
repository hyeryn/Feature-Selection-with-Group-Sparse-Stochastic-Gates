from model.baseline import Model
from dataset.data_provider import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, adjust_learning_rate2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

class Exp_Main():
    def __init__(self, args):
        self.args = args
        
    def _build_model(self, num_coeffs, group_sizes):
        model = Model(num_coeffs, group_sizes, self.args).to(self.args.device)
        return model

    def _get_data(self):
        train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader = data_provider(self.args)
        return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    
    def _select_optimizer(self): 
        layer1_optim = optim.Adam(self.model.first.parameters(), lr=self.args.lr, weight_decay=self.args.w_dec1)
        model_optim = optim.Adam(self.model.NN.parameters(), lr=self.args.lr, weight_decay=self.args.w_dec2)
        #model_optim = optim.Adam(self.model.NN.parameters(), lr=self.args.lr)
        selector_optim = optim.Adam(self.model.feature_selector.parameters(), lr=self.args.lr2)
        #return model_optim, selector_optim
        return layer1_optim, model_optim, selector_optim
    
    def _select_criterion(self):
        # regression
        if self.args.data in ('syn', 'gas'):
            criterion = nn.MSELoss()
        # classification
        else:
            criterion = nn.BCEWithLogitsLoss()
        return criterion
    
    def _perm_index(self, x):
        n_rows, n_cols = x.shape
        perm_index = torch.stack([torch.randperm(n_rows) for _ in range(n_cols)], dim=1)
        return perm_index

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.to(self.args.device)
                if self.args.permuted == None:
                    outputs = self.model(batch_x, None)
                else :
                    n_rows, n_cols = batch_x.shape
                    index = self._perm_index(batch_x)
                    permuted_x = batch_x[index, torch.arange(n_cols)]
                    outputs = self.model(batch_x, permuted_x)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.selector == 'HC':
                    reg_1 = self.model.regularizer_all().detach().cpu()
                    reg_2 = self.model.regularizer_group().detach().cpu()
                    loss = criterion(pred, true) + self.args.lamb1 * reg_1.mean() + self.args.lamb2 * reg_2.mean()
                else:
                    reg = self.model.regularizer().detach().cpu()
                    loss = criterion(pred, true) + self.args.lamb1 * reg.mean()

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader, vali_data, vali_loader, self.test_data, test_loader = self._get_data()
        '''check_dataset'''
        print(train_data.x.shape, vali_data.x.shape, self.test_data.x.shape)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.model = self._build_model(train_data.n, train_data.g)
        layer1_optim, model_optim, selector_optim = self._select_optimizer()
        #model_optim, selector_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                layer1_optim.zero_grad()
                model_optim.zero_grad()
                selector_optim.zero_grad()

                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                if self.args.permuted == None:
                    output = self.model(batch_x, None)
                else :
                    n_rows, n_cols = batch_x.shape
                    index = self._perm_index(batch_x)
                    permuted_x = batch_x[index, torch.arange(n_cols)]
                    output = self.model(batch_x, permuted_x)

                if self.args.selector == 'HC':
                    reg_1 = self.model.regularizer_all()
                    reg_2 = self.model.regularizer_group()
                    loss = criterion(output, batch_y) + self.args.lamb1 * reg_1.mean() + self.args.lamb2 * reg_2.mean()
                    train_loss.append(loss.item())
                else:
                    reg = self.model.regularizer()
                    loss = criterion(output, batch_y) + self.args.lamb1 * reg.mean()
                    train_loss.append(loss.item())

                loss.backward()
                layer1_optim.step()
                model_optim.step()
                selector_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(self.test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            #torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')   # always model save

            ''' early stopping & learning rate adjust'''
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)
            #adjust_learning_rate2(selector_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        f = open('./test_results/times.txt', 'a')
        times = time.time() - epoch_time
        f.write(setting + " \n")
        f.write(str(round(times,5)))
        f.write('\n')
        f.close()

        # probability save
        mu, self.prob = self.model.get_gates()
        np.save('./results/prob/{}'.format(setting), self.prob)

        plt.figure(figsize=(12,5))
        plt.title(f'{setting}')
        plt.bar(np.arange(0, len(self.prob)), self.prob, width=1.0)
        plt.xlabel('feature')
        plt.ylabel('prob')
        plt.show()
        plt.savefig('./results/image/'+setting+'.png')
        plt.close()

        return


    def test(self, setting):
        print('load model')
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        mu, self.prob = self.model.get_gates()
        self.prob = np.where(self.prob > self.args.threshold, self.prob, 0)

        ''' new model prediction '''
        prob_x = self.test_data.x * self.prob  # 720, 128
        X_train, X_test, y_train, y_test = train_test_split(prob_x, self.test_data.y,  
                                                    test_size = 0.25, random_state = 2024)

        xgb = XGBRegressor(
            n_estimators=50,
            max_depth=5,
            gamma = 0,
            importance_type='gain',
            reg_lambda=1,
            random_state=2024
        )

        xgb.fit(X_train, y_train)
        score = xgb.score(X_test, y_test)
        y_pred = xgb.predict(X_test)
        print(score)

        f = open('./test_results/metric.txt', 'a')
        f.write(setting + " \n")
        f.write(str(round(score,3)) + " | ")
        f.write(str(round(mean_squared_error(y_test, y_pred),3)) + " | ")
        f.write(str(round(mean_absolute_error(y_test, y_pred),3)) + " | ")
        f.write(str(round(mean_squared_error(y_test, y_pred,squared=False),3)))
        f.write('\n\n')
        f.close()

        ''' probability analysis'''
        # num of select total feature
        total = len(np.where(self.prob != 0)[0])

        # num of select group
        g_num = []
        st = 0
        for i in range(len(self.test_data.g)):
            end = st + self.test_data.g[i]
            g_prob = self.prob[st:end]
            st += self.test_data.g[i]
            g_num.append(len(np.where(g_prob != 0)[0]))

        # right probability
        # if self.args.data == 'syn':
        #     print(test_data.c)
        #     print(self.prob)

        f = open('./test_results/probability_analysis.txt', 'a')
        f.write(setting + " \n")
        f.write(str(total) + " | ")
        f.write(str(g_num))
        f.write('\n\n')
        f.close()

        return