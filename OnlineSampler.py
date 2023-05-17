import sys, os
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import itertools
import copy
import time
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch
from models import LogisticRegression, weights_init_normal, test_model

class RobustSampler(Sampler):
    def __init__(self, xz, y, model, alpha, gamma, delta, fairness_type, lb_init, lr_reg, warm_start, batch_size, w_init, device, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.device = device
        self.model = model
        self.x_data = xz
        self.y_data = y
        self.length = len(self.y_data)
        self.index = random.randint(0, self.length)
        self.d_curr = self.x_data[self.index]
        self.d_curr.requires_grad_()

        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.mu = torch.zeros((1, 1)).to(self.device)
        self.tau = 0
        self.lr_reg = lr_reg
        self.clean_ratio = 0
        self.selection = torch.zeros((self.length, 1)).to(self.device)
        # self.index_dict = self.__divider(self.y_data, self.z_data)
        #     fair slection
        self.fairness_type = fairness_type
        # print(self.fairness_type)
        self.lb1 = lb_init
        # print('checkpoint1', self.lb1)
        self.lb2 = lb_init
        self.loss_1 = [0]
        self.loss_2 = [0]
        self.loss_3 = [0]
        self.warm_start = warm_start
        self.batch_size = batch_size

        self.data_batch = torch.ones((self.batch_size, self.x_data.size(dim=1))).to(self.device) * 2
        self.label_batch = torch.ones(self.batch_size).to(self.device) * 2
        self.grad_loss_batch = torch.ones(self.batch_size).to(self.device) * 2
        self.index_batch = torch.ones(self.batch_size).to(self.device) * 2

        self.g_value = torch.zeros((self.length, 2)).to(self.device)
        self.w_init = w_init
        self.w = self.w_init

    def get_logit(self, x_data):
        """Runs forward pass of the intermediate model with the training data.

        Returns:
            Outputs (logits) of the model.

        """

        self.model.eval()
        logit = self.model(x_data)

        return logit
    def __adjust_lambda(self, top_range):
        """Adjusts the lambda values using FairBatch [Roh et al., ICLR 2021].
        See our paper for algorithm details.

        Args:
            logit: A torch tensor that contains the intermediate model's output on the training data.

        """

        z_item = [0, 1]
        y_item = [-1, 1]
        # yz_tuple = list(itertools.product(y_item, z_item))
        criterion = torch.nn.BCELoss(reduction='none')
        # loss = criterion((F.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        # print(f'run adj 1')
        if self.fairness_type == 'eqodd':

            yhat_yz = {}
            yhat_y = {}

            # eo_loss = criterion((torch.tanh(logit.squeeze()) + 1) / 2, (self.y_data.squeeze() + 1) / 2)

            yhat_yz[(-1, 0)] = float(sum(self.loss_l1)) / len(self.loss_l1)
            yhat_yz[(-1, 1)] = float(sum(self.loss_l2)) / len(self.loss_l2)
            yhat_yz[(1, 0)] = float(sum(self.loss_l3)) / len(self.loss_l3)
            yhat_yz[(1, 1)] = float(sum(self.loss_l4)) / len(self.loss_l4)
            # yhat_y[tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]

            y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
            y0_diff = abs(yhat_yz[(-1, 1)] - yhat_yz[(-1, 0)])

            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            if y1_diff > y0_diff:
                if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                    self.lb1 += self.alpha
                else:
                    self.lb1 -= self.alpha
            else:
                if yhat_yz[(-1, 1)] > yhat_yz[(-1, 0)]:
                    self.lb2 += self.alpha
                else:
                    self.lb2 -= self.alpha

            if self.lb1 < 0:
                self.lb1 = 0
            elif self.lb1 > 1:
                self.lb1 = 1

            if self.lb2 < 0:
                self.lb2 = 0
            elif self.lb2 > 1:
                self.lb2 = 1

        if self.fairness_type == 'eqopp':
            # print(f'run adj 2')
            yhat_yz = {}
            S_10 = len(self.loss_1)
            S_11 = len(self.loss_2)
            S_1 = S_10 + S_11
            S_0 = len(self.loss_3)
            # eo_loss = criterion((torch.tanh(logit.squeeze()) + 1) / 2, (self.y_data.squeeze() + 1) / 2)

            yhat_yz[(1, 0)] = float(sum(self.loss_1)) / len(self.loss_1)
            yhat_yz[(1, 1)] = float(sum(self.loss_2)) / len(self.loss_2)
            # yhat_y [tmp_y] = float(torch.sum(eo_loss[self.clean_y_index[tmp_y]])) / self.clean_y_len[tmp_y]
            # print('checkpoint2', yhat_yz[(1, 0)], yhat_yz[(1, 1)], self.lb1)
            # lb1 * loss_y1z1 + (1-lb1) * loss_y1z0
            # lb2 * loss_y0z1 + (1-lb2) * loss_y0z0

            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.lb1 -= self.alpha
            else:
                self.lb1 += self.alpha

            if self.lb1 < 0:
                self.lb1 = 0
            # elif self.lb1 > S_1/(S_0 + S_1):
            #     self.lb1 = S_1/(S_0 + S_1)
            elif self.lb1 > top_range:
                self.lb1 = top_range
        # print(f'lb1 {self.lb1}')


    # FIXME when len(dataset) != T, didn't work
    def g(self, x_iter, isAll):
        t = x_iter.size(dim=0)
        S_10 = len(self.loss_1)
        S_11 = len(self.loss_2)
        S_1 = S_10 + S_11
        S_0 = len(self.loss_3)
        self.__adjust_lambda(S_1 / (S_0 + S_1))
        index = self.selection[:t]
        g_0 = index - self.tau

        # # newly developed fairness constraints
        #
        # # the number of selected data in each sub group
        # S_10 = len(self.loss_1)
        # S_11 = len(self.loss_2)
        # S_1 = S_10 + S_11
        # S_0 = len(self.loss_3)
        # p_10 = self.lb1
        # p_11 = S_1 / (S_1 + S_0) - self.lb1
        # p_0 = S_0 / (S_0 + S_1)
        # v = torch.zeros((t, 3)).to(self.device)
        # v_1 = torch.Tensor([1 - p_10, - p_11, - p_0]).to(self.device)
        # v_2 = torch.Tensor([- p_10, 1 - p_11, - p_0]).to(self.device)
        # v_3 = torch.Tensor([- p_10, - p_11, 1 - p_0]).to(self.device)
        # for i in range(t):
        #     if self.index_dict[i] == '1, 0':
        #         v[i] = v_1
        #     elif self.index_dict[i] == '1, 1':
        #         v[i] = v_2
        #     else:
        #         v[i] = v_3
        # v[:, 0] = self.selection.squeeze() * v[:, 0]
        # v[:, 1] = self.selection.squeeze() * v[:, 1]
        # v[:, 2] = self.selection.squeeze() * v[:, 2]
        # v_mean = torch.sum(v, dim=0) # use sum here, can try to use mean
        # for i in range(i):
        #     if self.index_dict[i] == '1, 0':
        #         v[i] = v_1 + v_mean
        #     elif self.index_dict[i] == '1, 1':
        #         v[i] = v_2 + v_mean
        #     else:
        #         v[i] = v_3 + v_mean
        g_return = g_0
        # subgroup_list = np.array(list(self.index_dict.items()))
        # print(f'g_return {g_return[:10]} subgroup_list {subgroup_list[:10]}')
        if isAll:
            return g_return
        else:
            return g_return[self.index]
        # if isAll:
        #     return torch.concat((g_0, g_1), dim=1)
        # else:
        #     return torch.concat((g_0, g_1), dim=1)[self.index]
        
        # if isAll:
        #     return g_0
        # else:
        #     return g_0[self.index]


    # FIXME
    def plot(self):
        data_plot = self.x_data.cpu().detach().numpy()
        criterion = torch.nn.BCELoss(reduction='none')
        logit = self.get_logit(self.x_data)
        loss = criterion((torch.tanh(logit.squeeze())+1)/2, (self.y_data.squeeze()+1)/2)
        loss_plot = loss.cpu().detach().numpy()
        fig = plt.figure(dpi=400)
        ax = plt.axes(projection="3d")

        img = ax.scatter3D(data_plot[:,0], data_plot[:,1], data_plot[:,2], c=loss_plot, alpha=0.7, marker='.')
        fig.colorbar(img)
        plt.show()

    # Training
    def Sampler(self, t):
        # Warm start
        if t <= self.warm_start:
            index = torch.randint(len(self.y_data), (10,))
            # print('index', index)
            data_batch = self.x_data[index]
            label_batch = self.y_data[index]
            # print(f'testpoint1 {data_batch.shape} {label_batch.shape}')
            criterion = torch.nn.BCELoss(reduction='none')
            logit = self.get_logit(data_batch)
            loss = criterion((torch.tanh(logit.squeeze()) + 1) / 2, (self.y_data[index].squeeze() + 1) / 2)
            self.selection[index] = 1
            # print(f'testpoint2 {self.selection[index]}')
                # print(f't {t} num of (1, 0) {len(self.loss_1)} num of (1, 1) {len(self.loss_2)}')
            # print(f'time {t} num of 1, 0 {len(self.loss_1)} num of 1, 1 {len(self.loss_2)}')
            return data_batch, label_batch, 1

        else:
            data_batch = []
            label_batch = []
            index_list = []
            # when average gradient > threshold, update the batch
            for _ in range(self.batch_size):
                self.w = self.w_init
                self.index = random.randint(0, self.length - 1)
                self.d_curr = self.x_data[self.index]
                if self.w < 6:
                # if torch.mean(self.grad_loss_batch[:1,]) > self.delta:
                    aver_grad = 0


                    # FIXME static dataset here, change to online setting when publish

                    x_iter = self.x_data
                    y_iter = self.y_data

                    # FIXME

                    # update to select next sample

                    self.d_curr.requires_grad_()
                    criterion = torch.nn.BCELoss(reduction='none')
                    logit = self.get_logit(self.d_curr)
                    # print('d_curr:', self.d_curr.requires_grad)
                    # print(y_iter.shape)
                    loss = criterion((torch.tanh(logit.squeeze()) + 1) / 2, (y_iter[self.index].squeeze() + 1) / 2)
                    loss.backward()
                    # print('d_curr:', self.d_curr.requires_grad)
                    grad_loss = self.d_curr.grad
                    aver_grad += torch.norm(grad_loss)
                    self.g_value = self.g(x_iter, isAll=True)
                    opt_results = (x_iter - self.d_curr) @ grad_loss.reshape(self.x_data.size(dim=1),
                                                                             1) + 1 / 50 * self.g_value @ self.mu.T \
                                  + 1. / (2 * self.lr_reg) * torch.norm(x_iter - self.d_curr, p=2, dim=1).reshape(-1, 1)
                    # print(f'first term {(x_iter - self.d_curr) @ grad_loss.reshape(self.x_data.size(dim=1),1)[:10]} second term {self.g_value @ self.mu.T[:10]}')
                    # opt_result.shape = n*3 * 3*1 + n*2 * 2*1 + n*1
                    # print('mu:', self.mu)
                    self.index = torch.argmin(opt_results, dim=0)
                    # self.selection[self.index] = 1
                    self.d_curr = x_iter[self.index]
                    self.selection[self.index] += 1
                    self.mu = self.mu + self.gamma * self.g(x_iter, isAll=False)  # 1*constraints_num
                    self.mu[self.mu < 0] = 0
                    # 1*4 + 
                    # print(f'checkpoint 1: opt term 1 {(x_iter - self.d_curr) @ grad_loss} mu {self.mu} g_value {self.g(x_iter, isAll=False)} second term {self.gamma * self.g(x_iter, isAll=False)}')
                    self.w += 1
                data_batch.append(self.d_curr)
                label_batch.append(y_iter[self.index])
                index_list.append(self.index.cpu().detach().numpy())
            return torch.stack(data_batch).squeeze(dim=1), torch.stack(label_batch).squeeze(), index_list
                    # return data_batch_res, label_batch_res, torch.mean(grad_loss_batch_res), grad_loss_batch_res, self.g_value[self.index].squeeze().tolist(), True


class ITLM_sampler():
    def __init__(self, train_data, train_labels_noisy, model, tau, batch_size, device, dim, seed, warm_start):
        self.train_data = train_data
        self.train_labels_noisy = train_labels_noisy
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.dim = dim
        self.seed = seed
        self.tau = tau
        self.warm_start = warm_start
        np.random.seed(seed)
        random.seed(seed)

    def eval(self):
        criterion_eval = torch.nn.BCELoss(reduction='none')
        outputs_eval = self.model(self.train_data)
        loss_list = criterion_eval((F.tanh(outputs_eval.squeeze()) + 1) / 2, (self.train_labels_noisy.squeeze() + 1) / 2)
        return loss_list

    def Sampler(self, t):
        if t <= self.warm_start:
            index = torch.randint(len(self.train_labels_noisy), (10,))
            # print('index', index)
            data_batch = self.train_data[index]
            label_batch = self.train_labels_noisy[index]
            return data_batch, label_batch, index
        else:
            train_len = len(self.train_labels_noisy)
            k = int((1 - self.tau) * train_len)
            loss_list = self.eval()
            # print(f'profit.shape {loss_list} k {k}')
            (_, sorted_index) = torch.topk(loss_list, k, largest=True, sorted=True)
            # print(f'checkpoint {len(sorted_index)}')
            idx = np.random.choice(sorted_index.cpu(), size=self.batch_size, replace=False)
            return self.train_data[idx], self.train_labels_noisy[idx], idx



