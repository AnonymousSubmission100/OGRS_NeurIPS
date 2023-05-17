from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import argparse
from models import LogisticRegression, weights_init_normal, test_model
from OnlineSampler import RobustSampler, ITLM_sampler
import time
import math
import random
seed = 1234
np.random.seed(seed)
random.seed(seed)

# import local class
import models
import OnlineSampler



device = 2
device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")


# training function
def run_epoch(model, train_features, labels, optimizer, criterion):
    """Trains the model with the given train data.

    Args:
        model: A torch model to train.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.
        criterion: A torch criterion.

    Returns:
        loss values.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()
    optimizer.step()
    # print(train_features.requires_grad)
    return loss.item()

# generate synthetic dataset
class data_randomizer():
    """
    randomize label to create noisy data

    Args:
    dataset: (image, target)
    label_randomization_ratio: contol the ratio of randomized label
    returns:
    datasets, same as input

    """
    def __init__(self, train_data, train_labels, label_randomization_ratio=0) -> None:
        self.data = train_data
        self.labels = train_labels
        self.label_randomization_ratio = label_randomization_ratio
    def randomize_labels_(self, labels: torch.Tensor, ratio: float) -> None:
        labels_count = len(self.labels)
        to_pick = math.ceil(ratio * labels_count)
        indexes_to_change = random.sample(population=range(labels_count), k=to_pick)
        unique_ele = np.unique(self.labels)
        possible_labels = unique_ele
        for i in indexes_to_change:
            # labels[i] = possible_labels[0]
            self.labels[i] = np.random.choice(possible_labels)
        return indexes_to_change
    def get_dataset(self):
        if self.label_randomization_ratio >= 0:
            indexes_to_change = self.randomize_labels_(self.labels, self.label_randomization_ratio)
        return self.data, self.labels, indexes_to_change


def data_generator(mu, sigma, size):
    return np.random.multivariate_normal(mu, sigma, size)

size = int(20000 / 2)
mu_0 = [1, 1]
sigma_0 = np.array([[5, 1],
                  [1, 5]])
mu_1 = [-1, -1]
sigma_1 = np.array([[10, 1],
                  [1, 3]])
x_data_0 = data_generator(mu=mu_0, sigma=sigma_0, size=size)
x_data_1 = data_generator(mu=mu_1, sigma=sigma_1, size=size)

xz_data = np.concatenate((x_data_0, x_data_1))
y_data = np.concatenate((np.ones(size), - np.ones(size)))
xz_data, y_data = shuffle(xz_data, y_data, random_state=0)
xz_train_np = xz_data[:200]
y_train_np = y_data[:200]
xz_test = xz_data[100:-1]
y_test = y_data[100:-1]


# Randomize label to create noisy data
label_randomization_ratio_init = 0.6
randomizer = data_randomizer(xz_train_np, y_train_np, label_randomization_ratio=label_randomization_ratio_init)
xz_train, y_train, correctness = randomizer.get_dataset()



xz_train = torch.FloatTensor(xz_train)
y_train = torch.FloatTensor(y_train)

xz_test = torch.FloatTensor(xz_test)
y_test = torch.FloatTensor(y_test)

xz_train = xz_train.to(device)
y_train = y_train.to(device)

xz_test = xz_test.to(device)
y_test = y_test.to(device)
print("---------- Number of Data ----------" )
print(
    "Train data : %d, Test data : %d "
    % (len(y_train), len(y_test))
)




# define parameters
num=0
num_iter = 4
avg_loss = 0
T = 400
alpha = 0.001 # lr for fairness ratio adjustment
gamma = 50 / 200
# lr for constraint coefficient update
lr = 0.01 # 0.01 lr fot he LR model train
lr_reg = 50 / 500
class_num = 4
delta = 1.5
warm_start = 40
batch_size = 1
w_init = 1
avg_loss = 0


seeds = [seed]
def plot_data(model, X, Y):
    label_predicted = model.forward(X)
    criterion = torch.nn.BCELoss(reduction='none')
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (Y.squeeze() + 1) / 2)
    loss_plot = loss.cpu().detach().numpy()
    fig = plt.figure(dpi=400)
    ax = plt.axes(projection="3d")
    data_plot = X.cpu().detach().numpy()
    img = ax.scatter3D(data_plot[:, 0], data_plot[:, 1], loss_plot, c=loss_plot, alpha=0.7, marker='.', s=15)
    ax.view_init(10, 30)
    fig.colorbar(img)
    fig.savefig('3d_ogrs.png', dpi=400)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel('loss')
    fig.colorbar(img, orientation='horizontal')
    plt.show()


test_acc = []
avg_list = []
dim = 2
tau = 0.8
# training
for seed in seeds:
    print("< Seed: {} >".format(seed))
    model = LogisticRegression(2, 1).to(device)
    torch.manual_seed(seed)
    model.apply(weights_init_normal)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    print(f'data shape {xz_train.shape} {y_train.shape}')
    sampler = RobustSampler(xz_train, y_train, model, alpha, gamma, delta, 'eqopp', 0.5, lr_reg, warm_start, batch_size, w_init, device, seed)
    begin_time = time.time()
    for t in range(1, T):
        data, target, index = sampler.Sampler(t)
        if t > warm_start:
            selected_index.append(index)
        num += 1
        loss = run_epoch(model, data, target, optimizer, criterion)
        print(t, end="\r")
        if t % 10 == 0:
            tmp_test = test_model(model, xz_test, y_test)
            test_acc.append(tmp_test)
            avg_acc = sum(test_acc) / len(test_acc)
            avg_list.append(avg_acc)
            print(f'Time: {t} Test accuracy: {tmp_test}')
    end_time = time.time()
    tmp_test = test_model(model, xz_test, y_test)
    plot_data(model, xz_train, y_train)
    print(f'Time: {t} Test accuracy: {tmp_test}')
    print(avg_list)
    print("----------------------------------------------------------------------")


