import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


from mvnns.layers import *
#from mvnns.explicit_100_percent_upper_bound_mvnn import Explicit100UpperBoundMVNN


def plot_random_subsets_1dpath(model,
                               device,
                               train_dataset,
                               test_dataset,
                               seed,
                               loss_func,
                               exp_upper_bound_net,
                               plot=False,
                               log_path=None,
                               send_to=None):

    mean_model, ub_model = model
    mean_model.eval()
    ub_model.eval()

    num_items_train = train_dataset.tensors[0].sum(dim=1)
    bundle_value_train = train_dataset.tensors[1]
    mean_pred_train = mean_model(train_dataset.tensors[0]).detach().cpu().numpy()
    ub_pred_train = ub_model(train_dataset.tensors[0]).detach().cpu().numpy()

    upper_bound_train = exp_upper_bound_net(train_dataset.tensors[0]).detach().cpu().numpy()

    num_items_test = test_dataset.tensors[0].sum(dim=1)
    bundle_value_test = test_dataset.tensors[1]
    mean_pred_test = mean_model(test_dataset.tensors[0]).detach().cpu().numpy()
    ub_pred_test = ub_model(test_dataset.tensors[0]).detach().cpu().numpy()
    upper_bound_test = exp_upper_bound_net(test_dataset.tensors[0]).detach().cpu().numpy()

    # sort training and test points ------------------------------------------------
    num_items_train = num_items_train.detach().cpu().numpy()
    num_items_test = num_items_test.detach().cpu().numpy()
    num_items = np.concatenate((num_items_train, num_items_test))
    sorted_idx = np.argsort(num_items)

    mean_pred = np.concatenate((mean_pred_train, mean_pred_test))[sorted_idx]
    ub_pred = np.concatenate((ub_pred_train, ub_pred_test))[sorted_idx]
    upper_bound = np.concatenate((upper_bound_train, upper_bound_test))[sorted_idx]
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(16, 9))
    plt.scatter(num_items_train, bundle_value_train, color='black', s=200, marker='x')

    plt.plot(num_items[sorted_idx], mean_pred, label='Mean', color='darkred')
    plt.plot(num_items[sorted_idx], ub_pred, label='UB', color='b')
    plt.plot(num_items[sorted_idx], upper_bound, label='ExpUB', color='orange')

    plt.scatter(num_items_train, mean_pred_train, color='darkred', s=20)
    plt.scatter(num_items_train, ub_pred_train, color='b', s=20)
    plt.scatter(num_items_train, upper_bound_train, color='orange', s=20)

    plt.scatter(num_items_test, mean_pred_test, color='darkred', s=20)
    plt.scatter(num_items_test, ub_pred_test, color='b', s=20)
    plt.scatter(num_items_test, upper_bound_test, color='orange', s=20)

    plt.scatter(num_items_test, bundle_value_test, color='black', label='True Value', s=25)
    plt.xlabel('Num. items in bundle')
    plt.ylabel('Value')
    plt.legend()

    plt.title('1D-Path Plot')
    time = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    plt.savefig(os.path.join(log_path, f'Seed{seed}_1dpathPlot_{time}.pdf'), format='pdf')

    plt.show()
