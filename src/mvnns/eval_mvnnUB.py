import json
import logging
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.utils.data.dataset
from generate_SATS_data import generate_data
from mvnns.explicit_100_percent_upper_bound_mvnn import Explicit100UpperBoundMVNN
from mvnns.layers import *
from mvnns.mvnn import MVNN
from mvnns.plot_random_subsets_1dpath import plot_random_subsets_1dpath
from mvnns.test_mvnnUB import test
from mvnns.train_mvnnUB import train
from torch.utils.data import Dataset
from util import bidder_type_to_bidder_id, timediff_d_h_m_s


def eval_model(input_dim: int, target_max: float, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset,
               bidder_id: int, data_gen_method: str, seed: int, log_path: str, batch_size: int, l2: float,
               optimizer: str, epochs: int, send_to, new_test_plot, q: int, plot_history: bool, lr: float,
               dropout_prob: float, layer_type: str, loss_func: str, log_full_train_history: bool, *args, **kwargs):
    start_train = datetime.now()

    logs = {}
    loss_func = eval(loss_func)
    device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    exp_upper_bound_net = Explicit100UpperBoundMVNN(input_dim=train_dataset.tensors[0].shape[1],
                                                    layer_type=layer_type, target_max=target_max,
                                                    X_train=train_dataset.tensors[0],
                                                    y_train=train_dataset.tensors[1].flatten())

    metrics = defaultdict(dict)
    metrics['train'] = {}

    best_train_loss_attempts_and_epochs = np.inf
    best_mean_model = None
    best_ub_model = None
    best_epoch = None
    best_attempt = None

    attempt = 1
    ##################
    MAX_ATTEMPTS = 2
    ##################

    reattempt = (attempt <= MAX_ATTEMPTS)

    pi_exp_factor = 1.0
    lr_factor = 1.0
    logging.info('START Training')
    while reattempt:

        metrics['train'][attempt] = {}
        mean_model = MVNN(layer_type=layer_type, input_dim=input_dim, target_max=target_max, dropout_prob=dropout_prob,
                          **kwargs).to(device)
        ub_model = MVNN(layer_type=layer_type, input_dim=input_dim, target_max=target_max, dropout_prob=dropout_prob,
                        **kwargs).to(device)

        # do not use l2 reg on t's
        l2_reg_parameters = {'params': [], 'weight_decay': l2}
        no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}

        for p in [*mean_model.named_parameters(), *ub_model.named_parameters()]:
            if 'ts' in p[0] or 'lin_skip_layer' in p[0]:
                logging.debug(f'Setting L2-Reg. to 0.0 for {p[0]}.')
                no_l2_reg_parameters['params'].append(p[1])
            else:
                l2_reg_parameters['params'].append(p[1])

        if optimizer == 'Adam':
            torch_optimizer = optim.Adam([l2_reg_parameters, no_l2_reg_parameters], lr=lr * lr_factor)
        elif optimizer == 'SGD':
            torch_optimizer = optim.SGD([l2_reg_parameters, no_l2_reg_parameters], lr=lr * lr_factor, momentum=.9)
        else:
            raise NotImplementedError()

        init_dropout_prob = dropout_prob
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(torch_optimizer, epochs)
        dropout_prob = init_dropout_prob

        logging.info(f'Training attempt:{attempt}')
        for epoch in range(1, epochs + 1):
            metrics['train'][attempt][epoch] = train(model=(mean_model, ub_model), device=device,
                                                     train_loader=train_loader, optimizer=torch_optimizer,
                                                     epoch=epoch, target_max=target_max, loss_func=loss_func,
                                                     exp_upper_bound_net=exp_upper_bound_net, dropout_prob=dropout_prob,
                                                     q=q, *args, **kwargs)
            scheduler.step()
            dropout_prob = dropout_prob * kwargs['dropout_prob_decay']

            # NEW: best model criteria to select from all reattempts AND epochs:
            # uUB-model should fit the training data reasonably well (small loss_b)
            #     and should produce reasonably uUB (running average of loss_c small)
            #     and mean model should be ok (small 0.5*loss_a)
            if epoch == 1:
                running_c_loss = metrics['train'][attempt][epoch]['loss_c']
            else:
                running_c_loss = 0.5 * running_c_loss + 0.5 * metrics['train'][attempt][epoch]['loss_c']

            train_loss_epoch = (0.5 * metrics['train'][attempt][epoch]['loss_a'] + metrics['train'][attempt][epoch][
                'loss_b'] + running_c_loss)

            if train_loss_epoch < best_train_loss_attempts_and_epochs:
                best_mean_model = pickle.loads(pickle.dumps(mean_model))
                best_ub_model = pickle.loads(pickle.dumps(ub_model))
                best_train_loss_attempts_and_epochs = train_loss_epoch
                best_epoch = epoch
                best_attempt = attempt

        # NEW: reattempt criteria uUB should fit the training data reasonably (R2>=0.9) well
        if metrics['train'][attempt][best_epoch]['uUB-r2'] < 0.9 and (attempt + 1 <= MAX_ATTEMPTS):
            reattempt = True
            attempt += 1
            pi_exp_factor *= 0.25
            lr_factor *= 0.5
        else:
            reattempt = False

    end_train = datetime.now()
    logs['train_time_elapsed'] = (end_train - start_train).total_seconds()

    logging.info("TIME ELAPSED for training: {}d {}h:{}m:{}s".format(
        *timediff_d_h_m_s(end_train - start_train)) + " (" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")")

    logging.info(f'Best mean_model & uUB_model from attempt:{best_attempt} and epoch:{best_epoch}/{epochs}')
    logs['best_epoch'] = best_epoch
    logs['best_attempt'] = best_attempt
    logs['attempts'] = attempt
    logs['target_max'] = str(target_max)  # json serialize cannot do np.float32
    logs['bidder_id'] = str(bidder_id)

    # Use now th best mean and ub_model
    mean_model, ub_model = best_mean_model, best_ub_model
    # Transform their weights
    mean_model.transform_weights(), ub_model.transform_weights()

    if test_dataset:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, num_workers=0)

        # Plot 1D-random_subset_path
        if data_gen_method == 'random_subset_path':
            plot_random_subsets_1dpath((mean_model, ub_model), device, train_dataset, test_dataset,
                                       seed=seed, loss_func=loss_func, exp_upper_bound_net=exp_upper_bound_net,
                                       plot=False, log_path=log_path, send_to=send_to)

        # Evaluate Model on Test-Set
        metrics['test'] = {}
        if kwargs['eval_test']:
            metrics['test'] = test(model=(mean_model, ub_model), device=device, loader=test_loader, valid_true=False,
                                   target_max=target_max, seed=seed, loss_func=loss_func,
                                   exp_upper_bound_net=exp_upper_bound_net, plot=False, new_test_plot=new_test_plot,
                                   log_path=log_path, q=q, send_to=send_to)

    if log_full_train_history:
        logs['metrics'] = metrics
    else:
        logs['metrics'] = {'train': {best_attempt: {best_epoch: metrics['train'][best_attempt][best_epoch]}},
                           'test': metrics['test']}  # only log train history for best_attempt and best_epoch

    if kwargs['save_datasets']:
        metrics['datasets'] = {'train': train_dataset,
                               'val': val_dataset,
                               'test': test_dataset,
                               'target_max': target_max}
        metrics['mean-model'] = best_mean_model
        metrics['uuB-model'] = best_ub_model
        logs['metrics'] = metrics
    if log_path is not None:
        time = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
        json.dump(logs, open(os.path.join(log_path, f'Seed{seed}_Results_{time}.json'), 'w'))

    if plot_history and log_full_train_history:
        # HISTORY PLOT ---------------------------------------------------------------------------------------------------------
        history_total_loss = [v['loss'] for k, v in logs['metrics']['train'][best_attempt].items()]
        history_total_loss_b = [v['loss_b'] for k, v in logs['metrics']['train'][best_attempt].items()]
        history_total_loss_c = [v['loss_c'] for k, v in logs['metrics']['train'][best_attempt].items()]

        plt.figure(figsize=(16, 9))
        plt.grid()
        plt.plot(history_total_loss, label='Total loss')
        plt.plot(history_total_loss_b, 'g--', label='Loss b')
        plt.plot(history_total_loss_c, 'r--', label='Loss c')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.title('Train History')
        time = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
        plt.savefig(os.path.join(log_path, f'Seed{seed}_TrainHistory_{time}.pdf'), format='pdf')
        plt.close()
        # ----------------------------------------------------------------------------------------------------------------------

    return ub_model, logs, mean_model, exp_upper_bound_net


def eval_config(seed, SATS_domain, bidder_type, *args, **kwargs):
    # SEEDING ------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # ---------------------------

    # GENERATE DATA: TRAIN/VAL/TEST ----------------------------------------------------------------------------
    bidder_id = bidder_type_to_bidder_id(SATS_domain, bidder_type)  # sample bidder_id corresponding to bidder type
    train_dataset, val_dataset, test_dataset, dataset_info = generate_data(SATS_domain=SATS_domain, bidder_id=bidder_id,
                                                                           seed=seed, **kwargs)
    # -------------------------------------------------------------------------------------------------------------

    return eval_model(input_dim=dataset_info['M'], seed=seed, SATS_domain=SATS_domain, bidder_type=bidder_type,
                      target_max=dataset_info['target_max'], train_dataset=train_dataset, val_dataset=val_dataset,
                      test_dataset=test_dataset, bidder_id=bidder_id, **kwargs)
