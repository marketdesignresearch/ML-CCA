"""
FILE DESCRIPTION:

This file stores general helper functions.
"""

# libs
import itertools
import json
import logging
import os
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds / 3600), -int(td.seconds / 60) % 60, -(td.seconds % 60)
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


def generate_all_bundle_value_pairs(world, order=0, k=262144):
    N = world.get_bidder_ids()
    M = world.get_good_ids()
    print()
    if order == 0:
        bundle_space = list(itertools.product([0, 1], repeat=len(M)))
    elif order == 1:
        bundle_space = [[int(b) for b in bin(2 ** len(M) + k)[3:][::-1]] for k in np.arange(2 ** len(M))]
    elif order == 2:
        # for mrvm and srvm space too large -> sample instead
        bundle_space = [np.random.choice([0, 1], len(M)) for _ in range(k)]
        # Only use unique samples.
        bundle_space = np.unique(np.array(bundle_space), axis=0)
    else:
        raise NotImplementedError('Order must be either 0 or 1')
    s = time.time()
    bundle_value_pairs = np.array(
        [list(x) + [world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)])
    e = time.time()
    print('Elapsed sec: ', round(e - s))
    return (bundle_value_pairs)


def nested_max(nestedList):
    if not (isinstance(nestedList, list)):
        return np.max(nestedList)
    else:
        return max([nested_max(a) for a in nestedList])


def nested_min(nestedList):
    # print("nestedList: ",nestedList)
    if not (isinstance(nestedList, list)):
        return np.min(nestedList)
    else:
        return min([nested_min(a) for a in nestedList])


def total_max(*args):
    return max([nested_max(a) for a in args])


def total_min(*args):
    # print("args: ",args)
    return min([nested_min(a) for a in args])


def total_range(*args):
    return total_min(*args), total_max(*args)


def get_config(path, config, RE_RUN):
    # only select folders in that path
    existing_configs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    if not existing_configs:
        return 'config1'
    for c in existing_configs:
        json_path = os.path.join(path, c, c + '.json')
        try:
            with open(json_path, "r") as file:
                loaded_config = json.load(file)
            file.close()
        except FileNotFoundError:
            print(f'NO config.json AVAILABLE IN {c}!-> CHECK AND REMOVE THIS CONFIG')
            return
        if config == loaded_config and not RE_RUN:
            print(
                f'YOUR CONFIG WAS ALREADY RUN IN {c} AND YOU SELECTED RE_RUN:{RE_RUN}! -> THUS NOT RUNNING AGAIN THIS CONFIG')
            return None
        elif config == loaded_config and RE_RUN:
            print(f'YOUR CONFIG WAS ALREADY RUN IN {c} HOWEVER YOU SELECTED RERUN:{RE_RUN}!')
            return c
        else:
            continue
    c = 'config' + str(max([int(re.findall(r'\d+', x)[0]) for x in existing_configs]) + 1)
    return c


def get_hyps_from_dirname(dirname):
    hyps = dirname.split('_')
    hyps_in_dir = {}
    try:
        _, _, _, _, _, _, _, problem_instance, layer_type, num_train_data, _, bidder_type, _, _ = hyps
    except ValueError as e:
        _, _, _, _, _, _, _, problem_instance, layer_type, num_train_data, _, bidder_type1, bidder_type2, _, _ = hyps
        bidder_type = bidder_type1 + '_' + bidder_type2
    bidder_type = bidder_type if bidder_type.lower() != 'high_frequency' else 'high_frequency'
    hyps_in_dir.update(
        {'problem_instance': problem_instance, 'layer_type': layer_type, 'num_train_data': int(num_train_data),
         'bidder_type': bidder_type})
    return hyps_in_dir


def read_results_hpo_mvnnUB(path_prefix,
                            SATS_domain,
                            bidder_type,
                            evaluation_criteria,
                            config_file_identifier=None):
    path = os.path.join(path_prefix, SATS_domain, bidder_type)

    raw_config_results_dict = {}

    # loop over config folder
    try:
        folders = os.listdir(path)
    except:
        logging.warning(f'No results for {path}')
        return raw_config_results_dict

    for folder in folders:
        if os.path.isdir(os.path.join(path, folder)):
            logging.info(folder)

            if config_file_identifier:
                config_name = [x for x in os.listdir(os.path.join(path, folder)) if config_file_identifier in x]
                if len(config_name) > 1:
                    raise ValueError(f'Multiple .json config files in {path} -> CHECK AND REMOVE')
                config_filepath = os.path.join(path, folder, config_name[0])
            else:
                config_filepath = os.path.join(path, folder, folder + '.json')
            try:
                config = json.load(open(config_filepath, 'r'))
                config_keys = [x for x in config.keys() if bool(re.search(r'config[0-9]+', x))]
                if len(config_keys) == 1:
                    config = config[config_keys[0]]
                elif len(config_keys) > 1:
                    raise ValueError(f'.json config file in {path} potentially broken -> CHECK')
                else:
                    pass
            except FileNotFoundError:
                logging.warning(f'NO config.json AVAILABLE IN {folder}!-> CHECK AND REMOVE THIS CONFIG')
                continue
            raw_config_results = []

            # loop over seeds in respective config folder
            finished_seeds = []
            for result_file in os.listdir(os.path.join(path, folder)):

                if bool(re.search(r'Seed\d+_Results.*.json', result_file)):
                    seed = re.search(r'\d+', result_file).group()
                    finished_seeds.append(seed)
                    with open(os.path.join(path, folder, result_file), 'r') as f:
                        result = json.load(f)
                    f.close()
                    best_epoch = result['best_epoch']
                    best_attempt = result['best_attempt']
                    train_metrics = pd.DataFrame.from_dict(
                        result['metrics']['train'][str(best_attempt)][str(best_epoch)],
                        orient='index',
                        columns=[seed])
                    train_metrics.index = ['train-' + x for x in train_metrics.index]

                    test_metrics = pd.DataFrame.from_dict(result['metrics']['test'],
                                                          orient='index',
                                                          columns=[seed])

                    test_metrics.index = ['test-' + x for x in test_metrics.index]
                    test_metrics.loc['train-best_epoch'] = result['best_epoch']
                    test_metrics.loc['train-attempts'] = result['attempts']
                    test_metrics.loc['train-best_attempt'] = best_attempt
                    test_metrics.loc['train-time_min'] = (result['train_time_elapsed'] / 60)
                    # test_metrics.loc['bidder-id'] = int(result['bidder_id'])
                    metrics = pd.concat([test_metrics, train_metrics])
                    raw_config_results.append(metrics)

            if len(raw_config_results) > 0:
                raw_config_results_dict[folder] = {'config': config}
                raw_config_results = pd.concat(raw_config_results, axis=1)
                raw_config_results.columns = raw_config_results.columns.astype(int)
                raw_config_results.sort_index(axis=1, inplace=True)
                raw_config_results_dict[folder]['raw_results'] = raw_config_results

                # collect also average results
                raw_config_results_dict[folder]['average_results'] = raw_config_results.mean(axis=1)

                # collect numer of succesfully finished seeds:
                raw_config_results_dict[folder]['finished_seeds'] = finished_seeds

                eval_losses = 0
                for k, v in evaluation_criteria.items():
                    eval_losses += v * raw_config_results.loc[k]
                eval_losses = eval_losses.describe()
                eval_losses = eval_losses.rename("eval_loss")

                coverage_probability_CI = 0.95
                uQ = (coverage_probability_CI + 1) / 2
                QUANTILE = norm.ppf(uQ)

                eval_losses.loc["Upper-CI"] = (
                        eval_losses.loc["mean"] + QUANTILE * eval_losses.loc["std"] / np.sqrt(eval_losses.loc["count"]))
                eval_losses.loc["Lower-CI"] = (
                        eval_losses.loc["mean"] - QUANTILE * eval_losses.loc["std"] / np.sqrt(eval_losses.loc["count"]))

                raw_config_results_dict[folder]['eval_loss'] = eval_losses

    return raw_config_results_dict


def show_results_hpo_mvnnUB(R,
                            path_prefix,
                            SATS_domain,
                            bidder_type,
                            evaluation_criteria):
    path = os.path.join(path_prefix, SATS_domain, bidder_type)

    if not R:
        logging.warning(f'No results for {path}')
        return

    # collect specific qloss keys and values additionally
    all_keys = list(R[list(R.keys())[0]]['average_results'].index)
    keys_for_v = []
    keys_for_v += [x for x in all_keys if re.compile('test[-uUB]*-qloss').match(x)]  # collect all qlosses
    keys_for_v.sort(key=lambda x: float(re.findall("\d*\.*\d+", x)[0]))  # sort based on q

    keys_for_E = keys_for_v
    #

    E = {}
    for k, v in R.items():
        E[k] = {'eval-loss': v['eval_loss']['mean'],
                'lower-CI': v['eval_loss']['Lower-CI'],
                'upper-CI': v['eval_loss']['Upper-CI'],
                'n-seeds': len(v['finished_seeds']),
                'train-mae': v['average_results']['train-mae'],
                'train-uUB-mae': v['average_results']['train-uUB-mae'],
                'train-time_min': v['average_results']['train-time_min'],
                }
        # update
        for Ekey, vkey in zip(keys_for_E, keys_for_v):
            E[k][Ekey] = v['average_results'][vkey]
    EE = pd.DataFrame.from_dict(E, orient='index')
    EE.sort_values(by='eval-loss', inplace=True, axis=0)
    txt_file = ''
    str_criteria = ' + '.join([f'{v}*{k}' for k, v in evaluation_criteria.items()])
    txt_file += f'\nCONFIG EVALUATION WITH CRITERIA:{str_criteria} (shown are averages)'
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'
    txt_file += str(EE)
    best_config = EE.index[0]
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'

    # txt_file += f'\nWINNER CONFIG:{best_config}\n'
    # txt_file += ''.join(['-'] * 30) + '\n'
    # for k, v in R[best_config]['config'].items():
    # txt_file += f'{k}: {v}\n'
    C = {}
    for k, v in R.items():
        C[k] = v['config']
    CC = pd.DataFrame.from_dict(C, orient='index')
    CC = CC.reindex(EE.index)
    del CC['q'], CC['seeds']
    txt_file += f'\nCONFIG HYPERPARAMETERS (WINNER={best_config}):\n'
    txt_file += ''.join(['-'] * 30) + '\n'
    txt_file += str(CC)
    txt_file += '\n'

    best_config_results = pd.Series.to_dict(R[best_config]['raw_results'].mean(axis=1))
    txt_file += ''.join(['-'] * 30) + '\n'
    txt_file += f'\nDETAILED AVERAGE RESULTS for {best_config}:'
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'
    for k, v in best_config_results.items():
        txt_file += f'{k}: {v}\n'
    txt_file += ''.join(['-'] * 30) + '\n'

    str_criteria = str_criteria.replace('*', '')
    str_criteria = str_criteria.replace(' + ', '_and_')
    str_criteria = str_criteria.replace('-', '_')
    now = datetime.now()
    with open(os.path.join(path,
                           f'{SATS_domain}__{bidder_type}__hpo_winner__{str_criteria}__{now.strftime("%d_%m_%Y_%Hh%Mm%Ss")}.json'),
              'w') as f:
        json.dump({'eval_criterium': evaluation_criteria, best_config: R[best_config]['config'],
                   'results': best_config_results}, f)
    f.close()
    text_file = open(os.path.join(path,
                                  f'{SATS_domain}__{bidder_type}__summary__{str_criteria}__{now.strftime("%d_%m_%Y_%Hh%Mm%Ss")}.txt'),
                     "w")
    text_file.write(txt_file)
    text_file.close()


# for the results of the first hpo, now deprecated
def read_results_hpo_mvnnUB_old(path,
                                evaluation_criteria,
                                config_file_identifier=None):
    raw_config_results_dict = {}

    # loop over config folder
    try:
        folders = os.listdir(path)
    except:
        print(f'No results for {path}')
        return raw_config_results_dict

    for folder in folders:
        print(folder)
        if os.path.isdir(os.path.join(path, folder)):

            if config_file_identifier:
                config_name = [x for x in os.listdir(os.path.join(path, folder)) if config_file_identifier in x]
                if len(config_name) > 1:
                    raise ValueError(f'Multiple .json config files in {path} -> CHECK AND REMOVE')
                config_filepath = os.path.join(path, folder, config_name[0])
            else:
                config_filepath = os.path.join(path, folder, folder + '.json')
            try:
                config = json.load(open(config_filepath, 'r'))
                config_keys = [x for x in config.keys() if bool(re.search(r'config[0-9]+', x))]
                if len(config_keys) == 1:
                    config = config[config_keys[0]]
                elif len(config_keys) > 1:
                    raise ValueError(f'.json config file in {path} potentially broken -> CHECK')
                else:
                    pass
            except FileNotFoundError:
                print(f'NO config.json AVAILABLE IN {folder}!-> CHECK AND REMOVE THIS CONFIG')
                continue
            raw_config_results = []

            # loop over seeds in respective config folder
            finished_seeds = []
            for result_file in os.listdir(os.path.join(path, folder)):

                if bool(re.search(r'Seed\d+_Results.*.json', result_file)):
                    seed = re.search(r'\d+', result_file).group()
                    finished_seeds.append(seed)
                    with open(os.path.join(path, folder, result_file), 'r') as f:
                        result = json.load(f)
                    f.close()
                    #########
                    best_epoch = result['best_epoch']
                    # best_epoch = str(config['epochs']) old
                    #########
                    train_metrics = pd.DataFrame.from_dict(result['metrics']['train'][str(best_epoch)],
                                                           orient='index',
                                                           columns=[seed])

                    train_metrics.index = ['train-' + x for x in train_metrics.index]

                    test_metrics = pd.DataFrame.from_dict(result['metrics']['test'][str(config['epochs'])],
                                                          orient='index',
                                                          columns=[seed])

                    test_metrics.index = ['test-' + x for x in test_metrics.index]
                    test_metrics.loc['train-best_epoch'] = result['best_epoch']
                    test_metrics.loc['train-attempts'] = result['attempt']
                    # test_metrics.loc['bidder-id'] = int(result['bidder_id'])
                    metrics = pd.concat([test_metrics, train_metrics])
                    raw_config_results.append(metrics)

            if len(raw_config_results) > 0:
                raw_config_results_dict[folder] = {'config': config}
                raw_config_results = pd.concat(raw_config_results, axis=1)
                raw_config_results.columns = raw_config_results.columns.astype(int)
                raw_config_results.sort_index(axis=1, inplace=True)
                raw_config_results_dict[folder]['raw_results'] = raw_config_results

                # collect also average results
                raw_config_results_dict[folder]['average_results'] = raw_config_results.mean(axis=1)

                # collect numer of succesfully finished seeds:
                raw_config_results_dict[folder]['finished_seeds'] = finished_seeds

                eval_losses = 0
                for k, v in evaluation_criteria.items():
                    eval_losses += v * raw_config_results.loc[k]
                eval_losses = eval_losses.describe()
                eval_losses = eval_losses.rename("eval_loss")

                coverage_probability_CI = 0.95
                uQ = (coverage_probability_CI + 1) / 2
                QUANTILE = norm.ppf(uQ)

                eval_losses.loc["Upper-CI"] = (
                        eval_losses.loc["mean"] + QUANTILE * eval_losses.loc["std"] / np.sqrt(eval_losses.loc["count"]))
                eval_losses.loc["Lower-CI"] = (
                        eval_losses.loc["mean"] - QUANTILE * eval_losses.loc["std"] / np.sqrt(eval_losses.loc["count"]))

                raw_config_results_dict[folder]['eval_loss'] = eval_losses

    return raw_config_results_dict


# for the results of the first hpo, now deprecated
def show_results_hpo_mvnnUB_old(R,
                                path,
                                evaluation_criteria):
    if not R:
        print(f'No results for {path}')
        return

    # collect specific qloss keys and values additionally
    all_keys = list(R[list(R.keys())[0]]['average_results'].index)
    keys_for_v = []
    keys_for_v += [x for x in all_keys if re.compile('test[-uUB]*-qloss').match(x)]  # collect all qlosses
    keys_for_v.sort(key=lambda x: float(x[-3:]))  # sort based on q
    keys_for_E = keys_for_v
    #

    E = {}
    for k, v in R.items():
        E[k] = {'eval-loss': v['eval_loss']['mean'],
                'lower-CI': v['eval_loss']['Lower-CI'],
                'upper-CI': v['eval_loss']['Upper-CI'],
                'n-seeds': len(v['finished_seeds']),
                'train-mae': v['average_results']['train-mae'],
                'train-uUB-mae': v['average_results']['train-uUB-mae'],
                }
        # update
        for Ekey, vkey in zip(keys_for_E, keys_for_v):
            E[k][Ekey] = v['average_results'][vkey]
    EE = pd.DataFrame.from_dict(E, orient='index')
    EE.sort_values(by='eval-loss', inplace=True, axis=0)
    txt_file = ''
    str_criteria = ' + '.join([f'{v}*{k}' for k, v in evaluation_criteria.items()])
    txt_file += f'\nCONFIG EVALUATION WITH CRITERIA:{str_criteria} (shown are averages)'
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'
    txt_file += str(EE)
    best_config = EE.index[0]
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'

    # txt_file += f'\nWINNER CONFIG:{best_config}\n'
    # txt_file += ''.join(['-'] * 30) + '\n'
    # for k, v in R[best_config]['config'].items():
    # txt_file += f'{k}: {v}\n'
    C = {}
    for k, v in R.items():
        C[k] = v['config']
    CC = pd.DataFrame.from_dict(C, orient='index')
    CC = CC.reindex(EE.index)
    del CC['q'], CC['seeds']
    txt_file += f'\nCONFIG HYPERPARAMETERS (WINNER={best_config}):\n'
    txt_file += ''.join(['-'] * 30) + '\n'
    txt_file += str(CC)
    txt_file += '\n'

    best_config_results = pd.Series.to_dict(R[best_config]['raw_results'].mean(axis=1))
    txt_file += ''.join(['-'] * 30) + '\n'
    txt_file += f'\nDETAILED AVERAGE RESULTS for {best_config}:'
    txt_file += '\n' + ''.join(['-'] * 30) + '\n'
    for k, v in best_config_results.items():
        txt_file += f'{k}: {v}\n'
    txt_file += ''.join(['-'] * 30) + '\n'

    str_criteria = str_criteria.replace('*', '')
    str_criteria = str_criteria.replace(' + ', '_and_')
    str_criteria = str_criteria.replace('-', '_')
    now = datetime.now()
    with open(os.path.join(path, f'hpo_winner__{str_criteria}__{now.strftime("%d_%m_%Y_%Hh%Mm%Ss")}.json'), 'w') as f:
        json.dump({best_config: R[best_config]['config'], 'results': best_config_results}, f)
    f.close()
    text_file = open(os.path.join(path, f'summary__{str_criteria}__{now.strftime("%d_%m_%Y_%Hh%Mm%Ss")}.txt'), "w")
    text_file.write(txt_file)
    text_file.close()


def random_search(number_of_configs,
                  **kwargs
                  ):
    configs = {}

    for i in range(1, number_of_configs + 1):
        name = f'config{i}'
        configs[name] = {}

        for key, value in kwargs.items():

            if isinstance(value, list):
                n = len(value)
                random_idx = np.random.choice(range(n), size=1, replace=False)[0]

                if key == 'epochs':  # fix gradient steps
                    raw_epochs = value[random_idx]
                    configs[name][key] = int(raw_epochs * configs[name]['batch_size'] / configs[name]['num_train_data'])

                else:  # default, i.e., no adjustment
                    configs[name][key] = value[random_idx]

            elif isinstance(value, tuple) and len(value) == 2:

                if key in ['l2', 'lr', 'pi_exp', 'c_exp', 'init_Var', 'pi_above_mean',
                           'clip_grad_norm']:  # log uniform sampling
                    configs[name][key] = 10 ** \
                                         np.random.uniform(low=np.log10(value[0]), high=np.log10(value[1]), size=1)[0]
                else:  # default uniform sampling
                    configs[name][key] = np.random.uniform(low=value[0], high=value[1], size=1)[0]
            else:
                raise ValueError(f'Invalid argument:{value} for {key}! Must be either list or tuple of length 2!')

        # overwrite num_neurons and num_hidden_layers if architecture is specified:
        if configs[name]['architecture']:
            configs[name]['num_hidden_units'] = configs[name]['architecture'][0]
            configs[name]['num_hidden_layers'] = configs[name]['architecture'][1]

    return configs


def bidder_type_to_bidder_id(SATS_domain,
                             bidder_type):
    bidder_id_mappings = {'GSVM': {'national': [6], 'regional': [0, 1, 2, 3, 4, 5]},
                          'LSVM': {'national': [0], 'regional': [1, 2, 3, 4, 5]},
                          'SRVM': {'national': [5, 6], 'regional': [3, 4], 'high_frequency': [2], 'local': [0, 1]},
                          'MRVM': {'national': [7, 8, 9], 'regional': [3, 4, 5, 6], 'local': [0, 1, 2]}
                          }

    bidder_id = np.random.choice(bidder_id_mappings[SATS_domain][bidder_type], size=1, replace=False)[0]
    logging.info(f'BIDDER ID:{bidder_id}')

    return bidder_id


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
