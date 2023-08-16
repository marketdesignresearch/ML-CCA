from test_training import train_mvnn
import argparse


parser = argparse.ArgumentParser()



# SATS parameter arguments 
parser.add_argument("--SATS_seed", help="SATS seed", default= 1, type=int)
parser.add_argument("--SATS_domain", help="SATS domain", default= 'LSVM', type=str)

#train parameter arguments
parser.add_argument("--l2", help="l2 regularization", type=float, default= 1e-4)
parser.add_argument("--lr", help="learning rate", default= 0.005, type=float)
parser.add_argument("--epochs", help="number of epochs", default= 30, type=int)
parser.add_argument("--batch_size", help="batch size", type=int, default = 1)
parser.add_argument("--max_linear_prices_multiplier", help="max linear prices multiplier", default= 2, type=float)
parser.add_argument("--number_train_data_points", help="number of training data points", default= 50, type=int)
parser.add_argument("--number_val_data_points", help="number of validation data points", default= 100, type=int)
parser.add_argument("--train_data_seed", help="training data seed", default= 1, type=int)
parser.add_argument("--val_data_seed", help="validation data seed", default= 42, type=int)
parser.add_argument("--use_gradient_clipping", help="use gradient clipping", default= 'false', type=str)
parser.add_argument("--price_file", help="name of the price file, without domain", default= 'values_for_null_price', type=str)

# MVNN parameter arguments
parser.add_argument("--num_hidden_layers", help="number of hidden layers", default= 1, type=int)
parser.add_argument("--num_hidden_units", help="number of hidden units", default= 20, type=int)
parser.add_argument("--dropout_prob", help="dropout probability", default= 0, type=float)
parser.add_argument("--lin_skip_connection", help="initialization method", default= 'false', type=str)

args = parser.parse_args()

lin_skip_connection = (args.lin_skip_connection.lower() == 'true')
use_gradient_clipping = (args.use_gradient_clipping.lower() == 'true')


# %% SET PARAMETERS

# 0. W&B
# ----------------------------------------------
wandb_tracking =  True
# ----------------------------------------------

# 1. SATS Parameters
# ----------------------------------------------
SATS_parameters = {"SATS_domain": args.SATS_domain,
                   "isLegacy": False,
                   "SATS_seed": args.SATS_seed,
                    }
# ----------------------------------------------

# 2. Training Parameters
# ----------------------------------------------
TRAIN_parameters = {"number_train_data_points": args.number_train_data_points,
                    "train_data_seed":args.train_data_seed,
                    "number_val_data_points": args.number_val_data_points,
                    "val_data_seed":args.val_data_seed,
                    "max_linear_prices_multiplier": args.max_linear_prices_multiplier,  # sample from larger prices
                    "price_file_name": args.price_file,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'l2_reg': args.l2,
                    'learning_rate': args.lr,
                    'clip_grad_norm': 1,
                    'use_gradient_clipping': use_gradient_clipping,
                    'print_frequency': 1
                    }
# ----------------------------------------------

# 3. MVNN Parameters
# ----------------------------------------------
MVNN_parameters = {'num_hidden_layers': args.num_hidden_layers,
                   'num_hidden_units': args.num_hidden_units,
                   'layer_type': 'MVNNLayerReLUProjected',
                   'target_max': 1, # TODO: check
                   'lin_skip_connection': lin_skip_connection, # TODO: discuss if we ever want True
                   'dropout_prob': args.dropout_prob,
                   'init_method':'custom',
                   'random_ts': [0,1],
                   'trainable_ts': True,
                   'init_E': 1,
                   'init_Var': 0.09,
                   'init_b': 0.05,
                   'init_bias': 0.05,
                   'init_little_const': 0.1
                    }     


all_models, all_metrics = train_mvnn(SATS_parameters=SATS_parameters,
                                         TRAIN_parameters=TRAIN_parameters,
                                         MVNN_parameters=MVNN_parameters,
                                         wandb_tracking=wandb_tracking)    