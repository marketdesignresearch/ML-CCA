# Libs
import numpy as np
import torch
import sklearn.metrics
from scipy import stats as scipy_stats
import wandb
import time
import logging

# Own Libs
from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
from milps.gurobi_mip_mvnn_single_bidder_util_max import GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX
from milps.gurobi_mip_mvnn_generic_single_bidder_util_max import GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX 


# from pdb import set_trace
#%%
def dq_train_mvnn_helper(model,
                        optimizer,
                        use_gradient_clipping,
                        clip_grad_norm,
                        train_loader_demand_queries,
                        SATS_domain,
                        bidder_id,
                        GSVM_national_bidder_goods_of_interest,
                        device,
                        MIP_parameters
                        ):

    model.train()
    loss_dq_list = []

    for batch_idx, (demand_vector, price_vector) in enumerate(train_loader_demand_queries):
        price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
        optimizer.zero_grad()

        #--------------------------------
        # IMPORTANT: we need to transform the weights and the biases of the MVNN to be non-positive and non-negative, respectively.
        model.transform_weights()
        #--------------------------------

        # compute the network's predicted answer to the demand query
        if SATS_domain in ['GSVM', 'LSVM']:
            solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(model=model,
                                                            SATS_domain = SATS_domain,
                                                            bidder_id = bidder_id,
                                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
            solver.generate_mip(price_vector.numpy()[0])
        elif SATS_domain in ['SRVM', 'MRVM']: 
            solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=model)  # if the domain is generic -> use the new MIP 
            solver.generate_mip(price_vector.numpy()[0])
        else:
            raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')
        
        try: 
            predicted_demand = solver.solve_mip(outputFlag=False,
                                                verbose = False,
                                                timeLimit = MIP_parameters["timeLimit"],
                                                MIPGap = MIP_parameters["MIPGap"],
                                                IntFeasTol = MIP_parameters["IntFeasTol"],
                                                FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                )
            predicted_demand = np.array(predicted_demand)
        except:
            print('--- MIP is unbounded, skipping this sample! ---')
            continue 

        # get the predicted value for that answer
        predicted_value = model(torch.from_numpy(predicted_demand).float())

        predicted_utility = predicted_value - torch.dot(price_vector.flatten(), torch.from_numpy(predicted_demand).float())

        # get the predicted utility for the actual demand vector
        predicted_value_at_true_demand = model(demand_vector)

        predicted_utility_at_true_demand = predicted_value_at_true_demand - torch.dot(price_vector.flatten(), demand_vector.flatten())


        # compute the loss
        predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
        if predicted_utility_difference < 0:
            print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

        loss = torch.relu(predicted_utility_difference)   # for numerical stability
        loss_dq_list.append(loss.detach().numpy())
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

    return np.mean(loss_dq_list)


def dq_val_mvnn(trained_model,
                val_loader,
                val_loader_gen_only,
                train_loader,
                SATS_auction_instance,
                bidder_id,
                scale,
                SATS_domain,
                GSVM_national_bidder_goods_of_interest,
                device,
                MIP_parameters
                ):
    
    
    trained_model.eval()
    val_metrics = {}

    scaled_value_preds = []
    demand_vectors = []
    price_vectors = []
    with torch.no_grad():
        for demand_vector, price_vector in val_loader:
            price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
            scaled_value_prediction = trained_model(demand_vector)
            scaled_value_preds.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
            demand_vectors.extend(demand_vector.cpu().numpy().tolist())
            price_vectors.extend(price_vector.cpu().numpy().tolist())

    scaled_value_preds = np.array(scaled_value_preds)
    true_values = np.array([SATS_auction_instance.calculate_value(bidder_id, demand_vector) for demand_vector in demand_vectors])
    scaled_true_values = true_values/scale

    inferred_values = np.array([np.dot(price_vector, demand_vector) for (price_vector, demand_vector) in zip(price_vectors, demand_vectors)])

    value_preds = scaled_value_preds * scale

    common_scale = np.mean(true_values)
    common_scale_true_values = true_values / common_scale
    common_scale_value_preds = value_preds / common_scale

    # 1. generalization performance measures (on the validation set, that is drawn using price vectors)
    # --------------------------------------
    val_metrics['r2'] = sklearn.metrics.r2_score(y_true=true_values, y_pred= value_preds)  # This is R2 coefficient of determination
    val_metrics['kendall_tau'] = scipy_stats.kendalltau(scaled_value_preds, scaled_true_values).correlation
    val_metrics['mae'] = sklearn.metrics.mean_absolute_error(value_preds, true_values)
    val_metrics['mae_scaled'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds, common_scale_true_values)
    val_metrics['r2_centered'] = sklearn.metrics.r2_score(y_true=true_values - np.mean(true_values), y_pred= value_preds - np.mean(value_preds)) 
    # a centered R2, because constant shifts in model predictions should not really affect us  

    val_metrics['scaled_true_values'] = scaled_true_values  # also store all true /predicted values so that we can make true vs predicted plots
    val_metrics['scaled_predicted_values'] = scaled_value_preds
    val_metrics['scaled_inferred_values'] = inferred_values


    # --------------------------------------
    
    # 1.5. If there is a specific loader for generalization performance -> use those for the generalization performance measures (on the validation set, that is drawn by uniformly sampling bundles)
    # --------------------------------------
    if val_loader_gen_only is not None:
        scaled_value_preds_generalization = []
        demand_vectors_generalization = []
        price_vectors_generalization = []
        with torch.no_grad():
            for demand_vector, price_vector in val_loader_gen_only:
                price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
                scaled_value_prediction = trained_model(demand_vector)
                scaled_value_preds_generalization.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
                demand_vectors_generalization.extend(demand_vector.cpu().numpy().tolist())
                price_vectors_generalization.extend(price_vector.cpu().numpy().tolist())
    
        scaled_value_preds_generalization = np.array(scaled_value_preds_generalization)
        true_values_generalization = np.array([SATS_auction_instance.calculate_value(bidder_id, demand_vector) for demand_vector in demand_vectors_generalization])
        scaled_true_values_generalization = true_values_generalization/scale
        value_preds_generalization = scaled_value_preds_generalization * scale
        
        
        val_metrics['r2_generalization'] = sklearn.metrics.r2_score(y_true=true_values_generalization, y_pred=value_preds_generalization)  # This is R2 coefficient of determination
        val_metrics['kendall_tau_generalization'] = scipy_stats.kendalltau(true_values_generalization, value_preds_generalization).correlation
        val_metrics['mae_generalization'] = sklearn.metrics.mean_absolute_error(true_values_generalization, value_preds_generalization)
        val_metrics['r2_centered_generalization'] = sklearn.metrics.r2_score(y_true=true_values_generalization - np.mean(true_values_generalization), y_pred= value_preds_generalization - np.mean(value_preds_generalization))

        common_scale = np.mean(true_values_generalization)
        common_scale_generalazation = common_scale  # need the results for regret
        common_scale_true_values_generalization = true_values_generalization / common_scale
        common_scale_value_preds_generalization = value_preds_generalization / common_scale

        val_metrics['mae_scaled_generalization'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds_generalization, common_scale_true_values_generalization)
        
        val_metrics['scaled_true_values_generalization'] = scaled_true_values_generalization  # also store all true /predicted values so that we can make true vs predicted plots
        val_metrics['scaled_predicted_values_generalization'] = scaled_value_preds_generalization
        print(f'Val metrics, generalization only set. R2: {val_metrics["r2_generalization"]}, R2C: {val_metrics["r2_centered_generalization"]} Kendall Tau: {val_metrics["kendall_tau_generalization"]}, MAE: {val_metrics["mae_generalization"]}')

    # --------------------------------------
    # 1.6 If the training loader is given -> also measure predictive performance on that 
    if train_loader is not None:
        scaled_value_preds_train = []
        demand_vectors_train = []
        price_vectors_train = []
        with torch.no_grad():
            for demand_vector, price_vector in train_loader:
                price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
                scaled_value_prediction = trained_model(demand_vector)
                scaled_value_preds_train.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
                demand_vectors_train.extend(demand_vector.cpu().numpy().tolist())
                price_vectors_train.extend(price_vector.cpu().numpy().tolist())
    
        scaled_value_preds_train = np.array(scaled_value_preds_train)
        true_values_train = np.array([SATS_auction_instance.calculate_value(bidder_id, demand_vector) for demand_vector in demand_vectors_train])

        scaled_true_values_train = true_values_train/scale
        value_preds_train = scaled_value_preds_train * scale

        inferred_values_train = np.array([np.dot(price_vector, demand_vector) for (price_vector, demand_vector) in zip(price_vectors_train, demand_vectors_train)])

        common_scale = np.mean(true_values_train)
        common_scale_true_values_train = true_values_train / common_scale
        common_scale_value_preds_train = value_preds_train / common_scale
        
        val_metrics['r2_train'] = sklearn.metrics.r2_score(y_true=true_values_train, y_pred=value_preds_train)
        val_metrics['r2_centered_train'] = sklearn.metrics.r2_score(y_true=true_values_train - np.mean(true_values_train), y_pred= value_preds_train - np.mean(value_preds_train))
        val_metrics['kendall_tau_train'] = scipy_stats.kendalltau(true_values_train, value_preds_train).correlation
        val_metrics['mae_train'] = sklearn.metrics.mean_absolute_error(true_values_train, value_preds_train)
        val_metrics['mae_scaled_train'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds_train, common_scale_true_values_train)

        # also store all true /predicted values so that we can make true vs predicted plots
        val_metrics['scaled_true_values_train'] = scaled_true_values_train
        val_metrics['scaled_predicted_values_train'] = scaled_value_preds_train
        val_metrics['scaled_inferred_values_train'] = inferred_values_train

        print(f'Predictive performance on training set. R2: {val_metrics["r2_train"]}, R2C: {val_metrics["r2_centered_train"]} Kendall Tau: {val_metrics["kendall_tau_train"]}, MAE: {val_metrics["mae_train"]}')

    
    # 2. DQ loss performance measure (same as training loss)
    # --------------------------------------
    
    # Create the common MVNN MIP solver
    if SATS_domain in ['GSVM', 'LSVM']:
        solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(model=trained_model,
                                                        SATS_domain = SATS_domain,
                                                        bidder_id = bidder_id,
                                                        GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
        solver.generate_mip(price_vector.numpy()[0])
    elif SATS_domain in ['SRVM', 'MRVM']:
        solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=trained_model)
        solver.generate_mip(price_vector.numpy()[0])
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')
    
    val_dq_loss = 0 
    predicted_demands = [] 
    for (j, price_vector) in enumerate(price_vectors): 
        # update the prices in the MIP  objective to the price vector of the current datapoint
        solver.update_prices_in_objective(price_vector)

        # compute the network's predicted answer to the demand query
    
        try: 
            predicted_demand = solver.solve_mip(outputFlag=False,
                                                verbose = False,
                                                timeLimit = MIP_parameters["timeLimit"],
                                                MIPGap = MIP_parameters["MIPGap"],
                                                IntFeasTol = MIP_parameters["IntFeasTol"],
                                                FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                )
            predicted_demand = np.array(predicted_demand)
        except:
            print('MIP is unbounded, something is wrong!')
            predicted_demand = np.ones(demand_vector.shape[0])

        predicted_demands.append(predicted_demand)

        # get the predicted value for that answer
        predicted_value = trained_model(torch.from_numpy(predicted_demand).float()).item()

        predicted_utility = predicted_value - np.dot(price_vector, predicted_demand)

        # get the predicted utility for the actual demand vector
        demand_vector = demand_vectors[j]
        predicted_value_at_true_demand = scaled_value_preds[j]

        predicted_utility_at_true_demand = predicted_value_at_true_demand - np.dot(price_vector, demand_vector)

        # compute the loss
        predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
        val_dq_loss = val_dq_loss + predicted_utility_difference
        if predicted_utility_difference < 0:
            print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')
            # solver._print_info() # NOTE: if you print info on a MIP that is unbounded, it will crush... 


    val_metrics['scaled_dq_loss'] = val_dq_loss / len(price_vectors)
    # --------------------------------------


    # 3. Regret performance measure 
    # --------------------------------------
    regret = 0
    for (j, price_vector) in enumerate(price_vectors):
        # calculate the optimal true utility for the true demand vector
        scaled_true_value =  scaled_true_values[j]
        scaled_true_opt_utility = scaled_true_value - np.dot(price_vector, demand_vectors[j])

        # calculate the true utility for the predicted demand vector
        predicted_demand = predicted_demands[j]
        scaled_value_at_predicted_demand = SATS_auction_instance.calculate_value(bidder_id, predicted_demand) / scale
        scaled_utility_at_predicted_demand = scaled_value_at_predicted_demand - np.dot(price_vector, predicted_demand)

        regret = regret + (scaled_true_opt_utility - scaled_utility_at_predicted_demand)


    val_metrics['mean_regret'] = (regret * scale) / len(price_vectors)
    val_metrics['mean_regret_scaled'] = val_metrics['mean_regret'] / common_scale_generalazation # regret scaled by the common scale of the generalization set to make numbers interpretable
    # --------------------------------------

    return val_metrics

#%%
def dq_train_mvnn(SATS_auction_instance,
                  capacity_generic_goods,
                  P_train,
                  X_train,
                  P_val,
                  X_val,
                  P_val_gen_only,
                  X_val_gen_only,
                  SATS_parameters,
                  TRAIN_parameters,
                  MVNN_parameters,
                  MIP_parameters,
                  bidder_id,
                  bidder_scale,
                  GSVM_national_bidder_goods_of_interest,
                  wandb_tracking
                  ):
    

    SATS_domain = SATS_parameters['SATS_domain']
  
    batch_size = TRAIN_parameters['batch_size']
    if batch_size != 1:
        raise NotImplementedError('batch_size != 1 is not implemented yet')
    epochs = TRAIN_parameters['epochs'] 
    l2_reg = TRAIN_parameters['l2_reg']
    learning_rate = TRAIN_parameters['learning_rate']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    print_frequency = TRAIN_parameters['print_frequency']

    induced_values = []
    for i in range(len(P_train)):
        induced_values.append(np.dot(P_train[i], X_train[i]))


    train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(P_train).float()
                                                                  )


    train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                              batch_size= batch_size,
                                                              shuffle=True)
    if P_val is not None and X_val is not None:
        val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                    torch.from_numpy(P_val).float()
                                                                    )


        val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                batch_size= batch_size,
                                                                shuffle=True) 
        
    else:
        val_loader_demand_queries = None

    if P_val_gen_only is not None and X_val_gen_only is not None:
        val_dataset_gen_only = torch.utils.data.TensorDataset(torch.from_numpy(X_val_gen_only).float(),
                                                                    torch.from_numpy(P_val_gen_only).float()
                                                                    )


        val_loader_gen_only = torch.utils.data.DataLoader(val_dataset_gen_only,
                                                                batch_size= batch_size,
                                                                shuffle=False) 
        
    else:
        val_loader_gen_only = None



    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max'] 
    lin_skip_connection = MVNN_parameters['lin_skip_connection'] 
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']   
    init_little_const = MVNN_parameters['init_little_const']

    print('Creating MVNN model with parameters:')
    print(f'num_hidden_layers: {num_hidden_layers}')
    print(f'num_hidden_units: {num_hidden_units}')
    print(f'regularisation: {l2_reg}')
    print(f'learning_rate: {learning_rate}')
    print(f'clip_grad_norm: {clip_grad_norm}')

    if SATS_domain in ['GSVM', 'LSVM']:
        model = MVNN(input_dim=X_train.shape[1],
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const
                    )
    elif SATS_domain in ['SRVM', 'MRVM']:
        model = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                    num_hidden_layers=num_hidden_layers,
                    num_hidden_units=num_hidden_units,
                    layer_type=layer_type,
                    target_max=target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const,
                    capacity_generic_goods=capacity_generic_goods
                    )
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')



    # make sure ts have no regularisation (the bigger t the more regular)
    l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
    no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
    for p in [*model.named_parameters()]:
        if 'ts' in p[0]:
            no_l2_reg_parameters['params'].append(p[1])
        else:
            l2_reg_parameters['params'].append(p[1])

    optimizer = torch.optim.Adam([l2_reg_parameters,no_l2_reg_parameters],
                                 lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           float(epochs))

    metrics = {}

    if wandb_tracking: 
        for loss_str in ['train_loss_dq_scaled', 'val_loss_dq_scaled', 'mean_regret_scaled', 
                         'val_r2_scaled', 'val_KT_scaled', 'val_MAE_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs")

    for epoch in range(epochs):
        train_loss_dq = dq_train_mvnn_helper(model,
                                            optimizer,
                                            use_gradient_clipping,
                                            clip_grad_norm,
                                            train_loader_demand_queries,
                                            SATS_domain = SATS_domain,
                                            bidder_id = bidder_id,
                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest,
                                            device=torch.device('cpu'),
                                            MIP_parameters = MIP_parameters
                                            )
        if val_loader_demand_queries is not None:
            val_metrics = dq_val_mvnn(trained_model = model,
                                     val_loader = val_loader_demand_queries,
                                     val_loader_gen_only = val_loader_gen_only,
                                     train_loader = train_loader_demand_queries,
                                     SATS_auction_instance = SATS_auction_instance,
                                     SATS_domain= SATS_domain,
                                     GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                     bidder_id = bidder_id,
                                     scale = bidder_scale,
                                     device=torch.device('cpu'),
                                     MIP_parameters = MIP_parameters
                                     )

        scheduler.step()
        if val_loader_demand_queries is not None:
            metrics[epoch] = val_metrics
        else: 
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_dq_loss"] = train_loss_dq

        if wandb_tracking:
            wandb.log({f"Bidder_{bidder_id}_train_loss_dq": train_loss_dq, 
                       f"Bidder_{bidder_id}_val_loss_dq": val_metrics["scaled_dq_loss"], 
                       f"Bidder_{bidder_id}_mean_regret": val_metrics["mean_regret"], 
                       f"Bidder_{bidder_id}_val_r2": val_metrics["r2"], 
                       f"Bidder_{bidder_id}_val_KT": val_metrics["kendall_tau"], 
                       f"Bidder_{bidder_id}_val_MAE": val_metrics["mae"],
                       "epochs": epoch})

        # TODO: remove later since we have W&B
        if epoch % print_frequency == 0:
            if val_loader_demand_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}, val_dq_loss:{val_metrics["scaled_dq_loss"]:.5f}, val_mean_regret:{val_metrics["mean_regret"]:.5f}, val_r2:{val_metrics["r2"]:.5f}, val_kendall_tau:{val_metrics["kendall_tau"]:.5f}, val_mae:{val_metrics["mae"]:.5f}')
            else: 
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}')
    
    return model, metrics



def dq_train_mvnn_parallel(bidder_id,# bidder_id must be first position for joblib.parallel!!
                           capacity_generic_goods,
                           elicited_dqs,
                           dqs_val_data,
                           scales,
                           SATS_parameters,
                           TRAIN_parameters,
                           MVNN_parameters,
                           MIP_parameters,
                           GSVM_national_bidder_goods_of_interest,
                           wandb_tracking,
                           num_cpu_per_job
                        ):
    
    # Preparation for "PARALLEL TRAINING"
    # --------------------------------------
    train_start_time = time.time()

    torch.set_num_threads(num_cpu_per_job)

    bidder_name = f'Bidder_{bidder_id}'
    P_train = elicited_dqs[bidder_name][1] / scales[bidder_name]  # scale the data to the range [0,1]
    X_train = elicited_dqs[bidder_name][0]
    if dqs_val_data:
        P_val = dqs_val_data[bidder_name][1]
        X_val = dqs_val_data[bidder_name][0]
    else:
        P_val = None
        X_val = None
    TRAIN_parameters = TRAIN_parameters[bidder_name]
    MVNN_parameters = MVNN_parameters[bidder_name]
    # --------------------------------------

    SATS_domain = SATS_parameters['SATS_domain']
  
    batch_size = TRAIN_parameters['batch_size']
    if batch_size != 1:
        raise NotImplementedError('batch_size != 1 is not implemented yet')
    epochs = TRAIN_parameters['epochs'] #TODO: change
    l2_reg = TRAIN_parameters['l2_reg']
    learning_rate = TRAIN_parameters['learning_rate']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    print_frequency = TRAIN_parameters['print_frequency']

    induced_values = []
    for i in range(len(P_train)):
        induced_values.append(np.dot(P_train[i], X_train[i]))


    train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(P_train).float()
                                                                  )


    train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                              batch_size= batch_size,
                                                              shuffle=True)
    if P_val is not None and X_val is not None:
        val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                    torch.from_numpy(P_val).float()
                                                                    )


        val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                batch_size= batch_size,
                                                                shuffle=True)
    else:
        val_loader_demand_queries = None



    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max'] # TODO: check
    lin_skip_connection = MVNN_parameters['lin_skip_connection']
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']   
    init_little_const = MVNN_parameters['init_little_const']

    print('Creating MVNN model with parameters:')
    print(f'num_hidden_layers: {num_hidden_layers}')
    print(f'num_hidden_units: {num_hidden_units}')
    print(f'regularisation: {l2_reg}')
    print(f'learning_rate: {learning_rate}')
    print(f'clip_grad_norm: {clip_grad_norm}')

    if SATS_domain in ['GSVM', 'LSVM']:
        model = MVNN(input_dim=X_train.shape[1],
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const
                    )
    elif SATS_domain in ['SRVM', 'MRVM']:
        model = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                    num_hidden_layers=num_hidden_layers,
                    num_hidden_units=num_hidden_units,
                    layer_type=layer_type,
                    target_max=target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const,
                    capacity_generic_goods=capacity_generic_goods
                    )
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')



    # make sure ts have no regularisation (the bigger t the more regular)
    l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
    no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
    for p in [*model.named_parameters()]:
        if 'ts' in p[0]:
            no_l2_reg_parameters['params'].append(p[1])
        else:
            l2_reg_parameters['params'].append(p[1])

    optimizer = torch.optim.Adam([l2_reg_parameters,no_l2_reg_parameters],
                                 lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           float(epochs))

    metrics = {}

    if wandb_tracking: 
        for loss_str in ['train_loss_dq_scaled', 'val_loss_dq_scaled', 'mean_regret_scaled', 
                         'val_r2_scaled', 'val_KT_scaled', 'val_MAE_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs")

    for epoch in range(epochs):
        train_loss_dq = dq_train_mvnn_helper(model,
                                            optimizer,
                                            use_gradient_clipping,
                                            clip_grad_norm,
                                            train_loader_demand_queries,
                                            SATS_domain = SATS_domain,
                                            bidder_id = bidder_id,
                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest,
                                            device=torch.device('cpu'),
                                            MIP_parameters = MIP_parameters
                                            )
        val_metrics = None
        if val_loader_demand_queries is not None:
            raise NotImplementedError('Parallel training does not support validation yet')
            # SATS_auction_instance not pickable for parallel training
            """ val_metrics = dq_val_mvnn(trained_model = model,
                                     val_loader = val_loader_demand_queries,
                                     SATS_auction_instance = SATS_auction_instance,
                                     SATS_domain= SATS_domain,
                                     GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                     bidder_id = bidder_id,
                                     scale = TRAIN_parameters['scales'][f"Bidder_{bidder_id}"],
                                     device=torch.device('cpu'),
                                     MIP_parameters = MIP_parameters
                                     ) """

        scheduler.step()
        if val_loader_demand_queries is not None:
            metrics[epoch] = val_metrics
        else: 
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_dq_loss"] = train_loss_dq

        if wandb_tracking:
            if val_loader_demand_queries is not None:
                wandb.log({f"Bidder_{bidder_id}_train_loss_dq_scaled": train_loss_dq, 
                        f"Bidder_{bidder_id}_val_loss_dq_scaled": val_metrics["scaled_dq_loss"], 
                        f"Bidder_{bidder_id}_mean_regret_scaled": val_metrics["scaled_mean_regret"], 
                        f"Bidder_{bidder_id}_val_r2_scaled": val_metrics["scaled_r2"], 
                        f"Bidder_{bidder_id}_val_KT_scaled": val_metrics["scaled_kendall_tau"], 
                        f"Bidder_{bidder_id}_val_MAE_scaled": val_metrics["scaled_mae"],
                        "epochs": epoch})
            else:
                wandb.log({f"Bidder_{bidder_id}_train_loss_dq_scaled": train_loss_dq,
                           "epochs": epoch})

        # TODO: remove later since we have W&B
        if epoch % print_frequency == 0:
            if val_loader_demand_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}, val_dq_loss:{val_metrics["scaled_dq_loss"]:.5f}, val_mean_regret:{val_metrics["mean_regret"]:.5f}, val_r2:{val_metrics["r2"]:.5f}, val_kendall_tau:{val_metrics["kendall_tau"]:.5f}, val_mae:{val_metrics["mae"]:.5f}')
            else: 
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}')

    # NEW: for parallel training measure time here
    # --------------------------------------
    train_end_time = time.time()
    metrics["train_time_elapsed"] = train_end_time - train_start_time
    logging.info(f'Training time for {bidder_name}: {metrics["train_time_elapsed"]}')
    # --------------------------------------

    # New return format for parallel training
    # --------------------------------------
    return {bidder_name: [model, metrics]}