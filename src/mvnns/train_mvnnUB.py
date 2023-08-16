import numpy as np
import torch
from mvnns.losses import NOMU_loss
from mvnns.metrics import compute_metrics, compute_metrics_UB


def train(model, device, train_loader, num_train_data, optimizer, clip_grad_norm, epoch, pi_sqr,
          pi_exp, pi_above_mean, c_exp, n_aug, target_max, loss_func, exp_upper_bound_net,
          dropout_prob, q, *args, **kwargs):
    mean_model, ub_model = model
    mean_model.train()
    ub_model.train()

    mean_model.set_dropout_prob(dropout_prob)
    ub_model.set_dropout_prob(dropout_prob)

    preds, preds_UB, targets = [], [], []
    total_loss = 0
    loss_a_total = 0
    loss_b_total = 0
    loss_c_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        mean_output = mean_model(data)
        ub_output = ub_model(data)

        preds.extend(mean_output.detach().cpu().numpy().flatten().tolist())
        preds_UB.extend(ub_output.detach().cpu().numpy().flatten().tolist())
        targets.extend(target.detach().cpu().numpy().flatten().tolist())

        nbatch = len(preds)

        # Calculate NOMU loss for a single batch: returns loss_a ,loss_b, loss_c
        loss_a, loss_b, loss_c = NOMU_loss(mean_output=mean_output,
                                           ub_output=ub_output,
                                           target=target,
                                           loss_func=loss_func,
                                           pi_sqr=pi_sqr,
                                           pi_exp=pi_exp,
                                           pi_above_mean=pi_above_mean,
                                           c_exp=c_exp,
                                           n_aug=n_aug,
                                           din=data.shape[1],
                                           mean_model=mean_model,
                                           ub_model=ub_model,
                                           exp_upper_bound_net=exp_upper_bound_net,
                                           ntrain=num_train_data
                                           )

        ######
        loss = loss_a + loss_b + loss_c
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ub_model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += float(loss_a + loss_b + loss_c) * nbatch
        loss_a_total += float(loss_a) * nbatch
        loss_b_total += float(loss_b) * nbatch
        loss_c_total += float(loss_c) * nbatch

    # UPDATE METRICS
    metrics = {'loss': total_loss / num_train_data,
               'loss_a': loss_a_total / num_train_data,
               'loss_b': loss_b_total / num_train_data,
               'loss_c': loss_c_total / num_train_data}

    # Scaled metrics
    metrics.update(compute_metrics(preds, targets, q=q, scaled=True))
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q, scaled=True))

    preds, preds_UB, targets = (np.array(preds) * target_max).tolist(), \
                               (np.array(preds_UB) * target_max).tolist(), \
                               (np.array(targets) * target_max).tolist()

    # Unscaled metrics (original scale)
    metrics.update(compute_metrics(preds, targets, q=q))
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q))
    return metrics
