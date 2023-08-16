import numpy as np
import torch
import torch.nn.functional as F

def qloss(preds, targets, q):
    '''
    Returns Quantile-loss for preds given targets and a quantile q.
    '''
    e = np.array(targets) - np.array(preds)
    return np.mean(np.maximum(q * e, (q - 1) * e))


def NOMU_loss(mean_output,
              ub_output,
              target,
              loss_func,
              pi_sqr,
              pi_exp,
              pi_above_mean,
              c_exp,
              n_aug,
              din,
              mean_model,
              ub_model,
              exp_upper_bound_net,
              ntrain
              ):
    '''
    Returns NOMU loss for outputs given a single batch.
    '''

    # 1. LOSS (A)
    # ----------------------------------------------------------
    loss_a = loss_func(mean_output.flatten(), target.flatten())
    # ----------------------------------------------------------

    # 2. LOSS (B)
    # ----------------------------------------------------------
    # smooth-L1-loss(UB,y) penalizes if UB(x)!=y for (x,y) in D_train:
    B1 = torch.nn.SmoothL1Loss(beta=1 / 8)(ub_output.flatten(),target.flatten())
    # max(0,UB-y) penalizes if UB(x) > y for (x,y) in D_train:
    B2 = 0.001*torch.mean(F.relu(ub_output.flatten()-target.flatten()))
    # smooth-L1-loss(max(0,UB-y),0) penalizes UB(x) > y for (x,y) in D_train:
    B3 = 0.5*torch.nn.SmoothL1Loss(beta=1 / 128)(F.relu(ub_output.flatten() - target.flatten()),torch.zeros(target.size()).flatten())
    loss_b = pi_sqr * (B1 + B2 + B3)


    # 3. LOSS (C)
    # ----------------------------------------------------------
    mc_samples = torch.distributions.uniform.Uniform(low=0, high=1).sample((n_aug, din))  # generated per batch

    # max(0,UB-UB100%) penalizes if UB(x) > UB100%(x) for x in D_art:
    C1 = 0.25*F.relu(ub_model(mc_samples) - exp_upper_bound_net(mc_samples).detach()).flatten()
    # max(0,MEAN-UB) penalizes if MEAN(x) > UB(x) for x in D_art:
    C2 = pi_above_mean*F.relu(mean_model(mc_samples).detach() - ub_model(mc_samples)).flatten()
    # smooth-L1-loss(C1 + C2,0) penalizes if C1(x) + C2(x) > 0 for x in D_art:
    C3 = c_exp * torch.nn.SmoothL1Loss(beta=1 / 64)(C1 + C2, torch.zeros(n_aug, 1).flatten())

    # 1 + ELU{-c_exp*(0.01 + min(UB,UB100%) - MEAN)} pushed UB(x) up for x in D_art:
    C4 = torch.mean(1 + torch.nn.ELU()(- c_exp * (0.01 + torch.minimum(ub_model(mc_samples), exp_upper_bound_net(mc_samples).detach()) - mean_model(mc_samples).detach())))

    loss_c = pi_exp * (C3 + C4)/ntrain


    return loss_a, loss_b, loss_c
