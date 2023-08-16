import os
import torch.utils.data.dataset
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from mvnns.metrics import compute_metrics, compute_metrics_UB
from mvnns.layers import *
from util import total_range

def test(model,
         device,
         loader,
         valid_true,
         target_max,
         seed,
         loss_func,
         exp_upper_bound_net,
         plot=False,
         new_test_plot=False,
         log_path=None,
         q=None,
         send_to=None):

    mean_model, ub_model = model
    mean_model.eval()
    ub_model.eval()

    preds, preds_UB, targets, preds_EXP_UB = [], [], [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            mean_output = mean_model(data)
            ub_output = ub_model(data)
            exp_ub_output = exp_upper_bound_net(data)

            preds.extend(mean_output.detach().cpu().numpy().flatten().tolist())
            preds_UB.extend(ub_output.detach().cpu().numpy().flatten().tolist())
            preds_EXP_UB.extend(exp_ub_output.detach().cpu().numpy().flatten().tolist())
            targets.extend(target.detach().cpu().numpy().flatten().tolist())

    metrics = {}

    # Scaled metrics
    metrics.update(compute_metrics(preds, targets, q=q, scaled=True))
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q, scaled=True))

    preds, preds_UB, targets, preds_EXP_UB = (np.array(preds) * target_max).tolist(), \
                                             (np.array(preds_UB) * target_max).tolist(), \
                                             (np.array(targets) * target_max).tolist(), \
                                             (np.array(preds_EXP_UB) * target_max).tolist()

    # Unscaled metrics (original scale)
    metrics.update(compute_metrics(preds, targets, q=q))
    metrics.update(compute_metrics_UB(preds_UB, targets, q=q))

    # Check 0-to-0 mapping
    metrics['0-to-0'] = int(np.all(mean_model(torch.zeros((1, data.shape[1]))).detach().cpu().numpy() == 0))
    metrics['uUB-0-to-0'] = int(np.all(ub_model(torch.zeros((1, data.shape[1]))).detach().cpu().numpy() == 0))

    # Checking monotonicity
    inp = torch.zeros((1, data.shape[1]))
    mean_outputs = []
    ub_outputs = []
    for i in range(data.shape[1]):
        inp[0, i] = 1
        mean_outputs.append(float(mean_model(inp).detach().cpu().numpy()))
        ub_outputs.append(float(ub_model(inp).detach().cpu().numpy()))
    metrics['monotonicity_satisfied'] = int(sorted(mean_outputs) == mean_outputs)
    metrics['uUB-monotonicity_satisfied'] = int(sorted(ub_outputs) == ub_outputs)

    eval_type = 'valid' if valid_true else 'test'

    # NEW PLOT that measures quality of UB: use preds,preds_UB and targets which are on original scale
    if new_test_plot:


        indices = np.random.choice(len(preds), min(200,len(preds)), replace=False)

        preds_subset = np.array(preds)[indices]
        preds_UB_subset = np.array(preds_UB)[indices]
        targets_subset = np.array(targets)[indices]
        preds_EXP_UB_subset = np.array(preds_EXP_UB)[indices]

        os.makedirs(log_path, exist_ok=True)
        horizontal_axis = preds_EXP_UB_subset - preds_subset
        dat_min, dat_max = total_range(horizontal_axis)
        ymin, y_max = total_range(targets_subset - preds_subset, preds_UB_subset - preds_subset)
        # plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(16, 9))
        plt.plot([0, dat_max], [0, dat_max], color='orange', label=r'$UB_{100\%}-\hat{\mu}$')
        plt.plot([0, dat_max], [0, 0], color='darkred', label=r'$\hat{\mu}-\hat{\mu}$', linewidth=1)
        bad_indi = np.argwhere(targets_subset - preds_UB_subset > 0)
        good_indi = np.argwhere(targets_subset - preds_UB_subset <= 0)
        plt.vlines(x=horizontal_axis[good_indi], ymin=(targets_subset - preds_subset)[good_indi],
                   ymax=(preds_UB_subset - preds_subset)[good_indi], colors="green", linewidth=0.5)
        plt.vlines(x=horizontal_axis[bad_indi], ymin=(preds_UB_subset - preds_subset)[bad_indi],
                   ymax=(targets_subset - preds_subset)[bad_indi], colors="red", linewidth=0.5)
        plt.scatter(horizontal_axis, targets_subset - preds_subset, marker="x", color="k", s=10,
                    label=r'$y-\hat{\mu}$', linewidth=0.5)
        below_mean_bad_indi = np.argwhere(preds_subset - preds_UB_subset > 0)
        plt.scatter(horizontal_axis[below_mean_bad_indi], (preds_UB_subset - preds_subset)[below_mean_bad_indi],
                    marker="o", color="r", s=20, linewidth=1, facecolors='none')
        plt.scatter(horizontal_axis, preds_UB_subset - preds_subset, marker="x", color="b", s=15,
                    label=r'$UB-\hat{\mu}$', linewidth=1)
        plt.xlabel(r'$UB_{100\%}-\hat{\mu}$')
        plt.ylim(None, y_max + 0.5)
        plt.legend()
        if len(horizontal_axis) > 10000:
            filetype = "jpg"
        else:
            filetype = "pdf"

        if log_path is not None:
            time = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            plot_save_loc = os.path.join(log_path, f'Seed{seed}_UBAnalysisPlot_{time}.') + filetype
            plt.savefig(plot_save_loc, dpi=100)
            #plt.show()
            plt.close()
            if send_to == "JakobH":
                import send_bot_message as SBM
                SBM.send_JakobH(files=[plot_save_loc])
                plt.close()
        else:
            if send_to == "JakobH":
                plot_save_loc = f'Seed{seed}_UBAnalysisPlot_{time}.' + filetype
                import send_bot_message as SBM
                plt.savefig(plot_save_loc, dpi=100)
                SBM.send_JakobH(files=[plot_save_loc])
                os.remove(plot_save_loc)
            else:
                plt.show()

    # Pred vs. True plot of mean_model (not used)
    if False:
        dat_min, dat_max = min(min(preds), min(targets)), \
                           max(max(preds), max(targets))
        plt.figure(figsize=(16, 9))
        plt.scatter(np.array(preds), np.array(targets), s=2)
        plt.plot([dat_min, dat_max], [dat_min, dat_max], 'y')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title('kt: {:.2f} | R2: {:.2f} | mae: {:.4f}'.format(
            metrics['kendall_tau'], metrics['r2'], metrics['mae']))
        plt.tight_layout()
        if log_path is not None:
            time = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            plt.savefig(os.path.join(log_path, f'true_vs_pred_{eval_type}_{time}.jpg'), dpi=100)
        else:
            plt.show()

    return metrics
