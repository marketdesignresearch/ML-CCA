import sklearn.metrics
from scipy import stats as scipy_stats

from mvnns.losses import qloss


def compute_metrics(preds,
                    targets,
                    q=None,
                    scaled=False):
    if scaled:
        prefix = 'scaled-'
    else:
        prefix = ''

    metrics = {}

    # r2 = scipy_stats.linregress(preds, targets)[2] # This is pearson correlation coefficient
    r2 = sklearn.metrics.r2_score(y_true=targets, y_pred=preds)  # This is R2 coefficient of determination
    metrics[prefix + 'r2'] = r2

    kendall_tau = scipy_stats.kendalltau(preds, targets).correlation
    metrics[prefix + 'kendall_tau'] = kendall_tau

    mae = sklearn.metrics.mean_absolute_error(preds, targets)
    metrics[prefix + 'mae'] = mae

    if q:
        if not isinstance(q,list):
            q=[q]
        for scalar_q in q:
            metrics[prefix + 'qloss'+str(scalar_q)] = qloss(preds, targets, scalar_q)

    return metrics


def compute_metrics_UB(preds,
                       targets,
                       q,
                       scaled=False):
    if scaled:
        prefix = 'scaled-'
    else:
        prefix = ''

    metrics = {}

    r2 = sklearn.metrics.r2_score(y_true=targets, y_pred=preds)  # This is R2 coefficient of determination
    metrics[prefix + 'uUB-r2'] = r2

    mae = sklearn.metrics.mean_absolute_error(preds, targets)
    metrics[prefix + 'uUB-mae'] = mae

    if not isinstance(q,list):
        q=[q]
    for scalar_q in q:
        metrics[prefix + 'uUB-qloss'+str(scalar_q)] = qloss(preds, targets, scalar_q)

    return metrics
