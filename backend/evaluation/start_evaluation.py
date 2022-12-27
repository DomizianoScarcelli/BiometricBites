from .evaluation import open_set_identification_eval, verification_eval
from .plots import roc_auc_score
import numpy as np
import pandas as pd

template_list = [] #TO DEFINE!!

open_set_identification_metrics_by_thresholds = {}
verification_metrics_by_thresholds = {}
thresholds = np.arange(0.05, 1, 0.05)

for threshold in thresholds:
    DIR, FRR, FAR, GRR = open_set_identification_eval(template_list, threshold)
    open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
    GAR, GRR, FAR, FRR = verification_eval(template_list, threshold)
    verification_metrics_by_thresholds[threshold] = [GAR, GRR, FAR, FRR]

#---------Plot ROC-----------
open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
open_set_FAR_FRR = {"FAR": open_set_metrics.iloc[2], "FRR": open_set_metrics.iloc[1]}
print("ROC curve for Open Set Identification:")
roc_auc_score(thresholds, "openset", "VggFace", open_set_FAR_FRR)

verification_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
verification_FAR_FRR = {"FAR": verification_metrics.iloc[2], "FRR": verification_metrics.iloc[1]}
print("ROC curve for Verification:")
roc_auc_score(thresholds, "verification", "VggFace", verification_FAR_FRR)
