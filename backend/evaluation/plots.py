from matplotlib import pyplot as plt
from sklearn.metrics import auc
import numpy as np
import os

TICK_SIZE = 10
TITLE_SIZE = 15
DESCRIPTION_SIZE = 15

#Returns the ROC curve and also the value of the AUROC
def roc_auc_curve(eval_type, alg_name, metrics, save_path):
    '''
        Input:
        eval_type: openset, verification
        alg_name: e.g. Keras
        template_list: list of all templates
    '''
    eval_name = get_eval_name(eval_type)
    FAR_list = metrics["FAR"]
    GAR_list = metrics["GAR"]

    auroc = auc(FAR_list, GAR_list)
    print("The AUROC for " + alg_name + " in " + eval_name + " is: " + str(auroc))

    plt.plot(FAR_list, GAR_list, marker=".", label='ROC curve for ' + alg_name)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
    plt.xlabel("False Acceptance Rate", fontsize=DESCRIPTION_SIZE)
    plt.ylabel("Genuine Acceptance Rate (1-FRR)", fontsize=DESCRIPTION_SIZE)
    plt.title("ROC curve for " + alg_name + " in " + eval_name, fontsize=TITLE_SIZE)
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

def far_frr_curve(eval_type, alg_name, metrics, thresholds, save_path):
    eval_name = get_eval_name(eval_type)
    FAR_list = metrics["FAR"]
    FRR_list = metrics["FRR"]

    #zero_far = metrics["FRR"][np.max(np.where(np.array(metrics["FAR"]) == 0)[0])] if len(np.where(np.array(metrics["FAR"]) == 0)[0]) != 0 else "Not defined"
    #zero_frr = metrics["FAR"][np.min(np.where(np.array(metrics["FRR"]) == 0)[0])] if len(np.where(np.array(metrics["FRR"]) == 0)[0]) != 0 else "Not defined"
    #print("The ZeroFAR for " + alg_name + " in " + eval_name + " is: " + str(zero_far))
    #print("The ZeroFRR for " + alg_name + " in " + eval_name + " is: " + str(zero_frr))

    plt.plot(thresholds, FAR_list, linestyle="dashed", label="FAR")
    plt.plot(thresholds, FRR_list, linestyle="dashed", label="FRR")
    plt.xlabel("Thresholds", fontsize=DESCRIPTION_SIZE)
    #plt.ylabel("", fontsize=DESCRIPTION_SIZE)
    plt.title("FAR vs FRR for " + alg_name + " in " + eval_name, fontsize=TITLE_SIZE)
    plt.grid()
    plt.legend(loc='upper center')
    plt.savefig(save_path)
    plt.close()

def get_eval_name(eval_type):
    if eval_type == "openset":
        return "Open Set Identification"
    elif eval_type == "verification":
        return "Verification Single Template"
    else: return "Verification Multiple Template"

def save_plots(open_set_metrics, verification_metrics, verification_mul_metrics, thresholds, folder):
    open_set_FAR_FRR = {"FAR": open_set_metrics.iloc[2], "FRR": open_set_metrics.iloc[1], "GAR": 1-open_set_metrics.iloc[1]}
    roc_auc_curve("openset", "DeepFace", open_set_FAR_FRR, save_path=os.path.join(folder, "openset_roc"))
    far_frr_curve("openset", "DeepFace", open_set_FAR_FRR, thresholds, save_path=os.path.join(folder, "openset_far_frr"))

    verification_FAR_FRR = {"FAR": verification_metrics.iloc[2], "FRR": verification_metrics.iloc[1], "GAR": 1-verification_metrics.iloc[1]}
    roc_auc_curve("verification", "DeepFace", verification_FAR_FRR, save_path=os.path.join(folder, "verification_roc"))
    far_frr_curve("verification", "DeepFace", verification_FAR_FRR, thresholds, save_path=os.path.join(folder, "verification_far_frr"))

    verification_mul_FAR_FRR = {"FAR": verification_mul_metrics.iloc[2], "FRR": verification_mul_metrics.iloc[1], "GAR": 1-verification_mul_metrics.iloc[1]}
    roc_auc_curve("verification-mul", "DeepFace", verification_mul_FAR_FRR, save_path=os.path.join(folder, "verification_mul_roc"))
    far_frr_curve("verification-mul", "DeepFace", verification_mul_FAR_FRR, thresholds, save_path=os.path.join(folder, "verification_mul_far_frr"))
