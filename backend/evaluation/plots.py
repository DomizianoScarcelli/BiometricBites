from matplotlib import pyplot as plt
from sklearn.metrics import auc
import numpy as np
import json
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
    print(alg_name + ": The AUROC for " + alg_name + " in " + eval_name + " is: " + str(auroc))
    plt.plot(FAR_list, GAR_list, marker=".", label='ROC curve for ' + alg_name)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess') #TODO: maybe remove this in the case of open set because IDK if this is correct
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

    zero_far = FRR_list[np.min(np.where(np.array(FAR_list) <= 0.0001)[0])] if len(np.where(np.array(FAR_list) == 0)[0]) != 0 else "Not defined"
    zero_frr = FAR_list[np.max(np.where(np.array(FRR_list) <= 0.0001)[0])] if len(np.where(np.array(FRR_list) == 0)[0]) != 0 else "Not defined"
    print(alg_name + ": The ZeroFAR for " + alg_name + " in " + eval_name + " is: " + str(zero_far))
    print(alg_name + ": The ZeroFRR for " + alg_name + " in " + eval_name + " is: " + str(zero_frr))

    eer_threshold = None
    try:
        frr_far_min_diff = [np.abs(FAR_list[i]-FRR_list[i]) for i in range(len(thresholds))]
        min_diff_index = np.argmin(frr_far_min_diff) if np.min(frr_far_min_diff) <= 0.1 else "Undefined"
        if min_diff_index != "Undefined":
            eer_threshold = thresholds[min_diff_index]
            eer_value = (FAR_list[min_diff_index]+FRR_list[min_diff_index])/2
            print(alg_name + ": The Equal Error Rate (ERR) for " + alg_name + " in " + eval_name + " is: " + str(eer_value) + " (threshold: " + str(eer_threshold) + ")")
        else: print(alg_name + ": The Equal Error Rate (ERR) for " + alg_name + " in " + eval_name + " is: Undefined.")
    except:
        print(alg_name + ": Can't calculate EER for " + alg_name + " in " + eval_name)

    plt.plot(thresholds, FAR_list, linestyle="dashed", label="FAR")
    plt.plot(thresholds, FRR_list, linestyle="dashed", label="FRR")
    plt.xlabel("Thresholds", fontsize=DESCRIPTION_SIZE)
    #plt.ylabel("", fontsize=DESCRIPTION_SIZE)
    plt.title("FAR vs FRR for " + alg_name + " in " + eval_name, fontsize=TITLE_SIZE)
    plt.grid()
    plt.legend(loc='upper center')
    plt.savefig(save_path)
    plt.close()

    return eer_threshold

def dir_curve(alg_name, DIR_list, threshold, save_path):
    k = np.arange(1, len(DIR_list)+1)
    plt.plot(k, DIR_list, linestyle="dashed", label="DIR at rank k")
    plt.xlabel("Rank (k)", fontsize=DESCRIPTION_SIZE)
    plt.ylabel("DIR value", fontsize=DESCRIPTION_SIZE)
    plt.title("DIR curve for " + alg_name + " in Open Set Identification (thr = " + str(threshold) +")", fontsize=TITLE_SIZE-1)
    plt.grid()
    plt.legend(loc='lower center')
    plt.savefig(save_path)
    plt.close()
    

def get_eval_name(eval_type):
    if eval_type == "openset":
        return "Open Set Identification"
    elif eval_type == "verification":
        return "Verification Single Template"
    else: return "Verification Multiple Template"

def save_plots(alg_name, open_set_metrics, verification_metrics, verification_mul_metrics, thresholds, folder):
    if not os.path.exists(os.path.join(folder, "dir_curve")):
        os.mkdir(os.path.join(folder, "dir_curve"))

    FAR_open_set = np.array(open_set_metrics.iloc[2]).astype(np.float)
    FRR_open_set = np.array(open_set_metrics.iloc[1]).astype(np.float)
    if len(thresholds) != len(FAR_open_set): FAR_open_set = FAR_open_set[1:]
    if len(thresholds) != len(FRR_open_set): FRR_open_set = FRR_open_set[1:]
    GAR_open_set = 1-FRR_open_set

    open_set_FAR_FRR = {"FAR": FAR_open_set, "FRR": FRR_open_set, "GAR": GAR_open_set}
    roc_auc_curve("openset", alg_name, open_set_FAR_FRR, save_path=os.path.join(folder, "openset_roc"))
    err_threshold = far_frr_curve("openset", alg_name, open_set_FAR_FRR, thresholds, save_path=os.path.join(folder, "openset_far_frr"))

    DIR_open_set = open_set_metrics.iloc[0].values.tolist()
    for t in range(1, len(thresholds)+1, 10):
        DIR_t = DIR_open_set[t] if t != 0 else DIR_open_set[t+1]
        DIR_t = DIR_open_set[t-1] if t == 100 else DIR_open_set[t]
        if isinstance(DIR_t, str):
            DIR_t = json.loads(DIR_t)
        dir_curve(alg_name, DIR_t, "%.2f" % thresholds[t-1], save_path=os.path.join(folder, "dir_curve", "openset_dir_"+str(t-1)))
    try:
        DIR_err = DIR_open_set[int(err_threshold*100)]
        if isinstance(DIR_err, str):
            DIR_err = json.loads(DIR_err)
        dir_curve(alg_name, DIR_err, "%.2f" % thresholds[int(err_threshold*100)], save_path=os.path.join(folder, "dir_curve", "openset_dir_"+str(err_threshold)+"_eer.png"))
    except:
        pass

    FAR_verification = np.array(verification_metrics.iloc[2]).astype(np.float)
    FRR_verification = np.array(verification_metrics.iloc[1]).astype(np.float)
    if len(thresholds) != len(FAR_verification): FAR_verification = FAR_verification[1:]
    if len(thresholds) != len(FRR_verification): FRR_verification = FRR_verification[1:]
    GAR_verification = 1-FRR_verification

    verification_FAR_FRR = {"FAR": FAR_verification, "FRR": FRR_verification, "GAR": GAR_verification}
    roc_auc_curve("verification", alg_name, verification_FAR_FRR, save_path=os.path.join(folder, "verification_roc"))
    far_frr_curve("verification", alg_name, verification_FAR_FRR, thresholds, save_path=os.path.join(folder, "verification_far_frr"))

    FAR_verification_mul = np.array(verification_mul_metrics.iloc[2]).astype(np.float)
    FRR_verification_mul = np.array(verification_mul_metrics.iloc[1]).astype(np.float)
    if len(thresholds) != len(FAR_verification_mul): FAR_verification_mul = FAR_verification_mul[1:]
    if len(thresholds) != len(FRR_verification_mul): FRR_verification_mul = FRR_verification_mul[1:]
    GAR_verification_mul = 1-FRR_verification_mul

    verification_mul_FAR_FRR = {"FAR": FAR_verification_mul, "FRR": FRR_verification_mul, "GAR": GAR_verification_mul}
    roc_auc_curve("verification-mul", alg_name, verification_mul_FAR_FRR, save_path=os.path.join(folder, "verification_mul_roc"))
    far_frr_curve("verification-mul", alg_name, verification_mul_FAR_FRR, thresholds, save_path=os.path.join(folder, "verification_mul_far_frr"))
