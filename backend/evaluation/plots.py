from matplotlib import pyplot as plt
from sklearn.metrics import auc
import numpy as np

TICK_SIZE = 10
TITLE_SIZE = 15
DESCRIPTION_SIZE = 15

#Returns the ROC curve and also the value of the AUROC
def roc_auc_curve(eval_type, alg_name, metrics):
    '''
        Input:
        eval_type: openset, verification
        alg_name: e.g. Keras
        template_list: list of all templates
    '''
    eval_name = eval_name = "Open Set Identification" if eval_type == "openset" else "Verification"
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
    plt.show()

def far_frr_curve(eval_type, alg_name, metrics, thresholds):
    eval_name = eval_name = "Open Set Identification" if eval_type == "openset" else "Verification"
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
    plt.show()

"""
def test():
    metrics = {"FAR": [0, 0, 0, 0.34, 0.45, 0.52, 0.67, 0.72, 0.92, 1], "FRR": [1, 0.92, 0.78, 0.56, 0.49, 0.34, 0.21, 0.12, 0, 0], "GAR": [0, 0.82, 0.83, 0.85, 0.87, 0.92, 0.94, 0.95, 0.96, 1]}
    alg_name = "SVC"
    eval_type = "openset"
    thresholds = np.linspace(0, 1, num=len(metrics["FAR"]))
    roc_auc_curve(eval_type, alg_name, metrics)
    far_frr_curve(eval_type, alg_name, metrics, thresholds)

test()
"""