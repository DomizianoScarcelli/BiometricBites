from evaluation import open_set_identification_eval, verification_eval
from matplotlib import pyplot as plt

def roc_auc_score(thresholds, eval_type, alg_name, metrics):
    '''
        Input:
        thresholds: the list of thresholds to test (e.g. [0.05, 0.10, 0.15, ..., 0.90, 0.95])
        eval_type: openset, verification
        alg_name: e.g. Keras
        template_list: list of all templates
    '''

    '''
    FAR_list = []
    FRR_list = []
    if eval_type == "openset":
        for threshold in thresholds:
            metrics = open_set_identification_eval(template_list, threshold)
            FAR_list.append(metrics["FAR"])
            FRR_list.append(metrics["FRR"])
    elif eval_type == "verification":
        for threshold in thresholds:
            metrics = verification_eval(template_list, threshold)
            FAR_list.append(metrics["FAR"])
            FRR_list.append(metrics["FRR"])
    else:
        TypeError("Please, specify a correct evaluation type (either openset or verification)")
    '''
    FAR_list = metrics["FAR"]
    FRR_list = metrics["FRR"]

    plt.plot(FAR_list, FRR_list, marker=".")
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('True Rejection Rate')
    plt.title('ROC curve for ' + alg_name)
    plt.show()