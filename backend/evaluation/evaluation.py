from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.spatial import distance
from evaluation import Gallery

#THE WHOLE EVALUATION STRUCTURE WILL BE CHANGED DEPENDING ON THE RECOGNITION ALGORITHM WE CHOOSE
#gallery_set = Gallery() #To implement
#probe_set = Gallery()
gallery_set = []
#probe_set = []

#X_train, X_test, y_train, y_test = train_test_split(gallery.get_images(), labels)

def compute_similarities(template_list):
    all_similarities = []
    genuine_claims = 0
    impostor_claims = 0
    for template_i, i in enumerate(0, len(template_list)): #for every row (probe)
        row_similarities = []
        cur_probe = template_i
        for template_j, j in enumerate(0, len(template_list)): #for every column (template)
            cur_template = template_j
            if i != j: #Do not consider main diagonal elements
                if cur_probe == cur_template: #We have to compare the identities to template_list must be some labels..
                    genuine_claims += 1
                else:
                    impostor_claims += 1
            row_similarities.append(0) #Must substitute 0 with the similarity algorithm
        all_similarities.append(row_similarities)
    return genuine_claims, impostor_claims, all_similarities

def open_set_identification_eval(template_list, threshold):
    genuine_claims, impostor_claims, all_similarities = compute_similarities(template_list)
    DI = [0 for _ in range(template_list)] #Detection and Identification
    GR = FA = 0
    for template_i, i in enumerate(0, len(template_list)): #for every row (probe)
        cur_probe = template_i
        first_similarity = all_similarities[i][0] #L_i,1, the most similar template
        if first_similarity >= threshold:
            if cur_probe == cur_template: #the identity!! (to change)
                DI[0] += 1
                for template_j, j in enumerate(1, len(template_list)): #Parallel impostor case: jump the templates belonging to label(i) since i not in G
                    cur_template = template_j
                    k = None
                    if i != j: #Do not consider main diagonal elements so the case in which the template is compared to itself
                        if (k == None) and (cur_probe != cur_template) and all_similarities[i][j] >= threshold: #The first template != label(i) has a similarity >= t
                            k = j
                    if k != None:
                        FA += 1
                    else:
                        GR += 1
            else:
                for template_j, j in enumerate(1, len(template_list)): #If genuine yet not the first, look for higher ranks
                    cur_template = template_j
                    k = None
                    if i != j: #Do not consider main diagonal elements so the case in which the template is compared to itself
                        if (k == None) and (cur_probe == cur_template) and all_similarities[i][j] >= threshold: #The first template != label(i) has a similarity >= t
                            k = j
                    if k != None:
                        DI[k] += 1 #End of genuine
                    FA += 1 #Impostor in parallel, distance below t but different label. No need to jump since the first label is not the impostor                 
        else:
            GR += 1 #Impostor case counted directly, FR computed through DIR

        DIR = [0 for _ in range(template_list)] #Detection and Identification rate
        DIR[0] = DI[0] / genuine_claims
        FRR = 1 - DIR[0] 
        FAR = FA / impostor_claims
        GRR = GR / impostor_claims
        for k in range (1, len(template_list)):
            DIR[k] = DI[k] / genuine_claims + DIR[k-1]
        return DIR, FRR, FAR, GRR

