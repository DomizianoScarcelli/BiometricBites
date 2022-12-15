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
    DIR = [0 for _ in range(template_list)]
    GR = FA = 0
    for template_i, i in enumerate(0, len(template_list)): #for every row (probe)
        cur_probe = template_i
        for template_j, j in enumerate(0, len(template_list)): #for every column (template)
            cur_template = template_j
            if i != j: #Do not consider main diagonal elements
                cur_similarity = all_similarities[i][j]
                if cur_similarity >= threshold:
                    if cur_probe == cur_template: #the identitity!! (to change)
                        DIR[0] += 1
                        #...