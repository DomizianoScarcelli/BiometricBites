from tqdm import tqdm
import numpy as np

#All similarities are store in an array with the following structure: [(label_i, [label_j, similarities_ij]), (label_n, [label_m, similarities_nm]), ...]
def compute_similarities(probe_set, gallery_set, similarity_function: callable):
    all_similarities = []
    for i, (label_i, template_i) in enumerate(tqdm(probe_set, "Computing similarities")): #for every row (probe)
        row_similarities = []
        for j, (label_j, template_j) in enumerate(gallery_set): #for every column (template)
            # if i != j: #Do not consider main diagonal elements
            similarity = similarity_function(template_i, template_j)
            row_similarities.append(np.array([label_j, similarity])) #Must substitute 0 with the similarity algorithm
        all_similarities.append(np.array([label_i, row_similarities]))
    return all_similarities

def compute_similarities_svc(probe_set, model):

    def inverse_softmax(array):
            x = np.nextafter(0,1)
            C = np.log(np.exp(array).sum() + x)
            return np.log(array + x) + C
    
    all_similarities = []
    for i, (label_i, template_i) in enumerate(tqdm(probe_set, "Computing similarities")): #for every row (probe)
        probabilities = model.predict_proba([template_i])[0]
        probabilities = inverse_softmax(probabilities) #From [0,1] to [-inf, inf]
        similarities = 1 / (1 + np.exp(-probabilities)) #From [-inf, inf] to [0,1] but not with sum(probabilities) = 1
        row_similarities = []
        for j, similarity in enumerate(similarities):
            row_similarities.append((j, similarity))
        all_similarities.append((label_i, row_similarities))
    return all_similarities

#OpenSet Identification Multiple Template
def open_set_identification_eval(threshold, all_similarities):
    genuine_claims = 0
    impostor_claims = 0
    gallery_cardinality = len(all_similarities[0][1])
    DI = [0 for _ in range(gallery_cardinality)] #Detection and Identification
    GR = FA = 0
    for i, (label_i, similarities) in enumerate(tqdm(all_similarities, desc=f"Open set identification with threshold: {threshold}")): #for every row (probe)
        genuine_claims += 1
        impostor_claims += 1
        ordered_similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True) #Order the similarity vector in a descending order
        first_similarity = ordered_similarities[0] #L_i,1, the most similar template (shape = (label_j, simularity_value))
        if first_similarity[1] >= threshold:
            if label_i == first_similarity[0]: #the identity!! (to change)
                DI[0] += 1
                k = None
                for j, (label_j, similarity) in enumerate(ordered_similarities): #Parallel impostor case: jump the templates belonging to label(i) since i not in G
                    if label_i != label_j and similarity >= threshold: #The first template != label(i) has a similarity >= t
                        k = j
                        break
                if k != None:
                    FA += 1
                else:
                    GR += 1
            else:
                k = None
                for j, (label_j, similarity) in enumerate(ordered_similarities): #If genuine yet not the first, look for higher ranks
                    if (k == None) and (label_i == label_j) and similarity >= threshold: #The first template != label(i) has a similarity >= t
                        k = j
                        break
                if k != None:
                    DI[k] += 1 #End of genuine
                FA += 1 #Impostor in parallel, distance below t but different label. No need to jump since the first label is not the impostor                 
        else:
            GR += 1 #Impostor case counted directly, FR computed through DIR

    DIR = [0 for _ in range(gallery_cardinality)] #Detection and Identification rate
    DIR[0] = DI[0] / genuine_claims
    FRR = 1 - DIR[0] 
    FAR = FA / impostor_claims
    GRR = GR / impostor_claims
    for k in range(1, gallery_cardinality):
        DIR[k] = DI[k] / (genuine_claims + DIR[k-1])
    return DIR, FRR, FAR, GRR


#Verification Single Template
def verification_eval(threshold, all_similarities):
    genuine_claims = 0
    impostor_claims = 0
    GA = GR = FA = FR = 0
    for i, (label_i, similarities) in enumerate(tqdm(all_similarities, desc=f"Verification with threshold: {threshold}")): #for every row (probe)
        ordered_similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True) #Order the similarity vector in a descending order
        for j, (label_j, similarity) in enumerate(ordered_similarities): #for every column (template)
                if similarity >= threshold: #If the templates are similar enough
                    if label_i == label_j:
                        GA += 1
                        genuine_claims += 1
                    else:
                        FA += 1
                        impostor_claims += 1
                else:
                    if label_i == label_j:
                        FR += 1
                        genuine_claims += 1
                    else:
                        GR += 1
                        impostor_claims += 1
    GAR = GA / genuine_claims
    GRR = GR / impostor_claims
    FAR = FA / impostor_claims
    FRR = FR / genuine_claims
    return GAR, FRR, FAR, GRR

#Verification Multiple Template
def verification_mul_eval(threshold, all_similarities):
    genuine_claims = 0
    impostor_claims = 0
    GA = GR = FA = FR = 0
    for i, (label_i, similarities) in enumerate(tqdm(all_similarities, desc=f"Verification multiple template with threshold: {threshold}")): #for every row (probe)
        genuine_claims += 1
        ordered_similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True) #Order the similarity vector in a descending order
        best_similarities = {}
        for j, (label_j, similarity) in enumerate(ordered_similarities): #for every column (template)
            if label_j in best_similarities:
                if similarity >= best_similarities[label_j]:
                    best_similarities[label_j] = similarity
            else:
                best_similarities[label_j] = similarity
        for label_j, best_similarity in best_similarities.items():
            impostor_claims += 1
            if best_similarity >= threshold: #If the templates are similar enough
                if label_i == label_j:
                    GA += 1
                else:
                    FA += 1
            else:
                if label_i == label_j:
                    FR += 1
                else:
                    GR += 1
    GAR = GA / genuine_claims
    GRR = GR / impostor_claims
    FAR = FA / impostor_claims
    FRR = FR / genuine_claims
    return GAR, FRR, FAR, GRR