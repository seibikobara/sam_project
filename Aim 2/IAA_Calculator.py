'''
This script was used to calculate Inter Annotator Agreement
'''
from collections import defaultdict
import pandas as pd
from sklearn.metrics import cohen_kappa_score

#Load all the ids
sym_ids = []
temp = pd.read_csv("./covid_bc_sideeffect_dictionary_ver2.csv")
ids = temp["id"]
for line in ids:
    sym_ids.append(line.strip())
#print (sym_ids)
#print (len(sym_ids))

#Now that we have the sym_ids loaded,
#we need to add negated or non-negated information.
#A simple way can be to append a '-0' or a '-1' tag indicating
#if a concept is negated or not.
sym_ids_with_neg_marker = []
for sym_id in sym_ids:
    sym_ids_with_neg_marker.append(sym_id+'-0')
    sym_ids_with_neg_marker.append(sym_id+'-1')

#print(sym_ids_with_neg_marker)

#Now we load the annotation files
def get_flagged_sym_ids_from_annotated_file (filepath):
    f1 = pd.read_csv(filepath)

    # if empty
    f1.fillna("$$$$$$", inplace=True)

    f1_flagged_sym_ids = defaultdict(list)

    for index, row in f1.iterrows():
        id_ = row['4']
        sym_ids = row['Symptom ID'].split('$$$')
        neg_flags = row['Negation flag'].split('$$$')
        for sym_id,flag in zip(sym_ids,neg_flags):
            if len(sym_id)>0 and len(flag)>0:
                f1_flagged_sym_ids[id_].append(sym_id+'-'+str(flag))
    return f1_flagged_sym_ids


f1_flagged_sym_ids = get_flagged_sym_ids_from_annotated_file('./processed/secondround/bc_drug_discovered_manual_annotation1.csv')
f2_flagged_sym_ids = get_flagged_sym_ids_from_annotated_file('./processed/secondround/bc_drug_discovered_manual_annotation2.csv')
#print(f1_flagged_sym_ids)
#print(f2_flagged_sym_ids)
#print("HERE")

#Now we generate vectors for the computation...
#We only want to include IDs that are common in both files
#That is why we had stored them in dictionaries with IDs as the keys
commonids = list(set(f1_flagged_sym_ids.keys()).intersection(set(f2_flagged_sym_ids.keys())))
print(commonids)
print("intersection: " + str(len(commonids)))
f1_vec = []
f2_vec = []

for k in commonids:
    for c in sym_ids_with_neg_marker:
        if c in f1_flagged_sym_ids[k]:
            f1_vec.append(1)
        else:
            f1_vec.append(0)
        if c in f2_flagged_sym_ids[k]:
            f2_vec.append(1)
        else:
            f2_vec.append(0)

print (f1_vec)
print (f2_vec)
print(cohen_kappa_score(f1_vec,f2_vec))
