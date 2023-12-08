
'''
This script was used to evaluate our rule-based model for drug and side-effect discovery
'''




import pandas as pd
from collections import defaultdict



def load_labels(f_path):
    '''
    Loads the labels

    :param f_path:
    :return:
    '''
    labeled_df = pd.read_excel(f_path)
    labeled_dict = defaultdict(list)
    for index,row in labeled_df.iterrows():
        id_ = row[4]
        if not pd.isna(row['Symptom ID']) and not pd.isna(row['Negation flag']):
            cuis = row['Symptom ID'].split('$$$')[1:-1]
            neg_flags = row['Negation flag'].split('$$$')[1:-1]
            for cui,neg_flag in zip(cuis,neg_flags):
                labeled_dict[id_].append(cui + '-' + str(neg_flag))
    return labeled_dict

infile = './bc_drug_discovered_evalulation_summarised.xlsx'
outfile = './bc_drug_discovered_gold_standard_predicted.xlsx' 
#outfile = './UnlabeledSet_annotated_SK.xlsx' 

#infile = 's10_annotated.xlsx'
#outfile = 's10_annotated_result.xlsx'
gold_standard_dict = load_labels(infile)
submission_dict = load_labels(outfile)

tp = 0
tn = 0
fp = 0
fn = 0
for k,v in gold_standard_dict.items():
    for c in v:
        try:
            if c in submission_dict[k]:
                tp+=1
            else:
                fn+=1
                print('{}\t{}\tfn'.format(k, c))
        except KeyError:#if the key is not found in the submission file, each is considered
                        #to be a false negative..
            fn+=1
            print('{}\t{}\tfn'.format(k, c))
    for c2 in submission_dict[k]:
        if not c2 in gold_standard_dict[k]:
            fp+=1
            print('{}\t{}\tfp'.format(k, c2))
print('True Positives:',tp, 'False Positives: ', fp, 'False Negatives:', fn)
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1 = (2*recall*precision)/(recall+precision)
print('Recall: ',recall,'\nPrecision:',precision,'\nF1-Score:',f1)
print('{}\t{}\t{}'.format(precision, recall, f1))


