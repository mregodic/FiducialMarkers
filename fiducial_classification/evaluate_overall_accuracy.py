from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

def get_class_items(data, class_id, t1, t2):
    
    class_first = 1
    class_second = 0
    
    data1 = []
    data2 = []
    for r in data:
        if(r[0] != class_id): continue
        
        if(r[2] == class_id):
            r[2] = class_first # specificy as main class id
            data1.append(r)
            data2.append(r)
        elif(r[2] == t1):
            r[2] = class_second
            data1.append(r)
        elif(r[2] == t2):
            r[2] = class_second
            data2.append(r)

    return data1, data2     

def confusion_for_threshold(item, threshold):
    
    data = item
    df = pd.DataFrame(data)
    testy = df[2]
    model_probs = df[4]
        
    #print('data count:', len(testy))
    #print('multi class for: ', label)
        
    # softmax
    #print('threshold=', threshold)
 
    # Find prediction to the dataframe applying threshold
    data_pred = model_probs.map(lambda x: 1 if x >= threshold else 0)
    
    # Print confusion Matrix
    tn, fp, fn, tp = confusion_matrix(testy, data_pred).ravel()
   
    
#    print('TP=', tp)
#    print('FP=', fp)
#    print('FN=', fn)
#    print('TN=', tn)
    
    return tp, fp

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)

    print('sensitvity=', sensitivity*100)
    print('specificity=', specificity*100)
    print('accuracy=', accuracy*100)
    print('balanced accuracy=', balanced_accuracy_score(testy, data_pred))

def calculate_accuracy(class_0, class_1, threshold, total):
    
    label_1 = class_0[0]
    label_2 = class_1[0]
    
    # CLASS 0
    
    tp01, fp01 =  confusion_for_threshold(class_0[1], threshold[0])
    tp02, fp02 =  confusion_for_threshold(class_0[2], threshold[0])
 
    if tp01 != tp02:
        print('ERROR tp01 != tp12')
        return 0

    # CLASS 1
 
    tp10, fp10 =  confusion_for_threshold(class_1[1], threshold[1])
    tp12, fp12 =  confusion_for_threshold(class_1[2], threshold[1])
    
    if tp10 != tp12:
        print('ERROR tp01 != tp02')
        return 0
    
    # CLASS 2
    # manually calculate 3rd class
    tp2 =   total[2] - fp02 - fp12
    fp20 =  total[0] - tp01 - fp10
    fp21 =  total[1] - tp10 - fp01
    
    
    print('approach labels:', label_1, ' ', label_2)
    print('tp01 =', tp01, 'fp01 =', fp01, 'fp02 =', fp02)
    print('fp10 =', fp10, 'tp02 =', tp10, 'fp12 =', fp12)
    print('fp20 =', fp20, 'fp21 =', fp21, 'tp02 =', tp2)
    print('')

    SN_0 = tp01 / total[0] * 100
    SN_1 = tp10 / total[1] * 100
    SN_2 = tp2  / total[2] * 100

    print('SN_0 = ', SN_0)
    print('SN_1 = ', SN_1)
    print('SN_2 = ', SN_2)

    print('')
    
    SP_0 = (1 - (fp01 + fp02) / (total[1] + total[2])) * 100
    SP_1 = (1 - (fp10 + fp12) / (total[0] + total[2])) * 100
    SP_2 = (1 - (fp20 + fp21) / (total[0] + total[1])) * 100

    print('SP_0 = ', SP_0)
    print('SP_1 = ', SP_1)
    print('SP_2 = ', SP_2)

    print('')
    
    FPR_0 = (fp01 + fp02) / (total[1] + total[2]) * 100
    FPR_1 = (fp10 + fp12) / (total[0] + total[2]) * 100

    print('FPR_0 = ', FPR_0)
    print('FPR_1 = ', FPR_1)
    print('')
    
    meanSN = (SN_0 + SN_1 + SN_2) / 3
    meanSP = (SP_0 + SP_1 + SP_2) / 3 

    print('mean SN = ', meanSN)
    print('mean SP = ', meanSP)

    balanced_acc = 1/2 * (meanSN + meanSP)

    print('balanced accuracy = ', balanced_acc)
    
    print('')
    
def organize_items(dir_name, files, num_class, test_class1, test_class2):
    items = []
    for f in files:
        label = f[0]
        f_name = dir_name + f[1]
        df = pd.read_csv(f_name,delimiter=' ',header=None, lineterminator='\n')
        data_class, data_noise = get_class_items(df.values, num_class, test_class1, test_class2)
        items.append((label, data_class, data_noise))
    return items
    
def main():
    
    dir_name = 'Model_Results/DIRs/Not_Fiducial/'
    
    f_softmax = 'SoftMax_0_Not_Fiducial.txt'
    f_bg = 'BG_0_Not_Fiducial.txt'
    f_cross = 'Cross_0_Not_Fiducial.txt'
    f_ring = 'Ring_30.0_0_Not_Fiducial.txt'
    
    file_items = []
    file_items.append(('SoftMax', f_softmax))
    file_items.append(('Background', f_bg))
    file_items.append(('Entropic Open-Set', f_cross))
    file_items.append(('Objectosphere', f_ring))
    
    items_class_0 = organize_items(dir_name, file_items, num_class=0, test_class1=1, test_class2=2)
    items_class_1 = organize_items(dir_name, file_items, num_class=1, test_class1=0, test_class2=2)
      
    total = [241, 151, 1550]
        
    thresholds = []
    thresholds.append((0.999995,0.973358))
    thresholds.append((0.312236,0.967643))
    thresholds.append((0.917204,0.983185))
    thresholds.append((0.960895,0.931309))

    for i in range(4):
        if file_items[i][0] != items_class_0[i][0] != items_class_1[i][0]:
            print('error label')
        calculate_accuracy(items_class_0[i], items_class_1[i], thresholds[i], total)

main()  
        