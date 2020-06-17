import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool

def write_file_best_tresholds(TP, FP, positives, unknowns, label, thresholds):
        
    print(label)
    
    tpr = TP/positives
    fpr = FP/unknowns
    TN = unknowns - FP
    
    tpr *= 100
    fpr *= 100
    max_space = tpr - fpr
    
    apos = np.full(len(tpr), positives)
    aneg = np.full(len(tpr), unknowns)

    accuracy = ((TP + TN) / (positives + unknowns)) * 100.0
        
    stacked_file_data=np.stack((tpr,
                                fpr,
                                max_space,
                                accuracy,
                                TP,
                                apos,
                                FP,
                                aneg,
                                thresholds
                                ),axis=1)
    stacked_file_data=sorted(stacked_file_data, key=lambda x: x[2], reverse=True)

    filename = str('F:/evaluation_' + label + '.txt')
    np.savetxt(filename, stacked_file_data, delimiter=' ',fmt=['%f','%f','%f', '%f', '%f', '%f', '%f', '%f', '%f'])

def write_file_for_DIR(gt_y,pred_y,file_name,feature_vector=None,num_of_known_classes=2):
    """
    Writes a file for use by DIR evaluation Script. Format also supported by Bob.
    The format is:
        'Ground Truth Class ID','Repeat GT Class ID','Predicted Class ID','Sample Number','Similarity Score'
    """
    
    if pred_y.shape[1]==num_of_known_classes+1:
        pred_y=pred_y[:,:-1]

    sample_identifier=np.tile(np.arange(gt_y.shape[0]),num_of_known_classes)
    gt_y=np.tile(gt_y,num_of_known_classes)
    predicted_class_id = np.repeat(np.arange(num_of_known_classes),pred_y.shape[0])

    similarity_score = pred_y.flatten('F')
    if feature_vector is not None:
        file_name = ('/').join(file_name.split('/')[:-1])+"/Multiplying_with_mag_"+file_name.split('/')[-1]
        print("DIR score file saved at",file_name)
        similarity_score=similarity_score*np.tile(np.sqrt(np.sum(np.square(feature_vector),axis=1)),num_of_known_classes)
    
    stacked_file_data=np.stack((
                                predicted_class_id, # PREDICTED_CLASS_ID
                                predicted_class_id, # PREDICTED_TEMPLATE_ID
                                gt_y,               # GT_CLASS_ID
                                sample_identifier,  # SAMPLE_IDENTIFIER
                                similarity_score    # SIMILARITY_SCORE
                                ),axis=1)
    np.savetxt(file_name, stacked_file_data, delimiter=' ',fmt=['%d','%d','%d','%d','%f'])


    
# MR: single_class_eval is a parameter that specifies to evaluate only one class versus unknowns [known_class, unknown_class]
def process_each_file(file_name, known_labels, negative_label):
    print('process each file', file_name)
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    
    csv_content = pd.DataFrame(csv_content)
   
    # MR: added for the purpose to classify some of known classes as negatives
    # MR: otherwise if this array is empty, it will use standard representation as in Reducing Network Agnostophobia
    if len(known_labels) > 0:
        csv_content = csv_content[csv_content[0].isin(known_labels)]
        for ind, row in csv_content.iterrows():
            if row[2] not in known_labels:
                csv_content.loc[ind, 2] = negative_label


    for k, g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())

    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
        
    print('unknowns label=',unknowns_label)
    FP=[0]
    TP=[0]
    N=0
    N_above_UK=0
    UK_prob=1
    
    thresholds = [0]
    
    for ind,row in df.iterrows():
        # If Sample is Unknown
        if row[2]==unknowns_label:
            UK_prob=row[4]
            FP.append(FP[-1]+1)
            TP.append(N)
            thresholds.append(row[4])
        # If Sample is Known and Classified Correctly
        else:
            if row[1]==row[2]:
                N_above_UK+=1
                if row[4] < UK_prob: # MR: FIXED BUG, ADDED EQUAL SIGN
                    N=N_above_UK
        

    TP=np.array(TP[1:]).astype(np.float32)
    FP=np.array(FP[1:]).astype(np.float32)
    thresholds = np.array(thresholds[1:]).astype(np.float32)
    
    return TP,FP,positives,unknowns, thresholds

def process_files(files_to_process,labels,DIR_filename=None,out_of_plot=False, known_labels = [], negative_label = 2):
    print(files_to_process)
    #p=Pool(processes=1) # MR: not working with the method, gets stucked
    to_plot = [
            process_each_file(files_to_process[0], known_labels, negative_label),
            process_each_file(files_to_process[1], known_labels, negative_label),
            process_each_file(files_to_process[2], known_labels, negative_label),
            process_each_file(files_to_process[3], known_labels, negative_label)
            ]
    #to_plot=p.map(process_each_file,files_to_process)
    #p.close()
    #p.join()
    u = []
    fig, ax = plt.subplots(figsize=(5,3))
    for i,(TP,FP,positives,unknowns,thresholds) in enumerate(to_plot):
        #print(labels[i], ':', ' TP=', TP, ' FP=', FP, ' positives=', positives, ' unknowns=', unknowns)
        write_file_best_tresholds(TP, FP, positives, unknowns, labels[i], thresholds)
        ax.plot(FP/unknowns,TP/positives,label=labels[i])
        #ax.plot(FP/unknowns*100,TP/positives*100,label=labels[i])
        #ax.plot(FP,TP/positives,label=labels[i])
        #ax.plot(FP,TP,label=labels[i])

    u.append(unknowns)
    ax.set_xscale('log') #MR: disable in case of small sample size
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0,1]) #MR: disable in case of small sample size
    ax.set_ylabel('Correct Classification Rate', fontsize=12, labelpad=5)
    ax.set_xlabel('False Positive Rate', fontsize=12, labelpad=5)
    #ax.set_xlabel('False Positive Rate : Total Unknowns'+str(list(set(u))[0]), fontsize=18, labelpad=10)

    if out_of_plot:
        ax.legend(loc='lower center',bbox_to_anchor=(-0.75, 0.),ncol=1,fontsize=12,frameon=False)
    else:
        #ax.legend(loc="upper left")
        ax.legend(loc="lower right",frameon=True)

        
    if DIR_filename is not None:
        fig.savefig(DIR_filename+'.pdf', bbox_inches="tight")
    plt.show()

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_to_process', nargs='+', help='DIR file Names', required=True)
    parser.add_argument("--labels", nargs='+', help='DIR file Labels', required=True)
    parser.add_argument("--DIR_filename", help='Name of DIR filename to create', required=True)
    args = parser.parse_args()
    
    if len(args.files_to_process) != len(args.labels):
        print("Please provide a label for each file name ... Exiting!!!")
        exit()
        
    process_files(args.files_to_process,args.labels,args.DIR_filename)