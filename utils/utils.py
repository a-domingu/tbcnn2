import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'




def plot_confusion_matrix(cm, classes, lr2, feature_size, epoch, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    title_string = 'LR:'+str(lr2)+';  FS:'+str(feature_size)+';   Epoch:'+str(epoch)
    plt.suptitle(title_string)#, y=1.05, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    namefile = os.path.join('confusion_matrix', 'CM_'+str(lr2)+'_'+str(feature_size)+'_'+str(epoch)+'.png')
    plt.savefig(namefile, dpi=300, bbox_inches='tight')
    plt.clf()



def writer(message):
    with open('results.txt', 'a') as f:
        f.write(message)


def remover():
    if os.path.exists('results.txt'):
        os.remove('results.txt')


def accuracy(predicts, targets):
    with torch.no_grad():
        print('predicts', predicts)
        rounded_prediction = torch.round(predicts)

    # 1 if false negative
    # -1 if false positive
    difference = targets - rounded_prediction
    errors = torch.abs(difference).sum()

    #accuracy = (len(difference) - errors)/len(difference)

    return errors


def bad_predicted_files(validation_set, predicts, targets):
    with torch.no_grad():
        rounded_predicts = torch.round(predicts)

    # 1 if false negative
    # -1 if false positive    
    difference = targets - rounded_predicts
    
    message = '\n The prediction of the following files is incorrect: \n'
    i=0
    for data in validation_set:
        if difference[i] == 1:
            message = message + 'The file ' + data + ' is a false negative \n'
            i+=1
        elif difference[i] == -1:
            message = message + 'The file ' + data + ' is a false positive \n'
            i+=1
        else:
            i+=1

    return message  

    
def conf_matrix(predicts, targets):
    with torch.no_grad():
        rounded_prediction = torch.round(predicts)

    # 1 if false negative
    # -1 if false positive
    difference = targets - rounded_prediction

    # 0 if true negative
    # 2 if true positive
    addition = targets + rounded_prediction

    conf_matrix = torch.zeros(2,2, dtype=torch.int64)
    # x axis are true values, and y axis are predictions
    for i in range(len(addition)):
        if difference[i] == 1:
            conf_matrix[1,0] += 1
        elif difference[i] == -1:
            conf_matrix[0,1] += 1
        elif addition[i] == 0:
            conf_matrix[0,0] +=1
        else:
            assert addition[i] == 2
            conf_matrix[1,1] += 1
        
    return conf_matrix.numpy()