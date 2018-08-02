import numpy as np  
from predeal import *
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=3.0, binarize=0, fit_prior=True, class_prior=[0.8,0.2])
clf.fit(trn_data,lable_data)  
preds = clf.predict(test_data);  

TP = 0
FP = 0
FN = 0
TN = 0

preds = preds.tolist()

for j in range(len(lable_test)):
    lable_test[j] = int(lable_test[j])
print('lable',lable_test)

output = np.zeros(np.shape(preds))
for i,pred in enumerate(preds):
    output[i] = int(pred)
print(output)

for i,pred in enumerate(preds):
    if int(lable_test[i]) == 1:
        if int(pred) == 1:
            TP += 1
        else:
            FN += 1
    else:
        if int(pred) == 1:
            FP += 1
        else:
            TN += 1

print(int(TP+TN)/int(len(lable_test)))        
precision = int(TP)/int(TP+FP)
recall = int(TP)/int(TP+FN)
F_1 = int(2*TP)/int(2*TP + FP + FN)

print("precision",precision)
print('recall',recall)
print('F_1',F_1)