from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
from predeal import *
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

rf=RandomForestRegressor(n_estimators=100,
                         criterion='mse',
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0,
                        max_features='auto',
                        max_leaf_nodes=None,
                        min_impurity_decrease=1e-07,
                        bootstrap=True,
                        oob_score=True,
                        n_jobs=1,
                        random_state=None,
                        verbose=0,
                        warm_start=False)
rf.fit(trn_data,lable_data)  
preds = rf.predict(test_data)
joblib.dump(rf,'finalRF.model')

TP = 0
FP = 0
FN = 0
TN = 0

preds = preds.tolist()

for j in range(len(lable_test)):
    lable_test[j] = int(lable_test[j])
print(lable_test)


output = np.zeros(np.shape(preds))
for i,pred in enumerate(preds):
    if pred >= 0.5:
        output[i] = 1
    else:
        output[i] = 0
print(output)

print(lable_test==output)

for i,pred in enumerate(preds):
    if int(lable_test[i]) == 1:
        if pred >= 0.5:
            TP += 1
        else:
            FN += 1
    else:
        if pred >= 0.5:
            FP += 1
        else:
            TN += 1

print(TP,FP,FN,TN)

print(int(TP+TN)/int(len(lable_test)))        
precision = int(TP)/int(TP+FP)
recall = int(TP)/int(TP+FN)
F_1 = int(2*TP)/int(2*TP + FP + FN)

print("precision",precision)
print('recall',recall)
print('F_1',F_1)

