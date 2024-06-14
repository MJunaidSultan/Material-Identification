# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:47:56 2024

@author: Junaid
"""


import pandas as pd


import pandas as pd
import numpy as np


from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

data = pd.read_csv("data.csv")

parameter = (105,80) 

y = data["label"]

x = data.drop("label" ,axis = 1)



import numpy as np


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=45)

model = RandomForestClassifier()

model.fit(x_train, y_train) 

y_pred = model.predict(x_test)




from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import f1_score
f1_sc = f1_score(y_test, y_pred,  average='macro')
print(f1_sc)



from sklearn.metrics import recall_score
rc_sc = recall_score(y_test, y_pred,  average='macro')
print(rc_sc)

from sklearn.metrics import precision_score
Prec = precision_score(y_test, y_pred ,  average='macro')
print(Prec)




    

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

"""
import pickle
filename = 'AdaBoost92%Accuracy.sav'
pickle.dump(model, open(filename, 'wb'))
"""