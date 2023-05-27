import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.neural_network  #
import matplotlib.pyplot as plt
import pickle
# --------Artificial_neural_network model---------------------------------------------------------
# --------preprossing data---------------------------------------------------------
data = pd.read_csv('heart.csv')
print('--------All Dataset------------')
print(data)
print('Data set information: ',data.describe())
print('---------Head-----------')
print(data.head())
print('----------Tail----------')
print(data.tail())
print('----------Shape----------')
print(data.shape) #count Row & colmun
print('---------Dataset Type-----------')
# ==================================================
print(data.isna()) # is not avaialbal for example row10 * colum 10 = 100 cells if any cells mising return tru
# how to find cell
print(data.isna().any())  #used the any to know cell = output  tru/false
# ==================================================
# ===============================================================================================
#-----------------------------------------------------------------------------------
#Check duplicate rows in data
duplicate_rows = data[data.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)
#-----------------------------------------------------------------------------------
 #-----------------------------------------------------------------------------------
x = data.iloc[:,[0,1,2,3,4]].values
y = data.iloc[:,13].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25 , random_state=0)
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:12, :])
classifier = sklearn.neural_network.MLPClassifier(
                activation='logistic',
                max_iter=1000,
                hidden_layer_sizes=(2,),  # default (100,)
                solver='lbfgs')

classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
print ("Accuracy : ", accuracy_score(ytest, ypred))
#-----------------------------------------------------------------------------------
best_score = 1000
kfolds = 5
# -----------------------------------------------------------------
# C:float, default=1.0
# Inverse of regularization strength; must be a positive float.
# Like in support vector machines, smaller values specify stronger regularization.
# -----------------------------------------------------------------
for c in [0.01, 0.1, 0.3, 1, 2, 10, 100]:
    model = classifier
    scores = cross_val_score(model, xtrain, ytrain, cv=kfolds)
    score = np.mean(scores) # average every chunk
    score=1/score
    print("score=", score, ' ==> ', c)
    if score < best_score:
        best_score = score
        best_parameters = c
# -----------------------------------------------------------------
SelectedModel = classifier.fit(xtrain, ytrain)
test_score = SelectedModel.score(xtest, ytest)
print("Best score on validation set is:", best_score)
print("Best parameter for regularization (lambda) is:", best_parameters)
print("Test set score with best C parameter is", test_score)
# -----------------------------------------------------------------
prediction_proba = SelectedModel.predict(xtest)
print("accuracy_score",accuracy_score(ytest, ypred))
print("Prediction: " , prediction_proba )
print('Best score on validation set is: ' , best_score)
print('Best parameter for regularization (lambda) is: ' , best_parameters)
print('Test set score with best C parameter is: ' , test_score)
cm = pd.crosstab(ytest, ypred)#, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
print ("Confusion Matrix : \n", cm)
#=========================================================
pickle.dump(sc_x, open("scalerNerulNetwork.pickle", "wb"))
ssc = pickle.load(open("scalerNerulNetwork.pickle", 'rb'))
pickle.dump(classifier, open('NerrulNetworkClassifier.pkl','wb'))
model = pickle.load(open('NerrulNetworkClassifier.pkl','rb'))
sample =ssc.transform([xtest[0]])
print("the sampl" ,sample)
model.predict(sample)
print("predict model" , model.predict(sample))



