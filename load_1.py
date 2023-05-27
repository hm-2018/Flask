import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
#we can apply diffrent dataset and change feature
dataset = pd.read_csv('House-Price-Prediction-clean.csv')
x = dataset.iloc[:, [0, 22]].values
y = dataset.iloc[:, 23].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25, random_state = 0)
#---------------------------------------------
#load scaler
scaler_file = 'scaler.sav'
sc_x = pickle.load(open(scaler_file, 'rb'))
xtest = sc_x.transform(xtest)
#-------------------------------------
# load the model from disk
mpdel_file='model_1.sav'
classifier = pickle.load(open("model_1.sav", 'rb'))
#------------------------------------
y_pred = classifier.predict(xtest)
print ("Accuracy : ", accuracy_score(ytest, y_pred))



