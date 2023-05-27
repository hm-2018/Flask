import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
#we can apply diffrent dataset and change feature
data = pd.read_csv('heart.csv')
x = data.iloc[:, range(0,12)].values
y = data.iloc[:, 13].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25)
#---------------------------------------------
#load scaler
scaler_file = 'scaler_NN.sav'
sc_x = pickle.load(open(scaler_file, 'rb'))
xtest = sc_x.transform(xtest)
#-------------------------------------
# load the model from disk
model_file='model_NN.sav'
classifier = pickle.load(open(model_file, 'rb'))
#------------------------------------
y_pred = classifier.predict([xtest[3]])
#print ("Accuracy  it's on row: ", accuracy_score(ytest, y_pred))
#print ("Accuracy  it's on row: ", accuracy_score(ytest, y_pred))
print ("predict row  ", (y_pred))




