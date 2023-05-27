import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
#we can apply diffrent dataset and change feature
dataset = pd.read_csv('House-Price-Prediction-clean.csv')
x = dataset.iloc[:, [0, 22]].values
y = dataset.iloc[:, 23].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25, random_state = 0)
#--------------------------------
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)

#============================
classifier = LogisticRegression()
classifier.fit(xtrain, ytrain)

#save scaler
scalerfile = 'scaler.sav'
pickle.dump(sc_x, open(scalerfile, 'wb'))
# save the model to disk
filename = 'model_1.sav'
pickle.dump(classifier, open(filename, 'wb'))
