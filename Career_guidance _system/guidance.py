import numpy as np
import pandas as pd

# Dataset is now stored in a Pandas Dataframe
df = pd.read_csv('career_pred.csv')

df.head()

#1. Handling Missing values:
df.isna()

df=df.dropna()
df.isna().sum().sum()

# Data
data = df.iloc[:,:-1].values
label = df.iloc[:,-1]

#Label Encoding: COnverting To Numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

for i in range(14,38):
    data[:,i] = labelencoder.fit_transform(data[:,i])
    
    
#Normalizing the data
from sklearn.preprocessing import Normalizer
data1=data[:,:14]
normalized_data = Normalizer().fit_transform(data1)

data2=data[:,14:]
df1 = np.append(normalized_data,data2,axis=1)

#Combining into a dataset
df2=df.iloc[:,:-1]
dataset = pd.DataFrame(df1,columns=df2.columns)
dataset
    

# For label
label = df.iloc[:,-1]
original=label.unique() 
label=label.values
label2 = labelencoder.fit_transform(label)
y=pd.DataFrame(label2,columns=["Suggested Job Role"])
numeric=y["Suggested Job Role"].unique() 
Y = pd.DataFrame({'Suggested Job Role':original, 'Associated Number':numeric})



X = dataset.copy()

from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

#Decision Tree Classifier
from sklearn import tree

DT_model = tree.DecisionTreeClassifier()
DT_model = DT_model.fit(X_train, y_train)
DT_model

# Prediction
y_pred = DT_model.predict(X_test)
y_test_arr=y_test['Suggested Job Role']
Final = pd.DataFrame({'Predicted':y_pred, 'Actual':y_test_arr})
Final.reset_index()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("accuracy=",accuracy*100)

#Number of correct predicted
comparison_column = np.where(Final['Predicted'] == Final['Actual'], 1, 0)
Number_of_correct_predictions=list(comparison_column).count(1)
print("Number_of_correct_predictions : ",((Number_of_correct_predictions)/4000)*100,"%")
accuracy_DT=Number_of_correct_predictions



# Run svm with default hyper parameters
# Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters
# import SVC classifier
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score

# instantiate classifier with default hyperparameters
svc=SVC() 

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)*100))




from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train=pd.to_numeric(X_train.values.flatten())
X_train=X_train.reshape((16000,38))
X_test=pd.to_numeric(X_test.values.flatten())
X_test=X_test.reshape((4000,38))

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train['Suggested Job Role'])

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import pickle
pickle.dump(svc,open('svc.pkl','wb'))


