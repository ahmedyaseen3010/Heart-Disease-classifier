import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib as joblib
from sklearn.preprocessing import MinMaxScaler
import pickle

data=pd.read_csv('heart.csv')


X=data[["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]].values
Y=data[['HeartDisease']].values

#encoding

le=LabelEncoder()
#Sex: Sex of the patient [M:1 Male, F:0 Female]
X[:,1]=le.fit_transform(X[:,1])
#**ChestPainType**: Chest pain type [TA:0 Typical Angina, ATA:1 Atypical Angina, NAP:2 Non-Anginal Pain, ASY:3 Asymptomatic]
X[:,2]=le.fit_transform(X[:,2])
#RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
X[:,6]=le.fit_transform(X[:,6])
##ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
X[:,8]=le.fit_transform(X[:,8])
#ST_Slope: The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
X[:,10]=le.fit_transform(X[:,10])
#HeartDisease: Output class [1: heart disease, 0: Normal]
Y[:,0]=le.fit_transform(Y[:,0])


#scalling
# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()
data[["Age"]] = scaler.fit_transform(data[["Age"]])
data[["Cholesterol"]] = scaler.fit_transform(data[["Cholesterol"]])
data[["RestingBP"]] = scaler.fit_transform(data[["RestingBP"]])
data[["MaxHR"]] = scaler.fit_transform(data[["MaxHR"]])



#train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size= 0.20,random_state = 21)



# Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve

logisticRegression_model = LogisticRegression(solver="liblinear")
logisticRegression_model.fit(X_train, y_train) 

logisticRegression_test_pred = logisticRegression_model.predict(X_test)
logisticRegression_test_score = accuracy_score(y_test, logisticRegression_test_pred)




print(f"Test Score: {logisticRegression_test_score:0.10f}",
      f"Test Confusion Matrix:\n{confusion_matrix(y_test, logisticRegression_test_pred)}",
      f"Test Report:\n{classification_report(y_test, logisticRegression_test_pred)}", sep="\n")



input_data = (33,0,2,150,240,0,2,170,1,1,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = logisticRegression_model.predict([[20,0,1,100,200,1,2,170,0,0,0]])
# prediction = logisticRegression_model.predict([[33,0,2,150,240,0,2,170,1,1,0]])
print(prediction)

if (prediction[0] == 1):
  print('The patient has HeartDisease')
  
else:
  print('The patient dont have HeartDisease')   


pickle.dump(logisticRegression_model,open("model.pkl","wb"))