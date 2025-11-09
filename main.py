from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 
import numpy as np

# Loading Data
df=pd.read_csv('Loan.csv')

# Seperate Features X,y
X=df[['Income','Loan_Amount','Credit_Score']]
y=df['Defaulted ']

# Split Data into training and Testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Training Model
RVC=RandomForestClassifier(n_estimators=100,random_state=42)
RVC.fit(X_train,y_train)

# Model Predictions
y_pred=RVC.predict(X_test)

#Model Evaluation
print('Accuracy Score : \n',accuracy_score(y_test,y_pred))
print('\n Classification Report: \n',classification_report(y_test,y_pred))
print('\n Confusion Matrix: \n',confusion_matrix(y_test,y_pred))

# Making Predictions for New Applicant
new_applicant = [[30000, 15000, 600]]  # Income, Loan_Amount, Credit_Score
prediction = RVC.predict(new_applicant)

if prediction[0] == 1:
    print("\n🔴 The applicant is likely to DEFAULT on the loan.")
else:
    print("\n🟢 The applicant is likely to REPAY the loan.")

