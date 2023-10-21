import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# loading sklearn breast_cancer dataset
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
breast_cancer_dataset

# load it into datafrome
data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names,)
data_frame

data_frame.head()

# add the target colum to the dataframe
data_frame['target']=breast_cancer_dataset.target

# now dropping the target column from dataset
X=data_frame.drop(columns='target',axis=1)
y=data_frame['target']

# Splitting the data in to testing and training data
X_train,X_test,y_train,y_test=train_test_split(X,y)

# Logistic Regression
model=LogisticRegression(max_iter=50000)

# training the logistic regression model
model.fit(X_train,y_train)

Y_predicted=model.predict(X_test)

accuracy=accuracy_score(y_test,Y_predicted)

print(f"The accuracy of the model using the Fast Learner (Logistic Regression): {accuracy*100}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled)

# Evaluate the model
accuracylazy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model using Lazy Learner (KNN): {accuracylazy*100}%")