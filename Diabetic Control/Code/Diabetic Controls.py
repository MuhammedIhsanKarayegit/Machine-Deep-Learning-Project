# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Reading the dataset from a CSV file
df = pd.read_csv('Documents\\diabetes.csv')

# Extracting the target variable 'Outcome' and features 'x_raw_data'
y = df.Outcome.values
x_raw_data = df.drop(["Outcome"], axis=1)

#Normalization process performed
outo_normalize = MinMaxScaler()
x = outo_normalize.fit_transform(x_raw_data)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Creating a KNeighborsClassifier with k=7
knn = KNeighborsClassifier(n_neighbors=7)

# Training the KNN model
knn.fit(x_train, y_train)

# Making predictions on the test set
prediction = knn.predict(x_test)

# Printing the accuracy score on the test set for k=20
print("K = 7 için test verilerinin doğruluk skoru: ", knn.score(x_test, y_test))

#Here we decide what kind of output to give to the screen according to the machine's prediction based on patient data.
new_prediction = knn.predict(outo_normalize.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
if new_prediction[0] == 1:
    print("The patient whose information is entered is diabetic..")
else:
    print("The patient whose information was entered is not diabetic.")