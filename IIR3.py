# import numpy as np

# import matplotlib.pyplot as plt

# import pandas as pd

# #Importing the Artist

# dataset=pd.read_csv('m0isture_days.csv')

# X=dataset.iloc[:, :-1].values

# y=dataset.iloc[:, 1].values

# #SPlitting the dataset into the training set and test set

# from sklearn.model_selection import train_test_split

# X_train, X_test, y _train, y_test=train_test_split(X, y, test_size 1/3, random_state= 0)

# #Feature Scaling

# """from sklearn.preprocessing import StandardScaler
# sc_X StandardScaler()
# X_train=sc_X.fit_transform(X_train)
# X_tests=sc_X.transform(Xtext)
# sc_y=StandardScaler()
# y_train=sc_y.fit_transform(y_train)"""

# #Fitting Simple Linear Regression to the Training set

# from sklearn.linear_model import LinearRegression

# regressor=LinearRegression()

# regressor.fit(X_train, y_train)

# X_test[0]=500;

# X_test[1]=1000;

# #Predicting the test set results

# y_pred=regressor.predict(X_test)

# days=y_pred[0]-y_pred[1]

# print("\n \n \t Water needed after="+str(days)+"days")

# #Visualising the Training set results

# plt.scatter(X_train, y_train, color="red")

# plt.plot(X_train, regressor.predict(X_train), color="blue")

# plt.title("Time vs moisture (Training set)")

# plt.xlabel("Moisture")

# plt.ylabel("Time")

# plt.show()

# #Visualising the Test set results

# plt.scatter (X_test, y_test, color="red")

# plt.plot(X_train, regressor.predict(X_train), color=blue")

# plt.title("Time vs moisture (Training set)")

# plt.xlabel("Moisture")

# plt.ylabel("Time")

# plt.show()

# print("\n \n \t Water needed after="+str(days)="days")

# X_train, X_test, y_trein, y_test=train=test_split(X, y, test_set=0.1, random_state=0)

# #Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train=sc.fit=transform(X_train)
# X_test=sc.transform(X_test)

# import keras
# from keras.models import Sequential 
# from keras.layers import Dense

# classifier=Sequential()

# classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=2))

# classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))

# classifier.add(Dense(output_dim=6, init "uniform", activation="relu"))

# classifier.add(Dense(output_dim=6, init "uniform", activation="relu"))

# classifier.add(Dense(output dim=1, init="uniform", activation="sigmoid"))

# classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# classifier.fit(X_train, y_train, batch_size=5, nb_epoch=10)

# y_pred=classifier.predict(X_test)

# y_pred=(y_pred>0.5)

# from sklearn.metrics import confusion matrix 
# cm=confusion_matrix(y_test, y_pred)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('moisture_days.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Setting some test values
X_test[0] = 500
X_test[1] = 1000

# Predicting the test set results
y_pred = regressor.predict(X_test)

days = y_pred[0] - y_pred[1]

print("\n \n \t Water needed after " + str(days) + " days")

# Visualising the Training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Time vs moisture (Training set)")
plt.xlabel("Moisture")
plt.ylabel("Time")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Time vs moisture (Test set)")
plt.xlabel("Moisture")
plt.ylabel("Time")
plt.show()

print("\n \n \t Water needed after " + str(days) + " days")

# Continue with the Keras code, make sure you have your X_train and y_train defined before this section
