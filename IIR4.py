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


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the necessary libraries for building the neural network
import keras
from keras.models import Sequential 
from keras.layers import Dense

# Create a sequential model
classifier = Sequential()

# Add input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=2))

# Add additional hidden layers (you had a syntax error here)
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

# Add the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compile the model
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit the model to the training data
classifier.fit(X_train, y_train, batch_size=5, epochs=10)

# Predict the test set results
y_pred = classifier.predict(X_test)

# Convert the probabilities to binary predictions
y_pred = (y_pred > 0.5)

# Evaluate the model using a confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
