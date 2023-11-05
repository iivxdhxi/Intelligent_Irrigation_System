# import pandas as pd 
# import numpy as np 
# import matplotlib.pyplot as plt 
# import time 
# from sklearn.model_selection import train_test_split 
# from sklearn.naive_bayes import GaussianNB, BernoulliNB, MaltinomialNB

# #Importing dataset

# data=pd.read_csv("demo.csv")
# data("soil_cleaned"]=np.where(data["soil"]=="d",0,
# 				np.where(data["soil"]=="n",1)
# 					np.where(data["soil"]=="w",2,3)
# 					}
# 				}

# data=data[[
# 	"a0",
# 	"a1",
# 	"soil_cleansed",
# 	"moter",
# ]].dropna (axis=0, how='any')

# X_train, X_test=train_test_split(data, test_size=0.7, random_state=int(time.time()))

# gnb=GaussianNB()

# used features =[

# "soil_cleaned",
# "a0",
# "a1",

# ]
# gnb.fit{
# 	X_train[used_features].values, 
# 	X_train["moter"]
# }
# y_pred=gnb.predict(X_test[used_features]}

# #Print results
# print("Number of mislabeled points out of a total {} points: {}, performance {:05.2f}%"
# 	.format(
# 	X_test.shape[0],
# 	(X_test["moter"] != y_pred).sum(),
# 	100*(1-(X_test["moter"] != y_pred().sum()/X_test.shape[0])
# }}


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Importing dataset
data = pd.read_csv("demo.csv")

# Convert "soil" column to numerical values
data["soil_cleaned"] = np.where(data["soil"] == "d", 0, 
                                np.where(data["soil"] == "n", 1,
                                         np.where(data["soil"] == "w", 2, 3)))

data = data[["a0", "a1", "soil_cleaned", "moter"]].dropna(axis=0, how='any')

X_train, X_test = train_test_split(data, test_size=0.7, random_state=int(time.time()))

gnb = GaussianNB()

used_features = ["soil_cleaned", "a0", "a1"]

gnb.fit(X_train[used_features].values, X_train["moter"])

y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points: {}, performance {:05.2f}%"
      .format(X_test.shape[0], (X_test["moter"] != y_pred).sum(), 
              100 * (1 - (X_test["moter"] != y_pred).sum() / X_test.shape[0])))
