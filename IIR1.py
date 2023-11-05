# import random
# import csv
# import math

# split = 0.66

# with open('C:/Users/Seema Biltheria/Downloads/archive (3).zip') as csvfile:
#     lines = csv.reader(csvfile)
#     dataset = list(lines)

# random.shuffle(dataset)

# div = int(split * len(dataset))  # Added a missing parenthesis here
# train = dataset[:div]
# test = dataset[div:]

# # square root of the sum of the squared differences between the two arrays of numbers
# def euclideanDistance(instancel, instance2, length):
#     distance = 0
#     for x in range(length):
#         distance += pow((float(instancel[x]) - float(instance2[x]), 2)
#     return math.sqrt(distance)

# import operator

# def getNeighbors(trainingSet, testInstance, k):
#     distances = []
#     length = len(testInstance) - 1
#     for x in range(len(trainingSet):
#         dist = euclideanDistance(testInstance, trainingSet[x], length)
#         distances.append((trainingSet[x], dist))
#     distances.sort(key=operator.itemgetter(1))
#     neighbors = []
#     for x in range(k):
#         neighbors.append(distances[x][0])
#     return neighbors

# import collections

# def getResponse(neighbors):
#     classVotes = collections.defaultdict(int)
#     for x in range(len(neighbors)):
#         response = neighbors[x][-1]
#         classVotes[response] += 1
#     sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedVotes[0][0]

# def getAccuracy(testSet, predictions):
#     correct = 0
#     for x in range(len(testSet)):
#         if testSet[x][-1] == predictions[x]:
#             correct += 1
#     return (correct / float(len(testSet)) * 100.0)  # Added a missing parenthesis here

# k = 3
# trainSet = train
# testSet = test
# predictions = []

# for x in range(len(testSet)):
#     neighbors = getNeighbors(trainSet, testSet[x], k)
#     result = getResponse(neighbors)
#     predictions.append(result)
#     print(f"Predicted: {result}, Actual: {testSet[x][-1]}")

# accuracy = getAccuracy(testSet, predictions)
# print(f"Accuracy: {accuracy:.2f}%")


import random
import csv
import math

split = 0.66

# You cannot directly open a ZIP file with the csv.reader. You need to extract the CSV file from the ZIP file first.
# Assuming 'data.csv' is inside the ZIP file, you can do this:

import zipfile

with zipfile.ZipFile('C:/Users/Seema Biltheria/Downloads/archive (3).zip', 'r') as zip_file:
    # Assuming 'data.csv' is inside the ZIP file
    with zip_file.open('data.csv') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)

random.shuffle(dataset)

div = int(split * len(dataset))

train = dataset[:div]
test = dataset[div:]

# square root of the sum of the squared differences between the two arrays of numbers
def euclideanDistance(instancel, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instancel[x]) - float(instance2[x]), 2))
    return math.sqrt(distance)

import operator

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

import collections

def getResponse(neighbors):
    classVotes = collections.defaultdict(int)
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        classVotes[response] += 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet)) * 100.0)

k = 3
trainSet = train
testSet = test
predictions = []

for x in range(len(testSet)):
    neighbors = getNeighbors(trainSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print(f"Predicted: {result}, Actual: {testSet[x][-1]}")

accuracy = getAccuracy(testSet, predictions)
print(f"Accuracy: {accuracy:.2f}%")
