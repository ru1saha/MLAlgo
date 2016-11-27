__author__ = 'Rudra Saha, ASU ID : 1211135702'
'''
kNN implementation for Hot-or-Not.

kNN function will take the train and the test data with labels and
push out train and test error tuple list with k value viz, (kValue, test/train error).
We will use this tuple to plot our error rates.
Current implementation is just a single flow code, need to modularize it.
'''
import scipy.io as sio #Need to use as data in a .m file
import numpy as np
from scipy import stats #for calculating the mode
from scipy import spatial #cosine similarity as distance measure
import matplotlib.pyplot as plt

data = sio.loadmat('faces.mat')

trainData = data['traindata']
trainLabels = data['trainlabels']
testData = data['testdata']
testLabels= data['testlabels']

k = [1,10,20,30,40,50,60,70,80,90,100]

trainError = []
testError = []

numTrain = np.shape(trainData)[0]
numTest = np.shape(testData)[0]
cosineMatrixTest = np.zeros(shape=(numTest, numTrain))
cosineMatrixTrain = np.zeros(shape=(numTrain, numTrain))

#Precomputing the cosine distance and saving it.
#Doing it for both the training and the test data.
for i in range(numTrain):
    for j in range(numTrain):
        cosineMatrixTrain[i, j] = spatial.distance.cosine(trainData[i], trainData[j])

for i in range(numTest):
    for j in range(numTrain):
        cosineMatrixTest[i, j] = spatial.distance.cosine(testData[i], trainData[j])



NNTrainMatrix = np.zeros(shape=(numTrain, numTrain))
NNTestMatrix = np.zeros(shape=(numTest, numTrain))

'''
Using np.argsort to find the indices that will be looked in the label vector.
Voting criteria is majority voting, and ties are broken by choosing first label.
Will add voting criteria with kValue decay.
'''

NNTrainMatrix = np.argsort(cosineMatrixTrain)
NNTestMatrix = np.argsort(cosineMatrixTest)


for i in k:
    labelMatrixTrain = NNTrainMatrix[:, -i:]
    for m in range(numTrain):
        for n in range(i):
            labelMatrixTrain[m, n] = trainLabels[NNTrainMatrix[m, n]]
    predictedTrainLabels = np.zeros(shape=(numTrain, 1))
    predictedTrainLabels = stats.mode(labelMatrixTrain, axis=1)[0]
    error = np.sum(np.absolute(np.subtract(predictedTrainLabels, trainLabels)))
    errorRate = error/numTrain * 100
    trainError.append((i, errorRate))


for i in k:
    labelMatrixTest = NNTestMatrix[:, -i:]
    for m in range(numTest):
        for n in range(i):
            labelMatrixTest[m, n] = trainLabels[NNTestMatrix[m, n]]
    predictedTestLabels = np.zeros(shape=(numTest, 1))
    predictedTestLabels = stats.mode(labelMatrixTest, axis=1)[0]
    error = np.sum(np.absolute(np.subtract(predictedTestLabels, testLabels)))
    errorRate = error/numTest * 100
    testError.append((i, errorRate))


train1 = trainData[0, :]
train2 = trainData[1, :]
dis12 = spatial.distance.cosine(train1, train2) #To check the cosine similarity implementation, fixed value given


'''
print(trainData.shape, trainLabels.shape, testData.shape, testLabels.shape)
train1 = trainData[0,:]
train1 = train1.reshape(86, 86)
img = Image.fromarray(train1, '1')
img.show()
Just a way to visualize the image
'''

print("Training Error : " + str(trainError))
print("Test Error : " + str(testError))

test = zip(*testError)
train = zip(*trainError)
plt.title('Train Error v/s Test Error, kNN')
plt.xlabel('k Value')
plt.ylabel('Error')
plot1, = plt.plot(*test, marker='o')
plot2, = plt.plot(*train, marker='x')
plt.xticks(np.arange(0, 110, 10.0))
plt.yticks(np.arange(0, 50 + 10, 5.0))
plt.legend([plot1, plot2], ['test error', 'train error'], loc=4)
plt.grid()
plt.show()