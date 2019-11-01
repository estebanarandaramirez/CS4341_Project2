import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold

seed = np.random.seed(1)

def CrossValidation(model, trainingImages, trainingLabels):
    kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
    cvscores = []
    i = 1
    for train, test in kfold.split(trainingImages, trainingLabels):
        model.fit(trainingImages[train], trainingLabels[train], validation_split=0.16, batch_size=512, epochs=500, verbose=0)
        evaluation = model.evaluate(trainingImages[test], trainingLabels[test], verbose=2)
        print("Fold %i out of 3" % i)
        print("Accuracy: %.2f%%" % (evaluation[1] * 100))
        cvscores.append(evaluation[1] * 100)
        i += 1
    print("Mean(+/-Standard Deviation): %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

images = np.load('images.npy')
images = images.reshape(6500, 784)

labels = np.load('labels.npy')
encodedLabels = np.zeros((6500,10))
encodedLabels[np.arange(6500), labels]=1

pixels = 784
numbers = 10
trainingImages = np.zeros([5200, pixels])
testImages = np.zeros([1300, pixels])
trainingLabels = np.zeros([5200, numbers])
testLabels = np.zeros([1300, numbers])
for i in range(6500):
    if i < 5200:
        trainingImages[i] = images[i]
        trainingLabels[i] = encodedLabels[i]
    else:
        testImages[i-5200] = images[i]
        testLabels[i-5200] = encodedLabels[i]

model = Sequential()
model.add(Dense(50, input_dim=pixels, activation='relu'))
for i in range(9):
    model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
CrossValidation(model, trainingImages, trainingLabels)

evaluation = model.evaluate(testImages, testLabels, verbose=2)
print("Final Baseline Error for Test: %.2f%%" % (100-evaluation[1]*100))