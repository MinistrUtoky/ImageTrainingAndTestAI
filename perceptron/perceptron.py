import os
from pathlib import Path
import keras.utils as image
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import shuffle


p = Path('./Dataset')
dirs = p.glob("*")

image_data = []
labels = []

label_dict = {}
label_to_anime = {}
counter = 0
for i in dirs:
    label = str(i).split("\\")[-1]
    label_dict[label] = counter
    label_to_anime[counter] = label

    print(i)
    count = 0

    for img_path in i.glob("*.png"):
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(counter)
        count += 1

    print(count)
    counter += 1


X=np.array(image_data)
Y=np.array(labels)
#plt.imshow(X[0]/255)
def drawimage(image,label):
    plt.title(label_to_anime[label])
    plt.imshow(image)
    plt.show()
#drawimage(X[344]/255,Y[344])

random.seed(10)


X,Y = shuffle(X,Y,random_state=2)

#норм))()()
X = X/255.0

#драугр встает
X_ = np.array(X)
Y_ = np.array(Y)

#драугр качается
X = X_[:136, :]
Y = Y_[:136]

#драугр идет бить лицо
XTest = X_[:,:]
YTest = Y_[:]

print(X.shape,Y.shape)
print(XTest.shape,YTest.shape)


class NeuralNetwork:

    def __init__(self, input_size, layers, output_size):
        np.random.seed(0)

        model = {}  #словарик

        # первослой
        model['W1'] = np.random.randn(input_size, layers[0])
        model['b1'] = np.zeros((1, layers[0]))

        # второслой
        model['W2'] = np.random.randn(layers[0], layers[1])
        model['b2'] = np.zeros((1, layers[1]))

        # йой
        model['W3'] = np.random.randn(layers[1], layers[2])
        model['b3'] = np.zeros((1, layers[2]))

        # выводослой
        model['W4'] = np.random.randn(layers[2], output_size)
        model['b4'] = np.zeros((1, output_size))

        self.model = model
        self.activation_outputs = None

    def forward(self, x):
        W1, W2, W3, W4 = self.model['W1'], self.model['W2'], self.model['W3'], self.model['W4']
        b1, b2, b3, b4 = self.model['b1'], self.model['b2'], self.model['b3'], self.model['b4']

        z1 = np.dot(x, W1) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = np.tanh(z2)

        z3 = np.dot(a2, W3) + b3
        a3 = np.tanh(z3)

        z4 = np.dot(a3, W4) + b4
        y_ = softmax(z4)

        self.activation_outputs = (a1, a2, a3, y_)
        return y_

    def backward(self, x, y, learning_rate=0.001):
        W1, W2, W3, W4 = self.model['W1'], self.model['W2'], self.model['W3'], self.model['W4']
        b1, b2, b3, b4 = self.model['b1'], self.model['b2'], self.model['b3'], self.model['b4']
        m = x.shape[0]

        a1, a2, a3, y_ = self.activation_outputs

        delta4 = y_ - y
        dw4 = np.dot(a3.T, delta4)
        db4 = np.sum(delta4, axis=0)

        delta3 = (1 - np.square(a3)) * np.dot(delta4, W4.T)
        dw3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0)

        delta2 = (1 - np.square(a2)) * np.dot(delta3, W3.T)
        dw2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        delta1 = (1 - np.square(a1)) * np.dot(delta2, W2.T)
        dw1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # обновялем параметры модели градиентным спуском
        self.model["W1"] -= learning_rate * dw1
        self.model['b1'] -= learning_rate * db1

        self.model["W2"] -= learning_rate * dw2
        self.model['b2'] -= learning_rate * db2

        self.model["W3"] -= learning_rate * dw3
        self.model['b3'] -= learning_rate * db3

        self.model["W4"] -= learning_rate * dw4
        self.model['b4'] -= learning_rate * db4

        # :)

    def predict(self, x):
        y_out = self.forward(x)
        return np.argmax(y_out, axis=1)

    def summary(self):
        W1, W2, W3, W4 = self.model['W1'], self.model['W2'], self.model['W3'], self.model['W4']
        a1, a2, a3, y_ = self.activation_outputs

        print("W1 ", W1.shape)
        print("A1 ", a1.shape)


def softmax(a):
    e_pa = np.exp(a)
    ans = e_pa / np.sum(e_pa, axis=1, keepdims=True)
    return ans


def loss(y_oht, p):
    l = -np.mean(y_oht * np.log(p))
    return l


def one_hot(y, depth):
    m = y.shape[0]
    y_oht = np.zeros((m, depth))
    y_oht[np.arange(m), y] = 1
    return y_oht


def train(X, Y, model, epochs, learning_rate, logs=True):
    training_loss = []

    classes = 3
    Y_OHT = one_hot(Y, classes)

    for ix in range(epochs):

        Y_ = model.forward(X)
        l = loss(Y_OHT, Y_)

        model.backward(X, Y_OHT, learning_rate)
        training_loss.append(l)
        if (logs and ix % 50 == 0):
            print("Epoch %d Loss %.4f" % (ix, l))

    return training_loss

model = NeuralNetwork(input_size=12288,layers=[200,50,20],output_size=3)

X = X.reshape(X.shape[0],-1)
print(X.shape)

XTest = XTest.reshape(XTest.shape[0],-1)
print(XTest.shape)

#ЛЮТЫЙ ПАМПЕЦ ПРЯМО ИЗ КАТАКОМБ!!!
l = train(X,Y,model,1000,0.0005)

#--------------------------------------------------------------------------------------------------------
#кросивое
'''
import matplotlib.pyplot as plt
plt.style.use("dark_background")
plt.title("Training Loss vs Epochs")
plt.plot(l)

plt.show()
'''

def getAccuracy(X, Y, model):
    outputs = model.predict(X)
    acc = np.sum(outputs == Y) / Y.shape[0]
    return acc


print("Train Accuracy: %.4f  :)" % getAccuracy(X, Y, model))
print("Test Accuracy: %.4f :(" % getAccuracy(XTest, YTest, model))
#print("Real percentage of angwy fawses: %.4f >:з" % (41/131))
#print("Real percentage of cwying fawses: %.4f (/;^;)/" % (40/131))
#print("Real percentage of sad fawses: %.4f ('^')" % (50/131))


def class_folder_test_function():
    p = Path('./Dataset')
    dirs = p.glob("*")

    image_data = []
    labels = []

    label_dict = {}
    label_to_anime = {}
    counter = 0
    for i in dirs:
        label = str(i).split("\\")[-1]
        label_dict[label] = counter
        label_to_anime[counter] = label

        count = 0

        for img_path in i.glob("*.png"):
            img = image.load_img(img_path, target_size=(64, 64))
            img_array = image.img_to_array(img)
            image_data.append(img_array)
            labels.append(counter)
            count += 1

        counter += 1

    X = np.array(image_data)
    Y = np.array(labels)
    X = X / 255.0
    X_ = np.array(X)
    Y_ = np.array(Y)
    XTest = X_[:, :]
    YTest = Y_[:]
    XTest = XTest.reshape(XTest.shape[0], -1)

    print("Angwy accuracy: %.4f >:з" % getAccuracy(XTest[:42], YTest[:42], model))
    print("Purisudu accuracy: %.4f =^-^=" % getAccuracy(XTest[42:77], YTest[42:77], model))
    print("Shiokku accuracy: %.4f _(o-o)_" % getAccuracy(XTest[77:], YTest[77:], model))


class_folder_test_function()

#ищо кросивое
'''
print('Enter the image path: ')
img_path=input()

img = image.load_img(img_path,target_size=(64,64))
img = image.img_to_array(img)
print(img.shape)
img= img.reshape(1,-1)

y=model.predict(img)
for i in label_to_anime.keys():
    print(i)
    if i==y:
        print(label_to_anime[i])
        break'''
'''    
from sklearn.metrics import confusion_matrix
from visualize import plot_confusion_matrix
from sklearn.metrics import classification_report

output=model.predict(XTest)
conf_mat=confusion_matrix(output,YTest)
print(conf_mat)

print(classification_report(output,YTest))

plot_confusion_matrix(conf_mat,classes=["Bulbasaur","Meowth","Pikachu"],title="Confusion Matrix Test")
'''