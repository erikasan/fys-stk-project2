import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#calculate accuracy_score when using sklearn
def sklearnmethod(X_train, X_test, y_train, y_test):
    softReg = LogisticRegression(multi_class='multinomial', solver='newton-cg', C =10)
    softReg.fit(X_train, y_train)
    print("Test set accuracy with Logistic Regression: {:.5f}".format(softReg.score(X_test,y_test)))


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    softReg.fit(X_train_scaled, y_train)
    print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(softReg.score(X_test_scaled,y_test)))


#own method for logistic regression
class MultiClassLogisticRegression():
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(-1, 1)

    def p(self, X):
        pre_vals = (X@self.theta.T).reshape(-1, len(self.classes))
        return self.softmax(pre_vals)

    def cost_function(self,y, X):
        p = self.p(X)
        return -1*np.mean(y *np.log(p)) #using cross entropy

    def fit(self, X, y, epochs=1000, M=128, eta=0.001, lamb = 0):
        np.random.seed(0)
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        X = np.insert(X, 0, 1, axis =1)
        y = np.eye(len(self.classes))[y.reshape(-1)]
        self.theta = np.zeros(shape=(len(self.classes), X.shape[1]))
        n_M = np.round(int(epochs/M)) # number of minibatches
        minibatc_indices = np.split(np.arange(epochs),np.arange(1,n_M)*M)
        self.cost = []
        for i in range(epochs):
            self.cost.append(self.cost_function(y, X))
            for j in range(n_M):
                ind = np.random.choice(X.shape[0], n_M)
                xi = X[ind]
                yi = y[ind]
                p = self.p(xi)
                gradient = np.dot((yi-p).T, xi) + 2*lamb*self.theta
                self.theta += eta * gradient

    def predict(self, X):
        return self.p(np.insert(X, 0, 1, axis = 1))

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis =1)

    def accuracy(self, X, y):
        return np.mean(self.predict_classes(X) == y)


if __name__ == '__main__':

    np.random.seed(0)

    #import the digits dataset
    digits = datasets.load_digits()

    inputs = digits.images
    labels = digits.target

    #flatten the image
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)


    #split into test and train
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, train_size=0.8, test_size=0.2)

    lr = MultiClassLogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict_classes(X_test)
    print(lr.accuracy(X_test, y_test))

    lr_ridge = MultiClassLogisticRegression()
    lr_ridge.fit(X_train, y_train, lamb = 0.001)
    y_pred = lr_ridge.predict_classes(X_test)
    print(lr_ridge.accuracy(X_test, y_test))


    #plot cost function
    cost = lr.cost
    plt.plot(np.arange(len(cost)), cost)
    plt.title('Cost function as a function of epoch')
    plt.xlabel('epochs')
    plt.ylabel('cost function')
    plt.show()


    #See how the number of epochs affects the accuray score
    epochs = np.linspace(10, 1000, 100)

    acc = []
    for epoch in epochs:
        lr = MultiClassLogisticRegression()
        lr.fit(X_train, y_train, int(epoch))
        y_pred = lr.predict_classes(X_test)
        acc.append(lr.accuracy(X_test, y_test))

    plt.plot(epochs, acc)
    plt.title('Accuracy as a function of epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.show()

    #see how accuracy changes with learning rate
    learning_rate = np.linspace(0.0001, 0.1, 11)
    print(learning_rate)
    acc = []
    for learn in learning_rate:
        lr = MultiClassLogisticRegression()
        lr.fit(X_train, y_train, epochs=1000, M=128, eta=int(learn))
        y_pred = lr.predict_classes(X_test)
        acc.append(lr.accuracy(X_test, y_test))

    plt.plot(epochs, acc)
    plt.title('Accuracy as a function of epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.show()