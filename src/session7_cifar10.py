# path tools
import os
# generic tools
import numpy as np
import cv2
# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# defining function for loading data and preprocessing
def load_data():
    # loading in data set with training and test
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    # convert to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # compress pixel values
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # reshape values for training data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    # reshape values for test data
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train, X_test, lb, y_train, y_test, X_train_dataset, X_test_dataset

# defining function for model and it's architecture
def create_model(training_data, training_labels, testing_data):
    # define architecture 784x256x128x10
    model = Sequential() 
    model.add(Dense(256,
                    input_shape=(1024,), 
                    activation="relu")) 
    model.add(Dense(128, 
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax")) 
    # train model using SGD (stocastic grading descent)
    sgd = SGD(0.01) 
    model.compile(loss="categorical_crossentropy",
                optimizer=sgd, 
                metrics=["accuracy"]) 
    # using model.fit method to fit model onto training data and labels
    history = model.fit(training_data, training_labels, 
                        epochs=10, 
                        batch_size=32) 
    # evaluate network
    predictions = model.predict(testing_data, batch_size=32)
    return predictions

# defining main function
def main():
    # get data
    X_train, X_test, lb, y_train, y_test, X_train_dataset, X_test_dataset = load_data()
    # get model
    predictions = create_model(X_train_dataset, y_train, X_test_dataset)
    # creating classifictation report
    report = classification_report(y_test.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_])
    # defining path
    folderpath = os.path.join("src")
    # defining filename
    filename = "classification_report_cifar10.txt"
    # writing and saving classification report
    filepath = os.path.join(folderpath, filename)
    with open(filepath, "w") as f:
        f.write(report)

if __name__=="__main__":
    main() 

