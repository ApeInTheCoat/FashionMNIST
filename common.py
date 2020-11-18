from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def prepare_data(train_csv, test_csv):

    """
    Split the data to train, validate and test from two csv files

    :param train_csv: path to the csv file with train data
    :param test_csv: path to the csv file with test data
    :return: numpy arrays x_train, y_train, x_valid, y_valid, x_test, y_test
    """

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    train = train[test.shape[0]:]
    valid = train[0:test.shape[0]]

    x_train = train.drop(columns="label") / 255
    y_train = train.label
    x_valid = valid.drop(columns="label") / 255
    y_valid = valid.label
    x_test = test.drop(columns="label") / 255
    y_test = test.label

    y_train = tf.keras.utils.to_categorical(y_train)
    y_valid = tf.keras.utils.to_categorical(y_valid)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = x_train.values
    x_valid = x_valid.values
    x_test = x_test.values

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_mistakes(model, x_test, y_test):

    """
    Get the mistakes made by the model

    :param model: model that makes a prediction
    :param x_test: numpy array, data to be predicted
    :param y_test: numpy array, labels of the data
    :return: generator
    """

    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) != np.argmax(y_test[i]):
            yield i


def visualize_predictions(generator, model, x_test, y_test, convolutional=False):

    """
    Visualize Fashion MNIST samples via pyplot
    :param generator: indexes of the samples to be visualised
    :param model: model that makes a prediction
    :param x_test: numpy array, data to be predicted
    :param y_test: numpy array, labels of the data
    :param convolutional: whether the model is a convolutional neural network
    :return: None
    """

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    titles = {}
    for i in generator:
        if not convolutional:
            img_p = x_test[i].reshape(1, 28*28)
        else:
            img_p = x_test[i].reshape(1, 28, 28, 1)
        pred = model.predict_classes(img_p)
        pred_prob = model.predict_proba(img_p)
        pred_prob = "%.2f%%" % (pred_prob[0][pred]*100)
        titles[i] = f"{class_names[pred[0]]} ({bool(y_test[i][pred])}) — Score: {pred_prob}"
        plt.title(titles[i])
        plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
        plt.colorbar()
        plt.grid(False)
        plt.show()

