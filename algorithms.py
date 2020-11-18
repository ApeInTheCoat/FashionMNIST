import tensorflow as tf


"""
Predefined ml models made via tensorflow
"""

def logical_regression(x_train, x_valid, y_train, y_valid, callback):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation="softmax", input_shape=(784,)))
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid), callbacks=[callback])
    return model


def dense_neural_network(x_train, x_valid, y_train, y_valid, callback):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(784,)))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid), callbacks=[callback])
    return model


def convolutional_neural_network(x_train, x_valid, y_train, y_valid, callback, batch_normalization=False):

    # Reshaping data
    x_train = x_train.reshape(50000, 28, 28, 1)
    x_valid = x_valid.reshape(10000, 28, 28, 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    if batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, activation="relu"))
    if batch_normalization:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    model.summary()
    model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_valid, y_valid), callbacks=[callback])
    return model
