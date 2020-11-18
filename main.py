from tensorflow import keras
from algorithms import logical_regression, dense_neural_network, convolutional_neural_network
from common import prepare_data, get_mistakes, visualize_predictions

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data(train_csv="fashion-mnist_train.csv",
                                                                      test_csv="fashion-mnist_test.csv")
    # Register Tensorboard callbacks
    LR_callback = keras.callbacks.TensorBoard(log_dir="logs\\LR")
    DNN_callback = keras.callbacks.TensorBoard(log_dir="logs\\DNN")
    CNN_callback = keras.callbacks.TensorBoard(log_dir="logs\\CNN")
    BN_callback = keras.callbacks.TensorBoard(log_dir="logs\\BN")

    # Train different models, using train and validation sets
    log_reg = logical_regression(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                 callback=LR_callback)
    dense_nn = dense_neural_network(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                    callback=DNN_callback)
    conv_nn = convolutional_neural_network(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                           callback=CNN_callback)
    bn_conv_nn = convolutional_neural_network(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid,
                                              callback=BN_callback, batch_normalization=True)

    # Evaluate trained models on the test set
    log_reg.evaluate(x_test, y_test)
    dense_nn.evaluate(x_test, y_test)
    conv_nn.evaluate(x_test.reshape(10000, 28, 28, 1), y_test)
    bn_conv_nn.evaluate(x_test.reshape(10000, 28, 28, 1), y_test)

    # Visualize the results, how different models predicted the first 10 samples that logical regression got wrong
    mistakes = list(get_mistakes(log_reg, x_test, y_test))
    visualize_predictions(mistakes[:10], log_reg, x_test, y_test)
    visualize_predictions(mistakes[:10], dense_nn, x_test, y_test)
    visualize_predictions(mistakes[:10], conv_nn, x_test, y_test, convolutional=True)
    visualize_predictions(mistakes[:10], bn_conv_nn, x_test, y_test, convolutional=True)
