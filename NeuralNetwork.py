import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def data_cleaner(raw_data, limits):
    raw_data = raw_data[(raw_data['Velocity_X'] < limits)
                        & (raw_data['Velocity_X'] > -limits)]
    raw_data = raw_data[(raw_data['Velocity_Y'] < limits)
                        & (raw_data['Velocity_Y'] > -limits)]
    # raw_data = raw_data[(raw_data['q6'] < 1.0) & (raw_data['q6'] > -1.0)]
    cleaned_data = norm(raw_data)
    return cleaned_data


def output_format(data):
    # We need to separate the arrays because the model outputs two sets of arrays (Velocity X  & Velocity Y)
    y1 = data.pop("Velocity_X")
    y1 = np.array(y1)
    y2 = data.pop("Velocity_Y")
    y2 = np.array(y2)
    return y1, y2


def input_format(data):
    return np.array(data)


def norm(x):
    normalized_list = x.apply(
        lambda x: 2 * ((x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))) - 1 if x.name in ['Velocity_X',
                                                                                                'Velocity_Y'] else x)
    return normalized_list


def plot_diff(y_true, y_pred, bs, title='', w=1.2):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(-1 * w, 1 * w)
    plt.ylim(-1 * w, 1 * w)
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, bs, md, ylim=0):
    plt.title(title)
    plt.ylim(0.000005, ylim)
    plt.plot(md.history[metric_name], color='blue',
             label=metric_name, marker='o', markevery=50)
    plt.plot(md.history['val_' + metric_name], color='green',
             label='val_' + metric_name, marker='^', markevery=50)
    plt.yscale('log')
    plt.show()


def build_model(unit: int, activation) -> object:
    input_layer = Input(shape=(len(train.columns)))

    dense1 = Dense(units=unit, activation=activation)(input_layer)

    dense2 = Dense(units=unit, activation=activation)(dense1)

    dense3 = Dense(units=unit, activation=activation)(dense2)

    dense4 = Dense(units=unit, activation=activation)(dense3)

    # dense5 = Dense(units=unit, activation=activation)(dense4)

    # dense6 = Dense(units=unit, activation=activation)(dense5)
    #
    # dense7 = Dense(units=unit, activation=activation)(dense6)
    #
    # dense8 = Dense(units=unit, activation=activation)(dense7)

    # dense9 = Dense(units=unit, activation=activation)(dense8)
    #
    # dense10 = Dense(units=unit, activation=activation)(dense9)
    #
    # dense11 = Dense(units=unit, activation=activation)(dense10)
    #
    # dense12 = Dense(units=unit, activation=activation)(dense11)

    # dense13 = Dense(units=unit, activation=activation)(dense12)

    y1_output = Dense(units='1', name='Velocity_X_output',
                      activation='linear')(dense4)
    y2_output = Dense(units='1', name='Velocity_Y_output',
                      activation='linear')(dense4)

    defined_model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

    return defined_model


def fit_model(trainx, trainy, testx, testy, valx, valy, output):
    model = build_model(unit=128, activation='tanh')

    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    # optimizer = tfa.optimizers.MovingAverage(optimizer)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=optimizer,
                  loss={'Velocity_X_output': 'mse',
                        'Velocity_Y_output': 'mse'},
                  metrics={'Velocity_X_output': tf.keras.metrics.RootMeanSquaredError(),
                           'Velocity_Y_output': tf.keras.metrics.RootMeanSquaredError()})

    # es = tf.keras.callbacks.EarlyStopping(monitor='Velocity_X_output_root_mean_squared_error', patience=50)

    history = model.fit(trainx, trainy,
                        epochs=3000, batch_size=640,
                        validation_data=(testx, testy),
                        callbacks=[tensorboard_callback])

    # model.save('Finalized Model.h5')
    model.summary()

    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=valx, y=valy)
    Y_pred = model.predict(test_x)
    Velocity_X_pred = Y_pred[0]
    Velocity_Y_pred = Y_pred[1]

    # Plot Comparison between predicted and actual values
    plot_diff(test_y[0], Y_pred[0], bs=32, title='Velocity X (Normalized')
    plot_diff(test_y[1], Y_pred[1], bs=32, title='Velocity Y (Normalized)')

    # Plot RMSE
    plot_metrics(metric_name='Velocity_X_output_root_mean_squared_error', title='Velocity X RMSE', md=history,
                 bs=32,
                 ylim=0.2)
    plot_metrics(metric_name='Velocity_Y_output_root_mean_squared_error', title='Velocity Y RMSE', md=history,
                 bs=32,
                 ylim=0.2)

    # Plot loss
    plot_metrics(metric_name='Velocity_X_output_loss',
                 title='Velocity X LOSS', bs=32, md=history, ylim=0.0006)
    plot_metrics(metric_name='Velocity_Y_output_loss',
                 title='Velocity Y LOSS', bs=32, md=history, ylim=0.03)

    # plt.plot(history.history['Velocity_X_output_root_mean_squared_error'], label='Velocity X RMSE', marker='o',
    #          markevery=30)
    # plt.plot(history.history['val_Velocity_Y_output_root_mean_squared_error'], label='Velocity X Val RMSE', marker='^',
    #          markevery=30)
    # plt.ylabel('RMSE')
    # plt.xlabel('Epochs')
    # plt.legend(["Vel X RMSE", "Vel X RMSE Val"], loc="upper right")
    # plt.title('Max Epoch = ' + str(ne), pad=-40)

    output.loc[-1] = [loss, loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse]
    output.index = output.index + 1

    print('Overall Results for Max Epoch = ', 200)
    print(f'loss: {loss}')
    print(f'Velocity_X_loss: {Y1_loss}')
    print(f'Velocity_Y_loss: {Y2_loss}')
    print(f'Velocity_X_rmse: {Y1_rmse}')
    print(f'Velocity_Y_rmse: {Y2_rmse}')
    print('\n')


########################################################################################################################

# Import the Dataset and split into Training and Testing Data
dataset = pd.read_csv("2D_Hele_Shaw_Data (test).csv")
anomaly_limit = dataset.Velocity_X.std()
dataset = data_cleaner(dataset, (2 * anomaly_limit))
# dataset.to_csv("test pre data.csv")
# dataset.boxplot()
# plt.show()
# dataset = norm(dataset)
# dataset.to_csv('test cleaned data.csv')

# Split the data into train and test with 80 train / 20 test
train, test = train_test_split(dataset, test_size=0.3, random_state=1)
train, val = train_test_split(train, test_size=0.3, random_state=1)

# Prepare data into correct format and build the model
# Output format should be a tuple of two 1D arrays for Output data but np.array for Input data
train_y = output_format(train[['Velocity_X', 'Velocity_Y']])
test_y = output_format(test[['Velocity_X', 'Velocity_Y']])
val_y = output_format(val[['Velocity_X', 'Velocity_Y']])

train = train.drop(columns=['Velocity_X', 'Velocity_Y'])
test = test.drop(columns=['Velocity_X', 'Velocity_Y'])
val = val.drop(columns=['Velocity_X', 'Velocity_Y'])

train_x = input_format(train)
test_x = input_format(test)
val_x = input_format(val)

np.save('test_x.npy', test_x)

batch_size = [32, 64, 128, 320]
unit_num = [8, 16, 32, 64]
n_epoch = [100, 200, 300, 500]
learning_rates = [0.5, 0.05, 0.005, 0.0005, 0.00005, 0.00001]
activation_func = ['relu', 'tanh', 'sigmoid', 'softmax',
                   'softplus', 'softsign', 'exponential', 'linear']

output_data = pd.DataFrame(
    columns=['HL Units', 'loss', 'Velocity_X_loss', 'Velocity_Y_loss', 'Velocity_X_rmse', 'Velocity_Y_rmse'])

### This is for single runs ###
fit_model(train_x, train_y, test_x, test_y, val_x, val_y, output_data)

### This is for looping over certain hyperparameters ###
# for i in range(len(n_epoch)):
#     # Create each axe for each loop of batch_size
#     plot_no = 420 + (i + 1)
#     plt.subplot(plot_no)
#     fit_model(train_x, train_y, test_x, test_y, val_x, val_y, n_epoch[i], output_data)
#
# plt.show()
# print(output_data)
output_data.to_csv("Results.csv")
