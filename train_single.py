import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import *
from window_generator_reg import WindowGenerator, ShuffleWindowGenerator
from utils import get_dataframe, APPS, APPS_DICT

INPUT_STEPS = 144
OUTPUT_STEPS = 36
LSTM_UNITS = 16
BATCH_SIZE = 64
data_file_path = 'C:/data/CM003_10min.gzip'
fea_app = [20,9,5,34,35,42,48,28]
fea_ori = [20,9,5,34,35,42,48,28,'hour_cos','hour_sin','day_cos','day_sin','month_sin','month_cos']
fea_norm = ['norm_20', 'norm_9','norm_5','norm_34','norm_35','norm_42','norm_48','norm_28']
fea_without_time = [20,9,5,34,35,42,48,28,'gap_20','lagu_20','gap_9','lagu_9',
                    'gap_5','lagu_5','gap_34','lagu_34','gap_35','lagu_35','gap_42','lagu_42',
                    'gap_48','lagu_48','gap_28','lagu_28',]
fea_gap_lagu = ['gap_20','lagu_20','gap_9','lagu_9','gap_5','lagu_5','gap_34','lagu_34','gap_35','lagu_35',
                'gap_42','lagu_42', 'gap_48','lagu_48','gap_28','lagu_28',]
FEA = fea_norm
EPOCHS = 10
NUM_SLICES = 6
LEARNING_RATE = 1e-4


if __name__ == '__main__':

    df_std=get_dataframe(data_file_path, scale=2, shift=-1, normstd=5)
    ## dataset
    # win_generator = WindowGenerator(input_width=INPUT_STEPS, label_width=OUTPUT_STEPS, shift=OUTPUT_STEPS, 
    #                                 train_df=train_df, test_df=None, val_df=val_df,
    #                                 batch_size=BATCH_SIZE, features_columns=FEA, label_columns=APPS)
    win_generator = ShuffleWindowGenerator(input_width=INPUT_STEPS, label_width=OUTPUT_STEPS, shift=OUTPUT_STEPS, 
                                    dataframe=df_std,
                                    batch_size=BATCH_SIZE, features_columns=FEA, label_columns=FEA)

    ## test the datasets
    for test_inputs, test_labels in win_generator.val.take(1):
        print('Inputs Shape', test_inputs.shape)
        print('Labels Shape', test_labels.shape)
    print('Number of Batches in Training Set:', win_generator.train.cardinality().numpy())
    print('Number of Batches in Validation Set:', win_generator.val.cardinality().numpy())

    ## training
    # model = SL_LSTM_S2S(units=LSTM_UNITS, out_steps = OUTPUT_STEPS, out_dims=len(APPS), if_cnn=True, num_slices=NUM_SLICES)
    # model = Att_SL_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(FEA), attention_type='additive', num_slices=NUM_SLICES)
    model = Att_ML_S2S(
        units=LSTM_UNITS, 
        out_steps=OUTPUT_STEPS, 
        out_dims=len(FEA), 
        attention_type='multihead', 
        num_slices=NUM_SLICES, 
        if_cnn=True
        )
    model.build((BATCH_SIZE, INPUT_STEPS, len(FEA)))
    model.summary()
    history = model.custom_train(
        data=win_generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
        )
    print("Val:", model.evaluate(win_generator.val))
    print("Train:", model.evaluate(win_generator.train))

    # save models and history
    pd.DataFrame(history).to_csv('output/train_single/single_his.csv')
    model.save_weights('models/att_ml_s2s.h5')
    # history = pd.read_csv('output/train_single/single_his.csv')
    # model.load_weights('models/att_ml_s2s.h5')

    ## plot
    ## plot the loss graph
    plt.figure(dpi=200)
    plt.ylabel('Loss(SliceMSE)')
    plt.plot(history['loss'], c='red', label='Traininsg')
    plt.plot(history['val_loss'], c='blue', label='Validation')
    plt.legend()
    plt.savefig('./output/train_single/loss.png')
    ## plot the mae graph
    plt.figure(dpi=200)
    plt.ylabel('Mean Absolute Error')
    plt.plot(history['mean_absolute_error'], c='red', label='Training')
    plt.plot(history['val_mean_absolute_error'], c='blue', label='Validation')
    plt.legend()
    plt.savefig('./output/train_single/mae.png')
    ## plot the mape graph
    plt.figure(dpi=200)
    plt.ylabel('Mean Absolute Percentage Error')
    plt.plot(history['mean_absolute_percentage_error'], c='red', label='Training')
    plt.plot(history['val_mean_absolute_percentage_error'], c='blue', label='Validation')
    plt.legend()
    plt.savefig('./output/train_single/mape.png')
    ## plot test prediction
    num_test_win = 1
    Y,Y_hat,X = [], [], []
    for x,y in win_generator.val.take(1):
        y_hat = model(x)
        for i in range(num_test_win):
            Y.append(y[i])
            Y_hat.append(y_hat[i])
            X.append(x[i])
    Y = np.array(Y).reshape(-1,len(APPS))
    Y_hat = np.array(Y_hat).reshape(-1,len(APPS))
    X = np.array(X).reshape(-1,len(APPS))
    f = plt.figure(figsize=(16,9))
    plt.rc('legend', fontsize=16)
    x_index = np.arange(0, X.shape[0])
    y_index = np.arange(X.shape[0], X.shape[0]+Y.shape[0])
    for i, app in enumerate([20,9,5,34,35,42,48,28]):
        fig = plt.subplot(4,2,i+1)
        fig.plot(x_index, X[:,i], c='black', label='Input')
        fig.plot(y_index, Y[:,i], c='red', label='Label')
        fig.plot(y_index, Y_hat[:,i], c='blue', label='Prediction')
        fig.set_xlabel('Appliance ' + APPS_DICT[app])
    lines, labels = f.axes[-1].get_legend_handles_labels()
    plt.tight_layout()
    f.legend(lines, labels, loc='upper left')
    plt.savefig('./output/train_single/prediction.png')


