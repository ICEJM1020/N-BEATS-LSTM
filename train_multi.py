import tensorflow as tf
import gzip
import pickle
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
EPOCHS = 150
NUM_SLICES = 6
LEARNING_RATE = 1e-6

#  model
models = {
    'baseline_1': Baseline(out_steps=OUTPUT_STEPS, out_dims=len(APPS),label_shift=144, label_columns=FEA,num_slices=NUM_SLICES),
    'baseline_2': Baseline(out_steps=OUTPUT_STEPS, out_dims=len(APPS),label_shift=288, label_columns=FEA,num_slices=NUM_SLICES),
    'baseline_7': Baseline(out_steps=OUTPUT_STEPS, out_dims=len(APPS),label_shift=1008, label_columns=FEA,num_slices=NUM_SLICES),
    'sl_lstm': SL_LSTM(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=False, num_slices=NUM_SLICES),
    'ml_lstm': ML_LSTM(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=False, num_slices=NUM_SLICES),
    'sl_lstm_s2s': SL_LSTM_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=False, num_slices=NUM_SLICES),
    'ml_lstm_s2s': ML_LSTM_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=False, num_slices=NUM_SLICES),
    'cnn_sl_lstm_s2s': SL_LSTM_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=True, num_slices=NUM_SLICES),
    'cnn_ml_lstm_s2s': ML_LSTM_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), if_cnn=True, num_slices=NUM_SLICES),
    'addictive_sl': Att_SL_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), attention_type='additive', num_slices=NUM_SLICES),
    'addictive_ml': Att_ML_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), attention_type='additive', num_slices=NUM_SLICES),
    'multihead_sl': Att_SL_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), attention_type='multihead', num_slices=NUM_SLICES),
    'multihead_ml': Att_ML_S2S(units=LSTM_UNITS, out_steps=OUTPUT_STEPS, out_dims=len(APPS), attention_type='multihead', num_slices=NUM_SLICES),
    'nbeats':NBEATS(INPUT_STEPS, OUTPUT_STEPS, len(FEA), len(APPS), layer_width=LSTM_UNITS, num_blocks=4, num_slices=NUM_SLICES),
    'nbeats_lstm': NBEATS_LSTM(INPUT_STEPS, OUTPUT_STEPS, len(FEA), len(APPS), units=LSTM_UNITS, num_blocks=2, num_slices=NUM_SLICES, if_cnn=True, if_att=False),
    'nbeats_add_lstm': NBEATS_LSTM(INPUT_STEPS, OUTPUT_STEPS, len(FEA), len(APPS), units=LSTM_UNITS, num_blocks=2, num_slices=NUM_SLICES, if_cnn=True, if_att=True, att_type='additive'),
    'nbeats_mh_lstm': NBEATS_LSTM(INPUT_STEPS, OUTPUT_STEPS, len(FEA), len(APPS), units=LSTM_UNITS, num_blocks=2, num_slices=NUM_SLICES, if_cnn=True, if_att=True, att_type='multihead')
}

if __name__ == '__main__':

    df_std=get_dataframe(data_file_path, scale=2, shift=-1, normstd=5)
    models['baseline_1'].generate_data(df_std)
    models['baseline_2'].generate_data(df_std)
    models['baseline_7'].generate_data(df_std)
    ## dataset
    # win_generator = WindowGenerator(input_width=INPUT_STEPS, label_width=OUTPUT_STEPS, shift=OUTPUT_STEPS, 
    #                                 train_df=train_df, test_df=None, val_df=val_df,
    #                                 batch_size=BATCH_SIZE, features_columns=FEA, label_columns=APPS)
    win_generator = ShuffleWindowGenerator(input_width=INPUT_STEPS, label_width=OUTPUT_STEPS, shift=OUTPUT_STEPS, 
                                    dataframe=df_std,
                                    batch_size=BATCH_SIZE, features_columns=FEA, label_columns=FEA)

    ## test the datasets
    for test_inputs, test_labels in win_generator.train.take(1):
        print('Inputs Shape', test_inputs.shape)
        print('Labels Shape', test_labels.shape)
    print('Number of Batches in Training Set:', win_generator.train.cardinality().numpy())
    print('Number of Batches in Validation Set:', win_generator.val.cardinality().numpy())

    ## training

    performance = {}
    # performance = pd.read_csv('output/compare/compare.csv').to_dict()
    for key in models:
        print('============{}=========='.format(key))
        model = models[key]
        ## train
        if key.startswith('nbeats'):
            history = model.custom_train(data_generator=win_generator, epochs=EPOCHS, lr=LEARNING_RATE*0.1)
            performance[key] = [model.evaluate(win_generator.train), model.evaluate(win_generator.val)]
            pd.DataFrame(history).to_csv('output/compare/his_{}.csv'.format(key), index=None)
            model.save_weights('models/compare/comp_{}.h5'.format(key))
        elif key.startswith('baseline'):
            performance[key] = [model.evaluate(type='training'), model.evaluate(type='validation')]
        else:
            history = model.custom_train(
                data=win_generator,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE
                )
            performance[key] = [model.evaluate(win_generator.train), model.evaluate(win_generator.val)]
            pd.DataFrame(history).to_csv('output/compare/his_{}.csv'.format(key), index=None)
            model.save_weights('models/compare/comp_{}.h5'.format(key))


    pd.DataFrame(performance).to_csv('output/compare/compare.csv', index=None)
    ## plot
    x = np.arange(len(performance))
    width = 0.3
    train_mae, val_mae, train_loss, val_loss, train_mape, val_mape = [], [], [], [], [], []
    # for item in performance:
    #     train_mae.append(float(performance[item][0].strip('][').split(', ')[1]))
    #     val_mae.append(float(performance[item][1].strip('][').split(', ')[1]))
    #     train_loss.append(float(performance[item][0].strip('][').split(', ')[0]))
    #     val_loss.append(float(performance[item][1].strip('][').split(', ')[0]))
    #     train_mape.append(float(performance[item][0].strip('][').split(', ')[2]))
    #     val_mape.append(float(performance[item][1].strip('][').split(', ')[2]))
    for item in performance:
        train_mae.append(float(performance[item][0][1]))
        val_mae.append(float(performance[item][1][1]))
        train_loss.append(float(performance[item][0][0]))
        val_loss.append(float(performance[item][1][0]))
        train_mape.append(float(performance[item][0][2]))
        val_mape.append(float(performance[item][1][2]))


    plt.figure(figsize=(16,9))
    plt.rc('legend', fontsize=16)
    plt.rc('axes', labelsize=16)
    plt.ylabel('mean_absolute_error')
    plt.bar(x - 0.17, train_mae, width, label='Training')
    plt.bar(x + 0.17, val_mae, width, label='Validation')
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    plt.ylim(0,0.025)

    _ = plt.legend()
    plt.savefig('./output/compare/compare_mae.png')

    plt.figure(figsize=(16,9))
    plt.rc('legend', fontsize=16)
    plt.rc('axes', labelsize=16)
    plt.ylabel('loss')
    plt.bar(x - 0.17, train_loss, width, label='Training')
    plt.bar(x + 0.17, val_loss, width, label='Validation')
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    _ = plt.legend()
    plt.savefig('./output/compare/compare_loss.png')

    plt.figure(figsize=(16,9))
    plt.rc('legend', fontsize=16)
    plt.rc('axes', labelsize=16)
    plt.ylabel('mape')
    plt.bar(x - 0.17, train_mape, width, label='Training')
    plt.bar(x + 0.17, val_mape, width, label='Validation')
    plt.ylim(0,7)
    plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
    _ = plt.legend()
    plt.savefig('./output/compare/compare_mape.png')





