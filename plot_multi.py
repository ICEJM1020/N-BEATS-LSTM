# from loss_func import SliceMSELoss
from models import *
import tensorflow as tf
from window_generator_reg import WindowGenerator, ShuffleWindowGenerator
from loss_func import SliceMSELoss, SliceMAE, ActiveMAE
from utils import get_dataframe, APPS, APPS_DICT
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_multi import models


## build up the datasets
INPUT_STEPS = 144
OUTPUT_STEPS = 36
LSTM_UNITS = 32
BATCH_SIZE = 64
data_file_path = 'C:/data/CM003_10min.gzip'
fea_app = [20,9,5,34,35,42,48,28]
fea_ori = [20,9,5,34,35,42,48,28,'hour_cos','hour_sin','day_cos','day_sin','month_sin','month_cos']
fea_without_time = [20,9,5,34,35,42,48,28,'gap_20','lagu_20','gap_9','lagu_9',
                    'gap_5','lagu_5','gap_34','lagu_34','gap_35','lagu_35','gap_42','lagu_42',
                    'gap_48','lagu_48','gap_28','lagu_28',]
fea_gap_lagu = ['gap_20','lagu_20','gap_9','lagu_9','gap_5','lagu_5','gap_34','lagu_34','gap_35','lagu_35',
                'gap_42','lagu_42', 'gap_48','lagu_48','gap_28','lagu_28',]
fea_norm = ['norm_20', 'norm_9','norm_5','norm_34','norm_35','norm_42','norm_48','norm_28']

FEA = fea_norm
EPOCHS = 20
NUM_SLICES = 6
LEARNING_RATE = 1e-6

df_std=get_dataframe(data_file_path, scale=2, shift=-1, normstd=5)
models['baseline_1'].generate_data(df_std, stride=3)
models['baseline_2'].generate_data(df_std, stride=3)
models['baseline_7'].generate_data(df_std, stride=3)
## dataset
win_generator = ShuffleWindowGenerator(input_width=INPUT_STEPS, label_width=OUTPUT_STEPS, shift=OUTPUT_STEPS, 
                                dataframe=df_std, stride=3, if_shuffle=True,
                                batch_size=BATCH_SIZE, features_columns=FEA, label_columns=FEA)

# metrics=tf.keras.losses.MeanSquaredError()
metrics_list={
    'SliceMSE': SliceMSELoss(label_length=OUTPUT_STEPS, label_dims=len(APPS), num_slices=NUM_SLICES),
    'MeanAbsoluteError':tf.keras.metrics.MeanAbsoluteError(),
    'SlicedMAE_6':SliceMAE(label_length=OUTPUT_STEPS, label_dims=len(APPS), num_slices=6),
    'SlicedMAE_4':SliceMAE(label_length=OUTPUT_STEPS, label_dims=len(APPS), num_slices=9),
    'SlicedMAE_3':SliceMAE(label_length=OUTPUT_STEPS, label_dims=len(APPS), num_slices=12),
    # 'ActiveMAE':ActiveMAE(),
    'MAPE': tf.keras.metrics.MeanAbsolutePercentageError()
}

model_preds = {}
for model_name in models:
    if model_name=='nbeats': continue
    model = models[model_name]
    model_preds[model_name] = {}
    if model_name.startswith('baseline'): 
        model_preds[model_name]['train']=model.predict(type='training')
        model_preds[model_name]['val']=model.predict(type='validation')
    else:
        model.build((BATCH_SIZE, INPUT_STEPS, len(FEA)))
        model.load_weights('models/compare/comp_{}.h5'.format(model_name))
        for data_type in ['train', 'val']:
            if data_type=='train':
                data = win_generator.train
            else:
                data = win_generator.val
            # y_true, y_pred = None, None
            y_true, y_pred = [], []
            for (x,y) in data:
                if model_name.startswith('nbeats'):
                    y_hat,_ = model(x, training=True)
                else:
                    y_hat = model(x, training=True)
                # if y_true is None:
                #     y_pred = y_hat.numpy()
                #     y_true = y.numpy()
                # else:
                #     # print(y_pred.shape, y_hat.shape, y_true.shape, y.shape)
                #     y_pred = np.concatenate((y_pred, y_hat.numpy()), axis=0)
                #     y_true = np.concatenate((y_true, y.numpy()), axis=0)
                y_pred.append(y_hat.numpy())
                y_true.append(y.numpy())
            model_preds[model_name][data_type]=[y_pred, y_true]

for data_type in ['Training', 'Validation']:
    for metrics_name in metrics_list:
        print('{}_{} drawing'.format(data_type, metrics_name))
        metrics = metrics_list[metrics_name]
        all_metrics = {}
        labels = []
        for model_name in models:
            if model_name=='nbeats': continue
            labels.append(model_name)
            if data_type=='Training':
                y_pred, y_true = model_preds[model_name]['train'][0], model_preds[model_name]['train'][1]
            else:
                y_pred, y_true = model_preds[model_name]['val'][0], model_preds[model_name]['val'][1]

            metrics_value = []
            for pred, true in zip(y_pred, y_true):
                metrics_value.append(metrics(pred, true).numpy())
            metrics_value = np.array(metrics_value)
            metrics_value.sort()
            all_metrics[model_name] = metrics_value
        
        pd.DataFrame({ key:pd.Series(value) for key, value in all_metrics.items() })\
            .to_csv('output/violin/{}_{}.csv'.format(data_type, metrics_name).lower())
        plt.figure(figsize=(16,9))
        plt.xticks(ticks=np.arange(1, len(labels)+1), labels=labels, rotation=45)
        plt.xlabel(metrics_name)
        plt.violinplot(all_metrics.values())
        plt.savefig('output/violin/violin_{}_{}.png'.format(data_type, metrics_name))

