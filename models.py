import tensorflow as tf
from loss_func import SliceMSELoss, FocalSliceLoss
from window_generator_reg import BaselineGenerator
import time
import datetime
import numpy as np


MAX_EPOCHS = 999
BATCH_SIZE = 64
LOGGING = False


class BaseModel(tf.keras.Model):
    def __init__(self, out_steps, out_dims, num_slices=9):
        super().__init__()
        self.out_steps = out_steps
        self.out_dims = out_dims
        self.loss = SliceMSELoss(label_length=out_steps, label_dims=out_dims,num_slices=num_slices)
        # self.loss = FocalSliceLoss(label_length=out_steps, label_dims=out_dims, num_slices=num_slices)

    def append_dict(self, old, new):
        if old == None:
            old = {}
            for key in new.keys():
                old[key] = new[key]
        else:
            for key in old.keys():
                old[key].extend(new[key])
        return old

    def custom_train(self, data, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=1e-6):
        history = None
        metrics = [tf.metrics.MeanAbsoluteError(), tf.metrics.MeanAbsolutePercentageError()]
        
        # initial_learning_rate = lr * 5
        # final_learning_rate = lr / 5
        # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
        # steps_per_epoch = data.train.cardinality().numpy()
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #                 initial_learning_rate=initial_learning_rate,
        #                 decay_steps=steps_per_epoch,
        #                 decay_rate=learning_rate_decay_factor,
        #                 staircase=True)
        # optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=metrics,
        )
        callbacks=[]
        if LOGGING:
            log_dir = "C://logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-5,patience=10,mode='min')
        callbacks.append(early_stopping)
        _history = self.fit(
            data.train,
            epochs=epochs,
            validation_data=data.val,
            batch_size=batch_size,
            callbacks=callbacks
        )
        history = self.append_dict(history, _history.history)
        
        return history


# Single Layer LSTM
class SL_LSTM(BaseModel):
    def __init__(self, units=32, if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_cnn = if_cnn
        if if_cnn:
            self.cnn = tf.keras.layers.Conv1D(filters=units,kernel_size=5,activation='relu',padding='same')

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=units, return_sequences=False, dropout=0.2),
            tf.keras.layers.Dense(self.out_steps*self.out_dims),
            tf.keras.layers.Reshape([self.out_steps, self.out_dims])
        ])

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        x = self.model(x, *args, **kwargs)
        return x


# Multi Layer(s) LSTM
class ML_LSTM(BaseModel):
    def __init__(self, num_layers=3, units=32, if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_cnn = if_cnn
        if if_cnn:
            self.cnn = tf.keras.layers.Conv1D(filters=units,kernel_size=5,activation='relu',padding='same')

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=units, return_sequences=True, dropout=0.2) for i in range(num_layers-1)
        ])
        self.model.add(tf.keras.layers.LSTM(
            units=units, return_sequences=False, dropout=0.2))
        self.model.add(tf.keras.layers.Dense(self.out_steps*self.out_dims))
        self.model.add(tf.keras.layers.Reshape(
            [self.out_steps, self.out_dims]))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        x = self.model(x, *args, **kwargs)
        return x


# seq2seq sl lstm
class SL_LSTM_S2S(BaseModel):
    def __init__(self, units=32, if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_cnn = if_cnn
        if if_cnn:
            self.cnn = tf.keras.layers.Conv1D(filters=units,kernel_size=5,activation='relu',padding='same')
        self.encoder = tf.keras.layers.LSTM(units=units, return_state=True, dropout=0.2)

        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # self.decoder = tf.keras.layers.RNN(units=units, return_state=True)

        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.out_dims))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        ss, *state = self.encoder(x, *args, **kwargs)

        output = []
        for n in range(self.out_steps):
            x = ss
            x, state = self.lstm_cell(x, states=state, *args, **kwargs)
            ss = x
            output.append(ss)

        x = tf.stack(output)
        x = tf.transpose(x, [1, 0, 2])
        x = self.dense(x, *args, **kwargs)
        return x


# seq2seq ml lstm
class ML_LSTM_S2S(BaseModel):
    def __init__(self, num_layers=3, units=32, if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_cnn = if_cnn
        if if_cnn:
            self.cnn = tf.keras.layers.Conv1D(filters=units,kernel_size=5,activation='relu',padding='same')

        self.encoder_1 = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=units, return_sequences=True, dropout=0.2) for i in range(num_layers-1)
        ])
        self.encoder_2 = tf.keras.layers.LSTM(units=units, return_state=True, dropout=0.2)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.out_dims))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        ss = self.encoder_1(x, *args, **kwargs)
        ss, *state = self.encoder_2(ss, *args, **kwargs)

        output = []
        for n in range(self.out_steps):
            x = ss
            x, state = self.lstm_cell(x, states=state, *args, **kwargs)
            ss = x
            output.append(ss)

        x = tf.stack(output)
        x = tf.transpose(x, [1, 0, 2])
        x = self.dense(x, *args, **kwargs)
        return x


# seq2seq cnn + sl lstm + attention
class Att_SL_S2S(BaseModel):
    def __init__(self, units=32, attention_type='additive', if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_type = attention_type
        self.if_cnn = if_cnn

        self.cnn = tf.keras.layers.Conv1D(filters=units, kernel_size=5, padding='same', activation='relu')
        self.encoder = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dropout=0.2)

        if attention_type == 'additive':
            self.attention = tf.keras.layers.Attention()
        elif attention_type == 'multihead':
            self.attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=2)

        self.decoder = tf.keras.layers.LSTMCell(units)
        # self.decoder = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dropout=0.2)

        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.out_dims))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        enc_outputs, state_h, state_c = self.encoder(x, *args, **kwargs)

        state = [state_h, state_c]
        dec_outputs = []
        ss = state_h
        for n in range(self.out_steps):
            x = ss
            x, state = self.decoder(x, states=state, *args, **kwargs)
            ss = x
            dec_outputs.append(ss)

        dec_outputs = tf.stack(dec_outputs)
        dec_outputs = tf.transpose(dec_outputs, [1, 0, 2])

        if self.attention_type == 'additive':
            x = self.attention([dec_outputs, enc_outputs], *args, **kwargs)
        elif self.attention_type == 'multihead':
            x = self.attention(dec_outputs, enc_outputs, *args, **kwargs)

        x = self.dense(x, *args, **kwargs)
        return x


# seq2seq cnn + ml lstm + attention
class Att_ML_S2S(BaseModel):
    def __init__(self, num_layers=3, units=32, attention_type='additive', if_cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.attention_type = attention_type
        self.if_cnn = if_cnn

        self.cnn = tf.keras.layers.Conv1D(filters=units, kernel_size=5, padding='same', activation='relu')
        self.encoder_1 = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=units, return_sequences=True, dropout=0.2) for i in range(num_layers-1)
        ])
        self.encoder_2 = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dropout=0.2)

        if attention_type == 'additive':
            self.attention = tf.keras.layers.Attention()
        elif attention_type == 'multihead':
            self.attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=2)

        self.decoder = tf.keras.layers.LSTMCell(units)
        # self.decoder_2 = tf.keras.Sequential([
        #     tf.keras.layers.LSTM(units=units, return_sequences=True, dropout=0.2) for i in range(num_layers-1)
        # ])

        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.out_dims))

    def call(self, inputs, *args, **kwargs):
        x = inputs

        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        x = self.encoder_1(x, *args, **kwargs)
        enc_outputs, state_h, state_c = self.encoder_2(x, *args, **kwargs)

        state = [state_h, state_c]
        dec_outputs = []
        ss = state_h
        for n in range(self.out_steps):
            x = ss
            x, state = self.decoder(x, states=state, *args, **kwargs)
            ss = x
            dec_outputs.append(ss)

        dec_outputs = tf.stack(dec_outputs)
        dec_outputs = tf.transpose(dec_outputs, [1, 0, 2])

        # dec_outputs = self.decoder_2(dec_outputs)

        if self.attention_type == 'additive':
            x = self.attention([dec_outputs, enc_outputs], *args, **kwargs)
        elif self.attention_type == 'multihead':
            x = self.attention(dec_outputs, enc_outputs, *args, **kwargs)

        x = self.dense(x, *args, **kwargs)
        return x


class NBEATS_block(tf.keras.Model):
    def __init__(self, in_steps, out_steps, in_dims, out_dims, layer_width=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.in_dims = in_dims
        self.out_dims = out_dims

        self.dense = tf.keras.Sequential([tf.keras.layers.Dense(units=layer_width, activation='relu') for i in range(4)])
        self.dense.add(tf.keras.layers.Dense(units=layer_width, activation='linear'))

        self.forecast = tf.keras.layers.Dense(units=out_steps*out_dims)
        self.backcast = tf.keras.layers.Dense(units=in_steps*in_dims)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = tf.reshape(x, (-1, self.in_steps*self.in_dims))
        x = self.dense(x, *args, **kwargs)
        forecast = self.forecast(x, *args, **kwargs)
        forecast = tf.reshape(forecast, (-1, self.out_steps, self.out_dims))
        backcast = self.backcast(x, *args, **kwargs)
        backcast = tf.reshape(backcast, (-1, self.in_steps, self.in_dims))
        return forecast, backcast


class NBEATS_LSTM_block(tf.keras.Model):
    def __init__(self, out_steps, in_dims, out_dims, units=32, if_cnn=True, if_attention=False, attention_type='additive', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_steps = out_steps
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.if_attention = if_attention
        self.attention_type = attention_type
        self.if_cnn = if_cnn
        if if_cnn:
            self.cnn=tf.keras.layers.Conv1D(filters=units, kernel_size=5, padding='same', activation='relu')

        self.encoder = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True, dropout=0.2)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)

        if if_attention:
            if attention_type == 'additive':
                self.attention = tf.keras.layers.Attention()
            elif attention_type == 'multihead':
                self.attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=2)

        self.forecast = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(out_dims))
        self.backcast = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(in_dims))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.if_cnn:
            x = self.cnn(x, *args, **kwargs)
        backcast, forecast, state = self.encoder(x, *args, **kwargs)
        state = [forecast, state]

        output = []
        for n in range(self.out_steps):
            x = forecast
            x, state = self.lstm_cell(x, states=state, *args, **kwargs)
            forecast = x
            output.append(forecast)

        x = tf.stack(output)
        forecast = tf.transpose(x, [1, 0, 2])
        if self.if_attention:
            if self.attention_type == 'additive':
                forecast = self.attention([forecast, backcast], *args, **kwargs)
            elif self.attention_type == 'multihead':
                forecast = self.attention(forecast, backcast, *args, **kwargs)

        backcast = self.backcast(backcast, *args, **kwargs)
        forecast = self.forecast(forecast, *args, **kwargs)

        return forecast, backcast


class NBEATS_BASE(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if LOGGING:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'C://logs/' + current_time + '/train'
            test_log_dir = 'C://logs/' + current_time + '/validation'
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def call(self, inputs, *args, **kwargs):
        res = []
        mini_fore = []
        back = inputs
        forecast, backcast = self.stacks[0](back, *args, **kwargs)
        fore = forecast
        back = tf.math.subtract(back, backcast)
        res.append(back)
        mini_fore.append(fore)
        for i in range(1, self.num_blocks):
            forecast, backcast = self.stacks[i](back, *args, **kwargs)
            fore = tf.math.add(fore, forecast)
            back = tf.math.subtract(back, backcast)
            res.append(back)
            mini_fore.append(forecast)
        return fore, back, res, mini_fore

    @tf.function
    def _update(self, x_train, y_train):
        with tf.GradientTape() as tape:
            fore, back = self(x_train, training=True)
            loss_1 = self.loss_fore(y_train, fore)
            loss_2 = self.loss_back(x_train, back)
            _loss = [loss_1, loss_2]
        _grads = tape.gradient(_loss, self.trainable_variables)
        self.OPT.apply_gradients(zip(_grads, self.trainable_variables))
        return fore

    def epoch_train(self, data):
        # Training
        for (x_train, y_train) in data:
            fore = self._update(x_train, y_train)
            self.AVE.update_state(self.loss_fore(y_train, fore))
            self.MET_1.update_state(y_train, fore)
            self.MET_2.update_state(y_train, fore)

        feedback = [self.AVE.result().numpy(), self.MET_1.result().numpy(), self.MET_2.result().numpy()]
        self.AVE.reset_state()
        self.MET_1.reset_state()
        self.MET_2.reset_state()
        return feedback

    def epoch_test(self, data):
        for (x_test, y_test) in data:
            fore, _ = self(x_test, training=False)
            loss_1 = self.loss_fore(y_test, fore)
            self.AVE.update_state(loss_1)
            self.MET_1.update_state(y_test, fore)
            self.MET_2.update_state(y_test, fore)
        feedback = [self.AVE.result().numpy(), self.MET_1.result().numpy(), self.MET_2.result().numpy()]
        self.AVE.reset_state()
        self.MET_1.reset_state()
        self.MET_2.reset_state()
        return feedback

    def custom_train(self, data_generator, epochs, lr=0.001):
        # initial_learning_rate = 0.01
        # final_learning_rate = 0.0005
        # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
        # steps_per_epoch = data_generator.train.cardinality().numpy()
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #                 initial_learning_rate=initial_learning_rate,
        #                 decay_steps=steps_per_epoch,
        #                 decay_rate=learning_rate_decay_factor,
        #                 staircase=True)
        # self.OPT = tf.optimizers.Adam(learning_rate=lr_schedule)

        history = {
            'loss': [],
            'mean_absolute_error': [],
            'val_loss': [],
            'val_mean_absolute_error': [],
            'mean_absolute_percentage_error': [],
            'val_mean_absolute_percentage_error': []
        }

        for epoch in range(epochs):
            start = time.time()
            train_result = self.epoch_train(data_generator.train)
            if LOGGING:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('epoch_loss', train_result[0], step=epoch)
                    tf.summary.scalar('epoch_mean_absolute_error',train_result[1], step=epoch)
                    tf.summary.scalar('epoch_mean_absolute_percentage_error', train_result[2], step=epoch)
            history['loss'].append(train_result[0])
            history['mean_absolute_error'].append(train_result[1])
            history['mean_absolute_percentage_error'].append(train_result[2])

            test_result = self.epoch_test(data_generator.val)
            if LOGGING:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('epoch_loss', test_result[0], step=epoch)
                    tf.summary.scalar('epoch_mean_absolute_error',test_result[1], step=epoch)
                    tf.summary.scalar('epoch_mean_absolute_percentage_error', test_result[2], step=epoch)
            history['val_loss'].append(test_result[0])
            history['val_mean_absolute_error'].append(test_result[1])
            history['val_mean_absolute_percentage_error'].append(test_result[2])

            print("[{:.5f}s,Epoch {}: loss:{:.5f}, mean_absolute_error:{:.5f}, mean_absolute_percentage_error:{:.5f}, val_loss:{:.5f}, val_mean_absolute_error,{:.5f}, val_mean_absolute_percentage_error:{:.5f}]"
                  .format(time.time()-start, epoch+1, history['loss'][-1], history['mean_absolute_error'][-1], history['mean_absolute_percentage_error'][-1],
                          history['val_loss'][-1], history['val_mean_absolute_error'][-1], history['val_mean_absolute_percentage_error'][-1]))
        return history

    def evaluate(self, data):
        test_result = self.epoch_test(data)
        return test_result


class NBEATS(NBEATS_BASE):
    def __init__(self, in_steps, out_steps, in_dims, out_dims, layer_width=16, num_blocks=4, num_slices=9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.num_slices = num_slices
        self.loss_fore = SliceMSELoss(label_length=out_steps, label_dims=out_dims, num_slices=num_slices)
        self.loss_back = SliceMSELoss(label_length=in_steps, label_dims=in_dims, num_slices=num_slices)

        self.stacks = [
            NBEATS_block(in_steps, out_steps, in_dims, out_dims, layer_width=layer_width) for i in range(num_blocks)
        ]

        self.MAX_EPOCHS = 250
        self.AVE = tf.keras.metrics.Mean()
        self.MET_1 = tf.metrics.MeanAbsoluteError()
        self.MET_2 = tf.metrics.MeanAbsolutePercentageError()
        self.OPT = tf.optimizers.Adam()


class NBEATS_LSTM(NBEATS_BASE):
    def __init__(self, in_steps, out_steps, in_dims, out_dims, units=32, num_blocks=2, num_slices=9, if_cnn=True, if_att=False, att_type='additive', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.num_slices = num_slices
        self.loss_fore = SliceMSELoss(label_length=out_steps, label_dims=out_dims, num_slices=num_slices)
        self.loss_back = SliceMSELoss(label_length=in_steps, label_dims=in_dims, num_slices=num_slices)
        self.stacks = [
            NBEATS_LSTM_block(out_steps, in_dims, out_dims, units=units, if_cnn=if_cnn,if_attention=if_att,attention_type=att_type) for i in range(num_blocks)
        ]

        self.MAX_EPOCHS = 250
        self.AVE = tf.keras.metrics.Mean()
        self.MET_1 = tf.metrics.MeanAbsoluteError()
        self.MET_2 = tf.metrics.MeanAbsolutePercentageError()
        self.OPT = tf.optimizers.Adam()


# other day pattern baseline
# label_shift: the shift of the label
#        6 time-steps -> 1 hours
#        the next day shift: 24*6=144 time-step
#        the next week shift: 7*24*6=1008
class Baseline():
    def __init__(self, out_steps, out_dims, label_shift, label_columns, batch_size=256, num_slices=9):
        self.out_steps=out_steps
        self.label_shift=label_shift
        self.batch_size=batch_size
        self.label_columns=label_columns
        self.loss = SliceMSELoss(label_length=out_steps, label_dims=out_dims, num_slices=num_slices)
        self.out_dims = out_dims
        self.metrics_1 = tf.metrics.MeanAbsoluteError()
        self.metrics_2 = tf.metrics.MeanAbsolutePercentageError()
        self.avg = tf.keras.metrics.Mean()


    def generate_data(self, data, stride=3):
        self.data_generator = BaselineGenerator(
            label_width=self.out_steps,
            label_shift=self.label_shift,
            dataframe=data,
            stride=stride,
            batch_size=self.batch_size,
            label_columns=self.label_columns)


    def evaluate(self, type='training'):
        if type=='training':
            data = self.data_generator.train
        elif type=='validation':
            data = self.data_generator.val
        else:
            raise Exception('Wrong evaluation type: {}. It should be \'training\' or \'validation\''.format(type))
        self.avg.reset_state()
        self.metrics_1.reset_state()
        self.metrics_2.reset_state()
        for (x_test, y_test) in data:
            loss = self.loss(x_test, y_test)
            self.avg.update_state(loss)
            self.metrics_1.update_state(x_test, y_test)
            self.metrics_2.update_state(x_test, y_test)
        feedback = [self.avg.result().numpy(), self.metrics_1.result().numpy(), self.metrics_2.result().numpy()]
        return feedback
    

    def predict(self, type):
        if type=='training':
            data = self.data_generator.train
        elif type=='validation':
            data = self.data_generator.val
        y_true, y_pred = [], []
        for (x_test, y_test) in data:
            # if y_true is None:
            #     y_pred = y_test.numpy()
            #     y_true = x_test.numpy()
            # else:
            #     y_pred = np.concatenate((y_pred, y_test.numpy()), axis=0)
            #     y_true = np.concatenate((y_true, x_test.numpy()), axis=0)
            y_true.append(x_test)
            y_pred.append(y_test)
        
        return [y_pred, y_true]


if __name__ == '__main__':
    print('load model.py')
