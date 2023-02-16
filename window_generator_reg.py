"""
https://www.tensorflow.org/tutorials/structured_data/time_series#4_create_tfdatadatasets
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm



class WindowGenerator:
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 stride=3,
                 batch_size=32,
                 out_dims=None,
                 features_columns=None,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.features_columns = features_columns
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Sets input and output dimensions
        self.in_dims = len(train_df.columns)
        self.out_dims = out_dims if out_dims is not None else self.in_dims

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.batch_size = batch_size
        self.stride = stride


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.features_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in
                 self.features_columns],
                axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride,
            shuffle=True,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)


class BaselineGenerator:
    def __init__(self,
                 label_width,
                 label_shift,
                 dataframe,
                 stride=3,
                 batch_size=32,
                 label_columns=None):
        # Store the raw data.
        self.dataframe = dataframe

        # Work out the label column indices.
        self.label_columns = label_columns
        self.column_indices = {name: i for i, name in enumerate(dataframe.columns)}

        # Work out the window parameters.
        self.label_width = label_width
        self.label_shift = label_shift

        self.total_window_size = label_width + label_shift

        # self.input_slice = slice(0, label_width)
        # self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # self.label_start = self.total_window_size - self.label_width
        # self.labels_slice = slice(self.label_start, None)
        # self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.label_slice = slice(0, label_width)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

        self.input_start = self.total_window_size - self.label_width
        self.input_slice = slice(self.input_start, None)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.batch_size = batch_size
        self.stride = stride
        self.train = None
        self.val = None
        self.split_data(backup=label_shift)
            

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]

        if self.label_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.label_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride,
            batch_size=self.batch_size, )
        ds = ds.map(self.split_window)
        return ds


    def split_data(self, train_week=3, val_week=1, backup=0):
        start=0
        data_lenghth = self.dataframe.shape[0]
        while start < data_lenghth:
            end = start + (train_week+val_week)*144*7
            if end>=data_lenghth:
                self.train = self.train.concatenate(self.make_dataset(self.dataframe[start:]))

            if self.train==None:
                self.train = self.make_dataset(self.dataframe[start:start+train_week*144*7])
            else:
                self.train = self.train.concatenate(self.make_dataset(self.dataframe[start-backup:start+train_week*144*7]))

            if self.val==None:
                self.val = self.make_dataset(self.dataframe[end-val_week*144*7-backup:end])
            else:
                self.val = self.val.concatenate(self.make_dataset(self.dataframe[end-val_week*144*7-backup:end]))

            start = end

        self.train.cache()
        self.val.cache()


class ShuffleWindowGenerator:
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 dataframe,
                 stride=3,
                 if_shuffle=True,
                 batch_size=32,
                 out_dims=None,
                 features_columns=None,
                 label_columns=None):
        # Store the raw data.
        self.dataframe = dataframe

        # Work out the label column indices.
        self.label_columns = label_columns
        self.features_columns = features_columns
        self.column_indices = {name: i for i, name in
                               enumerate(dataframe.columns)}

        # Sets input and output dimensions
        self.in_dims = len(dataframe.columns)
        self.out_dims = out_dims if out_dims is not None else self.in_dims

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.batch_size = batch_size
        self.stride = stride
        self.if_shuffle = if_shuffle
        self.train = None
        self.val = None
        self.split_data()


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.features_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in
                 self.features_columns],
                axis=-1)

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride,
            shuffle=self.if_shuffle,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    def split_data(self, train_week=3, val_week=1):
        start=0
        data_lenghth = self.dataframe.shape[0]
        while start < data_lenghth:
            end = start + (train_week+val_week)*144*7
            if end>=data_lenghth:
                self.train = self.train.concatenate(self.make_dataset(self.dataframe[start:]))

            if self.train==None:
                self.train = self.make_dataset(self.dataframe[start:start+train_week*144*7])
            else:
                self.train = self.train.concatenate(self.make_dataset(self.dataframe[start:start+train_week*144*7]))

            if self.val==None:
                self.val = self.make_dataset(self.dataframe[end-val_week*144*7:end])
            else:
                self.val = self.val.concatenate(self.make_dataset(self.dataframe[end-val_week*144*7:end]))

            start = end

        self.train.cache()
        self.val.cache()
        if self.if_shuffle:
            self.train.shuffle(40000, reshuffle_each_iteration=True)
            self.val.shuffle(15000, reshuffle_each_iteration=True)


if __name__ == '__main__':
    train_split = 0.8
    data = np.ones((10000, 10)) * np.arange(10000).reshape(-1, 1)
    df = pd.DataFrame(data)
    n_samples = len(df)
    n_train_samples = round(n_samples * train_split)

    train_val_df = df.iloc[:n_train_samples]
    test_df = df.iloc[n_train_samples:]

    n_samples = len(train_val_df)
    n_train_samples = round(n_samples * train_split)

    train_df = train_val_df.iloc[:n_train_samples]
    val_df = train_val_df.iloc[n_train_samples:]

    fake_apps = list(range(10))

    gen = WindowGenerator(input_width=10, label_width=10, shift=10,
                          train_df=train_df, val_df=val_df, test_df=test_df,
                          apps=fake_apps,
                          label_columns=None)

    X, y = list(gen.train.take(1))[0]


