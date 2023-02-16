"""
https://www.tensorflow.org/tutorials/structured_data/time_series#4_create_tfdatadatasets
"""
import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator:
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 apps,
                 out_dims=None,
                 label_columns=None,
                 thresholds=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None: 
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.apps = apps
        self.app_indices = [self.column_indices[app] for app in self.apps]

        # Sets input and output dimensions
        self.in_dims = len(train_df.columns)
        self.out_dims = out_dims if out_dims is not None else self.in_dims

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.thresholds = thresholds

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        # features = tf.cast(features, tf.int32)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        # print(labels)
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        labels = tf.gather(labels, self.app_indices, axis=-1)
        if self.thresholds is not None:
                labels = tf.where(labels > self.thresholds, 1, 0)
                # determines whether any appliance is on at each time step
                # labels = tf.reduce_max(labels, axis=2)
                # labels = tf.reshape(labels, [-1, self.label_width, len(APPS)])

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        labels = tf.reshape(labels, [-1, self.label_width*self.out_dims])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

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