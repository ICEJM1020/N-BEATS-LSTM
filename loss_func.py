from scipy.stats import norm
import tensorflow as tf
import numpy as np

class NormalLoss(tf.keras.losses.Loss):
    def __init__(self, label_length, time_scope=7):
        super().__init__()
        self.time_scope = time_scope
        self.half = int(time_scope/2)
        self.mid = self.half
        self._norm = self._normal_probability()
        self.y_true_v = None
        self.label_length = label_length
        

    def _normal_probability(self):
        timestamp = np.linspace(-3, 3, self.time_scope)
        _norm_h = norm.cdf(timestamp[:self.mid+1])
        _norm = [_norm_h[0]]
        for i in range(1, self.time_scope):
            if i<self.mid:
                _norm.append(_norm_h[i]-_norm_h[i-1])
            elif i==self.mid:
                _norm.append((_norm_h[i]-_norm_h[i-1])*2)
            else:
                _norm.append(_norm[self.time_scope-i-1])
        return _norm

    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
        _y_min = tf.reduce_mean(y_true, axis=1, keepdims=True)
        y_min = tf.stack([_y_min for i in range(self.label_length)], axis=1)
        y_min = tf.squeeze(y_min)
        indices = tf.where(tf.math.subtract(y_true, y_mean)>0)

        temp_norm = tf.cast(self._norm, dtype=y_pred.dtype)
        length = self.label_length
        
        if self.y_true_v is None:
            self.y_true_v = tf.Variable(y_true, trainable=False, name='y_true_v_0')
        self.y_true_v.assign(y_min)
    
        for index in indices:
            _index = tf.cast(index, dtype=tf.int32)
            timestamp = _index[1]
            start = 0 if timestamp-self.half<0 else timestamp-self.half
            end = length if timestamp+self.half+1>length else timestamp+self.half+1
            multiplier = temp_norm[int(self.mid-(timestamp-start)) : int(self.mid + (end-timestamp))]
            self.y_true_v[_index[0],start:end,_index[2]].assign(\
                tf.math.add(self.y_true_v[_index[0],start:end,_index[2]],\
                    tf.math.multiply(self.y_true_v[_index[0],_index[1],_index[2]], multiplier)))
        y_norm = tf.cast(self.y_true_v, dtype=y_pred.dtype)
        return tf.reduce_mean(tf.math.squared_difference(y_norm, y_pred), axis=-1)


class SliceMSELoss(tf.keras.losses.Loss):
    def __init__(self, label_length, label_dims, num_slices=6):
        super().__init__()
        self.num_slices = num_slices
        self.label_length = label_length
        self.label_dims = label_dims
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        # self.mse_loss = tf.keras.losses.MeanAbsolutePercentageError()

        if (label_length%num_slices)==0:
            self.slice_size = tf.cast(label_length/num_slices, dtype=tf.int32)
        else:
            raise ValueError('Cannot cut the data (length:{}) into {} slices (remainer: {})'\
                .format(label_length, num_slices, label_length%num_slices))
        self.group_indices = None

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        
        if self.label_length==self.num_slices:
            _loss = self.mse_loss(y_true, y_pred)
        else:
            y_true = tf.reshape(y_true, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_true = tf.math.reduce_mean(y_true, axis=2, keepdims=False)
            y_pred = tf.reshape(y_pred, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_pred = tf.math.reduce_mean(y_pred, axis=2, keepdims=False)
            _loss = self.mse_loss(_sum_list_pred, _sum_list_true)
            # _loss = tf.sqrt(tf.reduce_mean(tf.math.square(_sum_list_pred - _sum_list_true)))
        return _loss


class FocalSliceLoss(tf.keras.losses.Loss):
    def __init__(self, label_length, label_dims, threshold=0.8, gamma=2, epsilon=1e-8, num_slices=6):
        super().__init__()
        self.num_slices = num_slices
        self.label_length = label_length
        self.label_dims = label_dims
        self.th = threshold
        self.gamma = gamma
        self.epsilon = epsilon

        if (label_length%num_slices)==0:
            self.slice_size = tf.cast(label_length/num_slices, dtype=tf.int32)
        else:
            raise ValueError('Cannot cut the data (length:{}) into {} slices (remainer: {})'\
                .format(label_length, num_slices, label_length%num_slices))
        self.group_indices = None

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        
        if self.label_length==self.num_slices:
            abs_error = tf.math.abs(y_pred - y_true)
            scaled_abs_error = tf.divide(abs_error-tf.reduce_min(abs_error, axis=1, keepdims=True), \
                tf.subtract(tf.reduce_max(abs_error, axis=1, keepdims=True), tf.reduce_min(abs_error, axis=1, keepdims=True)) + self.epsilon)
            _weight = tf.math.exp(self.gamma * (scaled_abs_error - self.th))
            _loss = tf.reduce_mean(tf.multiply(_weight, tf.math.square(y_pred - y_true)), axis=-1)
        else:
            y_true = tf.reshape(y_true, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_true = tf.math.reduce_mean(y_true, axis=2, keepdims=False)
            y_pred = tf.reshape(y_pred, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_pred = tf.math.reduce_mean(y_pred, axis=2, keepdims=False)
            abs_error = tf.math.abs(_sum_list_pred - _sum_list_true)
            scaled_abs_error = tf.divide(abs_error-tf.reduce_min(abs_error, axis=1, keepdims=True), \
                tf.subtract(tf.reduce_max(abs_error, axis=1, keepdims=True), tf.reduce_min(abs_error, axis=1, keepdims=True)) + self.epsilon)
            _weight = tf.math.exp(self.gamma * (scaled_abs_error - self.th))
            _loss = tf.reduce_mean(tf.multiply(_weight, tf.math.square(_sum_list_pred - _sum_list_true)), axis=-1)            
        return _loss


class SliceMAE(tf.keras.losses.Loss):
    def __init__(self, label_length, label_dims, num_slices=6):
        super().__init__()
        self.num_slices = num_slices
        self.label_length = label_length
        self.label_dims = label_dims
        self.mae_loss = tf.keras.metrics.MeanAbsoluteError()

        if (label_length%num_slices)==0:
            self.slice_size = tf.cast(label_length/num_slices, dtype=tf.int32)
        else:
            raise ValueError('Cannot cut the data (length:{}) into {} slices (remainer: {})'\
                .format(label_length, num_slices, label_length%num_slices))
        self.group_indices = None


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        
        if self.label_length==self.num_slices:
            _loss = self.mse_loss(y_true, y_pred)
        else:
            y_true = tf.reshape(y_true, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_true = tf.math.reduce_sum(y_true, axis=2, keepdims=False)
            y_pred = tf.reshape(y_pred, (-1, self.num_slices, self.slice_size, self.label_dims))
            _sum_list_pred = tf.math.reduce_sum(y_pred, axis=2, keepdims=False)
            _loss = self.mae_loss(_sum_list_pred, _sum_list_true)
            
        return _loss


class ActiveMAE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.mean_func = tf.keras.metrics.Mean()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_min = tf.reduce_min(y_true, axis=1, keepdims=True)
        indices = tf.where(tf.math.subtract(y_true, y_min)>0)
        self.mean_func.reset_state()
        for index in indices:
            _index = tf.cast(index, dtype=tf.int32)
            self.mean_func.update_state(tf.math.abs(y_true[_index[0], _index[1]] - y_pred[_index[0], _index[1]]))
            
        return self.mean_func.result()


if __name__ == '__main__':
    print('load loss_func.py')
    test = tf.random.Generator.from_seed(123)
    test = test.normal(shape=(2, 8, 2))
    loss = ActiveMAE(label_length=8, label_dims=2, num_slices=2)
    print(loss(test, test))
    
