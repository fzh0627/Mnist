import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, callbacks
from tensorflow.keras.callbacks import TensorBoard
import datetime

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.expand_dims(x, axis=-1)
    y = tf.cast(y, dtype=tf.int32)
    return x,y
(x, y), (x_test, y_test) = datasets.mnist.load_data()
batchsz = 128
y = tf.one_hot(y, depth=10, axis=1)
y_test = tf.one_hot(y_test, depth=10, axis=1)
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)
class RBF(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(RBF, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)
        out = tf.reduce_sum(tf.pow(inputs-self.kernel, 2), axis=1)
        return out

class LeNet(keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='valid') # 6@24*24
        self.maxpool1 = layers.MaxPool2D() # 6@12*12
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, padding='valid') # 16@10*10
        self.maxpool2 = layers.MaxPool2D()# 16@5*5
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=120, activation='relu')
        self.dense2 = layers.Dense(units=84, activation='relu')
        self.gaussianconnection = RBF(input_dim=84, output_dim=10)
        # self.dense1 = layers.Dense(units=84, activation='relu')
    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.gaussianconnection(x)
        # x = tf.reshape(x, [batchsz,-1]) # reshape成平面层
        # x = self.dense5(x) # 输出层
        return x
myLeNet = LeNet()
myLeNet.compile(optimizer=optimizers.Adam(lr=1e-3), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# tensorboard 可视化过程
tb = TensorBoard(
    log_dir='logs',
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    update_freq='batch',
    profile_batch=2,
    embeddings_freq=0,
    embeddings_metadata=None
)
tb.on_epoch_begin(0)

myLeNet.fit(db, epochs=10, validation_data = db_test, validation_freq = 1, callbacks=[tb])
myLeNet.evaluate(db_test)