import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, callbacks
from tensorflow.keras.callbacks import TensorBoard
import datetime

import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
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

# db_iter = iter(db)
# sample = next(db_iter)
# print('batch:', sample[0].shape, sample[1].shape)

class mnistClassify(keras.Model):
    def __init__(self):
        super(mnistClassify, self).__init__()
        self.dense1 = layers.Dense(units=256, activation='relu')
        self.dense2 = layers.Dense(units=128, activation='relu')
        self.dense3 = layers.Dense(units=64, activation='relu')
        self.dense4 = layers.Dense(units=32, activation='relu')
        self.dense5 = layers.Dense(units=10)

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, 28*28])
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x) # 输出层
        return x

myMnist = mnistClassify()
myMnist.compile(optimizer=optimizers.Adam(lr=1e-3), loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

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

myMnist.fit(db, epochs=5, validation_data = db_test, validation_freq = 1, callbacks=[tb])
myMnist.evaluate(db_test)


