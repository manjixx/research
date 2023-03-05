# -*- coding: utf-8 -*-

from random import randint
import tensorflow as tf
import numpy as np
import pandas
import csv
import os
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

tf.random.set_seed(2022)


class data_loader():
    def __init__(self):
        env = np.load('../dataset/npy/env.npy')
        body = np.load('../dataset/npy/body.npy')
        label = np.load('../dataset/npy/label.npy')
        feature = np.concatenate((env, body), axis=1)

        train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2)

        self.train_data = np.array(train_feature)
        self.test_data = np.array(test_feature)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)
        self.env_data = np.array(env)
        self.body_data = np.array(body)
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

        # train_feature = []
        # test_feature = []
        # train_label = []
        # test_label = []
        #
        # for i in range(len(data_gamble)):
        #     temp = [1, 0, 0]
        #     if i % 5 != 1:
        #         train_feature.append(data_gamble[i])
        #         train_label.append(temp[:])
        #     else:
        #         test_feature.append(data_gamble[i])
        #         test_label.append(temp[:])
        #
        # for i in range(len(data_porn)):
        #     temp = [0, 1, 0]
        #     if i % 5 != 1:
        #         train_feature.append(data_porn[i])
        #         train_label.append(temp[:])
        #     else:
        #         test_feature.append(data_porn[i])
        #         test_label.append(temp[:])
        #
        # for i in range(len(data_positive)):
        #     temp = [0, 0, 1]
        #     if i % 5 != 1:
        #         train_feature.append(data_positive[i])
        #         train_label.append(temp[:])
        #     else:
        #         test_feature.append(data_positive[i])
        #         test_label.append(temp[:])
        # self.train_data = np.array(train_feature)
        # self.test_data = np.array(test_feature)
        # self.train_label = np.array(train_label)
        # self.test_label = np.array(test_label)
        # self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]


class S_ATT(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(units=378, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(units=42, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(units=14, activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)
        self.drop = tf.keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=None, mask=None):
        dense = self.drop(inputs, training=training)
        dense = self.dense1(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense5(dense)
        output = tf.nn.softmax(dense)
        return output


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=0.00035)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = ['accuracy']
    ckpt = tf.keras.callbacks.ModelCheckpoint('save_love/model_classifier.ckpt', monitor='accuracy', verbose=2,
                                              save_best_only=True, save_weights_only=False, period=2)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=10, verbose=True)
    callbacks = [ckpt, earlystop]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    checkpoint = tf.train.Checkpoint(classifier=model)
    model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size,
              callbacks=callbacks)
    path = checkpoint.save('save_model/model_classifier.ckpt')
    print("model saved to %s" % path)


def test():
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('save_model/model_classifier.ckpt-1').expect_partial()
    num_batches = int(data_loader.num_test_data // batch_size)
    correct = 0
    print(data_loader.num_test_data)
    for batch_index in range(num_batches + 1):
        if batch_index < num_batches:
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            y_pred = model(data_loader.test_data[start_index: end_index], training=False)
        else:
            start_index, end_index = batch_index * batch_size, data_loader.num_test_data
            y_pred = model(data_loader.test_data[end_index - batch_size: end_index], training=False)[
                     batch_index * batch_size - end_index:]
        y_pred = tf.argmax(y_pred, 1)
        # print(y_pred)
        for index, label in enumerate(data_loader.test_label[start_index: end_index]):
            if label[y_pred[index]] == 1:
                correct += 1
            else:
                pass
    accuracy = correct / data_loader.num_test_data
    print("test accuracy: %f" % accuracy)


if __name__ == '__main__':
    model = S_ATT()
    num_epochs = 165
    batch_size = 128
    learning_rate = 0.001
    data_loader = data_loader()
    train()
    test()
