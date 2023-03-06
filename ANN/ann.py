# -*- coding: utf-8 -*-

import os
import csv
import json
import pandas
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def seed_tensorflow(seed=2022):
    tf.get_logger().setLevel('ERROR')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def data_loader():
    env1 = np.load('dataset/env.npy').astype(np.float32)
    env2 = np.load('synthetic/env.npy').astype(np.float32)
    env = np.concatenate((env1, env2), axis=0)
    body1 = np.load('dataset/body.npy').astype(np.float32)
    body2 = np.load('synthetic/body.npy').astype(np.float32)
    body = np.concatenate((body1, body2), axis=0)

    y1 = np.load('dataset/label.npy').astype(int)
    y2 = np.load('synthetic/label.npy').astype(int)
    y = np.concatenate((y1, y2), axis=0)
    x = np.concatenate((env, body), axis=1)
    train_feature, test_feature, train_label, test_label = train_test_split(x, y, test_size=0.2)

    print(f'train_feature shape: {len(train_feature)} * {len(train_feature[0])}')
    print(f'test_feature shape: {len(test_feature)} * {len(test_feature[0])}')

    return np.array(train_feature), np.array(test_feature), np.array(train_label), np.array(
        test_label)


class Classifier_Modeling(tf.keras.Model):
    def __init__(self):
        super(Classifier_Modeling, self).__init__()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

        self.dense_M1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_M2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense_Tsk1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_Tsk2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense_S1 = tf.keras.layers.Dense(units=16, activation=tf.nn.elu)
        self.dense_S2 = tf.keras.layers.Dense(units=16, activation=tf.nn.elu)

        self.dense_PMV1 = tf.keras.layers.Dense(units=16, activation=tf.nn.elu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=8, activation=tf.nn.elu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV5 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)

        self.dense_PMV11 = tf.keras.layers.Dense(units=8, activation=tf.nn.gelu)
        self.dense_PMV12 = tf.keras.layers.Dense(units=10, activation=tf.nn.gelu)
        self.dense_PMV13 = tf.keras.layers.Dense(units=10, activation=tf.nn.gelu)
        self.dense_PMV14 = tf.keras.layers.Dense(units=4, activation=tf.nn.gelu)
        self.dense_PMV15 = tf.keras.layers.Dense(units=3, activation=tf.nn.gelu)

    def call(self, inputs, training=None, mask=None):
        data = inputs['feature']  # [ta, hr, va, gender, age, weight, height, bmi]
        body = data[:, 3:]  # [gender, age, weight, height, bmi]
        environment = data[:, 0:3]  # [ta, hr, va]
        T = data[:, 0:1]  # Ta
        Pa = tf.math.log1p(T)

        M_input = self.drop(body, training=training)
        M = self.dense_M1(M_input)
        M = self.drop(M, training=training)
        M = self.dense_M2(M)

        environment = self.drop(environment, training=training)
        Tsk_input = self.drop(data, training=training)
        Tsk = tf.abs(self.dense_Tsk1(Tsk_input))
        Tsk_input = self.drop(data, training=training)
        Tsk = tf.abs(self.dense_Tsk2(Tsk_input))

        Psk = tf.math.log1p(Tsk)

        s_input = tf.concat([body, M, Tsk, Psk, environment, Pa], axis=1)

        s_input = self.drop(s_input, training=training)
        S = self.dense_S1(s_input)
        s_input = self.drop(S, training=training)
        S = self.dense_S2(s_input)

        pmv_input = tf.concat([body, M, Tsk, Psk, environment, Pa, S], axis=1)
        dense = self.dense_PMV1(pmv_input)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV5(dense)

        # dense = self.drop(data, training=training)
        # dense = self.dense_PMV11(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV12(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV13(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV14(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV15(dense)

        # print(dense)
        output = tf.nn.softmax(dense)
        # print(output)
        return output

    def get_embedding(self):
        return self.embedding.get_weights()


# def MSE_loss(y_true, y_pred):
#     loss = tf.keras.losses.mse(y_true, y_pred)
#     # loss= tf.square(y_true-y_pred)
#     # loss = tf.reduce_mean(loss)
#     return loss


def CE_loss(y_true, y_pred):
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    loss = tf.reduce_mean(loss)
    return loss


def Accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    # print(y_pred)
    return accuracy_score(y_pred, y_true)


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [CE_loss, Accuracy]
    loss = [CE_loss]
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='CE_loss', min_delta=0.0001, patience=10, verbose=1,
                                                 mode='min', restore_best_weights=True)
    callbacks = [earlyStop]
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x={'feature': train_feature},
              y=[train_label],
              epochs=num_epochs,
              batch_size=batch_size,
              validation_split=0.05,
              callbacks=callbacks,
              verbose=1,
              shuffle=True)
    checkpoint = tf.train.Checkpoint(classifier=model)
    path = checkpoint.save('save_model/model_ann.ckpt')
    print("model saved to %s" % path)


def test():
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('save_model/model_ann.ckpt-1').expect_partial()
    y_pred = model({'feature': test_feature}, training=False)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    print(accuracy_score(y_pred, test_label))
    # print("precision：")
    # print(precision_score(y_pred, test_label))
    # print("recall：")
    # print(recall_score(y_pred, test_label))
    # print("f1：")
    # print(f1_score(y_pred, test_label))


if __name__ == '__main__':
    seed_tensorflow(2022)
    train_feature, test_feature, train_label, test_label = data_loader()

    model = Classifier_Modeling()

    num_epochs, batch_size, learning_rate = 75, 16, 0.008
    train()
    test()
