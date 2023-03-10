# -*- coding: utf-8 -*-

import os
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
    env1 = np.load('./dataset/synthetic_v1/env.npy').astype(np.float32)
    env2 = np.load('./dataset/synthetic_v1/env.npy').astype(np.float32)
    env = np.concatenate((env1, env2), axis=0)
    body1 = np.load('./dataset/synthetic_v1/body.npy').astype(np.float32)
    body2 = np.load('./dataset/synthetic_v1/body.npy').astype(np.float32)
    body = np.concatenate((body1, body2), axis=0)

    y1 = np.load('./dataset/synthetic_v1/label.npy').astype(int)
    y2 = np.load('./dataset/synthetic_v1/label.npy').astype(int)
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

        self.dense_PMV1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        # self.dense_PMV5 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        # self.dense_PMV6 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV7 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV8 = tf.keras.layers.Dense(units=4, activation=tf.nn.leaky_relu)
        self.dense_PMV9 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)


    def call(self, inputs, training=None, mask=None):
        data = inputs['feature']  # [ta, hr, va, gender, age, weight, height, bmi]

        dense = self.drop(data, training=training)
        dense = self.dense_PMV1(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV5(dense)
        # dense = self.drop(dense, training=training)
        # dense = self.dense_PMV6(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV7(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV8(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV9(dense)

        output = tf.nn.softmax(dense)
        return output

    def get_embedding(self):
        return self.embedding.get_weights()


def MSE_loss(y_true, y_pred):
    loss = tf.keras.losses.mse(y_true, y_pred)
    return loss


def CE_loss(y_true, y_pred):
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss


def Accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_pred, y_true)


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [CE_loss, Accuracy]
    loss = [CE_loss, CE_loss]
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='CE_loss', min_delta=0.0001, patience=100, verbose=1,
                                                 mode='min', restore_best_weights=True)
    callbacks = [earlyStop]
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x={'feature': train_feature},
              y=[train_label, train_label],
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
    y_pred = np.argmax(y_pred, axis=1)
    print('????????????' + str(accuracy_score(y_pred, test_label)))
    print('????????? macro???' + str(precision_score(y_pred, test_label, average='macro')))
    print('????????? micro???' + str(precision_score(y_pred, test_label, average='micro')))
    print('????????? weighted???' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro???' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro???' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted???' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro???' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro???' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted???' + str(f1_score(y_pred, test_label, average='weighted')))


if __name__ == '__main__':
    seed_tensorflow(2022)
    train_feature, test_feature, train_label, test_label = data_loader()

    model = Classifier_Modeling()

    num_epochs, batch_size, learning_rate = 128, 32, 0.008
    train()
    test()
