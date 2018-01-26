#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import collections
import logging

import gensim
import os
import itertools

import numpy
import pandas
from keras.callbacks import TensorBoard, ModelCheckpoint

import lib
import models


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    train_observations = extract()
    train_observations, X, y = transform(train_observations)
    train_observations, X, y, cat_model = model(train_observations, X, y)
    load(train_observations, X, y, cat_model)
    pass


def extract():
    logging.info('Begin extract')

    # Read from file
    train_observations = pandas.read_csv(lib.get_conf('train_path'))

    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return train_observations


def transform(train_observations):
    logging.info('Begin transform')

    if lib.get_conf('create_histograms'):
        # Create histogram metrics
        train_observations['num_chars'] = train_observations['comment_text'].apply(len)
        train_observations['tokens'] = train_observations['comment_text'].apply(lambda x: list(gensim.utils.tokenize(x)))
        train_observations['num_tokens'] = train_observations['tokens'].apply(len)
        train_observations['percent_unique_tokens'] = train_observations['tokens'].apply(
            lambda x: float(len(set(x)))) / train_observations['num_tokens']

        # Create histograms
        histogram_vars = ['num_chars', 'num_tokens', 'percent_unique_tokens']
        for histogram_var in histogram_vars:
            logging.info('Creating histogram for: {}'.format(histogram_var))
            lib.var_histogram(train_observations, histogram_var)

        # Agg: Create data set level metrics
        # Agg: is_toxic
        train_observations['is_toxic'] = train_observations[lib.toxic_vars()].max(axis=1)
        logging.info('Data set containx {} toxic posts, of {} posts'.format(
            sum(train_observations['is_toxic']), len(train_observations.index)))

        # Agg: Breakdown by toxic type
        for toxic_type in lib.toxic_vars():
            logging.info('{}: {} of {} posts ({}%) are toxic type: {}'.format(
                toxic_type, sum(train_observations[toxic_type]), len(train_observations.index), sum(train_observations[toxic_type]) / float(len(train_observations.index)), toxic_type))

        # Agg: Vocab size
        all_tokens = [item for sublist in train_observations['tokens'] for item in sublist]
        logging.info('Raw vocab size: {}'.format(len(set(all_tokens))))

        # Agg word count distribution
        token_counts = collections.Counter(all_tokens).values()
        token_counts = filter(lambda x: x > 1000, token_counts)
        lib.histogram(token_counts, 'token_counts, count > 1000')

    # TODO Replace mockup X with actual values
    X = numpy.random.normal(size=(len(train_observations.index), 3))
    y = train_observations[lib.toxic_vars()].values
    logging.info('Created X with shape: {} and Y with shape: {}'.format(X.shape, y.shape))

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return train_observations, X, y


def model(train_observations, X, y):
    logging.info('Begin model')

    # Set up callbacks
    tf_log_path = os.path.join(os.path.expanduser('~/log_dir'), lib.get_batch_name())
    logging.info('Using Tensorboard path: {}'.format(tf_log_path))

    mc_log_path = os.path.join(lib.get_conf('model_checkpoint_path'),
                               lib.get_batch_name() + '_epoch_{epoch:03d}_val_loss_{val_loss:.2f}.h5py')
    logging.info('Using mc_log_path path: {}'.format(mc_log_path))
    callbacks = [TensorBoard(log_dir=tf_log_path),
                 ModelCheckpoint(mc_log_path)]
    cat_model = models.baseline_model(X, y)

    cat_model.fit(X, y, validation_split=.2, epochs=2, callbacks=callbacks)


    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    return train_observations, X, y, cat_model


def load(train_observations, X, y, cat_model):
    logging.info('Begin load')

    # Save observations
    train_observations.to_feather(os.path.join(lib.get_conf('load_path'), 'train_observations.feather'))
    train_observations.to_csv(os.path.join(lib.get_conf('load_path'), 'train_observations.csv'), index=False)

    # Save final model
    cat_model.save(lib.get_conf('model_path'))

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
