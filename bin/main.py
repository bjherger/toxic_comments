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

    train_observations, test_observations = extract()
    train_observations, train_X, train_y, cat_model = train(train_observations)
    test_observations, test_X, test_preds = infer(test_observations, cat_model)
    load(train_observations, train_X, train_y, cat_model)
    pass


def extract():
    logging.info('Begin extract')

    # Read from file
    train_observations = pandas.read_csv(lib.get_conf('train_path'))
    test_observations = pandas.read_csv(lib.get_conf('test_path'))

    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return train_observations, test_observations


def transform(observations, gen_y):
    logging.info('Begin transform')

    if lib.get_conf('create_histograms'):
        # Create histogram metrics
        observations['num_chars'] = observations['comment_text'].apply(len)
        observations['tokens'] = observations['comment_text'].apply(lambda x: list(gensim.utils.tokenize(x)))
        observations['num_tokens'] = observations['tokens'].apply(len)
        observations['percent_unique_tokens'] = observations['tokens'].apply(
            lambda x: float(len(set(x)))) / observations['num_tokens']

        # Create histograms
        histogram_vars = ['num_chars', 'num_tokens', 'percent_unique_tokens']
        for histogram_var in histogram_vars:
            logging.info('Creating histogram for: {}'.format(histogram_var))
            lib.var_histogram(observations, histogram_var)

        # Agg: Create data set level metrics
        # Agg: is_toxic
        observations['is_toxic'] = observations[lib.toxic_vars()].max(axis=1)
        logging.info('Data set containx {} toxic posts, of {} posts'.format(
            sum(observations['is_toxic']), len(observations.index)))

        # Agg: Breakdown by toxic type
        for toxic_type in lib.toxic_vars():
            logging.info('{}: {} of {} posts ({}%) are toxic type: {}'.format(
                toxic_type, sum(observations[toxic_type]), len(observations.index), sum(observations[toxic_type]) / float(len(observations.index)), toxic_type))

        # Agg: Vocab size
        all_tokens = [item for sublist in observations['tokens'] for item in sublist]
        logging.info('Raw vocab size: {}'.format(len(set(all_tokens))))

        # Agg word count distribution
        token_counts = collections.Counter(all_tokens).values()
        token_counts = filter(lambda x: x > 1000, token_counts)
        lib.histogram(token_counts, 'token_counts, count > 1000')

    # TODO Replace mockup X with actual values
    X = numpy.random.normal(size=(len(observations.index), 3))
    if gen_y:
        y = observations[lib.toxic_vars()].values
        logging.info('Created X with shape: {} and Y with shape: {}'.format(X.shape, y.shape))
    else:
        y = None
        logging.info('Created X with shape: {} and None Y'.format(X.shape))



    lib.archive_dataset_schemas('transform_y_{}'.format(gen_y), locals(), globals())
    logging.info('End transform')
    return observations, X, y


def train(train_observations):
    logging.info('Begin train')

    train_observations, train_X, train_y = transform(train_observations, gen_y=True)

    # Set up callbacks
    tf_log_path = os.path.join(os.path.expanduser('~/log_dir'), lib.get_batch_name())
    logging.info('Using Tensorboard path: {}'.format(tf_log_path))

    mc_log_path = os.path.join(lib.get_conf('model_checkpoint_path'),
                               lib.get_batch_name() + '_epoch_{epoch:03d}_val_loss_{val_loss:.2f}.h5py')
    logging.info('Using mc_log_path path: {}'.format(mc_log_path))
    callbacks = [TensorBoard(log_dir=tf_log_path),
                 ModelCheckpoint(mc_log_path)]
    cat_model = models.baseline_model(train_X, train_y)

    cat_model.fit(train_X, train_y, validation_split=.2, epochs=2, callbacks=callbacks)

    lib.archive_dataset_schemas('train', locals(), globals())
    logging.info('End train')
    return train_observations, train_X, train_y, cat_model


def infer(test_observations, cat_model):
    logging.info('Begin infer')
    test_observations, test_X, test_y = transform(test_observations, gen_y=False)
    test_preds = cat_model.predict(test_X)

    column_names = map(lambda x: x+'_pred', lib.toxic_vars())
    preds_df = pandas.DataFrame(test_preds, columns=column_names)
    preds_df['id'] = test_observations['id']

    test_observations = pandas.merge(left=test_observations, right=preds_df, on='id')

    lib.archive_dataset_schemas('infer', locals(), globals())
    logging.info('End infer')
    return test_observations, test_X, test_preds

def load(train_observations, X, y, cat_model):
    logging.info('Begin load')

    # Save observations, if object heavy histogram data set wasn't generated
    if not lib.get_conf('create_histograms'):
        train_observations.to_feather(os.path.join(lib.get_conf('load_path'), 'train_observations.feather'))
        train_observations.to_csv(os.path.join(lib.get_conf('load_path'), 'train_observations.csv'), index=False)

    # Save final model
    cat_model.save(os.path.join(lib.get_conf('model_path'), 'model.h5py'))

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
