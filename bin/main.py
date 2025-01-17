#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import collections
import logging
import os
import random
import re

import gensim
import numpy
import pandas
from keras.callbacks import TensorBoard, ModelCheckpoint

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('Begin model run: {}'.format(lib.get_batch_name()))
    print('Begin model run: {}'.format(lib.get_batch_name()))

    # Load all resources and data from file

    train_observations, test_observations = extract()

    # Train a model
    if lib.get_conf('run_train'):
        train_observations, cat_model = train(train_observations)

    # Use model for inference
    if lib.get_conf('run_infer'):
        test_observations = infer(test_observations)

    # Load all results to file
    load(train_observations, test_observations)

    logging.info('End model run: {}. Results can be found at: {}'.format(lib.get_batch_name(),
                                                                         os.path.abspath(lib.get_batch_output_folder())))
    print('End model run: {}. Results can be found at: {}'.format(lib.get_batch_name(),
                                                                  os.path.abspath(lib.get_batch_output_folder())))
    pass


def extract():
    logging.info('Begin extract')

    # Read from file
    train_observations = pandas.read_csv(lib.get_conf('train_path'))
    test_observations = pandas.read_csv(lib.get_conf('test_path'))

    if lib.get_conf('test_run'):
        numpy.random.seed(0)
        random.seed(0)
        logging.warning('Performing test run. Subsetting to 1000 samples each of train and test')
        train_observations = train_observations.sample(200)
        train_observations = train_observations.reset_index()
        test_observations = test_observations.sample(200)
        test_observations = test_observations.reset_index()

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
                toxic_type, sum(observations[toxic_type]), len(observations.index),
                sum(observations[toxic_type]) / float(len(observations.index)), toxic_type))

        # Agg: Vocab size
        all_tokens = [item for sublist in observations['tokens'] for item in sublist]
        logging.info('Raw vocab size: {}'.format(len(set(all_tokens))))

        # Agg word count distribution
        token_counts = collections.Counter(all_tokens).values()
        token_counts = filter(lambda x: x > 1000, token_counts)
        lib.histogram(token_counts, 'token_counts, count > 1000')

    # Generate X and y values
    X, ys = lib.gen_character_model_x_y(observations, 'comment_text', gen_y)

    lib.archive_dataset_schemas('transform_y_{}'.format(gen_y), locals(), globals())
    logging.info('End transform')
    return observations, X, ys


def train(train_observations):
    logging.info('Begin train')

    train_observations, train_X, train_ys = transform(train_observations, gen_y=True)

    # Set up callbacks
    tf_log_path = os.path.join(os.path.expanduser('~/log_dir'), lib.get_batch_name())
    logging.info('Using Tensorboard path: {}'.format(tf_log_path))

    mc_log_path = os.path.join(lib.get_model_checkpoint_path(),
                               lib.get_batch_name() + '_epoch_{epoch:03d}_val_loss_{val_loss:.2f}.h5py')
    logging.info('Using mc_log_path path: {}'.format(mc_log_path))
    callbacks = [TensorBoard(log_dir=tf_log_path),
                 ModelCheckpoint(mc_log_path)]

    cat_model = lib.get_model(train_X, train_ys)

    cat_model.fit(train_X, train_ys, validation_split=.2, epochs=lib.get_conf('num_epochs'), callbacks=callbacks)

    lib.archive_dataset_schemas('train', locals(), globals())
    logging.info('End train')
    return train_observations, cat_model


def infer(test_observations):
    logging.info('Begin infer')
    cat_model = lib.get_model()
    test_observations, test_X, test_ys = transform(test_observations, gen_y=False)
    test_preds = cat_model.predict(test_X)

    # Each probability is wrapped in its own array. This produces a flat array of probabilities
    test_preds = map(lambda x: x[:, 0], test_preds)

    column_names = map(lambda x: x + '_pred', lib.toxic_vars())
    preds_df = pandas.DataFrame(index=range(len(test_preds[0])))
    for index, column_name in enumerate(column_names):
        preds_df[column_name] = test_preds[index]

    preds_df['id'] = test_observations['id']
    preds_df = preds_df.reset_index()

    test_observations = pandas.merge(left=test_observations, right=preds_df, on='id')

    lib.archive_dataset_schemas('infer', locals(), globals())
    logging.info('End infer')
    return test_observations


def load(train_observations, test_observations):
    logging.info('Begin load')
    cat_model = lib.get_model()

    if not lib.get_conf('create_histograms'):
        # Save train observations, if object heavy histogram data set wasn't generated
        logging.info('Saving train observations')
        train_observations.to_feather(os.path.join(lib.get_batch_output_folder(), 'train_observations.feather'))
        train_observations.to_csv(os.path.join(lib.get_batch_output_folder(), 'train_observations.csv'), index=False)

        # Save test observations, if object heavy histogram data set wasn't generated
        logging.info('Saving test observations')
        test_observations.to_feather(os.path.join(lib.get_batch_output_folder(), 'test_observations.feather'))
        test_observations.to_csv(os.path.join(lib.get_batch_output_folder(), 'test_observations.csv'), index=False)

    # Save submission
    if lib.get_conf('run_infer') is True:
        logging.info('Saving submission')
        submission_columns = pandas.read_csv(lib.get_conf('sample_submission_path')).columns
        submissions = test_observations.copy()
        submissions.columns = map(lambda x: re.sub(r'_pred', '', x), submissions.columns)
        submissions = submissions[submission_columns]
        logging.info('Creating submission w/ columns: {}'.format(submissions.columns))
        submissions.to_csv(
            path_or_buf=os.path.join(lib.get_batch_output_folder(), 'final_epoch_submission.csv'),
            index=False)
    else:
        logging.info('Submission not saved, as infer was not run')

    # Save final model
    cat_model.save(os.path.join(lib.get_batch_output_folder(), 'final_model.h5py'))

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
