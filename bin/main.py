#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import collections
import logging

import gensim
import itertools
import pandas

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    train_observations = extract()
    transform(train_observations)
    model()
    load()
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

    # Create histogram metrics
    train_observations['num_chars'] = train_observations['comment_text'].apply(len)
    train_observations['tokens'] = train_observations['comment_text'].apply(lambda x: list(gensim.utils.tokenize(x)))
    train_observations['num_tokens'] = train_observations['tokens'].apply(len)
    train_observations['percent_unique_tokens'] = train_observations['tokens'].apply(
        lambda x: float(len(set(x)))) / train_observations['num_tokens']

    print train_observations

    # TODO Create histograms

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
    all_tokens = itertools.chain(train_observations['tokens'])
    logging.info('Vocab size: {}'.format(len(set(all_tokens))))

    # Agg word count distribution
    token_counts = collections.Counter(all_tokens).values()




    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    pass


def model():
    logging.info('Begin model')

    lib.archive_dataset_schemas('model', locals(), globals())
    logging.info('End model')
    pass


def load():
    logging.info('Begin load')

    lib.archive_dataset_schemas('load', locals(), globals())
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
