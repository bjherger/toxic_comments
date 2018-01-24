#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import pandas

import lib


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    extract()
    transform()
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


def transform():
    logging.info('Begin transform')

    # TODO Create histogram metrics

    # TODO Create histograms

    # TODO Create data set level metrics

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
