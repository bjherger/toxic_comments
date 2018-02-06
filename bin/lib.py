import datetime
import logging
import os
import re
import string

import matplotlib.pyplot as plt
import numpy
import pandas
import yaml
from keras import Model

from keras.preprocessing.sequence import pad_sequences
import keras

import models

# Global variables
CONFS = None
BATCH_NAME = None
BATCH_OUTPUT_FOLDER = None
MODEL_CHECKPOINT_PATH = None
TEMP_DIR = None
CHAR_INDICES = None
INDICES_CHAR = None
LEGAL_CHARS = None
CURRENT_BATCH_MODEL = None


def load_confs(confs_path='../conf/conf.yaml'):
    """
    Load configurations from file.

     - If configuration file is available, load it
     - If configuraiton file is not available attempt to load configuration template

    Configurations are never explicitly validated.
    :param confs_path: Path to a configuration file, appropriately formatted for this application
    :type confs_path: str
    :return: Python native object, containing configuration names and values
    :rtype: dict
    """
    global CONFS

    if CONFS is None:

        try:
            logging.info('Attempting to load conf from path: {}'.format(confs_path))

            # Attempt to load conf from confPath
            CONFS = yaml.load(open(confs_path))

        except IOError:
            logging.warn('Unable to open user conf file. Attempting to run with default values from conf template')

            # Attempt to load conf from template path
            template_path = confs_path + '.template'
            CONFS = yaml.load(open(template_path))

    return CONFS


def get_conf(conf_name):
    """
    Get a configuration parameter by its name
    :param conf_name: Name of a configuration parameter
    :type conf_name: str
    :return: Value for that conf (no specific type information available)
    """
    return load_confs()[conf_name]


def get_batch_name():
    """
    Get the name of the current run. This is a unique identifier for each run of this application
    :return: The name of the current run. This is a unique identifier for each run of this application
    :rtype: str
    """
    global BATCH_NAME

    if BATCH_NAME is None:
        logging.info('Batch name not yet set. Setting batch name.')
        batch_prefix = get_conf('batch_prefix')
        model_choice = get_conf('model_choice')
        datetime_str = str(datetime.datetime.utcnow().replace(microsecond=0).isoformat())+'Z'
        BATCH_NAME = '_'.join([batch_prefix, model_choice, datetime_str])
        logging.info('Batch name: {}'.format(BATCH_NAME))
    return BATCH_NAME

def get_batch_output_folder():
    global BATCH_OUTPUT_FOLDER
    if BATCH_OUTPUT_FOLDER is None:
        BATCH_OUTPUT_FOLDER = os.path.join(get_conf('load_path'), get_batch_name())
        os.mkdir(BATCH_OUTPUT_FOLDER)
        logging.info('Batch output folder: {}'.format(BATCH_OUTPUT_FOLDER))
    return BATCH_OUTPUT_FOLDER

def get_model_checkpoint_path():
    global MODEL_CHECKPOINT_PATH
    if MODEL_CHECKPOINT_PATH is None:
        MODEL_CHECKPOINT_PATH = os.path.join(get_batch_output_folder(), 'model_checkpoints')
        os.mkdir(MODEL_CHECKPOINT_PATH)
        logging.info('Model checkpoint path: {}'.format(MODEL_CHECKPOINT_PATH))
    return MODEL_CHECKPOINT_PATH




def get_model(X=None, y=None):
    global CURRENT_BATCH_MODEL

    if CURRENT_BATCH_MODEL is None:
        model_choice_string = get_conf('model_choice')
        logging.info('Attemption to load model w/ description: {}'.format(model_choice_string))

        # Pull serialized model, if requested
        if model_choice_string == 'serialized':
            logging.info('Attempting to load serialized model, from path: {}'.format(get_conf('serialized_model_choice_path')))
            CURRENT_BATCH_MODEL = keras.models.load_model(get_conf('serialized_model_choice_path'))

        # Attempt to pull a model from the models module, if requested
        elif model_choice_string in models.__dict__:
            if X is None or y is None:
                raise ValueError('Configurations request a new model, but X and y not provided. These are necessary'
                                 'for shape purposes.')

            CURRENT_BATCH_MODEL = models.__dict__[model_choice_string](X,y)

            if not isinstance(CURRENT_BATCH_MODEL, Model):
                raise ValueError('Specified model choice is not a Keras model. ')

        # Raise an error if no model generating
        else:
            raise ValueError('Could not find model choice with description: {}'.format(model_choice_string))

    return CURRENT_BATCH_MODEL


def archive_dataset_schemas(step_name, local_dict, global_dict):
    """
    Archive the schema for all available Pandas DataFrames

     - Determine which objects in namespace are Pandas DataFrames
     - Pull schema for all available Pandas DataFrames
     - Write schemas to file

    :param step_name: The name of the current operation (e.g. `extract`, `transform`, `model` or `load`
    :param local_dict: A dictionary containing mappings from variable name to objects. This is usually generated by
    calling `locals`
    :type local_dict: dict
    :param global_dict: A dictionary containing mappings from variable name to objects. This is usually generated by
    calling `globals`
    :type global_dict: dict
    :return: None
    :rtype: None
    """
    logging.info('Archiving data set schema(s) for step name: {}'.format(step_name))

    # Reference variables
    data_schema_dir = get_conf('data_schema_dir')
    schema_output_path = os.path.join(data_schema_dir, step_name + '.csv')
    schema_agg = list()

    env_variables = dict()
    env_variables.update(local_dict)
    env_variables.update(global_dict)

    # Filter down to Pandas DataFrames
    data_sets = filter(lambda (k, v): type(v) == pandas.DataFrame, env_variables.iteritems())
    data_sets = dict(data_sets)

    header = pandas.DataFrame(columns=['variable', 'type', 'data_set'])
    schema_agg.append(header)

    for (data_set_name, data_set) in data_sets.iteritems():
        # Extract variable names
        logging.info('Working data_set: {}'.format(data_set_name))

        local_schema_df = pandas.DataFrame(data_set.dtypes, columns=['type'])
        local_schema_df['data_set'] = data_set_name

        schema_agg.append(local_schema_df)

    # Aggregate schema list into one data frame
    agg_schema_df = pandas.concat(schema_agg)

    # Write to file
    agg_schema_df.to_csv(schema_output_path, index_label='variable')


def toxic_vars():
    return ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def var_histogram(df, column):
    x = df[column].values

    x = [value for value in x if not numpy.math.isnan(value)]
    histogram(x, column)


def histogram(x, name):
    # Stolen from https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html

    # Clear figure
    plt.clf()
    plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel(name)
    plt.title(name)
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.savefig(os.path.join(get_conf('histogram_path'), name))


def legal_characters():
    global LEGAL_CHARS
    if LEGAL_CHARS is None:
        chars = set(string.printable + '<>')
        # chars.remove('\n')
        # chars.remove('\r')
        LEGAL_CHARS = chars
    return LEGAL_CHARS


def get_char_indices():
    global CHAR_INDICES
    if CHAR_INDICES is None:
        chars = sorted(list(set(legal_characters())))
        CHAR_INDICES = dict((c, i) for i, c in enumerate(chars))
    return CHAR_INDICES


def get_indices_char():
    global INDICES_CHAR
    if INDICES_CHAR is None:
        chars = sorted(list(set(legal_characters())))
        INDICES_CHAR = dict((i, c) for i, c in enumerate(chars))
    return INDICES_CHAR


def gen_character_model_x_y(observations, x_column, gen_y=False):
    logging.info('Generating X and Y')

    # Reference vars
    char_indices = get_char_indices()
    indices_char = get_indices_char()
    cleaned_text_chars = list()
    cleaned_text_indices = list()

    # Prepare x
    for text in observations[x_column]:
        logging.debug('Raw text: {}'.format(text))

        text = map(lambda x: x.lower(), text)
        text = map(lambda x: x if x in legal_characters() else ' ', text)
        text = ''.join(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Add start and end characters
        text = re.sub('<', ' ', text)
        text = re.sub('>', ' ', text)
        text = '<' + text + '>'

        logging.debug('Cleaned text: {}'.format(text))
        cleaned_text_chars.append(text)

        text_indices = map(lambda x: char_indices[x], text)
        logging.debug('Cleaned text indices: {}'.format(text_indices))
        cleaned_text_indices.append(text_indices)

    X = pad_sequences(cleaned_text_indices, maxlen=get_conf('x_maxlen'), value=max(indices_char.keys()) + 1)

    # Prepare Ys
    if gen_y is True:
        ys = list()
        for toxic_var in toxic_vars():
            local_y = observations[toxic_var].values
            local_y = numpy.array(local_y, dtype=bool)
            ys.append(local_y)
        logging.info('Created X with shape: {} and Y_0 with shape: {}'.format(X.shape, ys[0].shape))
    else:
        ys = None
        logging.info('Created X with shape: {} and None Y'.format(X.shape))

    return X, ys
