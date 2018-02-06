import keras
import numpy
from keras import Input, Model, losses
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.optimizers import RMSprop

import lib


def baseline_model(X, y):
    model_input = Input(shape=(X.shape[1],))

    x = model_input
    output_layers = list()
    for toxic_var in lib.toxic_vars():
        local_output = Dense(units=1, activation='sigmoid', name=toxic_var + '_output')(x)
        output_layers.append(local_output)

    regression_model = Model(model_input, output_layers)
    regression_model.compile(loss=losses.binary_crossentropy,
                             optimizer='adam', metrics=['acc'])

    return regression_model


def bi_lstm(X, y):
    # Create Input layer
    # Input: Input length
    if len(X.shape) >= 2:
        model_input_length = int(X.shape[1])
    else:
        model_input_length = 1

    # Create Embedding layer
    # Embedding: Embedding input dimensionality is the same as the number of classes in the input data set
    embedding_input_dim = max(len(lib.legal_characters()), numpy.max(X)) + 1

    # Embedding: Embedding output dimensionality is determined by heuristic
    embedding_output_dim = int(min((embedding_input_dim + 1) / 2, 50))

    # Input: Use a smaller datatype, if possible. This explicit typing is necessary due to the OHE layer.
    if embedding_input_dim < 250:
        dtype = 'uint8'
    else:
        dtype = 'int32'
    sequence_input = keras.Input(shape=(model_input_length,), dtype=dtype, name='sigmoid')

    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                input_length=model_input_length,
                                trainable=True,
                                name='char_embedding')

    # Create model input and hidden layers
    x = sequence_input
    x = embedding_layer(x)
    x = Bidirectional(LSTM(128))(x)

    # Create model output layers
    output_layers = list()
    for toxic_var in lib.toxic_vars():
        local_output = Dense(units=1, activation='sigmoid', name=toxic_var + '_output')(x)
        output_layers.append(local_output)

    # Create model
    optimizer = RMSprop(lr=.001)
    bool_model = Model(sequence_input, output_layers)
    bool_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

    return bool_model
