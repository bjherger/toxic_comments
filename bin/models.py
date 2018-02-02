from keras import Input, Model, losses
from keras.layers import Dense

import lib


def baseline_model(X, y):

    model_input = Input(shape=(X.shape[1],))

    x = model_input
    output_layers = list()
    for toxic_var in lib.toxic_vars():
        local_output = Dense(units=1, activation='sigmoid', name=toxic_var + '_output' )(x)
        output_layers.append(local_output)

    regression_model = Model(model_input, output_layers)
    regression_model.compile(loss=losses.binary_crossentropy,
                             optimizer='adam', metrics=['acc'])
    
    return regression_model