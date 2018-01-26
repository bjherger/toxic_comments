from keras import Input, Model, losses
from keras.layers import Dense


def baseline_model(X, y):

    model_input = Input(shape=(X.shape[1],))

    x = model_input
    preds = Dense(units=y.shape[1], activation='sigmoid')(x)
    regression_model = Model(model_input, preds)
    regression_model.compile(loss=losses.categorical_crossentropy,
                             optimizer='adam', metrics=['acc'])
    return regression_model