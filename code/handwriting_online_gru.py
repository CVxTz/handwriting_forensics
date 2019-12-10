import json
from random import choice, sample

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D, GlobalAvgPool1D, Concatenate, Multiply, Dropout, \
    Subtract, Bidirectional, MaxPool1D, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops


def acc(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


def cap(seq, cap_val=0, max_size=40):
    seq = seq[:max_size]
    seq = seq + [cap_val] * (max_size - len(seq))
    return seq


def to_delta(x):
    x = [0] + x
    deltas = [b - a for a, b in zip(x[:-1], x[1:])]
    return deltas


def generate_sample_from_strokes(data, size_max=1000, pen_val=10):
    x = []
    y = []
    pen = []
    for stroke in data['strokes']:
        x += stroke['x']
        y += stroke['y']
        _pen = [0] * len(stroke['x'])
        _pen[0] = pen_val
        pen += _pen

    pen = cap(pen, cap_val=0, max_size=size_max)
    x = cap(x, cap_val=0, max_size=size_max)
    y = cap(y, cap_val=0, max_size=size_max)

    x = to_delta(x)
    y = to_delta(y)

    return list(zip(x, y, pen))


def read_sample(path):
    with open(path, "r") as f:
        data = json.load(f)
    return generate_sample_from_strokes(data)


def gen(writers_to_images_map, batch_size=16):
    writers = list(writers_to_images_map.keys())
    soft_val = 0.001
    while True:
        half_batch_writers = sample(writers, batch_size // 2)
        batch_tuples = list(zip(half_batch_writers, half_batch_writers))
        labels = [1 - soft_val] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(writers)
            p2 = choice(writers)

            if p1 != p2:
                batch_tuples.append((p1, p2))
                labels.append(soft_val)

        X1 = [choice(writers_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_sample(x) for x in X1]) / 10

        X2 = [choice(writers_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_sample(x) for x in X2]) / 10

        yield [X1, X2], labels


def encoder(seq_dim=3):
    in_seq_input = Input((None, seq_dim), name='in_sequence')

    in_seq = Bidirectional(GRU(128, return_sequences=True), name="gru_1")(in_seq_input)

    in_seq = MaxPool1D(pool_size=10)(in_seq)

    in_seq = Bidirectional(GRU(256, return_sequences=True), name="gru_2")(in_seq)

    in_seq = MaxPool1D(pool_size=10)(in_seq)

    in_seq = Bidirectional(GRU(1024, return_sequences=True), name="gru_3")(in_seq)

    model = Model(in_seq_input, in_seq)
    return model


def baseline_model(seq_dim=3):
    input_1 = Input(shape=(None, 3))
    input_2 = Input(shape=(None, seq_dim))

    base_model = encoder(seq_dim=3)

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool1D()(x1), GlobalAvgPool1D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool1D()(x2), GlobalAvgPool1D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=[acc], optimizer=Adam(0.0001))

    model.summary()

    return model


if __name__ == "__main__":

    with open('writer_json_mapping.json', 'r') as f:
        writer_json_mapping = json.load(f)

    train_writers_mapping = {k: v for k, v in writer_json_mapping.items() if int(k) % 10 != 1}
    val_writers_mapping = {k: v for k, v in writer_json_mapping.items() if int(k) % 10 == 1}

    file_path = "handwriting_online_gru.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    callbacks_list = [checkpoint, reduce_on_plateau]

    model = baseline_model()

    try:
        model.load_weights(file_path)
    except:
        print("No model to load")

    model.fit_generator(gen(train_writers_mapping, batch_size=32), use_multiprocessing=True,
                        validation_data=gen(val_writers_mapping, batch_size=32), epochs=1000, verbose=1,
                        workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

    eval = model.evaluate_generator(gen(val_writers_mapping, batch_size=32), steps=1000)

    print(eval)
    #[0.2315092908050865, 0.9208437]
