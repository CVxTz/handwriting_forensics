import json
from random import choice, sample

import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, \
    Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops


def acc(y_true, y_pred, threshold=0.5):
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  y_true = math_ops.cast(y_true > threshold, y_pred.dtype)

  return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


def read_img(path):
    img = cv2.imread(path)
    return img


with open('writer_images_mapping.json', 'r') as f:
    writer_images_mapping = json.load(f)

train_writers_mapping = {k: v for k, v in writer_images_mapping.items() if int(k) % 10 != 1}
val_writers_mapping = {k: v for k, v in writer_images_mapping.items() if int(k) % 10 == 1}


def gen(writers_to_images_map, batch_size=16):
    writers = list(writers_to_images_map.keys())
    soft_val = 0.000000000000001
    while True:
        half_batch_writers = sample(writers, batch_size // 2)
        batch_tuples = list(zip(half_batch_writers, half_batch_writers))
        labels = [1-soft_val] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(writers)
            p2 = choice(writers)

            if p1 != p2:
                batch_tuples.append((p1, p2))
                labels.append(soft_val)

        X1 = [choice(writers_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])/255

        X2 = [choice(writers_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])/255

        yield [X1, X2], labels


def baseline_model():
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))

    base_model = ResNet50(weights="imagenet", include_top=False)

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=[acc], optimizer=Adam(0.00001))

    model.summary()

    return model


file_path = "handwriting_baseline.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

model = baseline_model()

try:
    model.load_weights(file_path)
except:
    print("No model to load")

model.fit_generator(gen(train_writers_mapping, batch_size=8), use_multiprocessing=True,
                    validation_data=gen(val_writers_mapping, batch_size=8), epochs=1000, verbose=1,
                    workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

eval = model.evaluate_generator(gen(val_writers_mapping, batch_size=32), steps=1000)

print(eval)