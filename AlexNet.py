import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    BatchNormalization, GlobalAveragePooling2D,
    LeakyReLU, Dropout, Input
)
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Input(shape=(224, 224, 3)),

    Conv2D(64, (7,7), strides=2, padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(3, strides=2),

    Conv2D(128, (5,5), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D(3, strides=2),

    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    LeakyReLU(negative_slope=0.1),

    GlobalAveragePooling2D(),

    Dense(512),
    LeakyReLU(negative_slope=0.1),
    Dropout(0.4),

    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
