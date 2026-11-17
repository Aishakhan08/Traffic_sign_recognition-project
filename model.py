"""CNN architecture for traffic sign classification (TensorFlow / Keras)."""
from tensorflow import keras
from tensorflow.keras import layers

from config import IMG_SIZE


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)


def build_model(num_classes, img_size=IMG_SIZE):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Data augmentation (only active during training).
    # No horizontal flips: they would swap turn-left / turn-right signs.
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomContrast(0.2)(x)

    for filters in (32, 64, 128):
        x = conv_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="traffic_sign_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_model(13).summary()
