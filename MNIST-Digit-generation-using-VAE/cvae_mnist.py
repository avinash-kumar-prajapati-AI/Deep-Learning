

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set this to suppress excessive TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

latent_dim = 2
num_classes = 10
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)


def load_data():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = 255 - x_train  # Invert grayscale
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    y_train_ohe = keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train_ohe


# def sampling(args):
#     z_mean, z_log_var = args
#     epsilon = tf.random.normal(shape=tf.shape(z_mean))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder():
    image_input = keras.Input(shape=(28, 28, 1))
    label_input = keras.Input(shape=(num_classes,))
    x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(image_input)
    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, label_input])
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    return keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")


def build_decoder():
    latent_input = keras.Input(shape=(latent_dim,))
    label_input = keras.Input(shape=(num_classes,))
    x = layers.Concatenate()([latent_input, label_input])
    x = layers.Dense(7 * 7 * 64, activation='relu')(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    output_img = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return keras.Model([latent_input, label_input], output_img, name="decoder")


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

     # ✅ Add this method
    def call(self, inputs):
        x_img, x_label = inputs
        z_mean, z_log_var, z = self.encoder([x_img, x_label])
        return self.decoder([z, x_label])

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        (x_img, x_label) = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x_img, x_label])
            reconstruction = self.decoder([z, x_label])
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x_img, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            ))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}


def train_model():
    x_train, y_train_ohe = load_data()
    batch_size = 128
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_ohe))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    encoder = build_encoder()
    decoder = build_decoder()
    cvae = CVAE(encoder, decoder)
    cvae.compile(optimizer=keras.optimizers.Adam())
    cvae.fit(train_dataset, epochs=20)
    # cvae

    # 👇 Force model build before saving weights
    _ = cvae([tf.zeros((1, 28, 28, 1)), tf.zeros((1, 10))])

    # # 🔧 Build encoder and decoder to make sure model weights are initialized
    # _ = encoder([tf.zeros((1, 28, 28, 1)), tf.zeros((1, 10))])
    # _ = decoder([tf.zeros((1, latent_dim)), tf.zeros((1, 10))])

    encoder.save(os.path.join(model_dir, "encoder.h5"))
    decoder.save(os.path.join(model_dir, "decoder.h5"))
    cvae.save_weights(os.path.join(model_dir, "cvae.weights.h5"))


    print("✅ Model trained and saved.")


def generate_digit(digit=0, n=10):
    encoder = keras.models.load_model(os.path.join(model_dir, "encoder.h5"), compile=False)
    decoder = keras.models.load_model(os.path.join(model_dir, "decoder.h5"), compile=False)

    latent_samples = np.random.normal(size=(n, latent_dim))
    labels = keras.utils.to_categorical([digit] * n, num_classes=num_classes)
    generated = decoder.predict([latent_samples, labels])

    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(generated[i].squeeze(), cmap='gray')
        plt.axis("off")
    plt.suptitle(f"Generated Digit '{digit}'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the CVAE model")
    parser.add_argument("--generate", type=int, default=None, help="Generate digit (0-9)")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.generate is not None:
        generate_digit(digit=args.generate)
    else:
        print("⚠️ Use --train to train or --generate <digit> to generate images.")
