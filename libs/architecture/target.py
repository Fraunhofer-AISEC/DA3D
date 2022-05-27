import tensorflow as tf

from typing import List, Tuple, Dict
from pathlib import Path

from libs.network.network import add_symmetric_autoencoder, add_dense, parse_shape


class Autoencoder(tf.keras.Model):
    def __init__(
            self, p_dropout: float = 0.1,
            hidden_activation: str = "relu", out_activation: str = "sigmoid", layer_dims: List[int] = None,
            use_bias: bool = True, code_dim_override: int = None
    ):
        """
        Create an autoencoder
        :param p_dropout: dropout percentage
        :param hidden_activation: activation function of the hidden layers
        :param out_activation: activation function of the output layer
        :param layer_dims: hidden layer dimensions from the input to the code, if None use a convolutional AE
        :param use_bias: include the bias vector in the layers (e.g. DeepSVDD does not use it)
        """
        super(Autoencoder, self).__init__()

        # Model config
        self.p_dropout = p_dropout
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.layer_dims = layer_dims
        self.use_bias = use_bias
        self.code_dim_override = code_dim_override

        if (layer_dims is not None) and (code_dim_override is not None):
            # Ok, it was a bad idea to enforce tuples
            new_layer_dims = list(layer_dims)
            new_layer_dims[-1] = code_dim_override
            self.layer_dims = new_layer_dims

        # Layers
        self.m_enc = None
        self.m_dec = None

        # Activation extractors
        self.m_enc_act = None
        self.m_dec_act = None
        self.m_dec_act_on_code = None
        self.m_all_act = None

    # -- Autoencoder Architectures --
    # Conv AE based on https://blog.keras.io/building-autoencoders-in-keras.html
    def _conv_encoder(self, input_shape) -> tf.keras.Model:
        model = tf.keras.Sequential(name="encoder")

        model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", input_shape=input_shape[1:], use_bias=self.use_bias))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Activation(self.hidden_activation))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=self.use_bias))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Activation(self.hidden_activation))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=self.use_bias))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Activation(self.hidden_activation, name="code"))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        return model

    def _conv_decoder(self, input_shape, output_shape):
        model = tf.keras.Sequential(name="decoder")

        model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", input_shape=input_shape[1:], use_bias=self.use_bias))
        model.add(tf.keras.layers.Activation(self.hidden_activation))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", use_bias=self.use_bias))
        model.add(tf.keras.layers.Activation(self.hidden_activation))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        if output_shape[1] == 32:
            model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", use_bias=self.use_bias))
        else:
            model.add(tf.keras.layers.Conv2D(16, (3, 3), use_bias=self.use_bias))
        model.add(tf.keras.layers.Activation(self.hidden_activation))
        model.add(tf.keras.layers.UpSampling2D((2, 2)))
        model.add(tf.keras.layers.SpatialDropout2D(self.p_dropout))

        model.add(tf.keras.layers.Conv2D(
            output_shape[-1], (3, 3), activation=self.out_activation, padding="same", name="target_output", use_bias=self.use_bias
        ))

        return model

    def _dense_encoder(self, input_shape):
        model = tf.keras.Sequential(name="encoder")

        add_dense(
            model, layer_dims=self.layer_dims[:-1], p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # We add the last layer manually to name it accordingly
        model.add(tf.keras.layers.Dense(
            self.layer_dims[-1], activation=self.hidden_activation, name="code", use_bias=self.use_bias
        ))

        return model

    def _dense_decoder(self, input_shape, output_dim):
        model = tf.keras.Sequential(name="decoder")

        add_dense(
            model, layer_dims=list(reversed(self.layer_dims[:-1])), p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # The last layer reconstructs the input
        model.add(tf.keras.layers.Dense(
            output_dim[-1], activation=self.out_activation, use_bias=self.use_bias
        ))

        return model

    # == Keras functions ==
    def build(self, input_shape):
        # Based on the given layers, we use a dense or convolutional AE
        self.m_enc = self._conv_encoder(input_shape) if self.layer_dims is None \
            else self._dense_encoder(input_shape)
        self.m_dec = self._conv_decoder(self.m_enc.output_shape, input_shape) if self.layer_dims is None \
            else self._dense_decoder(self.m_enc.output_shape, input_shape)

        self.build_extractors()

    def compile(self, learning_rate=0.0001, loss="binary_crossentropy", optimizer=None, **kwargs):
        new_optimizer = tf.keras.optimizers.Adam(learning_rate)
        return super(Autoencoder, self).compile(optimizer=new_optimizer, loss=loss, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=60, verbose=2, **kwargs):
        return super(Autoencoder, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def build_extractors(self):
        # On top, we build the activation extractors
        t_enc_act = tf.keras.layers.Concatenate()([
            tf.keras.layers.Flatten()(cur_layer.output) for cur_layer in self.m_enc.layers
            if isinstance(cur_layer, tf.keras.layers.Activation) or isinstance(cur_layer, tf.keras.layers.LeakyReLU)
        ])
        t_dec_act = tf.keras.layers.Concatenate()([
            tf.keras.layers.Flatten()(cur_layer.output) for cur_layer in self.m_dec.layers
            if isinstance(cur_layer, tf.keras.layers.Activation) or isinstance(cur_layer, tf.keras.layers.LeakyReLU)
        ])
        # Activation extractors
        self.m_enc_act = tf.keras.Model(self.m_enc.inputs, t_enc_act, name="act_enc")
        self.m_dec_act_on_code = tf.keras.Model(self.m_dec.inputs, t_dec_act, name="act_dec_on_code")
        self.m_dec_act = tf.keras.Model(
            self.m_enc.inputs, self.m_dec_act_on_code(self.m_enc(self.m_enc.inputs)), name="act_dec"
        )

        # Concatenating both models gives us all activations
        t_all_act = tf.keras.layers.Concatenate()([
            self.m_enc_act(self.m_enc_act.inputs), self.m_dec_act(self.m_enc_act.inputs)
        ])
        self.m_all_act = tf.keras.Model(
            self.m_enc.inputs, t_all_act, name="act_all"
        )

    def call(self, inputs, training=False, mask=None):
        # Connect the encoder and decoder
        t_encoded = self.m_enc(inputs, training=training, mask=mask)
        t_decoded = self.m_dec(t_encoded, training=training, mask=mask)

        return t_decoded

    def get_config(self):
        config = {
            'p_dropout': self.p_dropout,
            'hidden_activation': self.hidden_activation,
            'out_activation': self.out_activation,
            'layer_dims': self.layer_dims,
        }

        base_config = super(Autoencoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdversarialAutoencoder(Autoencoder):
    def __init__(self, disc_dims: Tuple[int] = (40, 20, 10, 5), clipping_val: float = .01, **kwargs):

        # By default, use leaky ReLUs
        hidden_activation = {} if "hidden_activation" in kwargs else {"hidden_activation": "leakyrelu"}
        super(AdversarialAutoencoder, self).__init__(**hidden_activation, **kwargs)

        # Config
        self.disc_dims = disc_dims
        self.clipping_val = clipping_val

        # Models
        self.m_disc = None

        # Optimiser
        self.recon_opt = None
        self.disc_opt = None
        self.dec_opt = None

    def get_discriminator(self, input_shape):
        # Construct a simple feed-forward network
        m_disc = tf.keras.Sequential(name="discriminator")
        add_dense(
            m_disc, layer_dims=self.disc_dims, activation=self.hidden_activation, input_shape=input_shape[1:],
            p_dropout=self.p_dropout
        )
        m_disc.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        return m_disc

    def _dense_encoder(self, input_shape):
        model = tf.keras.Sequential(name="encoder")

        add_dense(
            model, layer_dims=self.layer_dims[:-1], p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:], use_bias=self.use_bias
        )
        # Code layer has no activation function
        model.add(tf.keras.layers.Dense(self.layer_dims[-1], name="code", use_bias=self.use_bias))

        return model

    # We use leaky ReLUs as inspired by DCGAN
    def _conv_encoder(self, input_shape, code_dim=16):
        inputs = tf.keras.layers.Input(shape=input_shape[1:])

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        z = tf.keras.layers.Conv2D(filters=code_dim if self.code_dim_override is None else self.code_dim_override, kernel_size=3, strides=2, padding='valid', name="code")(x)

        model = tf.keras.Model(inputs=inputs, outputs=z, name="encoder")
        return model

    def _conv_decoder(self, input_shape, output_dim):
        encoded = tf.keras.Input(shape=input_shape[1:])

        x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=4, strides=2, padding='same')(encoded)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if self.hidden_activation == "leakyrelu":
            x = tf.keras.layers.LeakyReLU(.01)(x)
        else:
            x = tf.keras.layers.Activation(self.hidden_activation)(x)

        decoded = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=self.out_activation)(x)
        decoder = tf.keras.Model(inputs=encoded, outputs=decoded, name="decoder")
        return decoder

    def _dense_encoder(self, input_shape):
        model = tf.keras.Sequential(name="encoder")

        add_dense(
            model, layer_dims=self.layer_dims[:-1], p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:],
        )
        # We add the last layer manually to name it accordingly
        model.add(tf.keras.layers.Dense(self.layer_dims[-1], name="code"))

        return model

    def _dense_decoder(self, input_shape, output_dim):
        model = tf.keras.Sequential(name="decoder")

        add_dense(
            model, layer_dims=list(reversed(self.layer_dims[:-1])), p_dropout=self.p_dropout,
            activation=self.hidden_activation, input_shape=input_shape[1:]
        )
        # The last layer reconstructs the input
        model.add(tf.keras.layers.Dense(
            output_dim[-1], activation=self.out_activation
        ))

        return model

    @staticmethod
    def recon_loss(y_true, y_pred, loss_f=tf.keras.losses.BinaryCrossentropy(from_logits=False)):
        return loss_f(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def disc_loss(y_real, y_fake, loss_f=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
        # Real => 1
        # real_loss = loss_f(
        #     # tf.random.normal(tf.shape(y_real), mean=1.0, stddev=0.01),
        #     y_true=tf.ones_like(y_real),
        #     y_pred=y_real
        # )
        real_loss = - tf.reduce_mean(y_real)

        # Fake => 0
        # fake_loss = loss_f(
        #     # tf.random.normal(tf.shape(y_fake), mean=0.0, stddev=0.01),
        #     y_true=tf.zeros_like(y_fake),
        #     y_pred=y_fake
        # )
        fake_loss = tf.reduce_mean(y_fake)

        return real_loss + fake_loss

    @staticmethod
    def enc_loss(y_fake, loss_f=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
        # Fake => Real = 1
        # fake_loss = loss_f(
        #     # tf.random.normal(tf.shape(y_fake), mean=1.0, stddev=0.01),
        #     y_true=tf.ones_like(y_fake),
        #     y_pred=y_fake
        # )
        fake_loss = - tf.reduce_mean(y_fake)

        return fake_loss

    # == Keras functions ==
    def compile(self, learning_rate=.0001, **kwargs):
        super(AdversarialAutoencoder, self).compile(learning_rate=learning_rate, **kwargs)

        self.recon_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.disc_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.dec_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def build(self, input_shape):
        super(AdversarialAutoencoder, self).build(input_shape)

        # Additionally, we need to build the discriminator
        self.m_disc = self.get_discriminator(tf.keras.layers.Flatten()(self.m_enc.output).shape)

    @tf.function
    def train_step(self, data):
        x_train = data[0]
        y_train = data[1]
        batch_size = x_train.shape[0]

        # 1) Overall AE should reconstruct the input samples
        with tf.GradientTape() as recon_tape:
            y_pred = self(x_train, training=True)
            recon_loss = self.recon_loss(y_true=x_train, y_pred=y_pred)

        recon_grad = recon_tape.gradient(recon_loss, self.m_enc.trainable_variables + self.m_dec.trainable_variables)
        self.recon_opt.apply_gradients(zip(recon_grad, self.m_enc.trainable_variables + self.m_dec.trainable_variables))

        # 2) Train the discriminator
        with tf.GradientTape() as disc_tape:
            # Get the code layer's reaction on the input
            t_code_x = self.m_enc(x_train, training=False)
            t_code_x = tf.keras.layers.Flatten()(t_code_x)
            t_code_noise = tf.random.normal(tf.shape(t_code_x))

            # Ask the discriminator what's real and what's fake
            t_disc_real = self.m_disc(t_code_noise, training=True)
            t_disc_fake = self.m_disc(t_code_x, training=True)

            disc_loss = self.disc_loss(y_real=t_disc_real, y_fake=t_disc_fake)

        disc_grad = disc_tape.gradient(disc_loss, self.m_disc.trainable_weights)
        if self.clipping_val:
            disc_grad, _ = tf.clip_by_global_norm(disc_grad, self.clipping_val)
        self.disc_opt.apply_gradients(zip(disc_grad, self.m_disc.trainable_weights))

        # 3) Train the encoder
        with tf.GradientTape() as dec_tape:
            # The code layer should look like "real" samples
            t_code_x = self.m_enc(x_train, training=True)
            t_code_x = tf.keras.layers.Flatten()(t_code_x)
            t_y_pred = self.m_disc(t_code_x, training=False)

            dec_loss = self.enc_loss(t_y_pred)

        dec_grad = dec_tape.gradient(dec_loss, self.m_enc.trainable_variables)
        if self.clipping_val:
            dec_grad, _ = tf.clip_by_global_norm(dec_grad, self.clipping_val)
        self.dec_opt.apply_gradients(zip(dec_grad, self.m_enc.trainable_variables))

        return {
            "Reconstruction Loss": recon_loss,
            "Discriminator Loss": disc_loss,
            "Encoder Loss": dec_loss
        }

    def call(self, inputs, training=None, mask=None):
        # Connect the encoder and decoder
        t_encoded = self.m_enc(inputs, training=training, mask=mask)
        t_decoded = self.m_dec(t_encoded, training=training, mask=mask)

        return t_decoded
