import tensorflow as tf
import tensorflow.keras as keras

from libs.architecture.target import Autoencoder


class fAnoGAN(keras.Model):
    def __init__(
            self, base_ae: Autoencoder,
            kappa=1.0, gen_dim: int = 8,
            name="fAnoGAN"
    ):
        """
        Implementation of the paper "f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks"
        originally by Schlegl et al.
        :param base_ae: autoencoder used to derive the architecture of AnoGAN
        :param kappa: weighting factor between the reconstruction and discriminator loss
        :param gen_dim: input dimension of the generator
        :param name: name of the AD method
        """
        super(fAnoGAN, self).__init__(name=name)

        # Configuration
        self.base_ae = base_ae
        self.kappa = kappa
        self.gen_dim = gen_dim

        # Optimisers
        self.opt_gen: keras.optimizers.Optimizer = None
        self.opt_disc: keras.optimizers.Optimizer = None
        self.opt_enc: keras.optimizers.Optimizer = None

        # Losses
        self.loss_mse = keras.losses.MeanSquaredError()

        # Models
        self.m_gen: keras.Model = None
        self.m_disc: keras.Model = None
        self.m_disc_inter: keras.Model = None
        self.m_enc: keras.Model = None

    def build(self, input_shape):

        # Our generator is the encoder plus an initial dense layer
        in_gen = keras.layers.Input((self.gen_dim, ))
        gen_tail = keras.models.clone_model(self.base_ae.m_dec)
        input_dim = gen_tail.input_shape[-1] if len(gen_tail.input_shape) == 2 \
            else gen_tail.input_shape[-1] * gen_tail.input_shape[-2] * gen_tail.input_shape[-3]
        gen_head = keras.layers.Dense(input_dim)
        gen_reshape = keras.layers.Reshape(gen_tail.input_shape[1:])
        self.m_gen = keras.Model(
            in_gen, gen_tail(gen_reshape(gen_head(in_gen)))
        )

        # The encoder is the opposite: takes the image and outputs our latent dimension
        m_enc = keras.models.clone_model(self.base_ae.m_enc)
        m_enc_flat = keras.layers.Flatten()
        m_enc_out = keras.layers.Dense(self.gen_dim)
        self.m_enc = keras.Model(
            m_enc.inputs, m_enc_out(m_enc_flat(m_enc(m_enc.inputs)))
        )

        # The discriminator will be similar to the encoder, but with a binary output
        m_disc = keras.models.clone_model(self.base_ae.m_enc)
        m_disc_flat = keras.layers.Flatten()
        m_disc_out = keras.layers.Dense(1, activation="sigmoid")
        self.m_disc = keras.Model(
            m_disc.inputs, m_disc_out(m_disc_flat(m_disc(m_disc.inputs)))
        )
        # ... without the final layer, it'll be our intermediate layer representation
        self.m_disc_inter = keras.Model(
            m_disc.inputs, m_disc(m_disc.inputs)
        )

    def compile(self, learning_rate: float = 1e-4):
        # We stick to adam
        self.opt_gen = keras.optimizers.Adam(learning_rate)
        self.opt_disc = keras.optimizers.Adam(learning_rate)
        self.opt_enc = keras.optimizers.Adam(learning_rate)
        
        super(fAnoGAN, self).compile(optimizer=keras.optimizers.Adam(learning_rate))

    @tf.function
    def train_step_gan(self, data):
        # Useful constants
        batch_size = tf.shape(data)[0]
        gen_shape = (batch_size, self.gen_dim)

        # 1) Train the GAN
        z_gen = tf.random.normal(gen_shape)

        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
            # Generate new images
            x_gen = self.m_gen(z_gen, training=True)

            # See what the discriminator is thinking
            y_real = self.m_disc(data, training=True)
            y_fake = self.m_disc(x_gen, training=True)

            # Generated samples should become more realistic
            loss_gen = tf.reduce_mean(y_fake)
            # Discriminator should become better in distinguishing between real and fake
            loss_disc = tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)

        # Calculate the gradients
        grad_gen = tape_gen.gradient(loss_gen, self.m_gen.trainable_weights)
        grad_disc = tape_disc.gradient(loss_disc, self.m_disc.trainable_weights)

        # Apply the gradients
        self.opt_gen.apply_gradients(zip(grad_gen, self.m_gen.trainable_weights))
        self.opt_disc.apply_gradients(zip(grad_disc, self.m_disc.trainable_weights))

        # Return the losses
        return {
            "Generator": loss_gen, "Discriminator": loss_disc,
        }

    @tf.function
    def train_step_enc(self, data):
        # 2) Train the encoder
        with tf.GradientTape() as tape_enc:
            # Encode
            z_in = self.m_enc(data, training=True)
            # Pass to the generator
            x_gen = self.m_gen(z_in, training=False)

            # Get the encoder's output
            f_gen = self.m_disc_inter(x_gen, training=False)
            f_in = self.m_disc_inter(data, training=False)

            # Calculate the losses
            loss_izi = self.loss_mse(data, x_gen)
            loss_inter = self.loss_mse(f_in, f_gen)
            loss_izif = loss_izi + self.kappa * loss_inter

        # Calculate the gradients
        grad_enc = tape_enc.gradient(loss_izif, self.m_enc.trainable_weights)

        # Apply the gradients
        self.opt_enc.apply_gradients(zip(grad_enc, self.m_enc.trainable_weights))

        # Return the losses
        return {
            "IZI": loss_izi, "Intermediate": loss_inter
        }

    @staticmethod
    def batch_mse(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]

        # MSE, but the batch dimension persists
        this_mse = tf.square(y_true - y_pred)
        this_mse = tf.reshape(this_mse, (batch_size, -1))
        this_mse = tf.reduce_mean(this_mse, axis=-1)

        return this_mse

    def anomaly_score(self, x_in, x_gen, f_in, f_gen):
        # Calculate the losses
        loss_izi = self.batch_mse(x_in, x_gen)
        loss_inter = self.batch_mse(f_in, f_gen)

        return loss_izi + self.kappa * loss_inter

    def call(self, inputs, training=None, mask=None):
        # Return all the outputs necessary to calculate the final loss
        z_in = self.m_enc(inputs, training=training)
        x_gen = self.m_gen(z_in, training=training)

        # Get the encoder's output
        f_in = self.m_disc_inter(inputs, training=training)
        f_gen = self.m_disc_inter(x_gen, training=training)

        return x_gen, f_in, f_gen

    def fit(
        self, x, y=None, validation_data=None, epochs=100, batch_size=128, verbose=None, **fit_params
    ):

        # Convert the ndarrays to a TF data set
        data_set = tf.data.Dataset.from_tensor_slices((x, y))
        data_set = data_set.shuffle(10*batch_size)
        batched_data = data_set.batch(batch_size)

        # 1) Train the GAN
        print("Training the GAN")
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")

            all_losses = {}
            for step, samples in enumerate(batched_data):
                all_losses = self.train_step_gan(samples[0], **fit_params)

            print(self.parse_losses(all_losses))

        # 2) Train the encoder
        print("Training the encoder")
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")

            all_losses = {}
            for step, samples in enumerate(batched_data):
                all_losses = self.train_step_enc(samples[0], **fit_params)

            print(self.parse_losses(all_losses))

    @staticmethod
    def parse_losses(loss_dict: dict) -> str:
        # Turn all losses to numpy
        all_losses = {cur_key: cur_val.numpy() for cur_key, cur_val in loss_dict.items()}
        # Show them as tabbed grid
        out_str = ""
        for cur_key, cur_val in all_losses.items():
            out_str += f"{cur_key}: {cur_val:.4f}\t"

        return out_str

