import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Union, Dict, Tuple
from pathlib import Path
from matplotlib import pyplot as plt

from libs.architecture.target import AdversarialAutoencoder
from libs.network.network import add_dense
from libs.constants import BASE_PATH


class DA3D(tf.keras.Model):
    def __init__(
            self, layer_dims: tuple, m_target: AdversarialAutoencoder = None, name="DA3D",
            hidden_activation: str = "leakyrelu", p_dropout: float = .1,
            sample_stddev: float = 2.0, simple_stddev: float = None, act_loc: str = "dec",
            gen_in_dim: int = 8, gen_hidden_dims: Tuple[int] = (50, 40, 30, 20),
            clip_val: float = .01, r_norm: float = None
    ):
        """
        Double-Adversarial Activation Anomaly Detection
        :param layer_dims: layer dimensions of the alarm network
        :param m_target: pretrained AAE
        :param name: name of the model
        :param hidden_activation: activation function of the alarm network
        :param p_dropout: drop between each alarm network layer
        :param sample_stddev: standard deviation for latent space anomalies
        :param simple_stddev: different standard deviation for simple anomalies, of None use sample_stddev
        :param act_loc: location of the extracted activations, may be "enc", "dec" or "all"
        :param gen_in_dim: input dimension of the anomaly generator
        :param gen_hidden_dims: layer dimensions of the generator network
        :param clip_val: clipping value on the gradient while training the Wasserstein GAN
        :param r_norm: mark samples outside this radius (measured in the AAE's code layer) as anomalous
        """
        super(DA3D, self).__init__(name=name)

        # Config
        self.layer_dims = layer_dims
        self.hidden_activation = hidden_activation
        self.p_dropout = p_dropout
        self.sample_stddev = sample_stddev
        self.simple_stddev = sample_stddev if simple_stddev is None else simple_stddev
        self.act_loc = act_loc

        self.gen_in_dim = gen_in_dim
        self.gen_hidden_dims = gen_hidden_dims
        self.dim_gen_in = None
        self.dim_gen_out = None
        self.clip_val = clip_val
        self.r_norm = r_norm

        # Network components
        self.m_aae: AdversarialAutoencoder = None
        self.m_critic: tf.keras.Model = None
        self.m_gen: tf.keras.Model = None
        self.m_alarm: tf.keras.Model = None
        if m_target is not None: self.add_target(m_target)

        # Losses
        self.loss_critic = None
        self.loss_alarm = None

        # Optimiser
        self.opt_critic: tf.keras.optimizers.Optimizer = None
        self.opt_gen: tf.keras.optimizers.Optimizer = None
        self.opt_alarm: tf.keras.optimizers.Optimizer = None

    # == Helper functions ==
    def add_target(self, m_target: AdversarialAutoencoder):
        self.m_aae = m_target

    # == Keras functions ==
    def compile(
            self,
            learning_rate=0.0001,
            loss_critic=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            loss_alarm=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            **kwargs
    ):
        super(DA3D, self).compile(**kwargs)

        self.loss_critic = loss_critic
        self.loss_alarm = loss_alarm

        # We'll use adam as default optimiser
        self.opt_critic = tf.keras.optimizers.Adam(learning_rate)
        self.opt_gen = tf.keras.optimizers.Adam(learning_rate)
        self.opt_alarm = tf.keras.optimizers.Adam(learning_rate)

    def build(self, input_shape):

        # Build the alarm network from the given dimensions
        m_alarm = tf.keras.models.Sequential(name="Alarm")
        add_dense(m_alarm, layer_dims=self.layer_dims, activation=self.hidden_activation, p_dropout=self.p_dropout)
        m_alarm.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.m_alarm = m_alarm

        # Build the generator
        m_gen = tf.keras.models.Sequential(name="DA3D-generator")
        add_dense(
            m_gen, layer_dims=self.gen_hidden_dims, input_shape=(self.gen_in_dim, ),
            activation=self.hidden_activation, p_dropout=self.p_dropout
        )
        m_gen.add(tf.keras.layers.Dense(self.m_aae.m_enc.output_shape[-1]))
        # The CNN needs some reshaping
        m_gen.add(tf.keras.layers.Reshape((self.m_aae.m_enc.output_shape[1:])))
        self.m_gen = m_gen
        self.dim_gen_in = self.m_gen.input_shape[1:]
        self.dim_gen_out = self.m_gen.output_shape[1:]

        # The critic takes the flattened version as the input
        m_critic = tf.keras.models.clone_model(self.m_aae.m_disc)
        in_critic = tf.keras.layers.Input(self.dim_gen_out)
        # Prepend a flattening layer
        self.m_critic = tf.keras.Model(
            in_critic, m_critic(tf.keras.layers.Flatten()(in_critic))
        )

    @tf.function
    def pred_on_gen(self, batch_size: int, train_alarm: bool = False, train_gen: bool = False):
        # Generate fresh anomalies
        h_anom = self.m_gen(tf.random.normal((batch_size, ) + self.dim_gen_in), training=train_gen)
        x_anom = self.m_aae.m_dec(h_anom, training=False)

        # Predict based on the hidden activations
        act_dec_anom = self.m_aae.m_dec_act_on_code(h_anom, training=False)
        act_enc_anom = self.m_aae.m_enc_act(x_anom, training=False)
        # Use the respective activations
        if self.act_loc == "enc":
            act_anom = act_enc_anom
        elif self.act_loc == "dec":
            act_anom = act_dec_anom
        elif self.act_loc == "all":
            act_anom = tf.concat([act_enc_anom, act_dec_anom], axis=1)
        else:
            raise AttributeError("Activations may only be 'enc', 'dec' or 'all'")
        y_pred_anom = self.m_alarm(act_anom, training=train_alarm)

        return y_pred_anom

    @tf.function
    def train_step(
            self, data,
            train_alarm: bool = True, train_generator: bool = True, train_critic: bool = True,
            w_norm: float = 1.0, w_generated: float = 1.0, w_trivial: float = 1.0, w_simple: float = 0.0,
            w_crit_gauss: float = 1.0, w_crit_norm: float = 2.0, w_crit_anom: float = 2.0
    ):

        # Extract the data
        x_norm = data[0]
        h_norm = self.m_aae.m_enc(x_norm, training=False)
        y_norm = data[1]
        batch_size = tf.shape(y_norm)[0]
        # Trivial anomalies
        x_triv = tf.random.normal(tf.shape(x_norm), mean=0.5, stddev=1.0)
        y_triv = tf.ones_like(y_norm)

        # 1) Train the alarm network
        with tf.GradientTape() as tape_alarm:
            # Normal data
            y_alarm_norm = self(x_norm, training=True)
            loss_alarm_norm = self.loss_alarm(y_true=y_norm, y_pred=y_alarm_norm)

            # Trivial anomalies
            y_pred_triv = self(x_triv, training=True)
            loss_alarm_triv = self.loss_alarm(y_true=y_triv, y_pred=y_pred_triv)

            # Simple anomalies
            h_simp = tf.random.normal(tf.shape(h_norm), mean=0.0, stddev=self.simple_stddev)
            act_simp_dec = self.m_aae.m_dec_act_on_code(h_simp, training=False)
            x_simp = self.m_aae.m_dec(h_simp, training=False)
            act_simp_enc = self.m_aae.m_enc_act(x_simp, training=False)
            if self.act_loc == "enc":
                act_simp = act_simp_enc
            elif self.act_loc == "dec":
                act_simp = act_simp_dec
            elif self.act_loc == "all":
                act_simp = tf.concat([act_simp_enc, act_simp_dec], axis=1)
            y_pred_simp = self.m_alarm(act_simp, training=True)
            loss_alarm_simp = self.loss_alarm(y_true=y_triv, y_pred=y_pred_simp)

            # Generated anomalies
            y_pred_gen = self.pred_on_gen(batch_size, train_alarm=True)
            loss_alarm_gen = self.loss_alarm(y_true=y_triv, y_pred=y_pred_gen)

            # Losses
            loss_alarm_tot = w_norm * loss_alarm_norm \
                    + w_generated * loss_alarm_gen \
                    + w_trivial * loss_alarm_triv \
                    + w_simple * loss_alarm_simp

        # Backpropagate the gradient to the alarm network
        grad_alarm = tape_alarm.gradient(loss_alarm_tot, self.m_alarm.trainable_weights)
        if train_alarm:
            self.opt_alarm.apply_gradients(zip(grad_alarm, self.m_alarm.trainable_weights))

        # 2) Train the generator
        with tf.GradientTape() as tape_gen:
            # Fool the alarm network
            y_pred_anom = self.pred_on_gen(batch_size, train_gen=True)
            loss_anom = tf.reduce_mean(y_pred_anom)

            # Fool the critic
            h_anom = self.m_gen(tf.random.normal((batch_size, ) + self.dim_gen_in), training=True)
            y_pred_crit = self.m_critic(h_anom, training=False)
            loss_crit = tf.reduce_mean(y_pred_crit)

            loss_gen_tot = loss_anom + loss_crit

        grad_gen = tape_gen.gradient(loss_gen_tot, self.m_gen.trainable_weights)
        if self.clip_val:
            grad_gen, _ = tf.clip_by_global_norm(grad_gen, self.clip_val)
        if train_generator:
            self.opt_gen.apply_gradients(zip(grad_gen, self.m_gen.trainable_weights))

        # 3) Train the critic
        with tf.GradientTape() as tape_crit:
            # Avoid normal
            y_crit_norm = self.m_critic(h_norm, training=True)
            loss_crit_norm = - tf.reduce_mean(y_crit_norm)

            # Incentivise exploration
            h_anom = self.m_gen(tf.random.normal((batch_size, ) + self.dim_gen_in), training=False)
            y_crit_anom = self.m_critic(h_anom, training=True)
            loss_crit_anom = - tf.reduce_mean(y_crit_anom)

            # Stay in boundaries
            h_gauss = tf.random.normal(tf.shape(h_norm), mean=0.0, stddev=self.sample_stddev)
            y_crit_gauss = self.m_critic(h_gauss, training=True)
            loss_crit_gauss = tf.reduce_mean(y_crit_gauss)

            loss_crit_tot = w_crit_gauss*loss_crit_gauss \
                            + w_crit_norm*loss_crit_norm \
                            + w_crit_anom*loss_crit_anom

        grad_crit = tape_crit.gradient(loss_crit_tot, self.m_critic.trainable_weights)
        if self.clip_val:
            grad_crit, _ = tf.clip_by_global_norm(grad_crit, self.clip_val)
        if train_critic:
            self.opt_critic.apply_gradients(zip(grad_crit, self.m_critic.trainable_weights))

        # Only return losses on what we trained
        all_losses = {}
        if train_alarm:
            all_losses["Alarm"] = loss_alarm_tot
        if train_critic:
            all_losses["Critic"] = loss_crit_tot
        if train_generator:
            all_losses["Generator"] = loss_gen_tot
        return all_losses

    def filter_data(self, x_in: np.ndarray, y_in: np.ndarray):
        if self.r_norm is None:
            return x_in, y_in

        # Filter the data based on its radius in the code layer
        x_code = self.m_aae.m_enc.predict(x_in)
        x_code = np.reshape(x_code, (x_code.shape[0], -1))
        r_code = np.square(x_code)
        r_code = np.sum(r_code, axis=-1)
        r_code = np.sqrt(r_code)
        y_adapted = y_in.copy()
        y_adapted[r_code > self.r_norm, :] = 1

        x_norm = x_in[y_adapted[:, 0] == 0, :]
        y_norm = y_adapted[y_adapted[:, 0] == 0, :]

        return x_norm, y_norm

    def fit(
            self, x, y, validation_data=None, plot_freq: int = None,
            pretrain=0, epochs=500, posttrain=0, batch_size=128, verbose=None, **fit_params
    ):

        # DA3D only considers normal data so far
        x_norm, y_norm = self.filter_data(x, y)

        # Convert the ndarrays to a TF data set
        data_set = tf.data.Dataset.from_tensor_slices((x_norm, y_norm))
        data_set = data_set.shuffle(10*batch_size)
        batched_data = data_set.batch(batch_size)

        # The fit parameters may contain "train_on_generated", which is turned off in the pretraining phase
        _fit_params = {
            cur_key: cur_val for cur_key, cur_val in fit_params.items() if cur_key != "w_generated"
        }

        # Train and print the losses
        for epoch in range(1, pretrain+epochs+posttrain+1):
            print(f"Epoch {epoch}/{pretrain+epochs+posttrain}")

            all_losses = {}
            for step, samples in enumerate(batched_data):
                # During pretraining, train on the normal samples and noise only
                if epoch < pretrain:
                    all_losses = self.train_step(samples, w_generated=0, **_fit_params)
                # During posttraining, keep the generator fixed
                if epoch > (epochs + pretrain):
                    all_losses = self.train_step(samples, **fit_params)
                # Otherwise train on everything
                else:
                    all_losses = self.train_step(samples, **fit_params)

            print(self.parse_losses(all_losses))

            # Plot
            if plot_freq and epoch % plot_freq == 0:
                print("Plotting...")
                self.visualise_generator(x_train=x, epoch=epoch)

            # TODO: Return val score

    @staticmethod
    def parse_losses(loss_dict: dict) -> str:
        # Turn all losses to numpy
        all_losses = {cur_key: cur_val.numpy() for cur_key, cur_val in loss_dict.items()}
        # Show them as tabbed grid
        out_str = ""
        for cur_key, cur_val in all_losses.items():
            out_str += f"{cur_key}: {cur_val:.4f}\t"

        return out_str

    def visualise_generator(
            self, x_train: np.ndarray, epoch: int, base_path: Path = BASE_PATH / "images",
            n_gen: int = 500, alpha: float = .1, xy_range: int = 6
    ):

        # Sample the latent space
        h_train = self.m_aae.m_enc.predict(x_train)[:n_gen, :]
        h_gen = self.m_gen.predict(tf.random.normal((n_gen, ) + self.dim_gen_in))
        h_trivial = self.m_aae.m_enc.predict(tf.random.normal((n_gen, ) + x_train.shape[1:], mean=0.5, stddev=1.0))
        h_simple = tf.random.normal(tf.shape(h_gen), mean=0.0, stddev=self.simple_stddev)

        # Look what this corresponds to in the image space - take the very first sample only
        x_gen = self.m_aae.m_dec.predict(h_gen)[0, :, :, 0]
        x_trivial = self.m_aae.m_dec.predict(h_trivial)[0, :, :, 0]
        x_simple = self.m_aae.m_dec.predict(h_simple)[0, :, :, 0]

        # Plot the image space
        for cur_name, cur_img in zip(["gen", "trivial", "simple"], [x_gen, x_trivial, x_simple]):
            plt.clf()
            plt.imshow(cur_img, cmap="gray")
            # https://www.delftstack.com/howto/matplotlib/hide-axis-borders-and-white-spaces-in-matplotlib/
            plt.axis("off")
            plt.savefig(base_path / f"{self.name}_{cur_name}_ep_{epoch}.png", bbox_inches="tight", pad_inches=0)

        # We might have weird dimensions in CNNs
        h_train = np.reshape(h_train, (h_train.shape[0], -1))
        h_gen = np.reshape(h_gen, (n_gen, -1))
        h_trivial = np.reshape(h_trivial, (n_gen, -1))
        h_simple = np.reshape(h_simple, (n_gen, -1))

        # Plot the latent space
        plt.clf()
        plt.scatter(x=h_train[:, 0], y=h_train[:, 1], c="green", label="Training")
        plt.scatter(x=h_gen[:, 0], y=h_gen[:, 1], c="red", label="Generated", alpha=alpha)
        plt.scatter(x=h_trivial[:, 0], y=h_trivial[:, 1], c="yellow", label="Trivial", alpha=alpha)
        plt.scatter(x=h_simple[:, 0], y=h_simple[:, 1], c="orange", label="Simple", alpha=alpha)
        # plt.legend()
        plt.xlim(-xy_range, xy_range)
        plt.ylim(-xy_range, xy_range)
        plt.axis("off")
        plt.savefig(base_path / f"{self.name}_latent_ep_{epoch}.png", bbox_inches="tight", pad_inches=0)

        # For LaTeX/PGF: table of the values from above
        df_scatter = pd.DataFrame(
            {
                "x_train": h_train[:, 0], "y_train": h_train[:, 1],
                "x_gen": h_gen[:, 0], "y_gen": h_gen[:, 1],
                "x_trivial": h_trivial[:, 0], "y_trivial": h_trivial[:, 1],
                "x_simple": h_simple[:, 0], "y_simple": h_simple[:, 1],
            }
        )
        df_scatter.to_csv(base_path / f"{self.name}_latent_ep_{epoch}.csv", index=False)

    def call(self, inputs, training=None, mask=None):

        # Extract the AAE's activations
        if self.act_loc == "enc":
            t_all_act = self.m_aae.m_enc_act(inputs, training=False)
        elif self.act_loc == "dec":
            t_all_act = self.m_aae.m_dec_act(inputs, training=False)
        elif self.act_loc == "all":
            t_all_act = self.m_aae.m_all_act(inputs, training=False)
        else:
            raise AttributeError("Activations may only be 'enc', 'dec' or 'all'")

        # Pass them through the alarm network
        y_pred = self.m_alarm(t_all_act, training=training, mask=mask)

        return y_pred

    def get_config(self):
        config = {
            "layer_dims": self.layer_dims,
            "hidden_activation": self.hidden_activation,
            "p_dropout": self.p_dropout,
            "gen_in_dim": self.gen_in_dim,
            "gen_hidden_dims": self.gen_hidden_dims,
            "clip_val": self.clip_val,
        }

        base_config = super(DA3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
