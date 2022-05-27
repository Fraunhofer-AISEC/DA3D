import tensorflow as tf

from typing import List, Iterable, Union

def add_dense(
        network: tf.keras.Sequential, layer_dims: Iterable[int], input_shape=None,
        activation: str = "relu", kernel_initialiser: str = None,
        first_l1: float = 0.0, first_l2: float = 0.0, p_dropout: float = None, first_name: str = None,
        add_batch_norm: bool = False, use_bias: bool = True,
        leaky_slope: float = .01, *args, **kwargs
):
    """
    Build a dense model with the given hidden state
    :param network: sequential Keras network
    :param layer_dims: list of hidden state dimensions
    :param input_shape: shape of the input
    :param activation: activation of the layers
    :param first_l1: L1 kernel regulariser on the first layer
    :param first_l2: L2 kernel regulariser on the first layer
    :param p_dropout: dropout percentage after the first layer
    :param first_name: name of the first layer
    :param add_batch_norm: add batch normalisation after each layer
    :param use_bias: add bias vector to the output
    :param leaky_slope: slope of the negative LeakyReLU side - only applicable when activation==leakyrelu
    :param args: passed to Keras dense layer
    :param kwargs: passed to Keras dense layer
    :return:
    """

    # First layer
    input_kwarg = {} if input_shape is None else {"input_shape": input_shape}
    network.add(tf.keras.layers.Dense(
        layer_dims[0], **input_kwarg,
        name=first_name,
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
        bias_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
        kernel_initializer=kernel_initialiser, use_bias=use_bias
    ))
    if add_batch_norm:
        network.add(tf.keras.layers.BatchNormalization())

    if activation == "leakyrelu":
        network.add(tf.keras.layers.LeakyReLU(leaky_slope))
    else:
        network.add(tf.keras.layers.Activation(activation))

    if p_dropout:
        network.add(tf.keras.layers.Dropout(p_dropout))

    # Middle layers
    for cur_dim in layer_dims[1:]:
        network.add(tf.keras.layers.Dense(cur_dim, kernel_initializer=kernel_initialiser, use_bias=use_bias, *args, **kwargs))

        if add_batch_norm:
            network.add(tf.keras.layers.BatchNormalization())

        if activation == "leakyrelu":
            network.add(tf.keras.layers.LeakyReLU(leaky_slope))
        else:
            network.add(tf.keras.layers.Activation(activation))

        if p_dropout:
            network.add(tf.keras.layers.Dropout(p_dropout))


def add_symmetric_autoencoder(
        network: tf.keras.Sequential, layer_dims: List[int], input_shape=None, activation="relu", *args, **kwargs
) -> tf.keras.Sequential:
    """
    Build autoencoder where the hidden state dimensions of the en- and decoder are the same
    :param network: sequential Keras network
    :param layer_dims: list of hidden state dimensions to the input
    :param input_shape: shape of the input
    :param activation: activation of the layers
    :param args: passed to add_dense
    :param kwargs: passed to add_dense
    :return: sequential Keras autoencoder model
    """

    # Encoder
    add_dense(
        network, layer_dims=layer_dims[:-1], input_shape=input_shape, activation=activation, *args, **kwargs
    )
    # Code
    network.add(tf.keras.layers.Dense(
        layer_dims[-1], name="code",
    ))
    network.add(tf.keras.layers.Activation(activation))
    # Decoder
    add_dense(
        network, layer_dims=list(reversed(layer_dims[:-1])), activation=activation, first_name="code_in",
        *args, **kwargs
    )


def parse_shape(in_shape: tuple, allow_single: bool = True) -> tuple:
    """
    Parse the shape information, e.g. remove the batch size
    :param in_shape: raw shape, e.g. from a Keras layer
    :param allow_single: allow single-valued dimensions
    :return: parsed shape usable as input
    """
    new_shape = in_shape

    # Get rid of the maybe undefined first dimension
    if new_shape[0] is None:
        new_shape = new_shape[1:]

    # Don't allow single-valued dimensions
    if not allow_single:
        new_shape = tuple(cur_el for cur_el in new_shape if cur_el != 1)

    # Extend to a proper tuple
    if len(new_shape) == 1:
        new_shape = (new_shape[0],)

    return new_shape
