import tensorflow as tf


class RhINN(tf.keras.Model):
    """Fully connected network (t, amplitude) -> response, with the sparse
    library coefficients ``lambda1`` stored as an extra trainable variable."""

    def __init__(self, lb, ub, n_terms, num_hidden_layers=5,
                 num_neurons_per_layer=40, activation="tanh",
                 kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.lambda1 = tf.Variable(tf.ones([n_terms, 1], dtype=tf.float32),
                                   trainable=True, name="lambda1")
        self.input_scale = tf.keras.layers.Lambda(lambda x: (x - lb) / (ub - lb))
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                                             activation=activation,
                                             kernel_initializer=kernel_initializer)
                       for _ in range(num_hidden_layers)]
        self.out = tf.keras.layers.Dense(1)

    def call(self, X):
        Z = self.input_scale(X)
        for layer in self.hidden:
            Z = layer(Z)
        return self.out(Z)

    @property
    def network_variables(self):
        """Network weights only (lambda1 excluded); trained between STRidge steps."""
        return [v for v in self.trainable_variables if v is not self.lambda1]
