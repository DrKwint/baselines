import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False, embed_constraint_state=True):
        constraint_placeholders = []
        if type(input_placeholder) is list:
            constraint_placeholders = input_placeholder[1:]
            input_placeholder = input_placeholder[0]
        with tf.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                # concat constraint state
                if constraint_placeholders != []:
                    if embed_constraint_state:
                        sizes = [x.get_shape().as_list()[-1] for x in constraint_placeholders]
                        embedding_sizes = [max(int(np.log2(x)), 2) for x in sizes]
                        state_embeddings = [tf.get_variable('embedding_{}'.format(i), shape=(orig, size)) for (i, (orig, size)) in enumerate(zip(sizes, embedding_sizes))]
                        constraint_values = [tf.nn.embedding_lookup(state_embeddings, tf.argmax(x, axis=1)) for x in constraint_placeholders]
                    else:
                        constraint_values = constraint_placeholders

                    action_out = tf.concat([action_out] + constraint_values, axis=-1)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    if constraint_placeholders != []:
                        state_out = tf.concat([state_out] + constraint_placeholders, axis=-1)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder
