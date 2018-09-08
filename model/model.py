import tensorflow as tf

def modeler(hidden_units,
            keep_rate,
            num_layers):
    def _get_lstm_cell(hidden_units, keep_rate):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_rate)
        return cell
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[
        _get_lstm_cell(hidden_units, keep_rate) for _ in range(num_layers)],
        state_is_tuple=True)
    return cell

def model_fn(features, labels, mode, params):
    word_embeddings = tf.get_variable("word_embeddings",
                                      [params['vocab_size'],
                                       params['embedding_size']])
    embedded_words = tf.nn.embedding_lookup(word_embeddings, features['word_id'])
    rnn_cell = modeler(params['hidden_units'],
                       params['keep_rate'],
                       params['num_layers'])

    outputs, _ = tf.nn.dynamic_rnn(
        rnn_cell,
        inputs=embedded_words,
        sequence_length=features['batch_length'],
        dtype=tf.float32
    )

    output_embedding_W = tf.get_variable('output_embedding_W',
                                         [params['hidden_units'],
                                          params['vocab_size']])
    output_embedding_b = tf.get_variable('output_embedding_b', [params['vocab_size']])

    def output_embedding(current_logits):
        return tf.add(
                tf.matmul(
                    current_logits,
                    output_embedding_W
                ), output_embedding_b)

    logits_ = tf.map_fn(output_embedding, outputs)
    logits = tf.reshape(logits_, [-1, params['vocab_size']])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'preds': tf.expand_dims(tf.argmax(logits_, -1), 0)
        }
        spec = tf.estimator.EstimatorSpec(mode, predictions)
    else:
        non_zero_weights = tf.sign(features['word_id'])
        cost = (tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(labels, [-1]), logits=logits) *
            tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32))
        loss = tf.reduce_sum(cost)

        trainable_params = tf.trainable_variables()
        optimizer = tf.train.AdagradOptimizer(params['lr'])
        gradients = tf.gradients(loss, trainable_params,
                                 colocate_gradients_with_ops=True)
        clipped_grads, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
        train_op = optimizer.apply_gradients(
            zip(clipped_grads, trainable_params),
            global_step=tf.train.get_global_step())

        spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return spec
