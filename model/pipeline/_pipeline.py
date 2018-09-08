import tensorflow as tf

def parse(line):
    line_split = tf.string_split([line])
    input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
    output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
    return input_seq, output_seq

def input_fn(filenames,
             batch_size=32,
             buffer_size=2000,
             shuffle=True,
             repeat=None,
             is_train=True,
             padding_val=9999):
    dataset = tf.data.TextLineDataset(filenames).map(parse)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    else:
        dataset = dataset.prefetch(buffer_size)

    dataset = dataset.padded_batch(batch_size,
                                   ([None], [None]), padding_values=(padding_val,padding_val)).repeat(repeat)

    if is_train:
        inputs, labels = dataset.make_one_shot_iterator().get_next()
        return {
            'word_id': inputs,
            'batch_length': tf.multiply(tf.ones(tf.shape(inputs)[0],
                                        dtype=tf.int32), tf.shape(inputs)[-1])
            }, labels
    inputs, _ = dataset.make_one_shot_iterator().get_next()
    return {'word_id': inputs,
            'batch_length': tf.multiply(tf.ones(tf.shape(inputs)[0],
                                        dtype=tf.int32), tf.shape(inputs)[-1])}

def predict_in_fn(in_ids, batch_size=32, buffer_size=2000):
    dataset = tf.data.Dataset.from_tensor_slices((in_ids,))
    dataset = dataset.batch(batch_size).prefetch(buffer_size)
    in_ids = dataset.make_one_shot_iterator().get_next()[0]
    return {
        'word_id': in_ids,
        'batch_length': tf.multiply(tf.ones(tf.shape(in_ids)[0],
                                    dtype=tf.int32), tf.shape(in_ids)[-1])
    }
