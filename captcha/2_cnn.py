import tensorflow as tf
import os


# load and preprocess image with given path
def decode_image(path):
    # path example: xxx\abcd.jpg
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # convert to floats in the [0,1] range.
    image = tf.image.resize(image, [60, 120])
    return image

# get labels
def get_labels(path, prefix):
    ''' this function is used within dataset.map(), 
        where eager execution is disables by default:
            check tf.executing_eagerly() returns False.
        So any Tensor.numpy() is not allowed in this function.
    '''
    labels = {}
    for i in range(4):
        c = tf.strings.substr(path, i-8, 1) # path example: b'xxx\abcd.jpg'
        label = tf.strings.unicode_decode(c, input_encoding='utf-8') - ord('a')
        labels[f'{prefix}_{i}'] = label
    return labels
          
# create image/labels dataset from path
def create_dataset_from_path(path_pattern, batch_size, label_prefix): 
    # create path dataset
    # by default, `tf.data.Dataset.list_files` gets filenames 
    # in a non-deterministic random shuffled order
    dataset = tf.data.Dataset.list_files(path_pattern).map(
        lambda image_path: (decode_image(image_path),
                            get_labels(image_path, label_prefix))).batch(batch_size)
    return dataset


# build model
def captcha_cnn(image_shape, n_labels=4, n_class=26, name='captcha', output_label='labels'):
    # input
    image_input = tf.keras.Input(shape=image_shape, name='input_image')

    # conv layer 1
    x = tf.keras.layers.Conv2D(64, (11, 23), padding='same')(image_input)
    x = tf.keras.layers.MaxPool2D((4, 4), padding='same')(x)

    # conv layer 2
    x = tf.keras.layers.Conv2D(32, (5, 11), padding='same')(x)
    x = tf.keras.layers.MaxPool2D((4, 4), padding='same')(x)

    # # conv layer 3
    # x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)

    # # conv layer 4
    # x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)

    # # conv layer 5
    # x = tf.keras.layers.Conv2D(64, (3, 7), padding='same')(x)
    # x = tf.keras.layers.MaxPool2D((2, 2), padding='same')(x)

    # dense layer 1
    x = tf.keras.layers.Flatten()(x) # flatten
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # # dense layer 2
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)

    # outputs
    labels = [tf.keras.layers.Dense(n_class, name=f'{output_label}_{i}')(x) for i in range(n_labels)] 

    # build model
    model = tf.keras.Model(inputs=image_input, outputs=labels, name=name)
    
    return model



if __name__ == '__main__':

    image_shape = (60, 120, 3)
    n_labels = 4
    n_class = 26

    model = captcha_cnn(image_shape, n_labels, n_class, name='captcha_cnn', output_label='labels')
    model.summary()

    # set loss for each output lables, or set a same loss for all labels
    # e.g. loss={ f'labels_{i}': 
    #   tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for i in range(n_labels)}
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    # create (image, label) dataset from generator
    train_ds = create_dataset_from_path(os.path.join('samples/train/*.jpg'), batch_size=128, label_prefix='labels')
    test_ds = create_dataset_from_path(os.path.join('samples/test/*.jpg'), batch_size=128, label_prefix='labels')

    # Write TensorBoard logs to `./tensorboard` directory
    # start visualization: tensorboard --logdir mylogdir
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join('tensorboard')) 
        ]

    # train model
    model.fit(train_ds, epochs=100, callbacks=callbacks)