import tensorflow as tf
import os
from prepare_dataset import create_dataset_from_path


# build model
def CNN_multi_outputs(image_shape, n_labels=4, n_class=26, name='captcha', output_label='labels'):
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

    image_shape = (60, 120, 1)
    n_labels = 4
    n_class = 26
    
    # set loss for each output lables, or set a same loss for all labels
    # e.g. loss={ f'labels_{i}': 
    #   tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for i in range(n_labels)}
    model = CNN_multi_outputs(image_shape, n_labels, n_class, name='cnn_multi_outputs', output_label='labels')
    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train model
    train_ds = create_dataset_from_path(os.path.join('samples/train/*.jpg'), batch_size=128, image_size=(60, 120), label_prefix='labels')
    test_ds = create_dataset_from_path(os.path.join('samples/test/*.jpg'), batch_size=128, image_size=(60, 120), label_prefix='labels')
    model.fit(train_ds, 
        epochs=100, 
        callbacks=[ tf.keras.callbacks.TensorBoard(log_dir=os.path.join('tensorboard')) ])