import tensorflow as tf
from prepare_dataset import create_dataset_from_tfrecord
import numpy as np


def CNN_multi_outputs(image_shape, n_labels=4, n_class=26, name='captcha', output_label='labels'):
    # input
    image_input = tf.keras.Input(shape=image_shape, name='input_image')

    # conv layer 1
    x = tf.keras.layers.Conv2D(32, (3, 3))(image_input)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    # conv layer 2
    x = tf.keras.layers.Conv2D(32, (3, 3))(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv layer 3
    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    
    # conv layer 4
    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # dense layer
    x = tf.keras.layers.Flatten()(x) # flatten
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # outputs
    labels = [tf.keras.layers.Dense(n_class, name=f'{output_label}{i}')(x) for i in range(n_labels)] 

    # build model
    model = tf.keras.Model(inputs=image_input, outputs=labels, name=name)
    
    return model


def CNN_crop_inputs(image_shape, n_class=26, name='captcha', output_label='labels'):
    # input
    image_input = tf.keras.Input(shape=image_shape, name='input_image')
    
    # split into two sets: 
    # - half image in left side: the first label
    # - hale image in right side: the last label
    H, W, C = image_shape
    x0 = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, int(W/2))), name='left_half')(image_input)
    x3 = tf.keras.layers.Cropping2D(cropping=((0, 0), (int(W/2), 0)), name='right_half')(image_input)
    
    # build conv layers
    x0 = _build_conv_layers(x0, [16, 32], 'Pre_A0')
    x3 = _build_conv_layers(x3, [16, 32], 'Pre_A3')
    x12 = tf.keras.layers.Concatenate(axis=2)([x0, x3]) # (s, h, w, c) concatenate on dimension w
    
    x0 = _build_conv_layers(x0, [32, 64, 64], 'A0')
    x12 = _build_conv_layers(x12, [32, 64, 64], 'A1_2')
    x3 = _build_conv_layers(x3, [32, 64, 64], 'A3')
    
    # dense layer
    x0 = tf.keras.layers.Flatten()(x0) # flatten
    x0 = tf.keras.layers.Dense(128, activation='relu')(x0)
    x0 = tf.keras.layers.Dropout(0.3)(x0)
    
    x12 = tf.keras.layers.Flatten()(x12) # flatten
    x12 = tf.keras.layers.Dense(256, activation='relu')(x12)
    x12 = tf.keras.layers.Dropout(0.3)(x12)
    
    x3 = tf.keras.layers.Flatten()(x3) # flatten
    x3 = tf.keras.layers.Dense(128, activation='relu')(x3)
    x3 = tf.keras.layers.Dropout(0.3)(x3)

    # combine multi-outputs
    labels = [
        tf.keras.layers.Dense(n_class, name=f'{output_label}0')(x0),
        tf.keras.layers.Dense(n_class, name=f'{output_label}1')(x12),
        tf.keras.layers.Dense(n_class, name=f'{output_label}2')(x12),
        tf.keras.layers.Dense(n_class, name=f'{output_label}3')(x3)
    ] 

    # build model
    model = tf.keras.Model(inputs=image_input, outputs=labels, name=name)
    
    return model


def evaluate_captcha(model, dataset):
    '''evaluate the model with test dataset:
       the prediction is good only when the four labels are predicted correctly.
    '''
    num = 0
    scores = [0] * 5 # total, label1-4
    for images, dict_labels in dataset.as_numpy_iterator():
        outputs = model.predict(images) # list
        for (image, *zip_labels) in zip(images, *dict_labels.values(), *outputs):
            num += 1
            labels = zip_labels[0:4] # true labels
            predict_labels = [np.argmax(label) for label in zip_labels[4:]]
            # get one score when the four predictions are correct
            flag = True
            for i, (a,b) in enumerate(zip(labels, predict_labels), start=1):
                if a==b:
                    scores[i] += 1
                else:
                    flag = False
            if flag:
                scores[0] += 1
    return [score / num if num else 0 for score in scores]


def _build_conv_layers(x, kernels, prefix):
    # conv layers
    for i, num in enumerate(kernels, start=1):
        # conv layer i
        x = tf.keras.layers.Conv2D(num, (3, 3), padding='same', name=f'{prefix}_conv_{i}')(x)
        x = tf.keras.layers.MaxPool2D((2, 2), padding='same', name=f'{prefix}_max_pool_{i}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_BN_{i}')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x



if __name__ == '__main__':

	import os
	
	image_shape = (60, 120, 3)
	n_labels = 4
	n_class = 26

	
	# create (image, label) dataset from generator
	train_ds = create_dataset_from_tfrecord('dataset/qq_captcha_train.tfrecords', 
	                                        batch_size=128, 
	                                        image_size=(60, 120), 
	                                        label_prefix='A',
	                                        buffer_size=1000).cache()
	test_ds = create_dataset_from_tfrecord('dataset/qq_captcha_test.tfrecords', 
	                                       batch_size=128, 
	                                       image_size=(60, 120), 
	                                       label_prefix='A',
	                                       buffer_size=1000).cache()
	
	# define model
	# model = CNN_multi_outputs(image_shape, 
	# 	n_labels, 
	# 	n_class, 
	# 	name='cnn_multi_outputs', 
	# 	output_label='A')

	model = CNN_crop_inputs(image_shape, 
		n_class, 
		name='cnn_crop_inputs', 
		output_label='A')

	model.summary()

	# set loss for each output lables, or set a same loss for all labels
	# e.g. loss={ f'labels_{i}': 
	#   tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) for i in range(n_labels)}
	model.compile(optimizer="adam", #tf.keras.optimizers.RMSprop(0.001, 0.9),
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])


	# train
	model.fit(train_ds, 
	          epochs=100, 
	          callbacks=[tf.keras.callbacks.TensorBoard(log_dir=os.path.join('tensorboard'))])

	# save the entire model
	model_dir = os.path.join('models', model.name) 
	model.save(model_dir)

	# evaluation
	res = evaluate_captcha(model, test_ds)
	print(res)