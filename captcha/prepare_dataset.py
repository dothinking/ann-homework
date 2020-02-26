import tensorflow as tf
import os

# --------------------------------------------
# Write raw image data to `images.tfrecords`
# --------------------------------------------
def write_images_to_tfrecord(image_file_pattern, # e.g. 'samples/*.jpg'
    dir_record, # folder for storing TFRecord files
    prefix_record='sample',
    split_test=0.1,  # split train/test rate
    buffer_size=1000000):
    ''' load images and save to TFRecord file
            - get image raw data and labels (filename)
            - split into train / test sets
            - write to TFRecord file
    '''
    # split path dataset
    train_dataset, test_dataset = _split_train_test(image_file_pattern, split_test, buffer_size)

    # read image in train set and save to TFRecord file
    train_record = os.path.join(dir_record, f'{prefix_record}_train.tfrecords')
    with tf.io.TFRecordWriter(train_record) as writer:
        for path in train_dataset.as_numpy_iterator():
            tf_example = _image_example(path)
            writer.write(tf_example.SerializeToString())

    # read image in test set and save to TFRecord file
    if test_dataset:
        test_record = os.path.join(dir_record, f'{prefix_record}_test.tfrecords')
        with tf.io.TFRecordWriter(test_record) as writer:
            for path in test_dataset.as_numpy_iterator():
                tf_example = _image_example(path)
                writer.write(tf_example.SerializeToString())
    else:
        test_record = None

    return train_record, test_record


# --------------------------------------------
# create dataset from path pattern
# --------------------------------------------
def create_dataset_from_path(path_pattern, 
    batch_size=32, 
    image_size=(60, 120), 
    label_prefix='labels'):            
    # create path dataset
    # by default, `tf.data.Dataset.list_files` gets filenames 
    # in a non-deterministic random shuffled order
    return tf.data.Dataset.list_files(path_pattern).map(
        lambda image_path: _parse_path_function(image_path, image_size, label_prefix)
    ).batch(batch_size)


# --------------------------------------------
# create dataset from TFRecord file
# --------------------------------------------
def create_dataset_from_tfrecord(record_file, 
    batch_size=32, 
    image_size=(60, 120), 
    label_prefix='labels',
    buffer_size=1000):
    '''create image/labels dataset from TFRecord file'''          
    return tf.data.TFRecordDataset(record_file).map(
        lambda example_proto: _parse_image_function(example_proto, image_size, label_prefix),
        num_parallel_calls=tf.data.experimental.AUTOTUNE # -1 any available CPUs
    ).shuffle(buffer_size).batch(batch_size)


def _split_train_test(file_pattern, test_rate=0.1, buffer_size=1000000):
    # by default, tf.data.Dataset.list_files always shuffles order during iteration
    # so set it false explicitly
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    
    # shuffle first and stop shuffling during each iteration
    # buffer_size is reccommanded to be larger than dataset size
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=False)

    # split train / test sets
    if test_rate:
        # define split interval
        sep = int(1.0/test_rate)
        is_test = lambda x, y: x % sep == 0
        is_train = lambda x, y: not is_test(x, y)
        recover = lambda x,y: y
        
        # split train/test set, and reset buffle mode: 
        # keep shuffle order different during iteration
        test_dataset = dataset.enumerate(start=1).filter(is_test).map(recover)
        train_dataset = dataset.enumerate(start=1).filter(is_train).map(recover)
    else:
        test_dataset, test_dataset = dataset, None
    
    return train_dataset, test_dataset


def _image_example(path):
    """Create a dictionary with features: image raw data, label
        path: image path, e.g. b'test\\lcrh.jpg
    '"""
    # get image raw data and labels
    image_string = open(path, 'rb').read()
    image_labels = tf.strings.substr(path, -8, 4)
    
    # preparation for tf.train.Example
    feature = {
      'labels'   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_labels.numpy()])),
      'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_path_function(path, image_size, label_prefix):
    '''parse image data and labels from path'''
    raw_image = open(path, 'rb').read()
    labels = tf.strings.substr(path, -8, 4) # path example: b'xxx\abcd.jpg'
    # decode image array and labels
    image_data = _decode_image(raw_image, image_size)
    dict_labels = _decode_labels(labels, label_prefix)

    return image_data, dict_labels


def _parse_image_function(example_proto, image_size, label_prefix):
    '''Parse the input tf.Example protocal using the dictionary describing the features'''
    image_feature_description = {
        'labels'   : tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    image_features = tf.io.parse_single_example(example_proto, image_feature_description)
    
    # decode image array and labels
    image_data = _decode_image(image_features['image_raw'], image_size)
    dict_labels = _decode_labels(image_features['labels'], label_prefix)
    
    return image_data, dict_labels


def _decode_image(image, resize=(60, 120)):
    '''preprocess image with given raw data
        - image: image raw data
    '''
    image = tf.image.decode_jpeg(image, channels=3)
    
    # convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, resize)
    
    # RGB to grayscale -> channels=1
    image = tf.image.rgb_to_grayscale(image)

    return image # shape=(h, w, 1)


def _decode_labels(labels, prefix):
    ''' this function is used within dataset.map(), 
        where eager execution is disables by default:
            check tf.executing_eagerly() returns False.
        So any Tensor.numpy() is not allowed in this function.
    '''
    dict_labels = {}
    for i in range(4):
        c = tf.strings.substr(labels, i, 1) # labels example: b'abcd'
        label = tf.strings.unicode_decode(c, input_encoding='utf-8') - ord('a')
        dict_labels[f'{prefix}{i}'] = label

    return dict_labels
