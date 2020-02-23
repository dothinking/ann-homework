import tensorflow as tf


# load and preprocess image with given path
def decode_image(path, resize=(60, 120)):
    # path example: xxx\abcd.jpg
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # convert to floats in the [0,1] range.
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, resize)
    
    # to gray scale: so channels=1
    image = tf.image.rgb_to_grayscale(image)
    return image # shape=(h, w, 1)

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
def create_dataset_from_path(path_pattern, batch_size=32, image_size=(60, 120), label_prefix='labels'):            
    # create path dataset
    # by default, `tf.data.Dataset.list_files` gets filenames 
    # in a non-deterministic random shuffled order
    dataset = tf.data.Dataset.list_files(path_pattern).map(
        lambda image_path: (decode_image(image_path, image_size),
                            get_labels(image_path, label_prefix))).batch(batch_size)
    return dataset