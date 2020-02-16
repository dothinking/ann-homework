import os
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST dataset
mnist = input_data.read_data_sets("dataset/",one_hot=True)


class NN:
    '''typical Neural Network structure'''
    def __init__(self, n_input, n_output, ckpt_dir='ckpt'): 
        '''define all tf variables, so that they can be initialized with
           tf.global_variables_initializer()
        '''
        # training step counter
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # placeholder: sample inputs and outputs
        self.sample_input = tf.placeholder(tf.float32, [None, n_input])
        self.sample_output = tf.placeholder(tf.float32, [None, n_output])
        
        # variables: weights and biases
        self.weights, self.biases = self.init_weights(n_input, n_output)
       
        # model
        self.model = self.build_model(self.weights, self.biases)
        
        # loss between model prediction and sample outputs
        self.loss = self.build_loss(self.model, self.sample_output)
        
        # optimizer
        self.optimizer = self.build_optimizer(self.loss)
        
        # accuracy between model prediction and sample outputs
        self.accuracy = self.build_accuracy(self.model, self.sample_output)
        
        # start session and initialize variables
        self.sess = tf.Session()
        self.init_var = tf.global_variables_initializer()

        # store variables
        self.ckpt_dir = ckpt_dir
        self.saver = tf.train.Saver(max_to_keep=3)
        tf.reset_default_graph()
 

    def init_weights(self, n_input, n_output):
        '''return tuple: (weights, biases)'''
        raise NotImplementedError
 

    def build_model(self, weights, biases):
        '''return model prediction'''
        raise NotImplementedError


    def build_loss(self, prediction, labels):
        '''cross entropy by default'''
        raise NotImplementedError
 

    def build_optimizer(self, target):
        '''Gradient Descent Optimizer by default'''
        raise NotImplementedError

    
    def build_accuracy(self, prediction, labels):
        '''to be implemented'''
        raise NotImplementedError

        
    def is_model_initialized(self):
        return self.sess.run(tf.is_variable_initialized(self.global_step))


    def restore_model(self, ckpt_dir=None, reset_step=False):
        '''restore model from check point files and return current step'''
        succ, step = False, 0

        if ckpt_dir==None: ckpt_dir = self.ckpt_dir
        if os.path.exists(ckpt_dir):
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # restore model                
                # needn't to run global_variables_initializer first
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                succ = True

                # read stored step
                if not reset_step:
                    step = self.sess.run(self.global_step)

        return succ, step

    
    def train(self, get_train_samples, batch_size = 50, 
        epochs = 1000, 
        display_interval = 200, 
        save_model = False,
        restore_model = False,
        save_interval = 500,
        train_feed_dict = None, # extra feed dict
        test_feed_dict = None):
        '''get_train_samples: function object return tuple (x, y)
                              where x is input sameple with shape: batch_size * n_input
        ''' 
        # create path to save model
        if save_model and not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
            
        # initializing variables
        self.sess.run(self.init_var)
        
        # restoring model before training
        step = 0
        if restore_model:
            _, step = self.restore_model()
        
        # training
        while step < epochs:
            step += 1
            batch_x, batch_y = get_train_samples(batch_size)
            inputs_dict = {self.sample_input: batch_x, self.sample_output: batch_y}
            if train_feed_dict and isinstance(train_feed_dict, dict):
                inputs_dict.update(train_feed_dict)
            self.sess.run(self.optimizer, feed_dict=inputs_dict)
            
            # outputs during iteration
            if step % display_interval == 0:
                if test_feed_dict and isinstance(test_feed_dict, dict):
                    inputs_dict.update(test_feed_dict)
                accuracy, loss = self.sess.run([self.accuracy, self.loss], feed_dict=inputs_dict)
                print(f"Epoch {step:>6d}, Iter {step*batch_size:>8d}: loss = {loss:<8.4f} training accuracy = {accuracy:<5.4%}")

            # save model
            if save_model and step % save_interval == 0:
                self.global_step.assign(step) # update step
                self.saver.save(self.sess, f'{self.ckpt_dir}/model', global_step=step)

                
    def test(self, test_x, test_y, feed_dict=None):
        '''test model'''
        # model in memory > restored model > error: not initialized
        if not self.is_model_initialized():
            loaded, _ = self.restore_model()
            if not loaded:
                raise Exception('Model is not initialized yet.')
        
        inputs_dict = {self.sample_input: test_x, self.sample_output: test_y}
        if feed_dict and isinstance(feed_dict, dict):
                inputs_dict.update(feed_dict)
        res = self.sess.run(self.accuracy, feed_dict=inputs_dict)
        return res


class MNIST:
    '''Neural Network for MNIST'''

    def build_loss(self, prediction, labels):
        '''cross entropy by default'''
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))


    def build_accuracy(self, prediction, labels):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def build_optimizer(self, target, learning_rate=0.5):
        '''Gradient Descent Optimizer by default'''
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(target)