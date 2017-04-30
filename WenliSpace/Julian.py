import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

class Julian:
    def __init__(self, learning_rate):
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.learning_rate = learning_rate
        
    def init_graph(self, n_inputs, n_hidden, n_outputs):
        X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
        
        with tf.name_scope("dnn"):
            hidden = fully_connected(X, n_hidden, activation_fn = tf.nn.elu,
                             weights_initializer = self.initializer)
            logits = fully_connected(hidden, n_outputs, activation_fn = None,
                             weights_initializer = self.initializer)
            outputs = tf.nn.softmax(logits)
            action = tf.multinomial(tf.log(outputs), num_samples=1)
            y = tf.one_hot(action, n_outputs)
        self.y = y
            
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                           labels=y, logits=logits)
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cross_entropy)
            self.gradients = [grad for grad, variable in grads_and_vars]
            gradient_placeholders = []
            grads_and_vars_feed = []
            for grad, variable in grads_and_vars:
                gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
                gradient_placeholders.append(gradient_placeholder)
                grads_and_vars_feed.append((gradient_placeholder, variable))
            self.training_op = optimizer.apply_gradients(grads_and_vars_feed)
            
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
    def run_graph(self):
        print("y", self.y)