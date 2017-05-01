import numpy as np
from WenliSpace import VonBraun
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, flatten, avg_pool2d, dropout

step_limit = 10
n_obs = 0

grid_size = (12,12)
window_scale = 25
v_range = 4

inputs_shape = [None, v_range*2 + 1, v_range*2 + 1, 1] # [batch, height, width, channels]

conv_1_n_filter = 9
conv_1_kernel_size = 2 # [height, width]
conv_1_stride = 1 # [height, width]

pool_1_kernel_size = 2
pool_1_stride = 2

conv_2_n_filter = 18
conv_2_kernel_size = 4 # [height, width]
conv_2_stride = 1 # [height, width]

fully_size = 81
n_outputs = 9

learning_rate = 0.01
dropout_keep_prob = 0.95

direction = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
        
np.set_printoptions(threshold = np.prod(grid_size))

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]

initializer = tf.contrib.layers.xavier_initializer_conv2d()

X = tf.placeholder(tf.float32, shape = inputs_shape, name = 'X')

with tf.name_scope("dnn"):
    conv_1 = conv2d(X, conv_1_n_filter, conv_1_kernel_size, conv_1_stride, padding = "VALID",
                    activation_fn = tf.nn.elu, weights_initializer = initializer)
    dropout_1 = dropout(conv_1, dropout_keep_prob)
    pool_1 = avg_pool2d(dropout_1, pool_1_kernel_size, pool_1_stride, padding = "VALID")
    
    conv_2 = conv2d(pool_1, conv_2_n_filter, conv_2_kernel_size, conv_2_stride, padding = "VALID",
                    activation_fn = tf.nn.elu, weights_initializer = initializer)
    dropout_2 = dropout(conv_2, dropout_keep_prob)
    fully = fully_connected(flatten(dropout_2), fully_size, activation_fn = tf.nn.elu,
                     weights_initializer = initializer)
    dropout_fully = dropout(fully, dropout_keep_prob)
    logits = fully_connected(dropout_fully, n_outputs, activation_fn = None,
                     weights_initializer = initializer)
    outputs = tf.nn.softmax(logits)
    action = tf.multinomial(tf.log(outputs), num_samples=1)
#    y = tf.one_hot(action, n_outputs, dtype=tf.int64)
    y = tf.one_hot(action, n_outputs)
    
with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                   labels=y, logits=logits)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()   

n_iterations = 100   # number of training iterations
n_max_steps = 30  # max steps per episode
n_games_per_update = 30 # train the policy every 10 episodes
save_iterations = 10    # save the model every 10 training iterations
discount_rate = 0.05

         
            
env = VonBraun.VonBraun(grid_size, window_scale)
env.set_display()
env.make(n_obs, v_range)
success_rates = []            

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []    # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        success = 0
        for game in range(n_games_per_update):
            current_rewards = []   # all raw rewards from the current episode
            current_gradients = [] # all gradients from the current episode
            obs_1, obs_2 = env.reset(show_display=True)
            for step in range(n_max_steps):
                #obs = np.append(obs_1.flatten(), obs_2.flatten())
                obs = np.array([obs_2]).reshape(inputs_shape[1],inputs_shape[2],1)
                action_val, gradients_val= sess.run(
                        [action, gradients],
                        feed_dict={X: [obs]}) # one obs
                next_move = direction[action_val.item(0)]
                obs_1, obs_2, done, reward = env.step(next_move[0], next_move[1])
                #env.render()
                if (iteration+1) % 10 == 0:
                    env.render()
                    time.sleep(1/30)
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    success += 1
                    print("Target reached! [" + str(game) + "/" + str(n_games_per_update) + "]")
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)],
                axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        success_rate = success/n_games_per_update
        success_rates.append(success_rate)
        print("Iteration", iteration, "; Success", success_rate)
        sess.run(training_op, feed_dict=feed_dict)


plt.plot(success_rates, 'b-')