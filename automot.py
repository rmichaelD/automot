import numpy as np
from WenliSpace import VonBraun
import time
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

step_limit = 10
n_obs = 1

grid_size = (18,18)
window_scale = 25

n_inputs = 5*5
n_hidden = 5
n_outputs = 9

learning_rate = 0.001

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

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')

with tf.name_scope("dnn"):
    hidden = fully_connected(X, n_hidden*3, activation_fn = tf.nn.elu,
                     weights_initializer = initializer)
    hidden_2 = fully_connected(hidden, n_hidden, activation_fn = tf.nn.elu,
                     weights_initializer = initializer)
    logits = fully_connected(hidden_2, n_outputs, activation_fn = None,
                     weights_initializer = initializer)
    outputs = tf.nn.softmax(logits)
    action = tf.multinomial(tf.log(outputs), num_samples=1)
#    y = tf.one_hot(action, n_outputs, dtype=tf.int64)
    y = action
    
with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                   labels=y[0], logits=logits)

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

n_iterations = 200   # number of training iterations
n_max_steps = 60   # max steps per episode
n_games_per_update = 25 # train the policy every 10 episodes
save_iterations = 10    # save the model every 10 training iterations
discount_rate = 0.01

         
            
env = VonBraun.VonBraun(grid_size, window_scale)
env.set_display()
env.make(n_obs)
step_count = 0            

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []    # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        for game in range(n_games_per_update):
            current_rewards = []   # all raw rewards from the current episode
            current_gradients = [] # all gradients from the current episode
            obs_1, obs_2 = env.reset(show_display=True)
            for step in range(n_max_steps):
                #obs = np.append(obs_1.flatten(), obs_2.flatten())
                obs = obs_2
                action_val, gradients_val, outputs_val = sess.run(
                        [action, gradients, cross_entropy],
                        feed_dict={X: obs.reshape(1, obs.size)}) # one obs
                next_move = direction[action_val.item(0)]
                obs_1, obs_2, done, reward = env.step(next_move[0], next_move[1])
                env.render()
                if iteration > 20:
                    time.sleep(1/60)
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        print(outputs_val)
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
        print("Now I'm learning! Iteration", iteration)
        sess.run(training_op, feed_dict=feed_dict)


#while True:
#    
#    env.step(1,1)
#    env.render()
#    #Check if step limit reached
#    step_count += 1
#    if step_count > step_limit:
#        print("Limit reached")
#        break
#    time.sleep(1)