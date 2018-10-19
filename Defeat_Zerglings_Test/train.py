import sys

import gflags as flags
import tensorflow as tf
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from Defeat_Zerglings_Test import Learn

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS

def main():
  FLAGS(sys.argv)
  with sc2_env.SC2Env(
      "DefeatZerglingsAndBanelings",
      #"DefeatRoaches",
      step_mul=step_mul,
      visualize=True,
      game_steps_per_episode=steps * step_mul) as env:


    #Network Parameters:
    n_hidden_1 = 100  # 1st layer number of features
    n_hidden_2 = 100  # 2nd layer number of features
    n_hidden_3 = 100  # 3rd layer number of features
    n_hidden_4 = 100  # 4th layer number of features
    n_input = 7  # MNIST data input (img shape: 28*28)
    n_output = 4096  # All possible locations on the map

    def model(data):
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([201, n_hidden_3])),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_output]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'b4': tf.Variable(tf.random_normal([n_hidden_4])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }
        print(weights["h1"])
        #First Layer = MLP + ELU
        layer1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
        layer1 = tf.nn.elu(layer1)

        #Second Layer = MLC + tanh
        layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
        layer2 = tf.nn.tanh(layer2)

        #Pool - Aggregate
        out1 = tf.reduce_mean(input_tensor=layer2, axis=0, keep_dims=True)
        out2 = tf.reduce_max(input_tensor=layer2, axis=0, keep_dims=True)
        out = tf.concat([out1, out2], axis=1, name='concat')
        out = tf.concat([out, [[1]]], axis=1, name='concat')
        #out = tf.concat(out, 1, 1)

        #Third Layer
        layer3 = tf.add(tf.matmul(out, weights['h3']), biases['b3'])
        layer3 = tf.nn.elu(layer3)

        #Fourth Layer
        layer4 = tf.add(tf.matmul(layer3, weights['h4']), biases['b4'])
        layer4 = tf.nn.relu(layer4)
        #Classify
        act = tf.add(tf.matmul(layer4, weights['out']), biases['out'])

        return act


    demo_replay = []
    act = Learn.learn(
      env,
      q_func=model,
      num_actions=3,
      lr=1e-4,
      max_timesteps=100000,
      buffer_size=100000,
      exploration_fraction=0.5,
      exploration_final_eps=0.01,
      train_freq=2,
      learning_starts=10000,
      target_network_update_freq=1000,
      gamma=0.99,
      prioritized_replay=True,
      demo_replay=demo_replay
    )
    #act.save("defeat_zerglings.pkl")
    #act.save("defeat_roaches.pkl")


if __name__ == '__main__':
  main()
