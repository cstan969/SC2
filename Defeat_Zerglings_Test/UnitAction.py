from pysc2.lib import actions as sc2_actions
from pysc2.lib import actions
import math
import numpy as np
import tensorflow as tf


def select_unit(env, obs, player_relative):
    #Select Unit
    obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_unit.id, [0, [player_relative[1], player_relative[0]]])])
    return obs

def take_action(feature_vector, u, q_func ):
    act = q_func(feature_vector)
    print(act)
#
    unit_index = int((feature_vector[u][2]-1)*64 + feature_vector[u][3])
    non_zero = np.zeros(64*64)
    inc = 2
    if unit_index - 64*inc - inc > 0:
        non_zero[unit_index - 64*inc - inc] = 1
    if unit_index - 64*inc > 0:
        non_zero[unit_index - 64*inc] = 1
    if unit_index - 64*inc + inc > 0:
        non_zero[unit_index - 64*inc + inc] = 1
    if unit_index - inc > 0:
        non_zero[unit_index - inc] = 1
    if unit_index + inc < 64*64:
        non_zero[unit_index + inc] = 1
    if unit_index + 64*inc - inc < 64*64:
        non_zero[unit_index + 64*inc - inc] = 1
    if unit_index + 64*inc < 64*64:
        non_zero[unit_index + 64*inc] = 1
    if unit_index + 64*inc + inc < 64*64:
        non_zero[unit_index + 64*inc + inc] = 1
    non_zero[unit_index] = 1

    for unit in feature_vector:
        if unit[1] == 4 & (((feature_vector[u][2] - unit[2]) ^ 2 + (feature_vector[u][3] - unit[3]) ^ 2) ^ .5 < 5):
            non_zero[int((unit[2] - 1) * 64 + unit[3] - 1)] = 1

    #print(non_zero)
    #print(max(non_zero))
    #multi = np.multiply(tf.Tensor.eval(act), non_zero)

    #print(max(multi))
    max_index = 2500
    #obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_unit.id,
                                           #          [0, [feature_vector[u][3], feature_vector[u][2]]])])

    #obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

    #obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_point.id,
                                               #      [[0], [math.floor(max_index / 64), max_index % 64]])])

    return [math.floor(max_index / 64), max_index % 64]





    # #MOVEMENTS ALLOWED
    # if unit_index - 64*inc - inc > 0:
    #     np.append(temp, act[0][unit_index - 64*inc - inc])
    #     np.append(temp_index, unit_index - 64*inc - inc)
    # if unit_index - 64*inc > 0:
    #     temp = [temp, act[0][unit_index - 64*inc]]
    #     temp_index = [temp_index, unit_index - 64*inc]
    # if unit_index - 64*inc + inc > 0:
    #     temp = [temp, act[0][unit_index - 64*inc + inc]]
    #     temp_index = [temp_index, unit_index - 64*inc + inc]
    # if unit_index - inc > 0:
    #     temp = [temp, act[0][unit_index - inc]]
    #     temp_index = [temp_index, unit_index - inc]
    # temp = [temp, act[0][unit_index]]
    # temp_index = [temp_index, unit_index]
    # if unit_index + inc < 64*64:
    #     temp = [temp, act[0][unit_index + inc]]
    #     temp_index = [temp_index, unit_index + inc]
    # if unit_index + 64*inc - inc < 64*64:
    #     temp = [temp, act[0][unit_index + 64*inc - inc]]
    #     temp_index = [temp_index, unit_index + 64*inc - inc]
    # if unit_index + 64*inc < 64*64:
    #     temp = [temp, act[0][unit_index + 64*inc]]
    #     temp_index = [temp_index, unit_index + 64*inc]
    # if unit_index + 64*inc + inc < 64*64:
    #     temp = [temp, act[0][unit_index + 64*inc + inc]]
    #     temp_index = [temp_index, unit_index + 64*inc + inc]
    #
    # #ATTACKS ALLOWED
    # for unit in feature_vector:
    #     if unit[1] == 4:
    #         index = int((unit[2] - 1) * 64 + unit[3] - 1)
    #         temp = [temp, act[0][index]]
    #         temp_index = [temp_index, index]

    #print(temp)
    #print(type(temp[0]))
    #Find max value - get index of that
    #a = temp.index(max(temp))
    #print(temp.count())
    #print(np.amax(temp, 0))

    #print(a)
    #max_index = temp_index[temp.index(max(temp))]

    # obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_unit.id,
    #                                                  [0, [feature_vector[u][3], feature_vector[u][2]]])])
    #
    # obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0],
    #       [math.floor(max_index / 64), max_index % 64]])])
    #
    # return obs
    # temp = 128
    # #MOVE
    # if action_index >=0 & action_index <= 7
    #     if action_index == 0:
    #         point = [feature_vector[u][3] - inc, feature_vector[u][2] - inc]
    #     elif action_index == 1:
    #         point = [feature_vector[u][3] - inc, feature_vector[u][2]]
    #     elif action_index == 2:
    #         point = [feature_vector[u][3] - inc, feature_vector[u][2] + inc]
    #     elif action_index == 3:
    #         point = [feature_vector[u][3] + inc, feature_vector[u][2] - inc]
    #     elif action_index == 4:
    #         point = [feature_vector[u][3] + inc, feature_vector[u][2]]
    #     elif action_index == 5:
    #         point = [feature_vector[u][3] + inc, feature_vector[u][2] + inc]
    #     elif action_index == 6:
    #         point = [feature_vector[u][3], feature_vector[u][2] - inc]
    #     elif action_index == 7:
    #         point = [feature_vector[u][3], feature_vector[u][2] + inc]
    #     if point[0] > 0 & point[1] > 0 & point[0] < temp & point[1] < temp:
    #         obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], point])])
    # elif action_index == 8:
    #     obs = env.step(actions=[sc2_actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], point])])
    #     #attack....somethging?

