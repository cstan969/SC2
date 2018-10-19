import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions


import gflags as flags

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_HEALTH = features.SCREEN_FEATURES.unit_hit_points.index
_SHIELD = features.SCREEN_FEATURES.unit_shields.index
_ENERGY = features.SCREEN_FEATURES.unit_energy.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_SELECT_ALL = [0]

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

FLAGS = flags.FLAGS




def PopulateFeatureVector(env, obs, feature_vector0, action_vector):


    ##==================PART 1 - FEATURES FROM FEATURE LAYERS=====================
    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
    health = obs[0].observation["screen"][_HEALTH]
    unit_type = obs[0].observation["screen"][_UNIT_TYPE]
    shield = obs[0].observation["screen"][_SHIELD]
    energy = obs[0].observation["screen"][_ENERGY]

    y, x = player_relative.nonzero()

    first = True
    for i in range(len(x)):
        if i % 4 != 0:
            continue
        if first:
            feature_vector = np.array([float(unit_type[y[i], x[i]]),
                float(player_relative[y[i], x[i]]),
                float(y[i]),
                float(x[i]),
                float(health[y[i], x[i]]),
                float(shield[y[i], x[i]]),
                float(energy[y[i], x[i]])
                   ], dtype='f')
            first = False
            feature_vector = np.expand_dims(feature_vector, axis=0)
        else:
            unit_vector = np.array([float(unit_type[y[i], x[i]]),
                float(player_relative[y[i], x[i]]),
                float(y[i]),
                float(x[i]),
                float(health[y[i], x[i]]),
                float(shield[y[i], x[i]]),
                float(energy[y[i], x[i]])
                                    ], dtype='f')
            unit_vector = np.expand_dims(unit_vector, axis=0)
            feature_vector = np.concatenate((feature_vector, unit_vector), axis=0)









    return feature_vector
