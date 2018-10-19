import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

from Defeat_Zerglings_Test import common
from Defeat_Zerglings_Test import FeatureObservation
from Defeat_Zerglings_Test import UnitAction
from Defeat_Zerglings_Test import TrackUnits

from s2clientprotocol import sc2api_pb2 as sc_pb



import gflags as flags

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

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

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

FLAGS = flags.FLAGS

class ActWrapper(object):
  def __init__(self, act):
    self._act = act
    #self._act_params = act_params

  @staticmethod
  def load(path, act_params, num_cpu=16):
    with open(path, "rb") as f:
      model_data = dill.load(f)
    act = deepq.build_act(**act_params)
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()
    with tempfile.TemporaryDirectory() as td:
      arc_path = os.path.join(td, "packed.zip")
      with open(arc_path, "wb") as f:
        f.write(model_data)

      zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
      U.load_state(os.path.join(td, "model"))

    return ActWrapper(act)

  def __call__(self, *args, **kwargs):
    return self._act(*args, **kwargs)

  def save(self, path):
    """Save model to a pickle located at `path`"""
    with tempfile.TemporaryDirectory() as td:
      U.save_state(os.path.join(td, "model"))
      arc_name = os.path.join(td, "packed.zip")
      with zipfile.ZipFile(arc_name, 'w') as zipf:
        for root, dirs, files in os.walk(td):
          for fname in files:
            file_path = os.path.join(root, fname)
            if file_path != arc_name:
              zipf.write(file_path, os.path.relpath(file_path, td))
      with open(arc_name, "rb") as f:
        model_data = f.read()
    with open(path, "wb") as f:
      dill.dump((model_data), f)


def load(path, act_params, num_cpu=16):
  """Load act function that was returned by learn function.

  Parameters
  ----------
  path: str
      path to the act function pickle
  num_cpu: int
      number of cpus to use for executing the policy

  Returns
  -------
  act: ActWrapper
      function that takes a batch of observations
      and returns actions.
  """
  return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)


def learn(env,
          q_func,
          num_actions=3,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0, #NO discounted reward
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          param_noise=False,
          param_noise_threshold=0.05,
          callback=None,
          demo_replay=[]
          ):


    #Create functions necessary to train the model
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput((64, 64), name=name)


    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)


    # Initialize U, reward, obs, environment
    U.initialize()
    #update_target() #WHAT DOES THIS DO OR HOW DO I DO THIS IF I AM NOT USING DEEPQ.BUILD_TRAIN
    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()

    # Init action_vector
    action_vector = [[]]
    track_unit_vector = np.array([[]])
    #Init feature_vector
    feature_vector = [[]]
    #feature_vector = FeatureObservation.PopulateFeatureVector(env, obs)
    #print(feature_vector)


    # Initialize Unit LastActionTaken Vector - This is for determining which units need to select actions still!
    time_between_actions = 9.0 #frames??
    #ActionVector = np.array([[]], dtype=[('unit_id', 'int'), ('x_pos', 'float'), ('y_pos', 'float'),
    #                                     ('last_action', 'float')])
    #unit_count_for_ActionVector = 0
    #for unit in feature_vector:
    #   if unit[1] == 1: #Add unit identifier to last action taken vector
    #       np.append(ActionVector, [unit_count_for_ActionVector, unit[2], unit[3], (-1) * time_between_actions], axis=1)
    #       unit_count_for_ActionVector = unit_count_for_ActionVector + 1

    reset = True # WHAT IS RESET

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        First = True
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            #EXPLORATION SPACE
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                if param_noise_threshold >= 0.:
                    update_param_noise_threshold = param_noise_threshold
                else:
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(
                        1. - exploration.value(t) + exploration.value(t) / float(num_actions))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True



            #Populate Observation Vector
            feature_vector = FeatureObservation.PopulateFeatureVector(env, obs, feature_vector, action_vector)
            print(feature_vector)
            track_unit_vector = TrackUnits.track(track_unit_vector, feature_vector)
            print(track_unit_vector)
            #for each unit in the vector, get an action for that unit...
            for u in range(0, feature_vector.shape[0]):
                if feature_vector[u][1] == 1:
                    xy = UnitAction.take_action(feature_vector, u, q_func)
                    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT,
                                                                     [[0], [feature_vector[u][3], feature_vector[u][2]]])])
                    #if movement then move
                    
                    #if action then action
                    obs = env.step(actions=[sc2_actions.FunctionCall(_ATTACK_SCREEN, [[0], xy])])
                    #update  track_unit_vector




            #DO ACTIONS
            obs, screen, player = common.select_marine(env, obs)
            #get action from training model thingy
            #action = act()
            reset = False
            rew = 0
            new_action = None
            #obs, new_action = common.marine_action(env, obs, player, action)

            new_screen = obs[0].observation["screen"][_PLAYER_RELATIVE]
            army_count = env._obs.observation.player_common.army_count
            rew += obs[0].reward / army_count
            game_info = sc_pb.ResponseGameInfo
            feature_vector = FeatureObservation.PopulateFeatureVector(env, obs)

            output = q_func(feature_vector)
            print(output)

            # available_actions = obs[0].observation["available_actions"]
            # for i in available_actions:
            #     print(i)
            # print("")

            # #select marine and see what we can do now...
            # obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], [feature_vector[0][2],
            #                                                                        feature_vector[0][3]]])])
            # obs = env.step(actions=[sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, [feature_vector[12][2],
            #                                                                                feature_vector[12][3]]])])
            # available_actions = obs[0].observation["available_actions"]
            # for i in available_actions:
            #     print(i)
            # print("")



        for t in range(max_timesteps):
            for unit in FeatureVector:
                game_info = sc_pb.ResponseGameInfo

            #obs, screen, player = common.select_marine(env, obs)
                action = act(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]
            #reset = False
            #rew = 0

            #new_action = None

            #obs, new_action = common.marine_action(env, obs, player, action)

    #Make decisions for each ally unit based on Feature Vector fed into
    #for unit in FeatureVector:
    #    if unit[1] == 1: # Then Friendly needs to make decision
            #do things

    #if ally army count > 0 make army actions
    try:
        if army_count > 0 and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
            obs = env.step(actions=new_action)
        else:
            new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
            obs = env.step(actions=new_action)
    except Exception as e:
        print(e)
        # Do nothing









