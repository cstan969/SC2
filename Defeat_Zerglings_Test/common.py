def init(env, player_relative, obs):

  #print("init")
  army_count = env._obs.observation.player_common.army_count

  if(army_count==0):
    return obs
  try:
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])

    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
  except Exception as e:
    print(e)
  for i in range(len(player_x)):
    if i % 4 != 0:
      continue

    xy = [player_x[i], player_y[i]]
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])

  group_id = 0
  group_list = []
  unit_xy_list = []
  for i in range(len(player_x)):
    if i % 4 != 0:
      continue

    if group_id > 9:
      break

    xy = [player_x[i], player_y[i]]
    unit_xy_list.append(xy)

    if(len(unit_xy_list) >= 1):
      for idx, xy in enumerate(unit_xy_list):
        if(idx==0):
          obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])
        else:
          obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[1], xy])])

      obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_SET], [group_id]])])
      unit_xy_list = []

      group_list.append(group_id)
      group_id += 1

  if(len(unit_xy_list) >= 1):
    for idx, xy in enumerate(unit_xy_list):
      if(idx==0):
        obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])
      else:
        obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[1], xy])])

    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_SET], [group_id]])])

    group_list.append(group_id)
    group_id += 1

  return obs

def select_marine(env, obs):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  screen = player_relative
  group_list = update_group_list(obs)

  if(check_group_list(env, obs)):
    obs = init(env, player_relative, obs)
    group_list = update_group_list(obs)

  # if(len(group_list) == 0):
  #   obs = init(env, player_relative, obs)
  #   group_list = update_group_list(obs)

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  player = []

  danger_closest, danger_min_dist = None, None
  for e in zip(enemy_x, enemy_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not danger_min_dist or dist < danger_min_dist:
        danger_closest, danger_min_dist = p, dist


  marine_closest, marine_min_dist = None, None
  for e in zip(friendly_x, friendly_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not marine_min_dist or dist < marine_min_dist:
        if dist >= 2:
          marine_closest, marine_min_dist = p, dist

  if(danger_min_dist != None and danger_min_dist <= 5):
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], danger_closest])])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if(len(player_y)>0):
      player = [int(player_x.mean()), int(player_y.mean())]

  elif(marine_closest != None and marine_min_dist <= 3):
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], marine_closest])])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if(len(player_y)>0):
      player = [int(player_x.mean()), int(player_y.mean())]

  else:

    # If there is no marine in danger, select random
    while(len(group_list)>0):
      # units = env._obs.observation.raw_data.units
      # marine_list = []          # for unit in units:
      #   if(unit.alliance == 1):
      #     marine_list.append(unit)

      group_id = np.random.choice(group_list)
      #xy = [int(unit.pos.y - 10), int(unit.pos.x+8)]
      #print("check xy : %s - %s" % (xy, player_relative[xy[0],xy[1]]))
      obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_RECALL], [group_id]])])

      selected = obs[0].observation["screen"][_SELECTED]
      player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
      if(len(player_y)>0):
        player = [int(player_x.mean()), int(player_y.mean())]
        break
      else:
        group_list.remove(group_id)

  if(len(player) == 2):

    if(player[0]>32):
      screen = shift(LEFT, player[0]-32, screen)
    elif(player[0]<32):
      screen = shift(RIGHT, 32 - player[0], screen)

    if(player[1]>32):
      screen = shift(UP, player[1]-32, screen)
    elif(player[1]<32):
      screen = shift(DOWN, 32 - player[1], screen)

  return obs, screen, player
