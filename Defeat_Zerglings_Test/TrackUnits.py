import numpy as np

def track(track_unit_vector0, feature_vector):
    #First instance of track_unit_vector?  INIT

    #temporary wrong code
    ids = np.zeros((feature_vector.shape[0], 1), dtype=np.int)
    for i in range(0, ids.shape[0]):
        ids[i][0] = i
    track_unit_vector = np.concatenate((ids, feature_vector[:, [0, 3, 2]]), axis=1)
    return track_unit_vector

    ##Working on it...
    #if track_unit_vector0.size == 0:
    #    #Assign arbitary id to each unit
    #    ids = np.zeros((feature_vector.shape[0], 1), dtype=np.int)
    #    for i in range(0, ids.shape[0]):
    #        ids[i][0] = i
    #    track_unit_vector = np.concatenate((ids, feature_vector[:, [0, 3, 2]]), axis=1)
    #    return track_unit_vector
    ##Actually track
    #else:
    #    list_of_possible_unit_ids = np.unique(feature_vector[:, [0]])
    #    print(list_of_possible_unit_ids)
    #    for unit_id in list_of_possible_unit_ids:
    #        row_idx = np.where(feature_vector[:, [0]] == unit_id)[0]
    #        print(row_idx)
    #        col_idx = np.array([3, 2])
    #        feature_vector_trim = feature_vector[row_idx[:, None], col_idx]
    #        print(feature_vector_trim)
    #        row_idx = np.where(track_unit_vector0[:, [1]] == unit_id)[0]
    #        col_idx = np.array([0, 1, 2])
    #        track_unit_vector_trim = track_unit_vector0[row_idx[:, None], col_idx]
    #        print(track_unit_vector_trim)
    #        for f in zip(feature_vector_trim[:, [1, 2]]):
    #            for t in zip(track_unit_vector_trim[:, [3, 2]]):
    #                dist = np.linalg.norm(np.array(t) - np.array(f))
    #                print(dist)





    #return track_unit_vector0


