"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.

Adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/) and Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

"""

data_path = './data'
JOINT_REGRESSOR_TRAIN_EXTRA = data_path + '/J_regressor_extra.npy'

MANO_FILE = data_path + '/MANO_RIGHT.pkl'
MANO_sampling_matrix = data_path + '/mano_downsampling.npz'

"""
Joint definition and mesh topology taken from 
open source project Manopth (https://github.com/hassony2/manopth)
"""
# Vertices number
VERT_NUM = 778
VERT_SUB_NUM = 195

# Finger tip indices right hand
FINGERTIPS_RIGHT = [745, 317, 444, 556, 673]

# Joints
J_NAME = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1',
'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')

# Joints number
JOIN_NUM = len(J_NAME)

# The wrist is considered the root joint
ROOT_INDEX = 0

# Skeleton definition
SKLT_DEF = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )