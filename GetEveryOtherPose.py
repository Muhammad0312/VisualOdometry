import numpy as np
import pdb

data_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

for data_sequence in data_sequences:
    print('Processing sequence: ', data_sequence)
    T_w_cam0_gt = []
    # Read the poses file
    poses_file = 'KITTI/OdomPoses/dataset/poses/' + data_sequence + '.txt'
    f = open(poses_file, 'r')
    gt = f.readlines()
    f.close()
    # Extract the ground truth trajectory
    for line in gt:
        T_w_cam0_gt.append(np.array(line.strip().split(' ')).astype(np.float32).reshape(3, 4))

    T_w_cam0_gt = np.array(T_w_cam0_gt)

    # Save every other pose to a file
    T_w_cam0_gt = T_w_cam0_gt[::2]
    np.savetxt('KITTI/GTPoses/' + data_sequence + '.txt', T_w_cam0_gt.reshape(-1, 12))