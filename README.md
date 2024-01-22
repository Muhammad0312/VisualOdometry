VO.py does visual odometry using ORB, it is not fully working
VO2.py does visual odometry using SIFT, it works

To run configure all the paths
path to data sequence
path to calib file
path to ground truth
path to store generated data

# evo_traj kitti VOPoses/SIFT_02.txt --ref=GTPoses/02.txt -p --plot_mode=xz
# evo_ape kitti GTPoses/02.txt VOPoses/SIFT_02.txt -va --plot --plot_mode xz --save_results results/SIFT02.zip
