import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import time

t1 = time.time()
data_sequence = '01'
data_path = 'KITTI/dataset/sequences/' + data_sequence
left_images_path = data_path + '/image_0/'
right_images_path = data_path + '/image_1/'
calib_file = data_path + '/calib.txt'

# Read the calibration file
f = open(calib_file, 'r')
lines = f.readlines()
f.close()

# Get the left and right images paths
left_images = [left_images_path + f for f in os.listdir(left_images_path) if f.endswith('.png')]
right_images = [right_images_path + f for f in os.listdir(right_images_path) if f.endswith('.png')]
left_images.sort()
right_images.sort()

# Camera matrix
P0 = np.array(lines[0].strip().split(' ')[1:]).astype(np.float32).reshape(3, 4)
P1 = np.array(lines[1].strip().split(' ')[1:]).astype(np.float32).reshape(3, 4)

Rot = np.eye(3)
tx = np.array([[0,   0,    0],
               [0,   0, -0.5372],
               [0, 0.5372, 0]])
E = tx @ Rot

K0 = P0[:, :3]
K1 = P1[:, :3]

F = np.linalg.inv(K1.T) @ E @ np.linalg.inv(K0)
'''======================== Plot the trajectory ========================'''
np.set_printoptions(suppress=True)

T_w_cam0_gt = []
if int(data_sequence) <= 10:
    # Read the poses file
    poses_file = 'KITTI/OdomPoses/dataset/poses/' + data_sequence + '.txt'
    f = open(poses_file, 'r')
    gt = f.readlines()
    f.close()
    # Extract the ground truth trajectory
    for line in gt:
        T_w_cam0_gt.append(np.array(line.strip().split(' ')).astype(np.float32).reshape(3, 4))

    T_w_cam0_gt = np.array(T_w_cam0_gt)


'''======================== Plot the trajectory ========================'''

distance_ratio = 0.85
max_z_threshold = 40
step = 2
images_to_process = len(left_images) - 1

# Keypoint detector and descriptor 
sift = cv2.SIFT_create()
# Keypoint matcher
bf = cv2.BFMatcher()

# array to store all the transformation matrices
T_w_cam0 = []
# identity matrix
T_w_cam0.append(np.eye(4))

last_transformation = np.eye(4)

for i in range(0, images_to_process, step):
    print('Processing images ' + str(i) + ' and ' + str(i + step))
    # if i % 100 == 0:
    #     print(str(i) + ' images processed')

    # Images at time k-1
    left_image_k1 = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    right_image_k1 = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)

    # Images at time k
    left_image_k = cv2.imread(left_images[i + step], cv2.IMREAD_GRAYSCALE)
    # right_image_k = cv2.imread(right_images[i + 1], cv2.IMREAD_GRAYSCALE)

    # find the keypoints and descriptors with ORB in left images at time k-1 and k
    keypoints_left_k1, descriptors_left_k1 = sift.detectAndCompute(left_image_k1, None)
    keypoints_left_k, descriptors_left_k = sift.detectAndCompute(left_image_k, None)

    # find the keypoints and descriptors with SIFT in right images at time k-1 and k
    keypoints_right_k1, descriptors_right_k1 = sift.detectAndCompute(right_image_k1, None)
    # keypoints_right_k, descriptors_right_k = sift.detectAndCompute(right_image_k, None)

    # compute the matches between the left images at time k-1 and k
    # Match descriptors using KNN (k-nearest neighbors)
    matches = bf.knnMatch(descriptors_left_k1, descriptors_left_k, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            good_matches.append(m)

    matches = good_matches

    # compute and draw the matches between left_k1 and right_k1 using the left_k1_matched keypoints
    matches1 = bf.knnMatch(descriptors_left_k1, descriptors_right_k1, k=2)
    # pdb.set_trace()

    # Apply ratio test keep the good matches
    good_matches = []
    for m, n in matches1:
        if m.distance < distance_ratio * n.distance:
            good_matches.append(m)
    
    matches1 = good_matches

    '''======================== Epipolar constraint ========================'''

    # convert points to homogeneous coordinates
    # pts1 = np.array([keypoints_left_k1[m.queryIdx].pt for m in matches1])
    # pts2 = np.array([keypoints_right_k1[m.trainIdx].pt for m in matches1])
    # matches1 = [m for i, m in enumerate(matches1) if abs(pts1[i, 1] - pts2[i, 1]) < 0.2]

    # for p1, p2 in zip(pts1, pts2):
    #     if abs(p1[1] - p2[1]) < 1:
    #         print(p1, p2)

    # pdb.set_trace()
    # pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    # pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    # pdb.set_trace()
    # # multiply the points with fundamental matrix
    # check = np.matmul(pts2, F)
    # check = np.matmul(check, pts1.T)
    # # get the vector of diagonal elements
    # diag = np.diagonal(check)
    # # get the indices of the points which are close to zero

    

    # Keep points with positive disparity
    pts1n = np.array([keypoints_left_k1[m.queryIdx].pt for m in matches1])
    pts2n = np.array([keypoints_right_k1[m.trainIdx].pt for m in matches1])
    matches1 = [m for i, m in enumerate(matches1) if pts1n[i, 0] - pts2n[i, 0] > 0]

    # Get points from the left_k image which are present in both left_k1 and right_k1 images
    matches_in_all = []
    for match1 in matches1:
        for match in matches:
            if match1.queryIdx == match.queryIdx:
                # store the indices of the matches as [left_k1, right_k1 and left_k]
                matches_in_all.append([match.queryIdx, match1.trainIdx, match.trainIdx])
                break

    matches_in_all = np.array(matches_in_all)

    ''' Draw the matches between left_k1 and right_k1 using the left_k1_matched keypoints '''
    # # draw these matches
    # matching_result = cv2.drawMatches(left_image_k1, keypoints_left_k1, left_image_k, keypoints_left_k, matches, None, flags=2)
    # matching_result1 = cv2.drawMatches(left_image_k1, keypoints_left_k1, right_image_k1, keypoints_right_k1, matches1, None, flags=2)

    # # Create a figure with two subplots
    # plt.figure(figsize=(10, 5))

    # # Plot the first image in the first subplot
    # plt.subplot(2, 1, 1)
    # plt.imshow(matching_result, cmap='gray')
    # plt.title('Left k-1 and Right k-1')

    # # Plot the second image in the second subplot
    # plt.subplot(2, 1, 2)
    # plt.imshow(matching_result1, cmap='gray')
    # plt.title('Left k-1 and Left k')

    # plt.show()

    # Triangulate using matches from left_k1 and right_k1
    points_left_k1 = np.array([keypoints_left_k1[match[0]].pt for match in matches_in_all])
    points_right_k1 = np.array([keypoints_right_k1[match[1]].pt for match in matches_in_all])

    # Triangulate the points using the two camera projections given they are rectified
    points_4d_hom = cv2.triangulatePoints(P0, P1, points_left_k1.T, points_right_k1.T)
    points_3d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    points_3d = points_3d[:3, :].T

    '''Filter points which are too far away from the camera'''
    xz_filter = np.sqrt(points_3d[:, 0] ** 2 + points_3d[:, 2] ** 2) < max_z_threshold
    points_3d = points_3d[xz_filter]


    # Get the 2D points from the left_k image
    points_left_k = np.array([keypoints_left_k[match[2]].pt for match in matches_in_all])
    '''Filter points which are too far away from the camera'''
    points_left_k = points_left_k[xz_filter]

    # Compute the relative transformation between the left_k1 and left_k images using PnP
    # the images are alreasy corrected for distortion
    # Distortion coefficients
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    if len(points_3d) >= 4:
        retval1, rvec1, tvec1, inliers1 = cv2.solvePnPRansac(points_3d, points_left_k, P0[:, :3], dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        # get transformation matrix from rotation vector and translation vector
        R, _ = cv2.Rodrigues(rvec1)
        T = np.hstack((R, tvec1))
        # convert the transformation matrix from the world coordinate to the camera coordinate
        T = np.vstack((T, [0, 0, 0, 1]))
        # compute the relative transformation between the left_k1 and left_k images
        T = np.linalg.inv(T)
        last_transformation = T
    else:
        print('Not enough points to compute the relative transformation')
        T = last_transformation
    
    # store the transformation matrix
    T_w_cam0.append(T_w_cam0[-1] @ T)

    # # plot the 3D points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], marker='.', c='r')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=0, azim=-90)
    # plt.title('3D points')
    # plt.show()

T_w_cam0 = np.array(T_w_cam0)
# Save the trajectory in a text file in the format required by the KITTI benchmark
np.savetxt('KITTI/VOPoses/SIFT_'+data_sequence+'.txt', T_w_cam0[:, :3, :].reshape(-1, 12), delimiter=' ')

'''Rough scale of the wasted time'''
t2 = time.time()
print('It took ' + str(round(t2 - t1, 3)) + ' seconds to result in a failure.')

'''======================== Plot the trajectory ========================'''
# Plot X and Z coordinates
print('Plotting the trajectory')
plt.plot(T_w_cam0[:, 0, 3], T_w_cam0[:, 2, 3], 'b-', label = 'VO')
if int(data_sequence) <= 10:
    plt.plot(T_w_cam0_gt[:images_to_process, 0, 3], T_w_cam0_gt[:images_to_process, 2, 3], 'r-',label = 'Ground Truth')
plt.legend()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Trajectory sequence '+data_sequence)
plt.savefig('KITTI/ComputedTracks/SIFT/Trajectory_sequence_'+data_sequence+'.png')
# plt.show()
