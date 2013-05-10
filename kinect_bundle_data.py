import numpy as np

import gzip
import argparse

import rftk.buffers as buffers
import kinect_utils as kinect_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-m', '--number_of_pixels_per_image', type=int, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths, labels = kinect_utils.load_data(args.pose_files_input_path, pose_filenames[0:args.number_of_images])
    pixel_indices, pixel_labels = kinect_utils.sample_pixels_from_images(labels, args.number_of_pixels_per_image)

    f = open(args.output_file, 'wb')
    np.save(f, depths)
    np.save(f, labels)
    np.save(f, pixel_indices)
    np.save(f, pixel_labels)
    

