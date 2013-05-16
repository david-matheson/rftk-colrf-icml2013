#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Offline training of kinect random forests
'''

import numpy as np
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os

import rftk
import kinect_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-p', '--train_poses', type=str, required=True)
    parser.add_argument('-n', '--number_of_pixels', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    args = parser.parse_args()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_training_data(args.train_poses)

    offline_run_folder = ("experiment_data_offline/offline-tree-%d-n-%d-%s-standard") % (
                            args.number_of_trees,
                            args.number_of_pixels,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(offline_run_folder):
        os.makedirs(offline_run_folder)


    # Randomly offset scales
    number_of_datapoints = min(args.number_of_pixels, pixel_indices_buffer.GetM())
    offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
    offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

    datapoint_indices = rftk.buffers.as_vector_buffer(np.array(np.arange(number_of_datapoints), dtype=np.int32))

    forest_learner = rftk.learn.create_vanilia_scaled_depth_delta_classifier(
                                  number_of_trees=args.number_of_trees,
                                  number_of_features=2000,
                                  max_depth=30,
                                  min_samples_split=10,
                                  # min_samples_leaf = 5,
                                  # min_impurity_gain = 0.01
                                  ux=75, uy=75, vx=75, vy=75,
                                  bootstrap=True,
                                  number_of_jobs=2)

    predictor = forest_learner.fit(depth_images=depths_buffer, 
                                  pixel_indices=pixel_indices_buffer.Slice(datapoint_indices), 
                                  offset_scales=offset_scales_buffer,
                                  classes=pixel_labels_buffer.Slice(datapoint_indices))

    predictions = predictor.predict(depth_images=depths_buffer,pixel_indices=pixel_indices_buffer)
    accurracy = np.mean(rftk.buffers.as_numpy_array(pixel_labels_buffer) == predictions.argmax(axis=1))
    print accurracy

    forest = predictor.get_forest()

    # Print forest stats
    forestStats = forest.GetForestStats()
    print forest.GetNumberOfTrees()
    forestStats.Print()

    #pickle forest and data used for training
    forest_pickle_filename = "%s/forest-1-%d.pkl" % (offline_run_folder, number_of_datapoints)
    pickle.dump(forest, gzip.open(forest_pickle_filename, 'wb'))