#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Online training of kinect random forests
'''

import numpy as np
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os

import rftk
import rftk.buffers as buffers

import kinect_utils as kinect_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-p', '--train_poses', type=str, required=True)
    parser.add_argument('-m', '--number_of_passes_through_data', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    parser.add_argument('-s', '--split_rate', type=float, required=True)
    parser.add_argument('-r', '--number_datapoints_split_root', type=float, required=True)
    parser.add_argument('-e', '--eval_split_period', type=int, required=True)
    parser.add_argument('-d', '--max_depth', type=int, required=True)
    parser.add_argument('--list_of_sample_counts', type=str, required=True)
    args = parser.parse_args()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_training_data(args.train_poses)

    online_run_folder = ("results/kinect-online-n-%d-m-%d-tree-%d-splitrate-%0.2f-splitroot-%0.2f-evalperiod-%d-maxdepth-%s-%s") % (
                            depths_buffer.GetL(),
                            args.number_of_passes_through_data,
                            args.number_of_trees,
                            args.split_rate,
                            args.number_datapoints_split_root,
                            args.eval_split_period,
                            args.max_depth,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(online_run_folder):
        os.makedirs(online_run_folder)

    print "Starting %s" % online_run_folder

    # Create learner
    forest_learner = rftk.learn.create_online_two_stream_consistent_depth_delta_classifier(                                      
                                      number_of_trees=args.number_of_trees,
                                      max_depth=args.max_depth,
                                      number_of_features=2000,
                                      number_of_splitpoints=10,
                                      min_impurity=0.01,
                                      number_of_data_to_split_root=args.number_datapoints_split_root,
                                      number_of_data_to_force_split_root=args.number_datapoints_split_root*4,
                                      split_rate_growth=args.split_rate,
                                      probability_of_impurity_stream=0.5,
                                      ux=75, uy=75, vx=75, vy=75,
                                      # poisson_sample=1,
                                      max_frontier_size=1000,
                                      impurity_update_period=args.eval_split_period)

    # Randomly offset scales
    number_of_datapoints = pixel_indices_buffer.GetM()
    offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
    offset_scales_buffer = rftk.buffers.as_matrix_buffer(offset_scales)

    # On the first pass through data learn for each sample counts
    list_of_sample_counts = eval(args.list_of_sample_counts)
    clipped_list_of_sample_counts = [min(s, pixel_labels_buffer.GetN()) for s in list_of_sample_counts]
    clipped_list_of_sample_ranges = zip([0] + clipped_list_of_sample_counts[:-1], clipped_list_of_sample_counts)
    print clipped_list_of_sample_ranges
    pass_id = 0
    for (start_index, end_index) in clipped_list_of_sample_ranges:
        print start_index
        print end_index

        # Slice data
        datapoint_indices = buffers.as_vector_buffer(np.array(np.arange(start_index, end_index), dtype=np.int32))
        sliced_pixel_indices_buffer = pixel_indices_buffer.Slice(datapoint_indices)
        sliced_offset_scales_buffer = offset_scales_buffer.Slice(datapoint_indices)
        sliced_pixel_labels_buffer = pixel_labels_buffer.Slice(datapoint_indices)

        # online_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices))
        predictor = forest_learner.fit(depth_images=depths_buffer, 
                                      pixel_indices=sliced_pixel_indices_buffer,
                                      offset_scales=sliced_offset_scales_buffer,
                                      classes=sliced_pixel_labels_buffer)

        #pickle forest and data used for training
        forest_pickle_filename = "%s/forest-%d-%d.pkl" % (online_run_folder, pass_id, end_index)
        pickle.dump(predictor.get_forest(), gzip.open(forest_pickle_filename, 'wb'))

        # Print forest stats
        forestStats = predictor.get_forest().GetForestStats()
        forestStats.Print()

    # For the rest of the passes use all of the data
    start_index = 0
    end_index = clipped_list_of_sample_counts[-1]
    for pass_id in range(1, args.number_of_passes_through_data):

        # Randomize the order
        perm = buffers.as_vector_buffer(np.array(np.random.permutation(pixel_labels_buffer.GetN()), dtype=np.int32))
        pixel_indices_buffer = pixel_indices_buffer.Slice(perm)
        pixel_labels_buffer = pixel_labels_buffer.Slice(perm)

        # Randomly offset scales
        number_of_datapoints = pixel_indices_buffer.GetM()
        offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
        offset_scales_buffer = buffers.as_matrix_buffer(offset_scales)

        predictor = forest_learner.fit(depth_images=depths_buffer, 
                                      pixel_indices=pixel_indices_buffer,
                                      offset_scales=offset_scales_buffer,
                                      classes=pixel_labels_buffer)

        #pickle forest and data used for training
        forest_pickle_filename = "%s/forest-%d-%d.pkl" % (online_run_folder, pass_id, end_index)
        pickle.dump(predictor.get_forest(), gzip.open(forest_pickle_filename, 'wb'))

        # Print forest stats
        forestStats = predictor.get_forest().GetForestStats()
        forestStats.Print()
