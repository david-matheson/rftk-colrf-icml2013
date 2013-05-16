import os
import numpy as np
import argparse
import itertools
import random as random
import cPickle as pickle

import rftk

import experiment_utils.measurements
import experiment_utils.management

from joblib import Parallel, delayed


def run_experiment(experiment_config, run_config):
    X_train, Y_train, X_test, Y_test = experiment_config.load_data(
        run_config.data_size,
        run_config.number_of_passes_through_data)
    (x_m,x_dim) = X_train.shape
    y_dim = int(np.max(Y_train) + 1)

    learner = rftk.learn.create_online_two_stream_consistent_classifier(
                            number_of_features=run_config.number_of_features,
                            number_of_trees=run_config.number_of_trees,
                            # max_depth=run_config.max_depth, set to 1000 in old version
                            number_of_splitpoints=run_config.number_of_thresholds,
                            min_impurity=run_config.min_impurity_gain,
                            number_of_data_to_split_root=run_config.number_of_data_to_split_root,
                            number_of_data_to_force_split_root=run_config.number_of_data_to_force_split_root,
                            split_rate_growth=run_config.split_rate,
                            probability_of_impurity_stream=run_config.impurity_probability,
                            # poisson_sample=1, disabled in old version
                            max_frontier_size=50000)

    forest = learner.fit(x=X_train, classes=Y_train)

    y_probs = forest.predict(x=X_test)
    y_hat = y_probs.argmax(axis=1)
    accuracy = np.mean(Y_test == y_hat)

    full_forest_data = forest.get_forest()
    stats = full_forest_data.GetForestStats()
    forest_measurement = experiment_utils.measurements.StatsMeasurement(
        accuracy=accuracy,
        min_depth=stats.mMinDepth,
        max_depth=stats.mMaxDepth,
        average_depth=stats.GetAverageDepth(),
        min_estimator_points=stats.mMinEstimatorPoints,
        max_estimator_points=stats.mMaxEstimatorPoints,
        average_estimator_points=stats.GetAverageEstimatorPoints(),
        total_estimator_points=stats.mTotalEstimatorPoints)

    if not experiment_config.measure_tree_accuracy:
        return forest_measurement, None

    tree_measurements = []
    for tree_id in range(full_forest_data.GetNumberOfTrees()):
        single_tree_forest_data = rftk.forest_data.Forest([full_forest_data.GetTree(tree_id)])
        single_tree_forest_predictor = rftk.predict.MatrixForestPredictor(single_tree_forest_data)
        y_probs = single_tree_forest_predictor.predict_proba(X_test)
        accuracy = np.mean(Y_test == y_probs.argmax(axis=1))
        tree_measurement = experiment_utils.measurements.AccuracyMeasurement(
            accuracy=accuracy)
        tree_measurements.append(tree_measurement)

    return forest_measurement, tree_measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Random Forest Training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-o', '--out', help='output file name', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file, fromlist=[""])
    experiment_config = config.get_experiment_config()
    run_config_template = config.get_online_config()

    configuration_domain = experiment_utils.management.ConfigurationDomain(
        run_config_template)

    def launch_job(configuration_domain, position):
        run_config = configuration_domain.configuration_at(position)
        forest_measurement, tree_measurement = run_experiment(experiment_config, run_config)
        return position, forest_measurement, tree_measurement

    # beware: these jobs take a lot of memory
    job_results = Parallel(n_jobs=5, verbose=5)(
        delayed(launch_job)(configuration_domain, position)
        for position in list(iter(configuration_domain)))

    forest_measurement_grid = experiment_utils.management.MeasurementGrid(
        configuration_domain,
        experiment_utils.measurements.StatsMeasurement)

    # this is going to break if we do tree measurements AND multiple
    # numbers of trees
    if experiment_config.measure_tree_accuracy:
        tree_measurement_grids = []
        for i in xrange(run_config_template.number_of_trees):
            tree_measurement_grids.append(
                experiment_utils.management.MeasurementGrid(
                    configuration_domain,
                    experiment_utils.measurements.AccuracyMeasurement))

    for position, forest_measurement, tree_measurements in job_results:
        forest_measurement_grid.record_at(position, forest_measurement)

        if experiment_config.measure_tree_accuracy:
            for tree_measurement_grid, tree_measurement in zip(tree_measurement_grids, tree_measurements):
                tree_measurement_grid.record_at(position, tree_measurement)

    results = {
        'measurements': forest_measurement_grid,
        'experiment_config': experiment_config,
        'run_config': run_config_template,
    }
    pickle.dump(results, open(args.out.format("forest"), "wb"))

    if experiment_config.measure_tree_accuracy:
        with open(args.out.format("trees"), 'wb') as tree_results_file:
            tree_results = [
                {
                    'measurements': tree_measurement_grid,
                    'experiment_config': experiment_config,
                    'run_config': run_config_template,
                }
                for tree_measurement_grid in tree_measurement_grids
                ]
            pickle.dump(tree_results, tree_results_file)
