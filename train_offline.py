import numpy as np
import argparse
import itertools
import cPickle as pickle
from datetime import datetime

import rftk

import experiment_utils.measurements
import experiment_utils.management

from joblib import Parallel, delayed


def run_experiment(experiment_config, run_config):
    X_train, Y_train, X_test, Y_test = experiment_config.load_data(
        run_config.data_size,
        run_config.number_of_passes_through_data)

    learner = rftk.learn.create_vanilia_classifier(                         
                            number_of_features=run_config.number_of_features, 
                            number_of_trees=run_config.number_of_trees,
                            max_depth=run_config.max_depth,
                            min_node_size=run_config.min_samples_split,
                            min_child_size=run_config.min_samples_leaf,
                            min_impurity=run_config.min_impurity_gain,
                            bootstrap=True,
                            number_of_jobs=run_config.number_of_jobs)
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

    return forest_measurement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline accuracy on data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-o', '--out', help='output file name', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file, fromlist=[""])
    experiment_config = config.get_experiment_config()
    run_config_template = config.get_offline_config()

    configuration_domain = experiment_utils.management.ConfigurationDomain(
        run_config_template)

    def launch_job(configuration_domain, position):
        run_config = configuration_domain.configuration_at(position)
        forest_measurement = run_experiment(experiment_config, run_config)
        return position, forest_measurement

    job_results = Parallel(n_jobs=5, verbose=5)(
        delayed(launch_job)(configuration_domain, position)
        for position in list(iter(configuration_domain)))

    forest_measurement_grid = experiment_utils.management.MeasurementGrid(
        configuration_domain,
        experiment_utils.measurements.StatsMeasurement)

    for position, forest_measurement in job_results:
        forest_measurement_grid.record_at(position, forest_measurement)

    results = {
        'measurements': forest_measurement_grid,
        'experiment_config': experiment_config,
        'run_config': run_config_template,
    }
    pickle.dump(results, open(args.out.format("forest"), "wb"))
