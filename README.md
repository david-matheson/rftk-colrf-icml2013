rftk-colrf-icml2013
===================

ICML 2013 experiments for "Consistency of Online Random Forests" using rftk.

Setup
--------------
- Clone https://github.com/david-matheson/rftk.git at (insert final commit)
- Build rftk and add it to your PYTHONPATH

    PYTHONPATH=/path/to/rftk/
    export PYTHONPATH

- Download and unzip the usps data (add urls) to rftk-colrf-icml2013/source_data
- Download and unzip the kinect data (add urls) to rftk-colrf-icml2013/source_data

Running experiments
--------------
    ./run_advantage_of_forest.sh
    ./run_compare_to_offline.sh
    ./run_kinect.sh
