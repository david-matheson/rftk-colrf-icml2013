rftk-colrf-icml2013
===================

ICML 2013 experiments for "Consistency of Online Random Forests" using rftk. To run the experiments you must clone and build rftk and rftk-colrf-icml2013. The rftk project has undergone significant refactoring since the paper was published. We recommend using the latest implementation but we also provide the commits required to reproduce the exact results from the paper. 

Setup
--------------
Clone both projects 

    git clone https://github.com/david-matheson/rftk.git
    git clone https://github.com/david-matheson/rftk-colrf-icml2013.git

If you wish to use the exact setup as the paper checkout the colrf-icml2013-camera-ready tag. 

    cd /path/to/rftk
    git checkout -b colrf-icml2013-camera-ready origin/colrf-icml2013-camera-ready
    cd /path/to/rftk-colrf-icml2013
    git checkout -b colrf-icml2013-camera-ready origin/colrf-icml2013-camera-ready

However, we recommend using the latest implementation in the colrf-icml2013 branch.

    cd /path/to/rftk
    git checkout -b colrf-icml2013 origin/colrf-icml2013
    cd /path/to/rftk-colrf-icml2013
    git checkout -b colrf-icml2013 origin/colrf-icml2013

Build rftk

    cd /path/to/rftk
    scons

Add rftk to your PYTHONPATH

    PYTHONPATH=/path/to/rftk/
    export PYTHONPATH

Download and unzip data

    cd /path/to/rftk-colrf-icml2013/source_data
    wget http://www.cs.ubc.ca/~mdenil/projects/random_forests/usps_mog.tar.bz2
    tar -jxvf usps_mog.tar.bz2
    wget http://www.cs.ubc.ca/~mdenil/projects/random_forests/kinect_data.zip
    unzip kinect_data.zip

Running experiments
--------------
    cd /path/to/rftk-colrf-icml2013/
    ./run_advantage_of_forest.sh
    ./run_compare_to_offline.sh
    ./run_kinect.sh
