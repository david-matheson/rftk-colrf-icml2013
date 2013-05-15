#!/bin/bash

RESULTS_FOLDER="results"
CONFIG_MODULE="experiment_config.compare_to_offline"
FIGURE_FILE="figures/compare_to_offline.pdf"
FIGURE_FILE_NEW="figures/compare_to_offline_n.pdf"

echo "Training consistent online forest..."
python train_online.py  -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_online-{}.pkl"

echo "Training saffari forest..."
python train_saffari.py -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_saffari-{}.pkl"

echo "Training offline forest..."
python train_offline.py -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_offline-{}.pkl"

echo "Plotting..."
python plot_compare_to_offline.py \
    --online "$RESULTS_FOLDER/usps_online-forest.pkl" \
    --saffari "$RESULTS_FOLDER/usps_saffari-forest.pkl" \
    --offline "$RESULTS_FOLDER/usps_offline-forest.pkl" \
    --out "$FIGURE_FILE"

echo "Training consistent online forest..."
python train_online_new.py  -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_online_n-{}.pkl"

echo "Training saffari forest..."
python train_saffari_new.py -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_saffari_n-{}.pkl"

echo "Training offline forest..."
python train_offline_new.py -c "$CONFIG_MODULE" -o "$RESULTS_FOLDER/usps_offline_n-{}.pkl"

echo "Plotting..."
python plot_compare_to_offline.py \
    --online "$RESULTS_FOLDER/usps_online_n-forest.pkl" \
    --saffari "$RESULTS_FOLDER/usps_saffari_n-forest.pkl" \
    --offline "$RESULTS_FOLDER/usps_offline_n-forest.pkl" \
    --out "$FIGURE_FILE_NEW"
