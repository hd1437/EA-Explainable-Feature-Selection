# EA-Explainable-Feature-Selection

This repository provides the source code of paper 'An Explainable Feature Selection Method through Statistical Reinforcement Learning'.

## Introduction

* `EA.py`: Implement the EA algorithm.
* `linear_test.py`: Explore the linearity of feature selection probability curves for different learning models.
* `record/fitness_iteration.py`: Compare the fitness values during training of feature selection methods.
* `record/view.py`: Visualize the feature selection probability curves.
* `dataset/generate_synthetic.py`: Generate the synthetic dataset.
* `dataset/rand.py`: Generate the random dataset.
* `Baselines`: Contains baselines used in the experiment.

## Usage
Put the datasets in `./dataset`. Run `EA.py` to save the intermediate variables and results. Run other scripts to explore the performace of EA.

