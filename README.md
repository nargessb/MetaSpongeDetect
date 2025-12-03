# MetaSpongeDetect
MetaSpongeDetect is a meta-learning–based framework for detecting sponge attacks, feature-collision poisoning, and other training-time manipulations in deep neural networks.
The method learns cross-scenario temporal patterns of instability rather than dataset-specific or architecture-specific cues, enabling generalization across models, datasets, and attack configurations.

Overview

MetaSpongeDetect works by:

Extracting temporal statistics from training dynamics
(e.g., sponge loss, neuron firing density, train/validation mismatch, source–ratio).

Feeding these sequences into a MAML-initialized RNN that meta-learns to distinguish
clean, sponge, and poisoned/FCA behaviors across diverse configurations.

Leveraging an energy-adaptive objective to make the detector sensitive to abnormal compute consumption patterns caused by sponge attacks.

The learned meta-detector captures structural instability introduced by attacks rather than superficial dataset-specific artifacts.

Repository Structure
MetaSpongeDetect/
│
├── data/                         # Example CSVs and extracted temporal features
├── feature_extraction/           # Temporal feature pipeline scripts
├── meta_learner/                 # MAML + RNN implementation
├── attacks/                      # Sponge, FCA, and poisoning scripts
├── evaluation/                   # Metrics, plots, and utilities
├── examples/                     # Notebooks and usage demos
└── README.md

License
MIT License (or specify another license here).
