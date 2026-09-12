# TTFS SNN Baseline and Experimental Extensions

This repository is a working copy of the code released with **“High-performance deep spiking neural networks with 0.3 spikes per neuron”** by Stanojevic et al., *Nature Communications* (2024), together with local experiment and evaluation material.

## Purpose

I used this codebase as a research baseline for studying time-to-first-spike neural networks, ANN-to-SNN mapping, gradient behavior, and evaluation before moving to dedicated ConvNeXt-based spiking experiments.

## Contents

The repository includes:

- dataset and preprocessing utilities;
- SNN/ReLU training code;
- model definitions;
- evaluation scripts;
- confusion-matrix outputs and experiment logs;
- local reports and supporting papers.

## Upstream Attribution

The original architecture, method, and released baseline are from Stanojevic et al. The corresponding publication is:

> Stanojevic, A., Woźniak, S., Bellec, G., Cherubini, G., Pantazi, A., & Gerstner, W. “High-performance deep spiking neural networks with 0.3 spikes per neuron.” *Nature Communications* 15, 6793 (2024).

This repository is presented as an experimental working copy and not as my original implementation of the published method.

## Related Work

More recent experiments that focus on my ConvNeXt-based spiking/TTFS research are maintained separately in `SpikingConvNeXt`, `SNN-only-Convnext`, and `ConvNeXt`.


## Installation

The upstream-style environment is recorded in `environment.yml`. Create it with Conda, activate the environment name declared in that file, and inspect the CLI before starting a run:

```bash
conda env create -f environment.yml
conda activate tf24
python main.py --help
```

If the environment name differs in the YAML file, use that name instead. `requirements.txt` provides an alternative package list but does not replace the CUDA and TensorFlow compatibility checks required for GPU execution.

## Working with the Repository

Use `main.py` for SNN/ReLU training and `evaluate_model.py` for evaluation. Keep datasets, mapped weights, logs, and the exact CLI arguments together when comparing runs. The checked-in reports and figures are historical experiment material and should not be treated as results from a fresh environment.
