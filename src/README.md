# Simulation of Network Learning in Structured Tasks

## Project Structure

```plaintext
sim_network_learning/
│
├── config/                          # Experiment configurations
│   ├── __init__.py
│   └── defaults.yaml                # Centralized hyper-parameters & seed management
│
├── tasks/                           # Generators of task data
│   ├── __init__.py
│   ├── base_generator.py            # Abstract class for input structure
│   ├── gaussian.py                  # Isotropic and low-rank generators
│   └── structured_task.py           # Task-structured input generation
│
├── models/                          # Network definitions and initializations
│   ├── __init__.py
│   ├── base_model.py                # Abstract interface
│   ├── linear.py                    # Linear feed-forward network
│   └── relu.py                      # ReLU feed-forward network
│
├── analysis/                        # Core representation analyses
│   ├── __init__.py
│   ├── representation_similarity.py
│   ├── dimensionality.py
│   ├── mode_alignment.py
│   ├── task_relevance.py
│   ├── covariance.py
│   ├── ntk.py
│   └── clustering.py
│
├── stats/                           # Statistical inference utilities
│   ├── __init__.py
│   ├── resampling.py                # Bootstrapping, permutation, CV
│   └── metrics.py                   # Confidence intervals, effect sizes
│
├── visualization/                   # Plotting utilities
│   ├── __init__.py
│   ├── rsm.py
│   ├── spectrum.py
│   ├── alignment.py
│   └── clustering.py
│
├── runner/                          # Experiment orchestration
│   ├── __init__.py
│   ├── experiment.py                # Class to manage full pipeline
│   └── sweeps.py                    # Grid/random sweep utilities
│
├── results/                         # Output directory for all experiment logs
│   └── ...

├── scripts/                         # CLI entry points for running experiments
│   ├── run_experiment.py
│   └── analyze_results.py
└── ...
```
