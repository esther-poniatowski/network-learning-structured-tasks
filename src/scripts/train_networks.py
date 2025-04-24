#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`train_networks` [script]

Main script for initializing, optionally training, and saving feedforward networks on structured
tasks under multiple initializations.

Main steps:
- Load configuration
- Set up logging
- Set random seed
- Instantiate tasks
- Instantiate models
- Optionally train models
- Save models and metadata
"""
from config.defaults import load_config
from utils.logging import init_logger
from utils.seed import SeedManager

from data.task_factory import TaskFactory
from models.model_factory import ModelFactory
from train.training_controller import TrainingController
from io.artifact_manager import ArtifactManager


def main(config_path: str = "config/defaults.yaml"):
    # === Load config, setup logging, seed manager =================================================
    config = load_config(config_path)
    logger = init_logger(config["log"]["dir"])
    seed_manager = SeedManager(config["seed"])

    logger.info("Training started.")
    logger.info(f"Loaded configuration from: {config_path}")

    # === Instantiate Task(s) ======================================================================
    task_factory = TaskFactory(config["task"])
    task_list = task_factory.build_all()

    # === Instantiate Model(s) =====================================================================
    model_factory = ModelFactory(config["model"])
    model_specs = model_factory.enumerate_specs()  # combinations of type, width, var
    model_list = []

    for spec in model_specs:
        for seed in seed_manager.sample_n(config["experiment"]["n_seeds"]):
            model = model_factory.build_model(spec, seed)
            model_list.append((spec, seed, model))

    logger.info(f"Instantiated {len(model_list)} model instances.")

    # === Training Loop (if enabled) ===============================================================
    if config["experiment"]["train"]:
        trainer = TrainingController(config["train"])
        for (spec, seed, model) in model_list:
            for task in task_list:
                logger.info(f"Training model [{spec}] on task [{task.name}] with seed [{seed}]")
                trainer.train(model, task)

    # === Save Models and Metadata =================================================================
    saver = ArtifactManager(config["paths"]["save_dir"])
    for (spec, seed, model) in model_list:
        for task in task_list:
            saver.save_model(model, spec=spec, seed=seed, task_name=task.name)
            logger.info(f"Saved model [{spec}] seed [{seed}] for task [{task.name}]")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
