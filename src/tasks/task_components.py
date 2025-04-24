#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`task_components` [module]

Defines task components that represent structured tasks with multiple contexts and generate samples
of input and output based on latent variables.


Classes
-------
TaskContext
    Represents a context for a task, including input and output bases.
StructuredTask
    Represents a structured task with several contexts.
LatentVariable
    Represents a latent variable with a distribution and a generator function.
"""
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np


class LatentVariable:
    """
    Represents a latent variable with a distribution and a generator function.

    Attributes
    ----------
    name : str, optional
        Name of the latent variable.
    dist : Callable
        Distribution function for the latent variable. It should not take any mandatory arguments,
        and could take a random seed as an optional argument. It should return a random sample from
        the distribution.

    Methods
    -------
    draw(seed: Optional[int] = None) -> int | float
        Draw a sample from the distribution.
    """

    def __init__(self, dist: Callable, name: Optional[str] = None) -> None:
        self.name = name
        self.dist = dist

    def draw(self, seed: Optional[int] = None) -> int | float:
        """
        Draw a sample from the distribution.

        Returns
        -------
        int | float
            Sample from the distribution.
        """
        if seed is not None:
            np.random.seed(seed)
        return self.dist()


class TaskContext:
    """
    Represents a context for a task, including input and output bases.

    Attributes
    ----------
    name : str, default=""
        Name of the context.
    input_basis : np.ndarray
        Input basis vectors. Shape: ``(n_bases_in, dim_in)``.
    output_basis : np.ndarray
        Output basis vectors. Shape: ``(n_bases_out, dim_out)``.
    input_names : List[str], optional
        Names of the input basis vectors. Length: ``n_bases_in``.
    output_names : List[str], optional
        Names of the output basis vectors. Length: ``n_bases_out``.
    n_bases_in : int
        Number of input basis vectors. Inferred from ``input_basis``.
    n_bases_out : int
        Number of output basis vectors. Inferred from ``output_basis``.
    dim_in : int
        Dimension of the input space. Inferred from ``input_basis``.
    dim_out : int
        Dimension of the output space. Inferred from ``output_basis``.
    latents : List[LatentVariable]
        Latent variables associated with the context.
    latents_names : List[str], optional
        Names of the latent variables. Length: ``n_latents``.
    n_latents : int
        Number of latent variables. Inferred from ``latents``.
    gen_coords_in : List[Callable]
        Generator functions for input coordinates. Length: ``n_bases_in``.
        Each function takes a realization of latent variables and produces a coordinate along one
        input basis vector.
    gen_coords_out : List[Callable]
        Generator functions for output coordinates. Length: ``n_bases_out``.

    Methods
    -------
    gen_sample() -> Tuple[np.ndarray, np.ndarray]
        Generates a sample of input and output.

    See Also
    --------
    LatentVariable :
        Represents a latent variable with a distribution and a generator function.
    """

    def __init__(
        self,
        input_basis: np.ndarray | Dict[str, np.ndarray],
        output_basis: np.ndarray | Dict[str, np.ndarray],
        latents: List[LatentVariable] | Dict[str, LatentVariable],
        gen_coords_in: List[Callable],
        gen_coords_out: List[Callable],
        name: Optional[str] = None,
    ) -> None:
        self.name = name or ""
        if isinstance(input_basis, dict):
            self.input_basis = np.array([input_basis[name] for name in input_basis])
            self.input_names = list(input_basis.keys())
        else:
            self.input_basis = np.array(input_basis)
            self.input_names = [str(i) for i in range(input_basis.shape[0])]
        if isinstance(output_basis, dict):
            self.output_basis = np.array([output_basis[name] for name in output_basis])
            self.output_names = list(output_basis.keys())
        else:
            self.output_basis = np.array(output_basis)
            self.output_names = [str(i) for i in range(output_basis.shape[0])]
        self.n_bases_in = self.input_basis.shape[0]
        self.n_bases_out = self.output_basis.shape[0]
        self.dim_in = self.input_basis.shape[1]
        self.dim_out = self.output_basis.shape[1]
        if isinstance(latents, dict):
            self.latents = [latents[name] for name in latents]
            self.latents_names = list(latents.keys())
        else:
            self.latents = latents
            self.latents_names = [str(i) for i in range(len(latents))]
        self.n_latents = len(self.latents)
        self.gen_coords_in = gen_coords_in
        self.gen_coords_out = gen_coords_out

    def __repr__(self) -> str:
        return (
            f"TaskContext(name={self.name}, "
            f"input_names={self.input_names}, "
            f"output_names={self.output_names}, "
            f"latents_names={self.latents_names})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def gen_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample of input and output.

        Returns
        -------
        x : np.ndarray
            Input sample. Shape: ``(dim_in,)``.
        y : np.ndarray
            Output sample. Shape: ``(dim_out,)``.
        """
        z = np.array([latent.draw() for latent in self.latents])
        coords_in = np.array([gen(z) for gen in self.gen_coords_in])
        coords_out = np.array([gen(z) for gen in self.gen_coords_out])
        x = np.dot(self.input_basis, coords_in)
        y = np.dot(self.output_basis, coords_out)
        return x, y


class StructuredTask:
    """
    Represents a structured task with several contexts.

    Attributes
    ----------
    name : str, optional
        Name of the task. Default: None
    contexts : List[TaskContext]
        Contexts included in the task. Length: ``n_contexts``.
    n_contexts : int
        Number of contexts in the task. Inferred from ``contexts``.
    context_names : List[str], optional
        Names of the contexts. Length: ``n_contexts``.
    dist_contexts : Callable
        Distribution function for the contexts. It should not take any mandatory arguments,
        and could take a random seed as an optional argument. It should return a random sample from
        the distribution.
        Default: Uniform distribution over the range of contexts.

    Methods
    -------
    gen_sample() -> Tuple[np.ndarray, np.ndarray]
        Generates a sample of input and output.
    """

    def __init__(
        self,
        contexts: List[TaskContext],
        name: Optional[str] = None,
        context_names: Optional[List[str]] = None,
        dist_contexts: Optional[Callable] = None,
    ) -> None:
        self.name = name or ""
        self.contexts = contexts
        self.n_contexts = len(contexts)
        self.context_names = context_names or [ctx.name for ctx in contexts]
        self.dist_contexts = dist_contexts or (lambda: np.random.randint(0, self.n_contexts))

    def __repr__(self) -> str:
        return (
            f"StructuredTask(name={self.name}, "
            f"n_contexts={self.n_contexts}, "
            f"context_names={self.context_names}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def gen_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample of input and output.

        Returns
        -------
        x : np.ndarray
            Input sample. Shape: ``(dim_in,)``.
        y : np.ndarray
            Output sample. Shape: ``(dim_out,)``.
        """
        ctx_idx = self.dist_contexts()
        context = self.contexts[ctx_idx]
        x, y = context.gen_sample()
        return x, y
