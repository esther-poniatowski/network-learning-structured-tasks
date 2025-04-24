#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`task_generator` [module]

Defines the base class for task generators that create structured tasks with multiple contexts.

Classes
-------

See Also
--------
sim_network_learning.tasks.task_components
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

from sim_network_learning.tasks.task_components import TaskContext, TaskInstance


class TaskBuilder:
    """
    Task builder for generating context-dependent tasks.

    Task specification: Each task is defined by adding contexts and their properties.


    - Define an input dimension, an output dimension
    - Define a number of contexts
    - Specify input and output basis vectors in each context, to define context-specific subspaces.
      For simple cases, it should be possible to specify only the number of input basis vectors in
      each context, and they will be drawn as mutually orthogonal vectors, and as many output basis
      vectors will be drawn too. For more flexibility, it should be possible to assign optional
      names to each basis vector, and to define overlaps between certain basis vectors, that will be
      taken into account when drawing the basis vectors.
    - Specify latent variables in each context. For simple cases, in each context, one single latent
      variable is created per context-specific, and will be drawn as a discrete uniform law over the
      number of input basis vectors. For more flexibility, it should be possible to define custom
      latent variables with optional names and alternative distributions, for instance, allowing
      several gaussian iid variables as common options.
    - Generate the coordinates along each basis vector in each context, through generative functions
      that apply context-wise, taking the context-specific latent variables in arguments. For simple
      cases, a single input basis vector has a coordinate 1 based on the value of the default latent
      variable, and the associated output basis vector too. For more flexibility, a custom
      generative function can be specified for the coordinate along each basis vector, and can
      include dependencies between multiple latent variables.

    To create a Task, the TaskBuilder should provide:

    - A list of task contexts.
    - A name for the task.
    - A list of context names.
    - A random distribution to sample the context.

    To create a context, the TaskBuilder should provide:

    - Input and output bases for the context (array or dict if named)
    - A list of latent variables for the context, (list of distributions or dict if named)
    - A list of coordinate generators for input and output bases (list of functions or dict if named)

    Attributes
    ----------
    dim_in : int
        Dimension of the input space.
    dim_out : int
        Dimension of the output space.
    n_contexts : int
        Number of contexts.
    input_basis : Dict[int, Dict[str, np.ndarray]]
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        latent_spec: Optional[Dict] = None,
        coord_generators: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        input_basis_spec : dict
            Specification of the input basis vectors for each context.
            Each context can have a different number of basis vectors and overlaps.
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.contexts = []  # names
        self.contexts_idx = {}  # names -> index
        self.input_basis = []  # names
        self.output_basis = []  # names
        self.input_vec = None
        self.output_vec = None
        self.rng = np.random.default_rng(seed)

    def build(self):
        contexts = []
        for name in self.contexts:
            input_basis = np.zeros()  # TODO
            output_basis = np.zeros()  # TODO
            input_bases_names = []  # TODO
            output_bases_names = []  # TODO
            gen_coords_in = []  # TODO
            gen_coords_out = []  # TODO

            latents = []  # TODO
            ctx = TaskContext(
                name=name,
                input_basis=input_basis,
                output_basis=output_basis,
                input_bases_names=input_bases_names,
                output_bases_names=output_bases_names,
                latents=latents,
                gen_coords_in=gen_coords_in,
                gen_coords_out=gen_coords_out,
            )
            contexts.append(ctx)

        task = StructuredTask()

    def add_context(
        self,
        name: Optional[str] = None,
        n_bases_in: int = 1,
        n_bases_out: int = 1,
        latents: Optional[List[str]] = None,
        gen_coords_in: Optional[List[Callable]] = None,
        gen_coords_out: Optional[List[Callable]] = None,
    ) -> TaskContext:
        """
        Add a new context to the future task.

        The context is defined by a name, and a set of names for
        """

    def generate_bases(
        self, dim: int, spec: Optional[Dict], label: str
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate bases for each context.

        By default basis vectors are, optionally with overlaps.
        """
        bases = {}
        for ctx in range(self.n_contexts):
            ctx_spec = spec.get(ctx, {}) if spec else {}
            names = ctx_spec.get("names", [f"{label}_{i}" for i in range(ctx_spec.get("n", 1))])
            overlaps = ctx_spec.get(
                "overlap", {}
            )  # e.g. {"name1": "name2"} forces shared direction
            basis_vecs = []
            used = set()
            for name in names:
                if name in overlaps and overlaps[name] in used:
                    vec = bases[ctx][overlaps[name]]  # reuse vector
                else:
                    vec = self.sample_orthogonal_vector(dim, basis_vecs)
                basis_vecs.append(vec)
                used.add(name)
            bases[ctx] = dict(zip(names, basis_vecs))
        return bases

    def sample_orthogonal_vector(self, dim: int, existing: List[np.ndarray]) -> np.ndarray:
        """
        Sample a new vector orthogonal to all existing ones.
        """
        vec = self.rng.normal(size=dim)
        for b in existing:
            vec -= np.dot(vec, b) * b
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else self.sample_orthogonal_vector(dim, existing)
