<p align="center">
  <img src="https://raw.githubusercontent.com/qpax-solver/qpax/main/docs/assets/images/logo_nobackground_cropped.png" alt="qpax logo">
</p>

<p align="center"><strong>Differentiable, batched, single-precision quadratic programming in JAX</strong></p>

<p align="center">
  <a href="https://pypi.org/project/qpax/"><img src="https://img.shields.io/pypi/v/qpax.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/qpax/"><img src="https://img.shields.io/pypi/pyversions/qpax.svg" alt="Python versions"></a>
  <a href="https://github.com/qpax-solver/qpax/actions/workflows/build.yaml"><img src="https://github.com/qpax-solver/qpax/actions/workflows/build.yaml/badge.svg" alt="Build status"></a>
  <a href="https://qpax-solver.github.io/qpax"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="https://qpax-solver.github.io/qpax">Documentation</a>
</p>

This package can be used for solving and differentiating (batched) convex quadratic programs of the following form:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{s.t.} & \quad  Ax = b,\\ 
                  & \quad  Gx \leq h
\end{align*}
$$


with decision variables $x \in \mathbb{R}^n$, and data matrices $Q \succeq 0$, $q \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$, $G \in \mathbb{R}^{p \times n}$ and $h \in \mathbb{R}^p$.



## Features
* __Differentiable__: Backpropagate through QPs and obtain smooth informative subgradients, even at active inequality constraints.
* __Single Precision__: Runs in `f32`, allowing for larger batch sizes and higher throughput.
* __Batchable__: Solves and differentiates lots of QPs in parallel with shared structure.
* __Infeasibility avoidance__: Avoids generating infeasible problems by solving an always-feasible "elastic" QP and providing informative gradients to encourage feasibility.


## Installation

To install directly from github using `pip`:

* CPU: `pip install qpax`
* NVIDIA GPU (cuda 12): `pip install "qpax[cuda12]"`
* NVIDIA GPU (cuda 13): `pip install "qpax[cuda13]"`

For further details, check [our documentation](https://qpax-solver.github.io/qpax).



## License
This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.


## Citing
If you use this solver, please cite our work(s):

```
@misc{arrizabalaga2026adifferentiable,
    title         = {A Differentiable Interior-Point Method in Single Precision},
    author        = {Jon Arrizabalaga, Kevin Tracy, Zachary Manchester},
    year          = {2026},
    eprint        = {XXXX},
    archivePrefix = {arXiv},
    primaryClass  = {math.OC}
}
```
```
@misc{tracy2024differentiability,
    title={On the Differentiability of the Primal-Dual Interior-Point Method},
    author={Kevin Tracy and Zachary Manchester},
    year={2024},
    eprint={2406.11749},
    archivePrefix={arXiv},
    primaryClass={math.OC}
}
```
