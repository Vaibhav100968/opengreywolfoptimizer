# Grey Wolf Optimizer with MPI (C++ & Python)

This repository contains two implementations of the Grey Wolf Optimizer (GWO) algorithm using MPI for parallelization:

- **C++ version** (`mpi.cpp`) using `mpic++`
- **Python version** (`mpi_gwo.py`) using `mpi4py` and `asyncio`

---

## Description

The Grey Wolf Optimizer is a nature-inspired metaheuristic optimization algorithm modeled after the leadership hierarchy and hunting strategy of grey wolves.

Both versions optimize the Sphere function, a common benchmark, in a high-dimensional space.

---

## Requirements

### For C++ Version

- MPI library and compiler with MPI support
  - Linux/Unix: OpenMPI or MPICH
  - Windows: Microsoft MPI (MS-MPI)
- C++ compiler with MPI support (`mpic++`)

### For Python Version

- Python 3.7 or higher
- `mpi4py` library
- `numpy` library
- `asyncio` (built-in with Python 3.7+)
- MPI installed on your system (e.g., OpenMPI or MPICH)

Install Python dependencies with:

```bash
pip install mpi4py numpy
```
