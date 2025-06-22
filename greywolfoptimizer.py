import numpy as np
from mpi4py import MPI
import asyncio

# Grey Wolf Optimizer (GWO) implementation with async MPI compatibility

def objective_function(x):
    # Example: Sphere function (min at 0)
    return np.sum(x ** 2)

class GreyWolfOptimizer:
    def __init__(self, obj_func, dim, bounds, n_wolves, max_iter):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.positions = np.random.uniform(bounds[0], bounds[1], (n_wolves, dim))
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf

    def update_leaders(self, fitness, positions):
        for i in range(self.n_wolves):
            if fitness[i] < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness[i]
                self.alpha_pos = positions[i].copy()
            elif fitness[i] < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness[i]
                self.beta_pos = positions[i].copy()
            elif fitness[i] < self.delta_score:
                self.delta_score = fitness[i]
                self.delta_pos = positions[i].copy()

    def update_positions(self, a):
        for i in range(self.n_wolves):
            for j in range(self.dim):
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta

                self.positions[i, j] = np.clip((X1 + X2 + X3) / 3, self.bounds[0], self.bounds[1])

    async def optimize(self, comm, rank, size):
        for iter in range(self.max_iter):
            # Scatter positions to all processes
            local_n = self.n_wolves // size
            if rank == size - 1:
                local_n += self.n_wolves % size
            local_positions = np.zeros((local_n, self.dim))
            comm.Scatterv([self.positions, (self.n_wolves // size) * self.dim * np.ones(size, dtype=int), None, MPI.DOUBLE], local_positions, root=0)

            # Each process evaluates its chunk
            local_fitness = np.array([self.obj_func(x) for x in local_positions])

            # Gather all fitnesses
            fitness = None
            if rank == 0:
                fitness = np.empty(self.n_wolves)
            comm.Gatherv(local_fitness, [fitness, (self.n_wolves // size) * np.ones(size, dtype=int), None, MPI.DOUBLE], root=0)

            # Update leaders and positions on root
            if rank == 0:
                self.update_leaders(fitness, self.positions)
                a = 2 - iter * (2 / self.max_iter)
                self.update_positions(a)
            # Broadcast updated positions and leaders
            comm.Bcast(self.positions, root=0)
            comm.Bcast(self.alpha_pos, root=0)
            comm.Bcast(self.beta_pos, root=0)
            comm.Bcast(self.delta_pos, root=0)
            comm.Bcast(np.array([self.alpha_score, self.beta_score, self.delta_score]), root=0)
            await asyncio.sleep(0)  # Yield control for async compatibility

        if rank == 0:
            return self.alpha_pos, self.alpha_score
        return None, None

async def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dim = 30
    bounds = (-10, 10)
    n_wolves = 100
    if n_wolves % size != 0:
        raise ValueError("Number of wolves must be divisible by the number of processes.")
    max_iter = 200


    gwo = GreyWolfOptimizer(objective_function, dim, bounds, n_wolves, max_iter)
    best_pos, best_score = await gwo.optimize(comm, rank, size)

    if rank == 0:
        print("Best position:", best_pos)
        print("Best score:", best_score)

if __name__ == "__main__":
    asyncio.run(main())