#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <numeric> // for std::accumulate
#include <vector>

using namespace std;

double sphereFunction(const vector<double> &x) {
  double sum = 0;
  for (double val : x)
    sum += val * val;
  return sum;
}

class GreyWolfOptimizer {
public:
  GreyWolfOptimizer(int dim, int n_wolves, int max_iter, double lower,
                    double upper)
      : dim(dim), n_wolves(n_wolves), max_iter(max_iter), lower(lower),
        upper(upper) {
    positions.resize(n_wolves, vector<double>(dim));
    srand(time(nullptr));

    // Random initialization of positions
    for (int i = 0; i < n_wolves; i++) {
      for (int j = 0; j < dim; j++) {
        double r = ((double)rand()) / RAND_MAX;
        positions[i][j] = lower + r * (upper - lower);
      }
    }

    alpha_pos.assign(dim, 0.0);
    beta_pos.assign(dim, 0.0);
    delta_pos.assign(dim, 0.0);

    alpha_score = numeric_limits<double>::infinity();
    beta_score = numeric_limits<double>::infinity();
    delta_score = numeric_limits<double>::infinity();
  }

  void updateLeaders(const vector<double> &fitness) {
    for (int i = 0; i < n_wolves; i++) {
      double fit = fitness[i];
      if (fit < alpha_score) {
        delta_score = beta_score;
        delta_pos = beta_pos;
        beta_score = alpha_score;
        beta_pos = alpha_pos;
        alpha_score = fit;
        alpha_pos = positions[i];
      } else if (fit < beta_score) {
        delta_score = beta_score;
        delta_pos = beta_pos;
        beta_score = fit;
        beta_pos = positions[i];
      } else if (fit < delta_score) {
        delta_score = fit;
        delta_pos = positions[i];
      }
    }
  }

  void updatePositions(double a) {
    for (int i = 0; i < n_wolves; i++) {
      for (int j = 0; j < dim; j++) {
        double r1 = ((double)rand()) / RAND_MAX;
        double r2 = ((double)rand()) / RAND_MAX;
        double A1 = 2 * a * r1 - a;
        double C1 = 2 * r2;
        double D_alpha = abs(C1 * alpha_pos[j] - positions[i][j]);
        double X1 = alpha_pos[j] - A1 * D_alpha;

        r1 = ((double)rand()) / RAND_MAX;
        r2 = ((double)rand()) / RAND_MAX;
        double A2 = 2 * a * r1 - a;
        double C2 = 2 * r2;
        double D_beta = abs(C2 * beta_pos[j] - positions[i][j]);
        double X2 = beta_pos[j] - A2 * D_beta;

        r1 = ((double)rand()) / RAND_MAX;
        r2 = ((double)rand()) / RAND_MAX;
        double A3 = 2 * a * r1 - a;
        double C3 = 2 * r2;
        double D_delta = abs(C3 * delta_pos[j] - positions[i][j]);
        double X3 = delta_pos[j] - A3 * D_delta;

        double new_pos = (X1 + X2 + X3) / 3.0;
        // Clamp within bounds
        if (new_pos < lower)
          new_pos = lower;
        if (new_pos > upper)
          new_pos = upper;

        positions[i][j] = new_pos;
      }
    }
  }

  // Accessor for positions data in flat format (needed for MPI scatter/gather)
  double *getPositionsFlat() {
    // flatten 2D vector into 1D vector
    flat_positions.clear();
    for (const auto &wolf : positions) {
      flat_positions.insert(flat_positions.end(), wolf.begin(), wolf.end());
    }
    return flat_positions.data();
  }

  // After updating positions flat array (e.g., from MPI_Bcast), sync back to 2D
  // vector
  void setPositionsFromFlat(const double *flat_data) {
    for (int i = 0; i < n_wolves; i++) {
      for (int j = 0; j < dim; j++) {
        positions[i][j] = flat_data[i * dim + j];
      }
    }
  }

  vector<vector<double>> positions;
  vector<double> alpha_pos, beta_pos, delta_pos;
  double alpha_score, beta_score, delta_score;

private:
  int dim, n_wolves, max_iter;
  double lower, upper;
  vector<double> flat_positions;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int dim = 30;
  const int n_wolves = 100;
  const int max_iter = 200;
  const double lower = -10;
  const double upper = 10;

  // Ensure divisible or handle remainder in code
  int base_chunk = n_wolves / size;
  int remainder = n_wolves % size;

  // Calculate send counts and displacements for Scatterv/Gatherv
  vector<int> sendcounts(size);
  vector<int> displs(size);
  int offset = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = (base_chunk + (i == size - 1 ? remainder : 0)) * dim;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  GreyWolfOptimizer gwo(dim, n_wolves, max_iter, lower, upper);

  vector<double> local_positions(sendcounts[rank]); // local chunk for positions
  vector<double> local_fitness(sendcounts[rank] / dim); // local fitness array

  for (int iter = 0; iter < max_iter; iter++) {
    // Scatter positions to all processes
    MPI_Scatterv(gwo.getPositionsFlat(), sendcounts.data(), displs.data(),
                 MPI_DOUBLE, local_positions.data(), sendcounts[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute fitness locally
    for (size_t i = 0; i < local_fitness.size(); i++) {
      vector<double> wolf_pos(local_positions.begin() + i * dim,
                              local_positions.begin() + (i + 1) * dim);
      local_fitness[i] = sphereFunction(wolf_pos);
    }

    // Gather all fitness values at root
    vector<double> fitness(n_wolves);
    MPI_Gatherv(local_fitness.data(), local_fitness.size(), MPI_DOUBLE,
                fitness.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
      gwo.updateLeaders(fitness);
      double a = 2.0 - iter * (2.0 / max_iter);
      gwo.updatePositions(a);

      cout << "Iteration " << iter << " Best score: " << gwo.alpha_score << "\r"
           << flush;
    }

    // Broadcast updated positions & leader info to all processes
    MPI_Bcast(gwo.getPositionsFlat(), n_wolves * dim, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(gwo.alpha_pos.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gwo.beta_pos.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gwo.delta_pos.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double leader_scores[3] = {gwo.alpha_score, gwo.beta_score,
                               gwo.delta_score};
    MPI_Bcast(leader_scores, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Sync updated positions from broadcast to local class variable
    if (rank != 0) {
      gwo.setPositionsFromFlat(gwo.getPositionsFlat());
      gwo.alpha_score = leader_scores[0];
      gwo.beta_score = leader_scores[1];
      gwo.delta_score = leader_scores[2];
    }
  }

  if (rank == 0) {
    cout << "\nBest position found (first 10 dims): ";
    for (int i = 0; i < min(10, dim); i++)
      cout << gwo.alpha_pos[i] << " ";
    cout << "\nBest fitness: " << gwo.alpha_score << endl;
  }

  MPI_Finalize();
  return 0;
}
