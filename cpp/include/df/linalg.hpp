// df/linalg.hpp — thin wrappers around Eigen for the kernel's three
// recurring linear-algebra needs:
//
//   1. Sample covariance matrix from a returns matrix (rows = days,
//      cols = stocks).
//   2. Symmetric eigendecomposition (top-k eigenvectors of a covariance
//      matrix), used by PCA.
//   3. Cholesky (LDLT) factorization, used inside the Kalman filter
//      for numerically stable covariance updates.
//
// The wrappers exist because callers shouldn't have to know Eigen's
// (excellent but verbose) API to do common things — and because we
// want to centralize input validation so PCA / Kalman code stays
// focused on their actual job.

#pragma once

#include "df/types.hpp"

#include <optional>
#include <stdexcept>

namespace df {

// ---------------------------------------------------------------------------
// Covariance estimators
// ---------------------------------------------------------------------------

/// Sample covariance matrix of a returns matrix.
///
/// Input shape: (T rows, N cols). Each row is one day, each column
/// one stock. Returns an N×N matrix.
///
/// Uses the unbiased estimator (1/(T-1)·X'X after demeaning columns).
/// For T < 2, returns a zero matrix — signals "not enough data" without
/// crashing.
[[nodiscard]] MatrixXd sample_covariance(MatrixCRef returns);

/// Ledoit-Wolf shrinkage estimator.
///
/// When the number of stocks N is large compared to T (the realistic
/// case for a 5-year backtest of the S&P 500: T ≈ 1,250 daily returns
/// vs N ≈ 500 stocks), the sample covariance has noisy eigenvalues
/// and a near-singular spectrum. Ledoit-Wolf shrinks toward the
/// identity matrix scaled to the average sample variance:
///
///     Σ_LW = (1 - δ) · Σ_sample + δ · μ · I
///
/// where δ is the shrinkage intensity (computed in closed form, no
/// hyperparameter tuning) and μ is the average diagonal element.
/// Result: tighter eigenvalue spectrum, much more stable PCA.
///
/// Reference: Ledoit & Wolf (2004), "A Well-Conditioned Estimator for
/// Large-Dimensional Covariance Matrices."
[[nodiscard]] MatrixXd ledoit_wolf(MatrixCRef returns);


// ---------------------------------------------------------------------------
// Symmetric eigendecomposition
// ---------------------------------------------------------------------------

/// Result of decomposing a symmetric matrix.
///
/// The Eigen library guarantees eigenvalues are sorted *ascending*;
/// we reverse to descending here because PCA conventionally lists the
/// largest factor first. eigenvectors[:, i] is the eigenvector for
/// eigenvalues[i].
struct EigenDecomposition {
    VectorXd eigenvalues;   // length N, sorted descending
    MatrixXd eigenvectors;  // N×N, columns are eigenvectors
};

/// Decompose a symmetric matrix. Throws std::invalid_argument if the
/// matrix isn't square.
///
/// We use Eigen's SelfAdjointEigenSolver: it's the LAPACK-quality
/// option for symmetric matrices, ~2× faster than the general-case
/// solver, and numerically stable for near-singular inputs.
[[nodiscard]] EigenDecomposition symmetric_eigen(MatrixCRef A);


// ---------------------------------------------------------------------------
// Cholesky / LDLT factorization
// ---------------------------------------------------------------------------

/// LDLT factorization: A = L · D · L'. Used by the Kalman filter for
/// covariance updates that must remain positive-definite.
///
/// Returns std::nullopt if the input isn't positive-semidefinite —
/// caller decides what to do (typically: increase regularization).
struct LdltFactor {
    MatrixXd L;     // lower triangular, unit-diagonal
    VectorXd D;     // diagonal of D matrix
};

[[nodiscard]] std::optional<LdltFactor> ldlt(MatrixCRef A) noexcept;


// ---------------------------------------------------------------------------
// Misc small helpers
// ---------------------------------------------------------------------------

/// Demean each column in-place.
void demean_columns(MatrixRef X);

}  // namespace df
