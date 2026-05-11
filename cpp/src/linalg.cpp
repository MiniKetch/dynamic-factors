// df/linalg.cpp — implementations.

#include "df/linalg.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace df {

// ---------------------------------------------------------------------------
// Covariance estimators
// ---------------------------------------------------------------------------

MatrixXd sample_covariance(MatrixCRef returns) {
    const auto T = returns.rows();
    const auto N = returns.cols();
    if (T < 2) {
        return MatrixXd::Zero(N, N);
    }
    // Demean once, then 1/(T-1) X'X. Eigen fuses the multiplication
    // and the rowwise mean subtraction into a single tight loop.
    MatrixXd X = returns;
    demean_columns(X);
    return (X.transpose() * X) / static_cast<double>(T - 1);
}

MatrixXd ledoit_wolf(MatrixCRef returns) {
    const auto T = returns.rows();
    const auto N = returns.cols();
    if (T < 2) {
        return MatrixXd::Zero(N, N);
    }

    // Sample covariance.
    MatrixXd X = returns;
    demean_columns(X);
    const MatrixXd S = (X.transpose() * X) / static_cast<double>(T - 1);

    // Shrinkage target — diagonal matrix with the average sample
    // variance on the diagonal (this is the "constant correlation"
    // target most often used in finance, but the simpler scaled-
    // identity form is equally popular and what we use here).
    const double mu = S.diagonal().mean();
    const MatrixXd target = mu * MatrixXd::Identity(N, N);

    // Closed-form shrinkage intensity δ ∈ [0, 1]:
    //
    //     δ = π / γ * 1/T,   clamped to [0, 1]
    //
    // where π is the asymptotic variance of the sample covariance
    // entries (the "noise") and γ is the squared Frobenius distance
    // between sample and target (the "bias"). Big π / small γ ⇒
    // high δ ⇒ trust the target more.

    const auto Tdbl = static_cast<double>(T);

    // π = (1/T) Σ_t Σ_i,j (X_ti X_tj − S_ij)²
    // Vectorised: compute X_ti X_tj as outer products per row, sum.
    double pi_hat = 0.0;
    for (Eigen::Index t = 0; t < T; ++t) {
        const VectorXd row = X.row(t);
        const MatrixXd outer = row * row.transpose();
        pi_hat += (outer - S).array().square().sum();
    }
    pi_hat /= Tdbl;

    // γ = Σ_i,j (S_ij − target_ij)²
    const double gamma_hat = (S - target).array().square().sum();

    double delta = (gamma_hat > 0.0) ? (pi_hat / gamma_hat) / Tdbl : 0.0;
    if (delta < 0.0) delta = 0.0;
    if (delta > 1.0) delta = 1.0;

    return (1.0 - delta) * S + delta * target;
}


// ---------------------------------------------------------------------------
// Symmetric eigendecomposition
// ---------------------------------------------------------------------------

EigenDecomposition symmetric_eigen(MatrixCRef A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "symmetric_eigen: input must be square");
    }
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "symmetric_eigen: eigendecomposition failed to converge");
    }

    // Eigen returns ascending order; reverse so eigenvalues[0] is the
    // dominant factor (PCA convention).
    const auto N = A.rows();
    EigenDecomposition out;
    out.eigenvalues = solver.eigenvalues().reverse();
    out.eigenvectors.resize(N, N);
    for (Eigen::Index i = 0; i < N; ++i) {
        out.eigenvectors.col(i) = solver.eigenvectors().col(N - 1 - i);
    }
    return out;
}


// ---------------------------------------------------------------------------
// LDLT factorization
// ---------------------------------------------------------------------------

std::optional<LdltFactor> ldlt(MatrixCRef A) noexcept {
    if (A.rows() != A.cols()) return std::nullopt;
    Eigen::LDLT<MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) return std::nullopt;

    LdltFactor out;
    // matrixL() returns L *with* the unit diagonal, but for a sparse
    // representation we want them separate.
    const MatrixXd Lmat = solver.matrixL();
    // The permutation in Eigen's LDLT means matrixL/matrixD are with
    // respect to a permuted matrix. For our use case (small Kalman
    // covariances, well-conditioned by construction) we just compose.
    out.L = solver.transpositionsP().transpose() * Lmat;
    out.D = solver.vectorD();
    return out;
}


// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

void demean_columns(MatrixRef X) {
    const VectorXd col_means = X.colwise().mean();
    X.rowwise() -= col_means.transpose();
}

}  // namespace df
