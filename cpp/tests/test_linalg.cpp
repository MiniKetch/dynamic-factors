// Tests for the linalg module — covariance estimators + eigendecomposition.

#include <doctest/doctest.h>

#include "df/linalg.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>

using namespace df;

// ---------------------------------------------------------------------------
// Sample covariance — pin against a tiny worked example.
// ---------------------------------------------------------------------------

TEST_CASE("sample_covariance — 2-stock textbook example") {
    // 4 observations, 2 stocks.
    // X = [[1, 2], [2, 4], [3, 6], [4, 8]]
    // Demeaned: [[-1.5,-3], [-0.5,-1], [0.5,1], [1.5,3]]
    // X'X / (T-1) = [[5/3·3, 10/3·3]/3, ...]
    // Manually: var(col1) = (1.5² + 0.5² + 0.5² + 1.5²)/3 = 5/1.5 = 5/1.5
    // Easier: stocks are perfectly correlated (col2 = 2·col1), so the
    // covariance matrix is [[var, 2·var], [2·var, 4·var]].
    MatrixXd X(4, 2);
    X << 1, 2,
         2, 4,
         3, 6,
         4, 8;
    const auto C = sample_covariance(X);
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);
    // var(col1) computed: ((1.5² + 0.5² + 0.5² + 1.5²)) / 3 = 5/3
    CHECK(C(0, 0) == doctest::Approx(5.0 / 3.0).epsilon(1e-12));
    CHECK(C(0, 1) == doctest::Approx(10.0 / 3.0).epsilon(1e-12));
    CHECK(C(1, 0) == doctest::Approx(10.0 / 3.0).epsilon(1e-12));
    CHECK(C(1, 1) == doctest::Approx(20.0 / 3.0).epsilon(1e-12));
}

TEST_CASE("sample_covariance — symmetric output, positive on diagonal") {
    // Random-ish data, just sanity-check structural properties.
    MatrixXd X(50, 5);
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 5; ++j) {
            X(i, j) = std::sin(0.1 * i + 0.5 * j) + 0.01 * (i - j);
        }
    }
    const auto C = sample_covariance(X);
    // Symmetric.
    CHECK((C - C.transpose()).norm() < 1e-12);
    // Positive diagonal (variances).
    for (int i = 0; i < 5; ++i) {
        CHECK(C(i, i) > 0.0);
    }
}


// ---------------------------------------------------------------------------
// Ledoit-Wolf — δ in [0, 1], output PSD, recovers sample cov for T >> N.
// ---------------------------------------------------------------------------

TEST_CASE("ledoit_wolf — δ in [0, 1] and output positive-semidefinite") {
    MatrixXd X(100, 8);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 8; ++j) {
            X(i, j) = 0.01 * (i + 2 * j) + std::cos(0.3 * i * j);
        }
    }
    const auto C = ledoit_wolf(X);
    // Symmetric.
    CHECK((C - C.transpose()).norm() < 1e-10);
    // PSD: smallest eigenvalue ≥ 0 (with a small numerical tolerance).
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(C);
    CHECK(solver.eigenvalues().minCoeff() >= -1e-10);
}


// ---------------------------------------------------------------------------
// Symmetric eigendecomposition — round-trip reconstruction + ordering.
// ---------------------------------------------------------------------------

TEST_CASE("symmetric_eigen — ordering descending and reconstruction") {
    // A simple symmetric matrix with known eigenvalues 1, 2, 3.
    // Diagonal matrix to make the answer obvious.
    MatrixXd A(3, 3);
    A << 3, 0, 0,
         0, 1, 0,
         0, 0, 2;
    const auto e = symmetric_eigen(A);
    // Descending order: 3, 2, 1.
    CHECK(e.eigenvalues(0) == doctest::Approx(3.0).epsilon(1e-10));
    CHECK(e.eigenvalues(1) == doctest::Approx(2.0).epsilon(1e-10));
    CHECK(e.eigenvalues(2) == doctest::Approx(1.0).epsilon(1e-10));
    // Reconstruction: V Λ V' == A.
    const MatrixXd Lambda = e.eigenvalues.asDiagonal();
    const MatrixXd reconstructed = e.eigenvectors * Lambda * e.eigenvectors.transpose();
    CHECK((reconstructed - A).norm() < 1e-10);
}

TEST_CASE("symmetric_eigen — orthonormal eigenvectors") {
    MatrixXd A(4, 4);
    A << 4, 1, 0, 0,
         1, 3, 1, 0,
         0, 1, 2, 1,
         0, 0, 1, 1;
    const auto e = symmetric_eigen(A);
    const MatrixXd VtV = e.eigenvectors.transpose() * e.eigenvectors;
    CHECK((VtV - MatrixXd::Identity(4, 4)).norm() < 1e-10);
}

TEST_CASE("symmetric_eigen — rejects non-square") {
    MatrixXd A(3, 2);
    A.setZero();
    CHECK_THROWS_AS(symmetric_eigen(A), std::invalid_argument);
}
