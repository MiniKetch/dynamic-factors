// Tests for PCA — dominant factor extraction.

#include <doctest/doctest.h>

#include "df/pca.hpp"

#include <cmath>
#include <random>

using namespace df;

// ---------------------------------------------------------------------------
// Synthetic factor model:
//   each stock's return = β_i · market_return + idiosyncratic_noise
// PCA should recover "market" as factor 1 and the loadings should
// match β up to a sign.
// ---------------------------------------------------------------------------

TEST_CASE("fit_pca — recovers single common factor on synthetic data") {
    constexpr int T = 500;
    constexpr int N = 8;
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Generate one shared market factor + per-stock noise.
    Eigen::VectorXd market(T);
    for (int t = 0; t < T; ++t) market(t) = normal(rng);

    // True betas — known answer the fit should recover.
    Eigen::VectorXd beta(N);
    beta << 0.8, 1.0, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3;

    Eigen::MatrixXd returns(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            returns(t, i) = beta(i) * market(t) + 0.1 * normal(rng);
        }
    }

    const auto pca = fit_pca(returns, /*k_factors=*/3, /*shrinkage=*/false);

    // Eigenvalue 1 should dominate — the variance of the market times
    // sum(β²) is ~9, idiosyncratic variance is just 0.01.
    CHECK(pca.eigenvalues(0) > 5.0 * pca.eigenvalues(1));

    // Factor 1 loadings should be (almost) parallel to β (up to sign).
    const Eigen::VectorXd L1 = pca.loadings.col(0);
    const double dot = std::abs(L1.dot(beta) / (L1.norm() * beta.norm()));
    CHECK(dot > 0.99);
}


TEST_CASE("fit_pca — variance explained sums to 1 across all factors") {
    constexpr int T = 200;
    constexpr int N = 5;
    std::mt19937 rng(7);
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd X(T, N);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < N; ++i) {
            X(t, i) = normal(rng);
        }
    }

    const auto pca = fit_pca(X, /*k_factors=*/N, /*shrinkage=*/false);
    // sum of kept eigenvalues == total variance when k=N.
    CHECK(pca.eigenvalues.sum()
          == doctest::Approx(pca.total_variance).epsilon(1e-10));
}


TEST_CASE("fit_pca — rejects k_factors out of range") {
    Eigen::MatrixXd X(10, 3);
    X.setRandom();
    CHECK_THROWS_AS(fit_pca(X, 0), std::invalid_argument);
    CHECK_THROWS_AS(fit_pca(X, 4), std::invalid_argument);
}


// Note: sign-alignment across rolling PCA windows is exercised in the
// Python tests (test_factors.py — fit_rolling_pca aligns columns from
// the previous window). The C++ helper used to live here but was dead
// code (never bound, never invoked) so it was removed.
