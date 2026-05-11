// df/pca.hpp — Principal Component Analysis.
//
// PCA decomposes a data matrix into orthogonal factors ordered by the
// variance they explain. For a daily-returns matrix of stocks, the
// first factor is almost always "the market" (everything moves
// together), the second is often a sector tilt, etc.
//
// The PCAResult exposes both the eigen-decomposition (raw) and the
// derived quantities the rest of the project actually uses
// (factor returns, loadings) so callers don't repeat the matrix
// gymnastics.

#pragma once

#include "df/types.hpp"

#include <cstddef>

namespace df {

/// Output of PCA on a (T × N) returns matrix.
///
/// `loadings` are the eigenvectors of the covariance matrix scaled
/// nothing — column i is the loading of each stock on factor i.
///
/// `factor_returns` are the daily returns of the factor portfolios:
/// factor_returns_t = returns_t · loadings, a (T × k) matrix where
/// column i is the daily P&L of holding factor portfolio i.
///
/// `eigenvalues` are the variances of those factor portfolios; the
/// share each one explains is `eigenvalues[i] / total_variance`.
struct PCAResult {
    VectorXd eigenvalues;        // length k, descending order
    MatrixXd loadings;           // (N × k), columns are factor loadings
    MatrixXd factor_returns;     // (T × k), the factor portfolios' daily returns
    double   total_variance;     // sum of all eigenvalues (full N-decomp)
};


/// Fit PCA on a returns matrix and keep the top `k_factors`.
///
/// @param returns   (T × N) matrix; rows are days, cols are stocks.
/// @param k_factors number of factors to keep (1 ≤ k ≤ N).
/// @param shrinkage if true, use Ledoit-Wolf shrunk covariance instead
///                  of the raw sample one. Recommended for any case
///                  where T ~ N or smaller (e.g., 252 daily returns
///                  for 500 stocks).
[[nodiscard]] PCAResult fit_pca(MatrixCRef returns,
                                std::size_t k_factors,
                                bool shrinkage = true);

// Note: PCA sign-alignment across rolling windows is implemented in
// Python (see fit_rolling_pca in factors/pca.py). The C++ kernel does
// not duplicate it since it would never be called — rolling PCA owns
// the time loop in Python and needs to retain the previous window
// to do the comparison.

}  // namespace df
