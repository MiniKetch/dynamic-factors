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


/// Align signs across two PCA fits.
///
/// PCA eigenvectors are determined only up to sign — flipping ALL
/// signs of one column produces an equally valid eigenvector. When
/// you re-fit PCA on a sliding window, the signs can flip
/// arbitrarily between consecutive windows, which makes the
/// time-evolution of factor loadings look like noise even when it
/// isn't.
///
/// This helper compares each factor's loading vector to the previous
/// window's loading vector (via dot product) and flips the sign if
/// the dot is negative — keeping the time series of loadings
/// continuous.
///
/// Modifies `current.loadings` and `current.factor_returns` in place.
void align_signs(PCAResult& current, const PCAResult& previous);

}  // namespace df
