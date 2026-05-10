// df/pca.cpp — implementation.

#include "df/pca.hpp"

#include "df/linalg.hpp"

#include <stdexcept>

namespace df {

PCAResult fit_pca(MatrixCRef returns,
                  std::size_t k_factors,
                  bool shrinkage) {
    const auto N = returns.cols();
    if (k_factors == 0) {
        throw std::invalid_argument("fit_pca: k_factors must be ≥ 1");
    }
    if (static_cast<Eigen::Index>(k_factors) > N) {
        throw std::invalid_argument(
            "fit_pca: k_factors cannot exceed the number of stocks");
    }

    // Compute the (possibly shrunk) covariance matrix, then decompose.
    const MatrixXd cov = shrinkage ? ledoit_wolf(returns)
                                    : sample_covariance(returns);
    const auto decomp = symmetric_eigen(cov);

    PCAResult out;
    out.total_variance = decomp.eigenvalues.sum();
    out.eigenvalues    = decomp.eigenvalues.head(static_cast<Eigen::Index>(k_factors));
    out.loadings       = decomp.eigenvectors.leftCols(
        static_cast<Eigen::Index>(k_factors));

    // Factor returns: project the (demeaned) original returns onto
    // the loadings. Demean to be consistent with how the covariance
    // was estimated.
    MatrixXd centered = returns;
    demean_columns(centered);
    out.factor_returns = centered * out.loadings;

    return out;
}

void align_signs(PCAResult& current, const PCAResult& previous) {
    const auto k = std::min(current.loadings.cols(),
                             previous.loadings.cols());
    for (Eigen::Index i = 0; i < k; ++i) {
        // If consecutive windows agree on the loading direction
        // (dot product positive), keep the sign; otherwise flip.
        const double dot = current.loadings.col(i)
                                  .dot(previous.loadings.col(i));
        if (dot < 0.0) {
            current.loadings.col(i)       *= -1.0;
            current.factor_returns.col(i) *= -1.0;
        }
    }
}

}  // namespace df
