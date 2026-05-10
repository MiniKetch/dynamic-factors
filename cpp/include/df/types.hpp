// df/types.hpp — common typedefs.
//
// Centralizing the matrix typedefs in one place keeps the rest of the
// kernel readable. We always use double — single precision saves
// memory on huge covariance matrices, but the eigendecomposition of a
// near-singular financial covariance can lose precision badly in
// float32, so the safety margin is worth it.

#pragma once

#include <Eigen/Core>

namespace df {

// Dynamic-size doubles — the workhorses for runtime-shaped matrices.
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using ArrayXd  = Eigen::ArrayXd;

// Const-row references: lets functions accept any matrix slice
// (including expressions) without copying. The contract is "I won't
// modify this; it must outlive the call."
using MatrixCRef = const Eigen::Ref<const MatrixXd>;
using VectorCRef = const Eigen::Ref<const VectorXd>;

// Mutating row references — same idea but the callee may write.
using MatrixRef  = Eigen::Ref<MatrixXd>;
using VectorRef  = Eigen::Ref<VectorXd>;

}  // namespace df
