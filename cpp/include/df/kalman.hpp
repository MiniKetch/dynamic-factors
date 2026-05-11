// df/kalman.hpp — Kalman filter for state-space models.
//
// Templated on the state dimension so the compiler can use Eigen's
// fixed-size matrix path (much faster for small dim, since the
// allocations and branches in the dynamic-size path become loops
// the optimiser can fully unroll).
//
// The model:
//
//     state_t   = state_{t-1} + process_noise_t       process_noise ~ N(0, Q)
//     observed_t = H_t · state_t + obs_noise_t         obs_noise ~ N(0, R)
//
// where:
//   * state_t is a length-k vector — for our use case, the loadings
//     of one stock on the k principal-component factors at time t.
//   * H_t is a (1 × k) row vector — the regressors at time t (i.e.
//     the factor returns at time t). One observation per step.
//   * observed_t is a scalar — the stock's return at time t.
//   * Q is a k×k *constant* covariance describing how fast the state
//     drifts. Tuning Q is the project's hardest knob.
//   * R is a scalar — the variance of the regression's residual.
//
// Implementation notes:
//
//   * We use the *information form* of the predict step (no matrix
//     inversion) and the standard form of the update (a single 1×1
//     "gain" division because we have a scalar observation).
//   * The covariance is symmetrised after every update via the
//     Joseph form (P = (I - KH) P (I - KH)' + K R K') to keep it
//     positive-semidefinite even after thousands of iterations of
//     accumulated floating-point round-off.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace df {

/// One Kalman filter for a k-dimensional state with scalar
/// observations. Templated for performance — k is usually 3-5 in
/// our application.
template <int K>
class KalmanFilter {
public:
    using State    = Eigen::Matrix<double, K, 1>;
    using StateCov = Eigen::Matrix<double, K, K>;
    using Regress  = Eigen::Matrix<double, 1, K>;

    /// Output of a batched run over a full history.
    struct BatchOut {
        /// (T × K) — filtered state at the end of each step.
        Eigen::Matrix<double, Eigen::Dynamic, K> state_path;
        /// (T,) — innovation at each step (NaN if the observation was NaN).
        Eigen::VectorXd residuals;
    };

    KalmanFilter(const State& initial_state,
                 const StateCov& initial_cov,
                 const StateCov& process_noise,
                 double observation_noise)
        : state_(initial_state),
          cov_(initial_cov),
          Q_(process_noise),
          R_(observation_noise),
          log_likelihood_(0.0),
          degenerate_steps_(0) {
        if (R_ <= 0.0) {
            throw std::invalid_argument(
                "KalmanFilter: observation_noise R must be positive");
        }
    }

    /// One filter step: predict then update with the (regressors,
    /// observation) pair.
    ///
    /// Returns the *innovation* (observed − predicted) — useful for
    /// downstream residual analysis.
    double step(const Regress& H, double observed) {
        // ----- Predict -----
        // State transition is the identity (we model loadings as a
        // random walk), so state_pred = state_, and cov_pred =
        // cov_ + Q.
        cov_ += Q_;

        // ----- Update -----
        // Innovation y = observed - H · state.
        const double y = observed - (H * state_).value();

        // Innovation covariance S = H P H' + R (a scalar).
        const double S = (H * cov_ * H.transpose()).value() + R_;
        if (!(S > 0.0)) {
            // Pathological: covariance went non-positive due to
            // numerical drift. Re-symmetrize, count it, and bail with
            // the current state. The caller can query degenerate_steps()
            // after the run to detect that the filter was unhealthy.
            cov_ = 0.5 * (cov_ + cov_.transpose());
            ++degenerate_steps_;
            return y;
        }

        // Kalman gain K = P H' / S, a (k × 1) vector.
        const Eigen::Matrix<double, K, 1> Kgain = (cov_ * H.transpose()) / S;

        // State update: state += K · y
        state_ += Kgain * y;

        // Covariance update via Joseph form for numerical stability:
        //
        //     P = (I - K H) P (I - K H)' + K R K'
        //
        // This costs more arithmetic than the Kalman-typical
        // P -= K H P, but the result is symmetric-PSD even after
        // thousands of updates.
        const StateCov I = StateCov::Identity();
        const StateCov IKH = I - Kgain * H;
        cov_ = IKH * cov_ * IKH.transpose() + R_ * (Kgain * Kgain.transpose());

        // Accumulate log-likelihood: −0.5·(log(2π·S) + y²/S).
        // Used by the tuning loop to pick Q via maximum likelihood.
        constexpr double kLog2Pi = 1.8378770664093454835;  // log(2π)
        log_likelihood_ += -0.5 * (kLog2Pi + std::log(S) + y * y / S);

        return y;
    }

    [[nodiscard]] const State&    state() const noexcept { return state_; }
    [[nodiscard]] const StateCov& covariance() const noexcept { return cov_; }
    [[nodiscard]] double log_likelihood() const noexcept { return log_likelihood_; }
    [[nodiscard]] int    degenerate_steps() const noexcept { return degenerate_steps_; }

    void reset(const State& initial_state, const StateCov& initial_cov) {
        state_ = initial_state;
        cov_   = initial_cov;
        log_likelihood_ = 0.0;
        degenerate_steps_ = 0;
    }

    /// Run the filter over an entire history in one C++ call.
    ///
    /// @param H  (T × K) regressor matrix — row t is the regressors at time t.
    /// @param y  (T,)    observation vector.
    ///
    /// If `y[t]` is NaN, the predict-update is skipped for that index:
    /// the current state is recorded into state_path.row(t) and the
    /// residual at t is set to NaN. This means a stock with sporadic
    /// missing trading days can be filtered without contaminating the
    /// state with synthetic zeros — important on real S&P 500 data
    /// where individual stocks miss listing days, halt days, etc.
    BatchOut run_batch(
        const Eigen::Matrix<double, Eigen::Dynamic, K>& H,
        const Eigen::VectorXd& y
    ) {
        const Eigen::Index T = H.rows();
        if (y.size() != T) {
            throw std::invalid_argument(
                "KalmanFilter::run_batch: H.rows() must equal y.size()");
        }
        BatchOut out;
        out.state_path.resize(T, K);
        out.residuals.resize(T);

        for (Eigen::Index t = 0; t < T; ++t) {
            const double obs = y(t);
            if (std::isnan(obs)) {
                // No observation today: skip predict+update so the
                // state and covariance don't drift on synthetic data.
                out.state_path.row(t) = state_.transpose();
                out.residuals(t) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const Regress h_row = H.row(t);
            const double innovation = step(h_row, obs);
            out.state_path.row(t) = state_.transpose();
            out.residuals(t) = innovation;
        }
        return out;
    }

private:
    State    state_;
    StateCov cov_;
    StateCov Q_;
    double   R_;
    double   log_likelihood_;
    int      degenerate_steps_;
};


// Convenience aliases for the dimensions we actually use.
using Kalman3 = KalmanFilter<3>;
using Kalman5 = KalmanFilter<5>;

}  // namespace df
