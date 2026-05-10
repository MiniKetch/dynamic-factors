// Kalman filter tests.
//
// Strategy: simulate a state-space model with known parameters, run
// the filter, verify it converges to the true state.

#include <doctest/doctest.h>

#include "df/kalman.hpp"

#include <random>

using namespace df;

TEST_CASE("KalmanFilter — recovers a constant state under noise") {
    // Scalar state β=2.5. Observations: y_t = β + N(0, 0.5²).
    // Use K=1 (1-dim state) — degenerate Kalman = exponential
    // average of observations.
    KalmanFilter<1> kf(
        /*initial_state=*/Eigen::Matrix<double, 1, 1>::Zero(),
        /*initial_cov=*/  Eigen::Matrix<double, 1, 1>::Constant(10.0),
        /*process_noise=*/Eigen::Matrix<double, 1, 1>::Constant(1e-6),
        /*observation_noise=*/0.25
    );

    constexpr double true_beta = 2.5;
    std::mt19937 rng(13);
    std::normal_distribution<double> noise(0.0, 0.5);

    for (int t = 0; t < 200; ++t) {
        const double y = true_beta + noise(rng);
        Eigen::Matrix<double, 1, 1> H;
        H << 1.0;
        kf.step(H, y);
    }
    // After 200 obs, the filter should be tightly around 2.5.
    CHECK(kf.state()(0) == doctest::Approx(true_beta).epsilon(0.05));
    // Posterior covariance should have shrunk well below the prior.
    CHECK(kf.covariance()(0, 0) < 0.01);
}


TEST_CASE("KalmanFilter — tracks a slowly drifting state") {
    // State drifts linearly: β_t = 1 + 0.005 · t (drifts from 1 to 2 over 200 steps).
    // We need Q large enough that the filter can keep up.
    KalmanFilter<1> kf(
        Eigen::Matrix<double, 1, 1>::Zero(),
        Eigen::Matrix<double, 1, 1>::Constant(1.0),
        Eigen::Matrix<double, 1, 1>::Constant(1e-3),  // Q allows drift
        0.01                                           // R = small obs noise
    );

    std::mt19937 rng(31);
    std::normal_distribution<double> noise(0.0, 0.1);
    double final_state = 0.0;
    for (int t = 0; t < 200; ++t) {
        const double true_beta = 1.0 + 0.005 * t;
        const double y = true_beta + noise(rng);
        Eigen::Matrix<double, 1, 1> H;
        H << 1.0;
        kf.step(H, y);
        if (t == 199) final_state = kf.state()(0);
    }
    // Should be near 2.0 (drifted endpoint), not stuck at the start.
    CHECK(final_state == doctest::Approx(2.0).epsilon(0.10));
}


TEST_CASE("KalmanFilter — multivariate state recovers loadings") {
    // 2-factor regression: y_t = β1·x1_t + β2·x2_t + noise.
    using KF = KalmanFilter<2>;
    KF::State    init_state;  init_state.setZero();
    KF::StateCov init_cov;    init_cov << 10.0, 0.0, 0.0, 10.0;
    KF::StateCov Q;           Q << 1e-7, 0.0, 0.0, 1e-7;
    KF kf(init_state, init_cov, Q, 0.04);

    std::mt19937 rng(99);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::normal_distribution<double> obs_noise(0.0, 0.2);

    const Eigen::Vector2d true_beta(1.5, -0.7);

    for (int t = 0; t < 500; ++t) {
        Eigen::Matrix<double, 1, 2> H;
        H << normal(rng), normal(rng);
        const double y = (H * true_beta).value() + obs_noise(rng);
        kf.step(H, y);
    }

    CHECK(kf.state()(0) == doctest::Approx(1.5).epsilon(0.10));
    CHECK(kf.state()(1) == doctest::Approx(-0.7).epsilon(0.10));
}


TEST_CASE("KalmanFilter — log-likelihood is finite and changes per step") {
    // Each step adds −0.5·(log(2π·S) + y²/S) to the running total.
    // That contribution can be positive (tight innovation, small S)
    // or negative (large innovation or large S), so we only check
    // that the total is finite and non-zero after a series of steps.
    KalmanFilter<1> kf(
        Eigen::Matrix<double, 1, 1>::Zero(),
        Eigen::Matrix<double, 1, 1>::Constant(1.0),
        Eigen::Matrix<double, 1, 1>::Constant(1e-4),
        0.1
    );
    Eigen::Matrix<double, 1, 1> H;
    H << 1.0;
    for (int t = 0; t < 50; ++t) {
        kf.step(H, 0.5 + 0.3 * std::sin(0.1 * t));
        CHECK(std::isfinite(kf.log_likelihood()));
    }
    // After 50 non-trivial observations the total should have moved
    // away from zero — not a monotonic check, just a "filter is
    // accumulating something" sanity check.
    CHECK(std::abs(kf.log_likelihood()) > 1e-6);
}


TEST_CASE("KalmanFilter — rejects non-positive observation noise") {
    Eigen::Matrix<double, 1, 1> z;
    z << 0.0;
    Eigen::Matrix<double, 1, 1> Q;
    Q << 1e-6;
    CHECK_THROWS_AS((KalmanFilter<1>(z, Q, Q, 0.0)),
                    std::invalid_argument);
    CHECK_THROWS_AS((KalmanFilter<1>(z, Q, Q, -1.0)),
                    std::invalid_argument);
}
