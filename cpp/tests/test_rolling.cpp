// Rolling-statistics tests.

#include <doctest/doctest.h>

#include "df/rolling_stats.hpp"

#include <cmath>

using namespace df;

TEST_CASE("RollingStats — mean and variance match closed-form on a small window") {
    RollingStats rs(/*window=*/4);
    rs.push(1.0);
    rs.push(2.0);
    rs.push(3.0);
    rs.push(4.0);
    // mean = 2.5; sample variance with N-1 = 3 denominator:
    // (1.5² + 0.5² + 0.5² + 1.5²) / 3 = 5/3 = 1.667
    CHECK(rs.mean() == doctest::Approx(2.5).epsilon(1e-12));
    CHECK(rs.variance() == doctest::Approx(5.0 / 3.0).epsilon(1e-10));
}

TEST_CASE("RollingStats — slides correctly when window fills") {
    RollingStats rs(3);
    rs.push(1.0);
    rs.push(2.0);
    rs.push(3.0);
    // Window now [1, 2, 3]; pushing 4 should evict 1.
    rs.push(4.0);
    // Window now [2, 3, 4]; mean = 3.
    CHECK(rs.mean() == doctest::Approx(3.0).epsilon(1e-12));
    CHECK(rs.size() == 3);
}

TEST_CASE("RollingStats — z-score is zero when sample equals the mean") {
    RollingStats rs(5);
    rs.push(1.0);
    rs.push(2.0);
    rs.push(3.0);
    rs.push(4.0);
    const double z = rs.push(2.5);  // 2.5 is the mean of [1, 2, 3, 4, 2.5]?
    // Actually mean is (1+2+3+4+2.5)/5 = 12.5/5 = 2.5 — push returns
    // the z-score *of the just-pushed sample*, so z(2.5) at mean=2.5
    // should be 0.
    CHECK(z == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("RollingStats — z-score sign correct for outliers") {
    RollingStats rs(20);
    for (int i = 0; i < 20; ++i) rs.push(0.0);  // all zeros
    // First non-zero will have undefined z (stdev=0), so push another zero
    // first to keep things finite, then a clear outlier.
    const double z = rs.push(5.0);  // way above mean 0
    CHECK(z > 0.0);  // sign positive
}

TEST_CASE("RollingStats — rejects window < 2") {
    CHECK_THROWS_AS(RollingStats(0), std::invalid_argument);
    CHECK_THROWS_AS(RollingStats(1), std::invalid_argument);
}
