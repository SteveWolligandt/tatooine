#include <tatooine/doublegyre.h>
#include <tatooine/grid_sampler.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_sampler_indexing_singletime",
          "[grid_sampler][indexing][singletime]") {
  using namespace tatooine::numerical;
  using namespace tatooine::interpolation;

  const double t = 0;
  grid         discrete_domain{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  doublegyre   v_analytical;
  auto         v_discrete =
      resample<hermite, hermite>(v_analytical, discrete_domain, t);
  for (size_t y = 0; y < discrete_domain.dimension(1).size(); ++y) {
    for (size_t x = 0; x < discrete_domain.dimension(0).size(); ++x) {
      REQUIRE(v_discrete.sampler()[x][y] ==
              v_analytical(discrete_domain(x, y), t));
    }
  }
}
//==============================================================================
TEST_CASE("grid_sampler_indexing_multitime",
          "[grid_sampler][indexing][multitime]") {
  using namespace tatooine::numerical;
  using namespace tatooine::interpolation;

  const linspace discrete_temporal_domain{0.0, 10.0, 101};
  grid discrete_spatial_domain{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  doublegyre v_analytical;
  auto       v_discrete = resample<hermite, hermite, linear>(
      v_analytical, discrete_spatial_domain, discrete_temporal_domain);
  for (size_t t = 0; t < discrete_temporal_domain.size(); ++t) {
    for (size_t y = 0; y < discrete_spatial_domain.dimension(1).size(); ++y) {
      for (size_t x = 0; x < discrete_spatial_domain.dimension(0).size(); ++x) {
        REQUIRE(v_discrete.sampler()[x][y][t] ==
                v_analytical(discrete_spatial_domain(x, y),
                             discrete_temporal_domain[t]));
      }
    }
  }
}
//==============================================================================
TEST_CASE("grid_sampler_sampling_singletime",
          "[grid_sampler][doublegyre][dg][numerical][singletime][sample]") {
  using namespace tatooine::numerical;
  using namespace tatooine::interpolation;
  grid         discrete_domain{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  const double t = 0;
  doublegyre   v_analytical;
  auto         v_discrete =
      resample<hermite, hermite>(v_analytical, discrete_domain, t);

  REQUIRE(v_discrete.sampler().boundingbox().min(0) == 0.0);
  REQUIRE(v_discrete.sampler().boundingbox().max(0) == 2.0);
  REQUIRE(v_discrete.sampler().boundingbox().min(1) == 0.0);
  REQUIRE(v_discrete.sampler().boundingbox().max(1) == 1.0);
  REQUIRE(v_discrete.sampler().data().size(0) == 21);
  REQUIRE(v_discrete.sampler().data().size(1) == 11);

  for (auto v : v_discrete.sampler().vertices()) {
    auto x = v.position();
    INFO("position: " << x);
    auto vel_dg = v_analytical(x, t);
    INFO("doublegyre: " << vel_dg);
    auto vel_dgr = v_discrete(x, t);
    INFO("resampled: " << vel_dgr);
    REQUIRE(approx_equal(vel_dg, vel_dgr, 1e-15));
  }
}
//==============================================================================
TEST_CASE("grid_sampler_sampling_multitime",
          "[grid_sampler][doublegyre][dg][numerical][multitime][sample]") {
  using namespace tatooine::numerical;
  using namespace tatooine::interpolation;
  const grid     discrete_spatial_domain{linspace{0.0, 2.0, 21},
                                     linspace{0.0, 1.0, 11}};
  const linspace discrete_temporal_domain{0.0, 10.0, 101};
  doublegyre     v_analytical;
  auto           v_discrete = resample<hermite, hermite, linear>(
      v_analytical, discrete_spatial_domain, discrete_temporal_domain);

  REQUIRE(v_discrete.sampler().boundingbox().min(0) == 0.0);
  REQUIRE(v_discrete.sampler().boundingbox().min(1) == 0.0);
  REQUIRE(v_discrete.sampler().boundingbox().min(2) == 0.0);
  REQUIRE(v_discrete.sampler().boundingbox().max(0) == 2.0);
  REQUIRE(v_discrete.sampler().boundingbox().max(1) == 1.0);
  REQUIRE(v_discrete.sampler().boundingbox().max(2) == 10.0);
  REQUIRE(v_discrete.sampler().data().size(0) == 21);
  REQUIRE(v_discrete.sampler().data().size(1) == 11);
  REQUIRE(v_discrete.sampler().data().size(2) == 101);

  for (auto v : v_discrete.sampler().vertices()) {
    const auto gridpos = v.position();
    const vec  x{gridpos(0), gridpos(1)};
    const auto t = gridpos(2);
    INFO("position: " << x);
    auto vel_dg = v_analytical(x, t);
    INFO("doublegyre: " << vel_dg);
    auto vel_dgr = v_discrete(x, t);
    INFO("resampled: " << vel_dgr);
    REQUIRE(approx_equal(vel_dg, vel_dgr, 1e-15));
  }
}
//==============================================================================
TEST_CASE("grid_sampler_resample_twice",
          "[grid_sampler][doublegyre][dg][numerical][resample]") {
  using namespace tatooine::numerical;
  using namespace tatooine::interpolation;
  const grid coarse_discrete_spatial_domain{linspace{0.0, 2.0, 4},
                                            linspace{0.0, 1.0, 4}};
  const grid fine_discrete_spatial_domain{linspace{0.0, 2.0, 201},
                                          linspace{0.0, 1.0, 101}};
  doublegyre v_analytical;
  SECTION("linear") {
    auto v_coarse = resample<linear, linear>(v_analytical,
                                             coarse_discrete_spatial_domain, 0);
    auto v_fine =
        resample<linear, linear>(v_coarse, fine_discrete_spatial_domain, 0);

    v_fine.sampler().write_vtk("resampled_twice_linear.vtk");
  }
  SECTION("hermite") {
    auto v_coarse = resample<hermite, hermite>(
        v_analytical, coarse_discrete_spatial_domain, 0);
    auto v_fine =
        resample<hermite, hermite>(v_coarse, fine_discrete_spatial_domain, 0);

    v_fine.sampler().write_vtk("resampled_twice_hermite.vtk");
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
