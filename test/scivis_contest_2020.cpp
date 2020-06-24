#include <tatooine/netcdf.h>
#include <tatooine/for_loop.h>
#include <tatooine/grid.h>
#include <tatooine/random.h>
#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/line.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
using namespace netCDF::exceptions;
static std::string const file_path = "2011013100.nc";
//==============================================================================
TEST_CASE("scivis_contest_2020_properties",
          "[scivis_contest_2020][print][properties]") {
  netcdf::file              f{file_path, netCDF::NcFile::read};
  std::cerr << "attributes:\n";
  for (auto const& [key, val] : f.attributes()) { std::cerr << key << '\n'; }
  std::cerr << "dimensions:\n";
  for (auto const& [key, val] : f.dimensions()) { std::cerr << key << '\n'; }
  std::cerr << "groups:\n";
  for (auto const& [key, val] : f.groups()) { std::cerr << key << '\n'; }
  std::cerr << "variables:\n";
  for (auto const& var : f.variables<double>()) { std::cerr << var.name() << '\n'; }
  auto u = f.variable<double>("U");
  auto v = f.variable<double>("V");
  auto w = f.variable<double>("W");
  std::cerr << "file.num_dimensions(): " << f.num_dimensions() << '\n';
  for (auto const& [name, dim] : f.dimensions()) {
    std::cerr << "dimension \"" << name << "\": " << dim.getSize() << '\n';
  }
  std::cerr << "u.num_dimensions(): " << u.num_dimensions() << '\n';
  std::cerr << "u.size(0): " << u.dimension_name(0) << " - "
            << u.size(0) << '\n';
  std::cerr << "u.size(1): " << u.dimension_name(1) << " - "
            << u.size(1) << '\n';
  std::cerr << "u.size(2): " << u.dimension_name(2) << " - "
            << u.size(2) << '\n';
  std::cerr << "u.size(3): " << u.dimension_name(3) << " - "
            << u.size(3) << '\n';

  std::cerr << "v.num_dimensions(): " << v.num_dimensions() << '\n';
  std::cerr << "v.size(0): " << v.dimension_name(0) << " - "
            << v.size(0) << '\n';
  std::cerr << "v.size(1): " << v.dimension_name(1) << " - "
            << v.size(1) << '\n';
  std::cerr << "v.size(2): " << v.dimension_name(2) << " - "
            << v.size(2) << '\n';
  std::cerr << "v.size(3): " << v.dimension_name(3) << " - "
            << v.size(3) << '\n';

  std::cerr << "w.num_dimensions(): " << w.num_dimensions() << '\n';
  std::cerr << "w.size(0): " << w.dimension_name(0) << " - "
            << w.size(0) << '\n';
  std::cerr << "w.size(1): " << w.dimension_name(1) << " - "
            << w.size(1) << '\n';
  std::cerr << "w.size(2): " << w.dimension_name(2) << " - "
            << w.size(2) << '\n';
  std::cerr << "w.size(3): " << w.dimension_name(3) << " - "
            << w.size(3) << '\n';
}
//==============================================================================
TEST_CASE("scivis_contest_2020_dimensions",
          "[scivis_contest_2020][axes]") {
  auto f           = netcdf::file{file_path, netCDF::NcFile::read};
  auto t_ax_var    = f.variable<double>("T_AX");
  auto z_mit40_var = f.variable<double>("Z_MIT40");
  auto xg_var      = f.variable<double>("XG");
  auto xc_var      = f.variable<double>("XC");
  auto yg_var      = f.variable<double>("YG");
  auto yc_var      = f.variable<double>("YC");

  linspace const xg_axis{xg_var.read_single(0), xg_var.read_single(499), 500};
  linspace const xc_axis{xc_var.read_single(0), xc_var.read_single(499), 500};
  linspace const yg_axis{yg_var.read_single(0), yg_var.read_single(499), 500};
  linspace const yc_axis{yc_var.read_single(0), yc_var.read_single(499), 500};
  linspace const t_axis{t_ax_var.read_single(0), t_ax_var.read_single(59), 60};
  auto const     z_axis = z_mit40_var.read_as_vector();

  SECTION("XG") {
    auto const data = xg_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        REQUIRE(*next(it) - *it ==
                Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    SECTION("check approx with linspace") {
      auto lin_it = begin(xg_axis);
      for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
        REQUIRE(*it == Approx(*lin_it));
      }
    }
  }
  SECTION("XC") {
    auto const data = xc_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        REQUIRE(*next(it) - *it ==
                Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    SECTION("check approx with linspace") {
      auto lin_it = begin(xc_axis);
      for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
        REQUIRE(*it == Approx(*lin_it));
      }
    }
  }
  SECTION("YG") {
    auto const data = yg_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        REQUIRE(*next(it) - *it ==
                Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    SECTION("check approx with linspace") {
      auto lin_it = begin(yg_axis);
      for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
        REQUIRE(*it == Approx(*lin_it));
      }
    }
  }
  SECTION("YC") {
    auto const data = yc_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        REQUIRE(*next(it) - *it ==
                Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    SECTION("check approx with linspace") {
      auto lin_it = begin(yc_axis);
      for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
        REQUIRE(*it == Approx(*lin_it));
      }
    }
  }
  SECTION("T_AX spacing") {
    auto const data = t_ax_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        REQUIRE(*next(it) - *it ==
                Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    SECTION("check approx with linspace") {
      auto lin_it = begin(t_axis);
      for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
        REQUIRE(*it == Approx(*lin_it));
      }
    }
  }
  // NOTE: Z_MIT40 is not linear
  SECTION("Z_MIT40 spacing") {
    auto const data = z_mit40_var.read_as_vector();
    SECTION("check linear spacing") {
      for (auto it = begin(data); it != prev(end(data), 2); ++it) {
        //CHECK(*next(it) - *it ==
        //        Approx(*next(it, 2) - *next(it)).margin(1e-5));
      }
    }
    //SECTION("check approx with linspace") {
    //  auto const z_mit40_size = z_mit40_var.dimension(0);
    //  linspace   z_mit40_lin{z_mit40_var.read_single(0),
    //                       z_mit40_var.read_single(z_mit40_size - 1),
    //                       z_mit40_size};
    //  auto       lin_it = begin(xg_axis);
    //  for (auto it = begin(data); it != prev(end(data), 2); ++it, ++lin_it) {
    //    CHECK(*it == Approx(*lin_it));
    //  }
    //}
  }

  //chunked_multidim_array<double> u{{0, 0, 0, 0}, {10, 10, 10, 10}};
  //u_var.read(u);
}
//==============================================================================
TEST_CASE("scivis_contest_2020_as_grid_property",
          "[scivis_contest_2020][grid][property]") {
  auto f           = netcdf::file{file_path, netCDF::NcFile::read};
  auto t_ax_var    = f.variable<double>("T_AX");
  auto z_mit40_var = f.variable<double>("Z_MIT40");
  auto xg_var      = f.variable<double>("XG");
  auto yc_var      = f.variable<double>("YC");
  auto u_var        = f.variable<double>("U");

  linspace  xg_axis{xg_var.read_single(0), xg_var.read_single(499), 500};
  linspace  yc_axis{yc_var.read_single(0), yc_var.read_single(499), 500};
  linspace  t_axis{t_ax_var.read_single(0), t_ax_var.read_single(59), 60};
  auto z_axis = z_mit40_var.read_as_vector();
}
//==============================================================================
TEST_CASE("scivis_contest_2020_field",
          "[scivis_contest_2020][field]") {
  using V = fields::scivis_contest_2020_ensemble_member;
  V         v{file_path};
  V::pos_t  x{38, 22, 3};
  V::real_t t = 964205;

  std::cerr << "u(199, 299, 0, 23): "
            <<  v.m_u->data_at(199, 299, 0, 23) << '\n';
  std::cerr << "u(199, 299, 0, 24): "
            <<  v.m_u->data_at(199, 299, 0, 24) << '\n';
  
  auto const      measured_vel = v(x, t);
  vec const       expected_vel{0.0477014, 0.126074, 1.16049e-06};
  CAPTURE(measured_vel, expected_vel);
  REQUIRE(approx_equal(measured_vel, expected_vel));
  
  std::vector<parameterized_line<V::real_t, 3, interpolation::linear>> pathlines;
  size_t const num_pathlines = 10;
  pathlines.reserve(2 * num_pathlines);
  ode::vclibs::rungekutta43<V::real_t, 3> solver;

  vec<V::real_t, 3> sample;
  bool in_domain = false;
  for (size_t i = 0; i < num_pathlines; ++i) {
   do {
      x         = {random_uniform(v.xg_axis.front(), v.xg_axis.back())(),
                   random_uniform(v.yg_axis.front(), v.yg_axis.back())(),
                   random_uniform(v.z_axis.front(), v.z_axis.back())()};
      t         = random_uniform(v.t_axis.front(), v.t_axis.back())();
      in_domain = !v.in_domain(vec{x(0), x(1), x(2)}, t);
      if (in_domain) { sample = v(vec{x(0), x(1), x(2)}, t); }
    } while (!in_domain || sqr_length(sample) == 0);

    std::cerr << "found position! (" << i << ")\n";
    pathlines.emplace_back();
    solver.solve(v, vec{x(0), x(1), x(2)}, t, 1000000,
                 [&pathlines](auto t, const auto& y) {
                   std::cerr << t << '\n';
                   pathlines.back().push_back(y, t);
                 });
    pathlines.emplace_back();
    solver.solve(v, vec{x(0), x(1), x(2)},t,  -1000000,
                 [&pathlines](auto t, const auto& y) {
                   std::cerr << t << '\n';
                   pathlines.back().push_back(y, t);
                 });
  }
  pathlines[0].write_vtk("red_sea0.vtk");
  pathlines[1].write_vtk("red_sea1.vtk");
  write_vtk(pathlines, "read_sea.vtk");
}
//==============================================================================
}  // namespace tatooine::netcdf
//==============================================================================
