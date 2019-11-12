#ifndef DATASETS_H
#define DATASETS_H

//#include <H5Cpp.h>
#include <tatooine/doublegyre.h>
#include <tatooine/boussinesq.h>
#include <tatooine/fixed_time_field.h>
//#include <tatooine/movinggyre.h>
#include <tatooine/grid.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/interpolation.h>
#include <tatooine/sinuscosinus.h>
#include <sstream>
#include <utility>
#include "real_t.h"

//==============================================================================
static std::string dataset_dir = DATASET_DIR + std::string("/");

//==============================================================================
struct fixed_time_doublegyre
    : tatooine::fixed_time_field<tatooine::numerical::doublegyre<real_t>,
                                 real_t, 2, 2> {
  using this_t = fixed_time_doublegyre;
  using parent_t =
      tatooine::fixed_time_field<tatooine::numerical::doublegyre<real_t>,
                                     real_t, 2, 2>;
  fixed_time_doublegyre() : parent_t{tatooine::numerical::doublegyre<real_t>{}, 0.0} {}
};

//==============================================================================
struct laminar : tatooine::field<laminar, real_t, 2, 2> {
  using this_t = laminar;
  using parent_t = tatooine::field<this_t, real_t, 2, 2>;
  using parent_t::pos_t;
  using parent_t::tensor_t;

  tensor_t evaluate(const pos_t& /*pos*/, real_t /*t*/) const {
    return {1.0 / sqrt(2.0), 1.0 / sqrt(2.0)};
  }

  constexpr bool in_domain(const pos_t& /*pos*/, real_t /*t*/) const {
    return true;
  }
};

//==============================================================================
struct cavity : tatooine::field<cavity, real_t, 2, 2> {
  using grid_t = tatooine::grid_sampler<
      real_t, 3, tatooine::vec<real_t, 2>, tatooine::interpolation::hermite,
      tatooine::interpolation::hermite, tatooine::interpolation::linear>;
  grid_t grid;
  using parent_t = tatooine::field<cavity, real_t, 2, 2>;
  using parent_t::pos_t;
  using parent_t::real_t;
  using parent_t::tensor_t;
  static constexpr tatooine::vec<size_t, 3> res{256, 96, 100};
  static constexpr tatooine::grid domain{
      tatooine::linspace{-1.0, 8.1164, res(0)},
      tatooine::linspace{-1.0, 1.5, res(1)},
      tatooine::linspace{0.0, 10.0, res(2)}};

  cavity()
      : grid(dataset_dir + "2DCavity/Cavity2DTimeFilter3x3x7_100_bin.am") {}

  tensor_t evaluate(const pos_t& x, real_t t) const {
    return grid(x(0), x(1), t);
  }

  bool in_domain(const pos_t& x, real_t t) const {
    return grid.in_domain(x(0), x(1), t) && !(x(0) < -0.1 && x(1) < 0.03) &&
           !(x(0) > 4 && x(1) < 0.03);
  }
};

//==============================================================================
// struct MovingGyre
//     : tatooine::UnsteadyGridSamplerVF<
//           2, real_t, 2, tatooine::interpolation::linear,
//           tatooine::interpolation::linear, tatooine::interpolation::linear> {
//   MovingGyre()
//       : tatooine::UnsteadyGridSamplerVF<
//             2, real_t, 2, tatooine::interpolation::linear,
//             tatooine::interpolation::linear, tatooine::interpolation::linear>(
//             dataset_dir + "movinggyre.am") {}
// };

//==============================================================================
struct rbc : tatooine::field<rbc, real_t, 2, 2> {
  using this_t = rbc;
  using parent_t = tatooine::field<this_t, real_t, 2, 2>;
  using parent_t::pos_t;
  using parent_t::tensor_t;
  using grid_t = tatooine::grid_sampler<real_t, 2, tatooine::vec<real_t, 2>,
                                        tatooine::interpolation::linear,
                                        tatooine::interpolation::linear>;
  static constexpr std::array dim{512ul, 128ul, 201ul};
  static constexpr tatooine::grid domain{
      tatooine::linspace{0.00390625, 3.99609375, dim[0]},
      tatooine::linspace{0.00390625, 0.99609375, dim[1]},
      tatooine::linspace{2000.0, 2020.0, dim[2]}};

  //============================================================================
  rbc() { read_from_binary(); }

  //----------------------------------------------------------------------------
  void read_from_binary();
  tensor_t evaluate(const pos_t& pos, real_t t) const;

  //----------------------------------------------------------------------------
  bool in_domain(const pos_t& p, real_t t) const {
    auto& times = domain.dimension(2);
    return times.front() <= t && t <= times.back() &&
           grids.front().in_domain(p(0), p(1));
  }

  std::vector<grid_t> grids;
};

//==============================================================================
//struct FlappingWing : tatooine::vectorfield<2, real_t, FlappingWing> {
//  using this_t = FlappingWing;
//  using parent_t = tatooine::vectorfield<2, real_t, this_t>;
//  using parent_t::pos_t;
//  using parent_t::tensor_t;
//
//  using grid_t = tatooine::grid_sampler<2, real_t, tatooine::vec<real_t, 2>,
//                                        tatooine::interpolation::linear,
//                                        tatooine::interpolation::linear>;
//  FlappingWing() { load(); }
//  tensor_t evaluate(const pos_t& pos, real_t t) const;
//
//  bool in_domain(const tensor_t& p, real_t t) const;
//  void load();
//  void load_times();
//  void load_data();
//
//  void create_filenames();
//
//  std::vector<real_t> times;
//  std::vector<std::string> filenames;
//  std::vector<grid_t> grids;
//};

//==============================================================================
//struct cylinder : tatooine::vectorfield<2, real_t, cylinder> {
//  using grid_t = tatooine::grid_sampler<
//      3, real_t, tatooine::vec<real_t, 2>, tatooine::interpolation::hermite,
//      tatooine::interpolation::hermite, tatooine::interpolation::linear>;
//  grid_t grid;
//  using parent_t = tatooine::vectorfield<2, real_t, cylinder>;
//  using parent_t::pos_t;
//  using parent_t::real_t;
//  using parent_t::tensor_t;
//  static constexpr tatooine::vec res{560, 160, 61};
//  static constexpr tatooine::grid domain{tatooine::linspace{0.0, 559.0, res[0]},
//                                         tatooine::linspace{0.0, 159.0, res[1]},
//                                         tatooine::linspace{0.0, 60.0, res[2]}};
//
//  cylinder() : grid(dataset_dir + "cylinder.am") {}
//
//  tensor_t evaluate(const pos_t& x, real_t t) const { return grid(x(0), x(1), t); }
//
//  bool in_domain(const pos_t& x, real_t t) const {
//    return grid.in_domain(x(0), x(1), t) &&
//           tatooine::norm(tatooine::vec{79.5, 79.5} - x) > 8.1545;
//  }
//};

//==============================================================================
//struct Cylinder2 : tatooine::vectorfield<2, real_t, Cylinder2> {
//  static auto files();
//  tatooine::grid_sampler<
//      3, real_t, tatooine::vec<real_t, 2>, tatooine::interpolation::hermite,
//      tatooine::interpolation::hermite, tatooine::interpolation::linear>
//      grid;
//  Cylinder2() { load(); }
//  tensor_t evaluate(const pos_t& pos, real_t t) const;
//  bool in_domain(const pos_t& p, real_t t) const;
//  void load();
//};

#endif
