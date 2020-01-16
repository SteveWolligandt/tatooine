#ifndef DATASETS_H
#define DATASETS_H

//#include <H5Cpp.h>
#include <tatooine/boussinesq.h>
#include <tatooine/doublegyre.h>
#include <tatooine/field.h>
#include <tatooine/fixed_time_field.h>
#include <tatooine/grid.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/interpolation.h>
#include <tatooine/sinuscosinus.h>

#include <sstream>
#include <utility>

//==============================================================================
namespace tatooine {
//==============================================================================
static std::string dataset_dir = DATASET_DIR + std::string("/");

//==============================================================================
template <typename Real>
struct laminar : field<laminar<Real>, Real, 2, 2> {
  using this_t   = laminar<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //----------------------------------------------------------------------------
  tensor_t evaluate(const pos_t& /*pos*/, Real /*t*/) const {
    return {1.0 / sqrt(2.0), 1.0 / sqrt(2.0)};
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& /*pos*/, Real /*t*/) const {
    return true;
  }
};
//==============================================================================
struct cavity : field<cavity, double, 2, 2> {
  using parent_t = field<cavity, double, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;

  using grid_t = grid_sampler<double, 3, vec<real_t, 2>, interpolation::hermite,
                              interpolation::hermite, interpolation::linear>;
  grid_t                            sampler;
  static constexpr vec<size_t, 3>   res{256, 96, 100};
  static constexpr grid             domain{linspace{-1.0, 8.1164, res(0)},
                                           linspace{-1.0,    1.5, res(1)},
                                           linspace{ 0.0,   10.0, res(2)}};

  cavity()
      : sampler(dataset_dir + "2DCavity/Cavity2DTimeFilter3x3x7_100_bin.am") {}

  tensor_t evaluate(const pos_t& x, real_t t) const {
    return sampler(x(0), x(1), t);
  }
  bool in_domain(const pos_t& x, real_t t) const {
    return sampler.in_domain(x(0), x(1), t) && !(x(0) < -0.1 && x(1) < 0.03) &&
           !(x(0) > 4 && x(1) < 0.03);
  }
};

//==============================================================================
// struct MovingGyre
//     : UnsteadyGridSamplerVF<
//           2, real_t, 2, interpolation::linear,
//           interpolation::linear, interpolation::linear> {
//   MovingGyre()
//       : UnsteadyGridSamplerVF<
//             2, real_t, 2, interpolation::linear,
//             interpolation::linear, interpolation::linear>(
//             dataset_dir + "movinggyre.am") {}
// };

//==============================================================================
struct rbc : field<rbc, double, 2, 2> {
  using this_t   = rbc;
  using parent_t = field<this_t, double, 2, 2>;
  using parent_t::pos_t;
  using parent_t::real_t;
  using parent_t::tensor_t;
  using grid_t = grid_sampler<real_t, 2, vec<real_t, 2>, interpolation::linear,
                              interpolation::linear>;
  static constexpr std::array dim{512ul, 128ul, 201ul};
  static constexpr grid       domain{linspace{0.00390625, 3.99609375, dim[0]},
                               linspace{0.00390625, 0.99609375, dim[1]},
                               linspace{2000.0, 2020.0, dim[2]}};

  //============================================================================
  rbc() { read_from_binary(); }

  //----------------------------------------------------------------------------
  void     read_from_binary();
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
// struct FlappingWing : vectorfield<2, real_t, FlappingWing> {
//  using this_t = FlappingWing;
//  using parent_t = vectorfield<2, real_t, this_t>;
//  using parent_t::pos_t;
//  using parent_t::tensor_t;
//
//  using grid_t = grid_sampler<2, real_t, vec<real_t, 2>,
//                                        interpolation::linear,
//                                        interpolation::linear>;
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
// struct cylinder : vectorfield<2, real_t, cylinder> {
//  using grid_t = grid_sampler<
//      3, real_t, vec<real_t, 2>, interpolation::hermite,
//      interpolation::hermite, interpolation::linear>;
//  grid_t grid;
//  using parent_t = vectorfield<2, real_t, cylinder>;
//  using parent_t::pos_t;
//  using parent_t::real_t;
//  using parent_t::tensor_t;
//  static constexpr vec res{560, 160, 61};
//  static constexpr grid domain{linspace{0.0, 559.0, res[0]},
//                                         linspace{0.0, 159.0, res[1]},
//                                         linspace{0.0, 60.0, res[2]}};
//
//  cylinder() : grid(dataset_dir + "cylinder.am") {}
//
//  tensor_t evaluate(const pos_t& x, real_t t) const { return grid(x(0), x(1),
//  t); }
//
//  bool in_domain(const pos_t& x, real_t t) const {
//    return grid.in_domain(x(0), x(1), t) &&
//           norm(vec{79.5, 79.5} - x) > 8.1545;
//  }
//};

//==============================================================================
// struct Cylinder2 : vectorfield<2, real_t, Cylinder2> {
//  static auto files();
//  grid_sampler<
//      3, real_t, vec<real_t, 2>, interpolation::hermite,
//      interpolation::hermite, interpolation::linear>
//      grid;
//  Cylinder2() { load(); }
//  tensor_t evaluate(const pos_t& pos, real_t t) const;
//  bool in_domain(const pos_t& p, real_t t) const;
//  void load();
//};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
