#ifndef TATOOINE_CAVITY_H
#define TATOOINE_CAVITY_H

#include "field.h"
#include "grid_sampler.h"
#include "interpolation.h"
#include "tensor.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct cavity : field<cavity, double, 2, 2> {
  using parent_t = field<cavity, double, 2, 2>;
  using parent_t::real_t;
  using grid_t = grid_sampler<
      real_t, 3, vec<real_t, 2>, interpolation::hermite,
      interpolation::hermite, interpolation::linear>;
  grid_t grid;
  using parent_t::pos_t;
  using parent_t::tensor_t;
  static constexpr vec<size_t, 3> res{256, 96, 100};
  //static constexpr grid domain{
  //    linspace{-1.0, 8.1164, res(0)},
  //    linspace{-1.0, 1.5, res(1)},
  //    linspace{0.0, 10.0, res(2)}};

  cavity(const std::string& path)
      : grid(path) {}

  tensor_t evaluate(const pos_t& x, real_t t) const {
    return grid(x(0), x(1), t);
  }

  bool in_domain(const pos_t& x, real_t t) const {
    return grid.in_domain(x(0), x(1), t) && !(x(0) < -0.1 && x(1) < 0.03) &&
           !(x(0) > 4 && x(1) < 0.03);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
