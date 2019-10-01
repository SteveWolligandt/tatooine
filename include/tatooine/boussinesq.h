#ifndef TATOOINE_BOUSSINESQ_H
#define TATOOINE_BOUSSINESQ_H

#include "field.h"
#include "grid_sampler.h"

//==============================================================================
namespace tatooine {
//==============================================================================

struct boussinesq : field<boussinesq, double, 2, 2> {
  using real_t = double;
  using sampler_t =
      grid_sampler<real_t, 3, vec<real_t, 2>, interpolation::hermite,
                   interpolation::hermite, interpolation::linear>;
  sampler_t sampler;
  using parent_t = field<boussinesq, real_t, 2, 2>;
  using parent_t::pos_t;
  using parent_t::tensor_t;
  static constexpr vec<size_t, 3> res{100, 300, 1601};
  static constexpr grid           domain{linspace{-0.5, 0.5, 100},
                               linspace{-0.5, 2.5, 300},
                               linspace{0.0, 20.0, 1601}};
  static constexpr vec            center{0.0, -0.4};
  static constexpr real_t         radius = 0.07;

  boussinesq(const std::string& filepath) : sampler(filepath) {}

  tensor_t evaluate(const pos_t& x, real_t t) const {
    return sampler(x(0), x(1), t);
  }

  bool in_domain(const pos_t& x, real_t t) const {
    return sampler.dimension(0).front() <= x(0) &&
           x(0) <= sampler.dimension(0).back() &&
           sampler.dimension(1).front() <= x(1) &&
           x(1) <= sampler.dimension(1).back() &&
           sampler.dimension(2).front() <= t &&
           t <= sampler.dimension(2).back() && distance(center, x) > radius;
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
