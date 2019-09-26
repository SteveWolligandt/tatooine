#ifndef TATOOINE_SAMPLED_FIELD_H
#define TATOOINE_SAMPLED_FIELD_H

#include <memory>
#include "field.h"
#include "geometry/primitive.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Sampler, typename Real, size_t N, size_t... TensorDims>
struct sampled_field : field<sampled_field<Sampler, Real, N, TensorDims...>,
                             Real, N, TensorDims...> {
  using this_t   = sampled_field<Sampler, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  std::shared_ptr<Sampler> m_sampler;
  std::vector<std::unique_ptr<geometry::primitive<Real, N>>> m_obstacles;

  //============================================================================
  sampled_field(const Sampler& _sampler)
      : m_sampler{std::make_shared<Sampler>(_sampler)} {}
  sampled_field(Sampler&& _sampler)
      : m_sampler{std::make_shared<Sampler>(std::move(_sampler))} {}

  template <typename... Args>
  sampled_field(Args&&... args)
      : m_sampler{std::make_shared<Sampler>(std::forward<Args>(args)...)} {}

  sampled_field(const sampled_field& other)
      : m_sampler{other.m_sampler} {}
  sampled_field(sampled_field&& other)
      : m_sampler{std::move(other.m_sampler)} {}

  //----------------------------------------------------------------------------
  constexpr decltype(auto) evaluate(const pos_t&            x,
                                    [[maybe_unused]] Real t = 0) const {
    if constexpr (Sampler::num_dimensions() == N) {
      return invoke_unpacked(
          [&](const auto... xs) { return m_sampler->sample(xs...); },
          unpack(x));
    } else {
      return invoke_unpacked(
          [&](const auto... xs) { return m_sampler->sample(xs..., t); },
          unpack(x));
    }
  }

  //----------------------------------------------------------------------------
  constexpr decltype(auto) in_domain(const pos_t&          x,
                                     [[maybe_unused]] Real t) const {
    auto on_grid = [&] {
      if constexpr (Sampler::num_dimensions() == N) {
        return invoke_unpacked(
            [&](const auto... xs) { return m_sampler->in_domain(xs...); },
            unpack(x));
      } else {
        return invoke_unpacked(
            [&](const auto... xs) { return m_sampler->in_domain(xs..., t); },
            unpack(x));
      }
    }();

    if (!on_grid) { return false; }
    for (const auto& obstacle : m_obstacles) {
      if (obstacle->is_inside(x)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  auto&       sampler() {
  return *m_sampler;
}
  const auto& sampler() const { return *m_sampler; }

  //----------------------------------------------------------------------------
  template <typename Obstacle>
  auto& add_obstacle(const Obstacle& obstacle) {
    m_obstacles.push_back(std::make_unique<Obstacle>(obstacle));
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
