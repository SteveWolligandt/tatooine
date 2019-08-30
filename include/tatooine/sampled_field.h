#ifndef TATOOINE_SAMPLED_FIELD_H
#define TATOOINE_SAMPLED_FIELD_H

#include <memory>
#include "field.h"

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
  constexpr decltype(auto) in_domain(const pos_t&            x,
                                     [[maybe_unused]] Real t) const {
    if constexpr (Sampler::num_dimensions() == N) {
      return invoke_unpacked(
          [&](const auto... xs) { return m_sampler->in_domain(xs...); },
          unpack(x));
    } else {
      return invoke_unpacked(
          [&](const auto... xs) { return m_sampler->in_domain(xs..., t); },
          unpack(x));
    }
  }

  //----------------------------------------------------------------------------
  auto&       sampler() { return *m_sampler; }
  const auto& sampler() const { return *m_sampler; }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
