#ifndef TATOOINE_SAMPLED_FIELD_H
#define TATOOINE_SAMPLED_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/sampler.h>

#include <memory>
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
  Sampler m_sampler;
  //============================================================================
  sampled_field(Sampler sampler)
      : m_samplersampler} {}
  //----------------------------------------------------------------------------
  sampled_field(sampled_field const& other) : m_sampler{other.m_sampler} {}
  sampled_field(sampled_field&& other)
      : m_sampler{std::move(other.m_sampler)} {}
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t&          x,
                                        [[maybe_unused]] Real t = 0) const
      -> tensor_t final {
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
  constexpr auto in_domain(const pos_t& x, [[maybe_unused]] Real t) const
      -> bool final {
    return [&] {
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
  }
  //----------------------------------------------------------------------------
  auto sampler() -> auto& { return m_sampler; }
  auto sampler() const -> auto const& { return m_sampler; }
};
//==============================================================================
template <typename Grid, typename Real, size_t ... TensorDims, bool HasNonConstReference,
          template <typename> typename... InterpolationKernels>
sampled_field(
    sampler<typed_multidim_property<Grid, tensor<Real, TensorDims...>, HasNonConstReference,
                                    InterpolationKernels...>>)
    ->sampled_field<sampler<typep_multidim_property<Grid, tensor<Real, TensorDims...>,
                                                    HasNonConstReference>>,
                    Real, Grid::num_dimensions(), TensorDims...>
//==============================================================================
template <typename Grid, typename Real, size_t M, size_t N, bool HasNonConstReference,
          template <typename> typename... InterpolationKernels>
sampled_field(
    sampler<typed_multidim_property<Grid, mat<Real, M, N>, HasNonConstReference,
                                    InterpolationKernels...>>)
    ->sampled_field<sampler<typep_multidim_property<Grid, mat<Real, M, N>,
                                                    HasNonConstReference>>,
                    Real, Grid::num_dimensions(), M, N>
//==============================================================================
template <typename Grid, typename Real, size_t N, bool HasNonConstReference,
          template <typename> typename... InterpolationKernels>
sampled_field(
    sampler<typed_multidim_property<Grid, vec<Real, N>, HasNonConstReference,
                                    InterpolationKernels...>>)
    ->sampled_field<sampler<typep_multidim_property<Grid, vec<Real, N>,
                                                    HasNonConstReference>>,
                    Real, Grid::num_dimensions(), N>
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference,
          template <typename> typename... InterpolationKernels>
sampled_field(
    sampler<typed_multidim_property<Grid, Real, HasNonConstReference,
                                    InterpolationKernels...>>)
    ->sampled_field<sampler<typep_multidim_property<Grid, Real,
                                                    HasNonConstReference>>,
                    Real, Grid::num_dimensions()>
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
