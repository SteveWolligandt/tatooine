#ifndef TATOOINE_SAMPLER_H
#define TATOOINE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/interpolation.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/crtp.h>
#include <tatooine/exceptions.h>
#include <tatooine/internal_value_type.h>
#include <tatooine/base_tensor.h>

#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct sampler;
//==============================================================================
template <typename TopSampler, typename ValueType,
          template <typename> typename... InterpolationKernels>
struct sampler_view;
//==============================================================================
template <typename Sampler, typename ValueType,
          template <typename> typename... InterpolationKernels>
struct base_sampler_at;
//------------------------------------------------------------------------------
template <typename Sampler, typename ValueType,
          template <typename> typename InterpolationKernel0,
          template <typename> typename InterpolationKernel1,
          template <typename> typename... TailInterpolationKernels>
struct base_sampler_at<Sampler, ValueType, InterpolationKernel0, InterpolationKernel1,
                       TailInterpolationKernels...> {
  using value_type =
      sampler_view<Sampler, ValueType, InterpolationKernel1,
                   TailInterpolationKernels...>;
};
//------------------------------------------------------------------------------
template <typename Sampler, typename ValueType,
          template <typename> typename InterpolationKernel>
struct base_sampler_at<Sampler, ValueType, InterpolationKernel> {
  using value_type       = std::decay_t<ValueType>&;
};
//==============================================================================
template <typename Sampler,
         typename ValueType,
          template <typename> typename... InterpolationKernels>
using base_sampler_at_t =
    typename base_sampler_at<Sampler, ValueType, InterpolationKernels...>::value_type;
//==============================================================================
/// CRTP inheritance class for sampler and sampler_view
template <typename Sampler, typename ValueType,
          template <typename> typename HeadInterpolationKernel,
          template <typename> typename... TailInterpolationKernels>
struct base_sampler : crtp<Sampler> {
  template <typename, typename, template <typename> typename,
            template <typename> typename...>
  friend struct base_sampler;
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using this_t = base_sampler<Sampler, ValueType, HeadInterpolationKernel,
                              TailInterpolationKernels...>;
  using indexing_t =
      base_sampler_at_t<this_t, ValueType, HeadInterpolationKernel,
                        TailInterpolationKernels...>;
  using value_type       = ValueType;
  static constexpr auto current_dimension_index() {
    return Sampler::current_dimension_index();
  }
  static constexpr auto num_dimensions() {
    return sizeof...(TailInterpolationKernels) + 1;
  }
  static constexpr auto num_components() {
    return tatooine::num_components<value_type>;
  }
  using crtp<Sampler>::as_derived;
  //============================================================================
  auto property() -> auto& { return as_derived().property(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto property() const -> auto const& { return as_derived().property(); }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto data_at(Is const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().data_at(is...);
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return as_derived().grid(); }
  ////----------------------------------------------------------------------------
  // auto stencil_coefficients(size_t const dim_index, size_t const i) const {
  //  return as_derived().stencil_coefficients(dim_index, i);
  //}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto position_at(Is const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().position_at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// sampler_view with i as fixed index
  auto at(size_t i) const -> decltype(auto) {
    if constexpr (num_dimensions() > 1) {
      return indexing_t{*this, i};
    } else {
      return data_at(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t i) const -> decltype(auto) { return at(i); }
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize>
  // auto diff_at(unsigned int num_diffs, integral auto const... is) const
  //    -> decltype(auto) {
  //  static_assert(sizeof...(is) == num_dimensions(),
  //                "Number of indices is not equal to number of dimensions.");
  //  return as_derived().template diff_at<DimIndex, StencilSize>(num_diffs,
  //                                                              is...);
  //}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <size_t DimensionIndex, arithmetic X>
#else
  template <size_t DimensionIndex, typename X, enable_if<is_arithmetic<X>> = true>
#endif
  auto cell_index(X const x) const -> decltype(auto) {
    return as_derived().template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
 private:
  auto diff_stencil_coefficients(size_t const vertex_index,
                                 size_t const left_index,
                                 size_t const right_index) const {
    if (left_index == vertex_index - 1 && right_index == vertex_index + 1) {
      return grid().diff_stencil_coefficients_n1_0_p1(current_dimension_index(),
                                                      vertex_index);
    }
    if (left_index == vertex_index && right_index == vertex_index + 2) {
      return grid().diff_stencil_coefficients_0_p1_p2(current_dimension_index(),
                                                      vertex_index);
    }
    if (left_index == vertex_index - 2 && right_index == vertex_index) {
      return grid().diff_stencil_coefficients_n2_n1_0(current_dimension_index(),
                                                      vertex_index);
    }
    if (left_index == vertex_index - 1 && right_index == vertex_index) {
      return grid().diff_stencil_coefficients_n1_0(current_dimension_index(),
                                                   vertex_index);
    }
    if (left_index == vertex_index && right_index == vertex_index + 1) {
      return grid().diff_stencil_coefficients_0_p1(current_dimension_index(),
                                                   vertex_index);
    }
    // TODO calculate actual stencil coefficients
    return std::vector<double>{};
  }
  //----------------------------------------------------------------------------
  /// Calcuates derivative from samples and differential coefficients.
  auto differentiate(std::vector<value_type> const& samples,
                     std::vector<double> const& coeffs, size_t const left_index,
                     size_t const right_index) const {
    value_type df_dx{};
    for (size_t i = left_index; i <= right_index; ++i) {
      if (coeffs[i] != 0) {
        df_dx += coeffs[i] * samples[i];
      }
    }
    return df_dx;
  }
  //----------------------------------------------------------------------------
  template <typename CITHead, typename... CITTail>
  auto sample_cit_zero_derivative(CITHead const& cit_head,
                                  CITTail const&... cit_tail) const {
    auto const& cell_index           = cit_head.first;
    auto const& interpolation_factor = cit_head.second;
    if constexpr (num_dimensions() == 1) {
      return HeadInterpolationKernel<value_type>{
          at(cell_index), at(cell_index + 1)}(interpolation_factor);
    } else {
      return HeadInterpolationKernel<value_type>{
          at(cell_index).sample_cit(cit_tail...),
          at(cell_index + 1).sample_cit(cit_tail...)}(interpolation_factor);
    }
  }
  //----------------------------------------------------------------------------
  template <typename CITHead, typename... CITTail>
  auto sample_cit_one_derivative(CITHead const& cit_head,
                                 CITTail const&... cit_tail) const {
    auto const& cell_index           = cit_head.first;
    auto const& interpolation_factor = cit_head.second;
    auto const& dim = grid().template dimension<current_dimension_index()>();
    auto const  dy  = dim[cell_index + 1] - dim[cell_index];

    size_t left_index_left  = cell_index == 0 ? cell_index : cell_index - 1;
    size_t right_index_left = cell_index == 0 ? cell_index + 2 : cell_index + 1;

    size_t left_index_right =
        cell_index == dim.size() - 2 ? cell_index - 1 : cell_index;
    size_t right_index_right =
        cell_index == dim.size() - 2 ? cell_index + 1 : cell_index + 2;

    std::vector<value_type> samples;
    samples.reserve(right_index_right - left_index_left + 1);
    // get samples
    for (size_t i = left_index_left; i <= right_index_right; ++i) {
      if constexpr (num_dimensions() == 1) {
        samples.push_back(at(i));
      } else {
        samples.push_back(at(i).sample_cit(cit_tail...));
      }
    }
    auto const coeffs_left = diff_stencil_coefficients(
        cell_index, left_index_left, right_index_left);
    auto const dleft_dx = differentiate(samples, coeffs_left, 0,
                                        right_index_left - left_index_left);

    auto const coeffs_right = diff_stencil_coefficients(
        cell_index + 1, left_index_right, right_index_right);
    auto const dright_dx =
        differentiate(samples, coeffs_right, left_index_right - left_index_left,
                      right_index_right - left_index_left);
    if constexpr (num_dimensions() == 1) {
      return HeadInterpolationKernel<value_type>{
          samples[cell_index - left_index_left],
          samples[cell_index - left_index_left + 1], dleft_dx * dy,
          dright_dx * dy}(interpolation_factor);
    } else {
      return HeadInterpolationKernel<value_type>{
          samples[cell_index - left_index_left],
          samples[cell_index - left_index_left + 1], dleft_dx * dy,
          dright_dx * dy}(interpolation_factor);
    }
  }
  //----------------------------------------------------------------------------
  /// Decides if first derivative is needed or not.
  template <typename... CITs>
  constexpr auto sample_cit(CITs const&... cits) const {
    constexpr auto num_derivatives_needed =
        HeadInterpolationKernel<value_type>::num_derivatives;
    if constexpr (num_derivatives_needed == 0) {
      return sample_cit_zero_derivative(cits...);
    } else if constexpr (num_derivatives_needed == 1) {
      return sample_cit_one_derivative(cits...);
    }
  }
  //------------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <size_t... Is, arithmetic... Xs>
#else
  template <size_t... Is, typename... Xs, enable_if<is_arithmetic<Xs...>> = true>
#endif
  constexpr auto sample(std::index_sequence<Is...> /*seq*/,
                        Xs const... xs) const {
    return sample_cit(cell_index<Is>(xs)...);
  }
  //----------------------------------------------------------------------------
  /// Recursive sampling by interpolating using interpolation kernels.
  ///
  /// Calculates cell indices and interpolation factors of every position
  /// component so that only once a binary search must be done and than calls
  /// sample_cit .
 public:
#ifdef __cpp_concepts
  template <arithmetic... Xs>
#else
  template <typename... Xs, enable_if<is_arithmetic<Xs...>> = true>
#endif
  constexpr auto sample(Xs const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    return sample(std::make_index_sequence<num_dimensions()>{}, xs...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
      base_tensor<Tensor, TensorReal, num_dimensions()>const& x) const {
    return invoke_unpacked([this](auto const... xs) { return sample(xs...); },
                           unpack(x));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic... Xs>
#else
  template <typename... Xs, enable_if<is_arithmetic<Xs...>> = true>
#endif
  constexpr auto operator()(Xs const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    return sample(std::make_index_sequence<num_dimensions()>{}, xs...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto operator()(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return sample(x);
  }
};
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct sampler
    : base_sampler<sampler<GridVertexProperty, InterpolationKernels...>,
                   typename GridVertexProperty::value_type,
                   InterpolationKernels...> {
  static_assert(sizeof...(InterpolationKernels) ==
                GridVertexProperty::num_dimensions());
  using property_t = GridVertexProperty;
  using this_t     = sampler<property_t, InterpolationKernels...>;
  using parent_t = base_sampler<this_t, typename GridVertexProperty::value_type,
                                InterpolationKernels...>;
  using value_type = typename parent_t::value_type;
  //============================================================================
  static constexpr size_t current_dimension_index() { return 0; }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() {
    return sizeof...(InterpolationKernels);
  }
  //----------------------------------------------------------------------------
  static_assert(is_floating_point<internal_value_type<value_type>>);
  //============================================================================
 private:
  property_t const& m_property;
  //============================================================================
 public:
  sampler(property_t const& prop) : m_property{prop} {}
  //----------------------------------------------------------------------------
  sampler(sampler const& other)     = default;
  sampler(sampler&& other) noexcept = default;
  //============================================================================
  auto property() const -> auto const& { return m_property; }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return m_property.grid(); }
  auto grid() -> auto& { return m_property.grid(); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(
      sizeof...(Is) ==
      GridVertexProperty::grid_t::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) ==
                       GridVertexProperty::grid_t::num_dimensions())> = true>
#endif
  auto data_at(Is const... is)
      const -> decltype(auto) {
    return m_property(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
  requires(
      sizeof...(Is) ==
      GridVertexProperty::grid_t::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) ==
                       GridVertexProperty::grid_t::num_dimensions())> = true>
#endif
  auto data_at(Is const... is)
      -> decltype(auto) {
    return m_property(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto position_at(Is const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return grid().position_at(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <size_t DimensionIndex, arithmetic X>
#else
  template <size_t DimensionIndex, typename X, enable_if<is_arithmetic<X>> = true>
#endif
  auto cell_index(X const x) const -> decltype(auto) {
    return grid().template cell_index<DimensionIndex>(x);
  }
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize, size_t... Is>
  // auto diff_at(unsigned int const                   num_diffs,
  //             std::array<size_t, num_dimensions()> is,
  //             std::index_sequence<Is...> [>seq<]) const {
  //  static_assert(DimIndex < num_dimensions());
  //  auto const [first_idx, coeffs] =
  //      stencil_coefficients<DimIndex, StencilSize>(is[DimIndex], num_diffs);
  //  value_type d{};
  //  is[DimIndex] = first_idx;
  //  for (size_t i = 0; i < StencilSize; ++i, ++is[DimIndex]) {
  //    if (coeffs(i) != 0) {
  //      d += coeffs(i) * data_at(is);
  //    }
  //  }
  //  return d;
  //}
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize>
  // auto diff_at(unsigned int const num_diffs, integral auto... is) const {
  //  static_assert(DimIndex < num_dimensions());
  //  static_assert(sizeof...(is) == num_dimensions(),
  //                "Number of indices does not match number of dimensions.");
  //  return diff_at<DimIndex, StencilSize>(
  //      num_diffs, std::array{static_cast<size_t>(is)...},
  //      std::make_index_sequence<num_dimensions()>{});
  //}
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize>
  // auto stencil_coefficients(size_t const       i,
  //                          unsigned int const num_diffs) const {
  //  return m_grid->template stencil_coefficients<DimIndex, StencilSize>(
  //      i, num_diffs);
  //}
};
//==============================================================================
/// holds an object of type TopSampler which can either be
/// sampler or sampler_view and a fixed index of the top
/// sampler
template <typename TopSampler, typename ValueType, 
          template <typename> typename... InterpolationKernels>
struct sampler_view
    : base_sampler<sampler_view<TopSampler, ValueType, InterpolationKernels...>,
                   typename TopSampler::value_type, InterpolationKernels...> {
  //============================================================================
  static constexpr auto data_is_changeable() {
    return TopSampler::data_is_changeable();
  }
  using this_t   = sampler_view<TopSampler, ValueType, InterpolationKernels...>;
  using parent_t = base_sampler<this_t, ValueType, InterpolationKernels...>;
  using value_type = ValueType;
  //============================================================================
  static constexpr auto num_dimensions() {
    return TopSampler::num_dimensions() - 1;
  }
  //============================================================================
  static constexpr auto current_dimension_index() {
    return TopSampler::current_dimension_index() + 1;
  }
  //============================================================================
  TopSampler const& m_top_sampler;
  size_t            m_fixed_index;
  //============================================================================
  sampler_view(TopSampler const& top_sampler, size_t const fixed_index)
      : m_top_sampler{top_sampler}, m_fixed_index{fixed_index} {}
  //============================================================================
  constexpr auto property() -> auto& { return m_top_sampler.property(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto property() const -> auto const& {
    return m_top_sampler.property();
  }
  //------------------------------------------------------------------------------
  /// returns data of top sampler at m_fixed_index and index list is...
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  constexpr auto data_at(Is const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler.data_at(m_fixed_index, is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <size_t DimensionIndex, arithmetic X>
#else
  template <size_t DimensionIndex, typename X, enable_if<is_arithmetic<X>> = true>
#endif
  constexpr auto cell_index(X const x) const -> decltype(auto) {
    return m_top_sampler.template cell_index<DimensionIndex>(x);
  }
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize>
  // auto diff_at(unsigned int num_diffs, integral auto... is) const
  //    -> decltype(auto) {
  //  static_assert(sizeof...(is) == num_dimensions(),
  //                "Number of indices is not equal to number of dimensions.");
  //  return m_top_sampler.template diff_at<DimIndex, StencilSize>(
  //      num_diffs, m_fixed_index, is...);
  //}
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return m_top_sampler.grid(); }
  ////----------------------------------------------------------------------------
  // auto stencil_coefficients(size_t const dim_index, size_t const i) const {
  //  return m_top_sampler.stencil_coefficients(dim_index, i);
  //}
};
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct differentiated_sampler;
//==============================================================================
template <typename GridVertexProperty>
struct differentiated_sampler<GridVertexProperty,
                              interpolation::linear, interpolation::linear> {
  static constexpr auto num_dimensions() { return 2; }

 private:
  sampler<GridVertexProperty, interpolation::linear,
          interpolation::linear> const& m_sampler;

 public:
  differentiated_sampler(
      sampler<GridVertexProperty, interpolation::linear,
              interpolation::linear> const& sampler)
      : m_sampler{sampler} {}
  //----------------------------------------------------------------------------
 public:
  constexpr auto sample(arithmetic auto x, arithmetic auto y) const {
    auto const [ix, u] = m_sampler.template cell_index<0>(x);
    auto const [iy, v] = m_sampler.template cell_index<1>(y);
    decltype(auto) a   = m_sampler.data_at(ix, iy);
    decltype(auto) b   = m_sampler.data_at(ix + 1, iy);
    decltype(auto) c   = m_sampler.data_at(ix, iy + 1);
    decltype(auto) d   = m_sampler.data_at(ix + 1, iy + 1);

    auto const k  = d - c - b + a;
    auto const dx = k * v + b - a;
    auto const dy = k * u + c - a;
    using value_type = typename std::decay_t<decltype(m_sampler)>::value_type;
    if constexpr (is_arithmetic<value_type>) {
      return vec{dx, dy};
    } else if constexpr (is_vec<value_type>) {
      return mat{{dx(0), dy(0)},
                 {dx(1), dy(1)}};
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return invoke_unpacked([this](auto const... xs) { return sample(xs...); },
                           unpack(x));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(arithmetic auto const x, arithmetic auto const y) const {
    return sample(x,y);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto operator()(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return invoke_unpacked([this](auto const... xs) { return sample(xs...); },
                           unpack(x));
  }
};
//==============================================================================
template <typename GridVertexProperty>
struct differentiated_sampler<GridVertexProperty, interpolation::linear,
                              interpolation::linear, interpolation::linear> {
  static constexpr auto num_dimensions() { return 3; }

 private:
  sampler<GridVertexProperty, interpolation::linear, interpolation::linear,
          interpolation::linear> const& m_sampler;

 public:
  differentiated_sampler(
      sampler<GridVertexProperty, interpolation::linear, interpolation::linear,
              interpolation::linear> const& sampler)
      : m_sampler{sampler} {}
  //----------------------------------------------------------------------------
 public:
  constexpr auto sample(arithmetic auto x, arithmetic auto y,
                        arithmetic auto z) const {
    auto const [ix, u] = m_sampler.template cell_index<0>(x);
    auto const [iy, v] = m_sampler.template cell_index<1>(y);
    auto const [iz, w] = m_sampler.template cell_index<2>(z);
    decltype(auto) a   = m_sampler.data_at(ix, iy, iz);
    decltype(auto) b   = m_sampler.data_at(ix + 1, iy, iz);
    decltype(auto) c   = m_sampler.data_at(ix, iy + 1, iz);
    decltype(auto) d   = m_sampler.data_at(ix + 1, iy + 1, iz);
    decltype(auto) e   = m_sampler.data_at(ix, iy, iz + 1);
    decltype(auto) f   = m_sampler.data_at(ix + 1, iy, iz + 1);
    decltype(auto) g   = m_sampler.data_at(ix, iy + 1, iz + 1);
    decltype(auto) h   = m_sampler.data_at(ix + 1, iy + 1, iz + 1);

    auto const k  = h - g - f + e - d + c + b - a;
    auto const dx = (k * v + f - e - b + a) * w + (d - c - b + a) * v + b - a;
    auto const dy = (k * u + g - e - c + a) * w + (d - c - b + a) * u + c - a;
    auto const dz = (k * u + g - e - c + a) * v + (f - e - b + a) * u + e - a;
    using value_type = typename std::decay_t<decltype(m_sampler)>::value_type;
    if constexpr (is_arithmetic<value_type>) {
      return vec{dx, dy, dz};
    }
    else if constexpr (is_vec<value_type>) {
      return mat{{dx(0), dy(0), dz(0)},
                 {dx(1), dy(1), dz(1)},
                 {dx(2), dy(2), dz(2)}};
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return invoke_unpacked(unpack(x),
                           [this](auto const... xs) { return sample(xs...); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(arithmetic auto const x, arithmetic auto const y, arithmetic auto const z) const {
    return sample(x,y,z);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto operator()(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return invoke_unpacked([this](auto const... xs) { return sample(xs...); },
                           unpack(x));
  }
};
//==============================================================================
template <typename GridVertexProperty>
auto diff(
    sampler<GridVertexProperty, interpolation::linear, interpolation::linear,
            interpolation::linear> const& sampler) {
  return differentiated_sampler<GridVertexProperty, interpolation::linear,
                                interpolation::linear, interpolation::linear>{
      sampler};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename GridVertexProperty>
auto diff(sampler<GridVertexProperty, interpolation::linear,
                  interpolation::linear> const& sampler) {
  return differentiated_sampler<GridVertexProperty, interpolation::linear,
                                interpolation::linear>{sampler};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
