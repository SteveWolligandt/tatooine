#ifndef TATOOINE_RECTILINEAR_GRID_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_RECTILINEAR_GRID_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/concepts.h>
#include <tatooine/crtp.h>
#include <tatooine/exceptions.h>
#include <tatooine/field.h>
#include <tatooine/internal_value_type.h>
#include <tatooine/interpolation.h>
#include <tatooine/invoke_unpacked.h>

#include <vector>
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename Derived, typename Real, size_t N, typename Tensor>
struct field;
//------------------------------------------------------------------------------
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct vertex_property_sampler;
//==============================================================================
template <typename TopSampler, typename Real, typename ValueType,
          template <typename> typename... InterpolationKernels>
struct vertex_property_sampler_view;
//==============================================================================
template <typename DerivedSampler, typename Real, typename ValueType,
          template <typename> typename... InterpolationKernels>
struct base_vertex_property_sampler_at;
//------------------------------------------------------------------------------
template <typename DerivedSampler, typename Real, typename ValueType,
          template <typename> typename InterpolationKernel0,
          template <typename> typename InterpolationKernel1,
          template <typename> typename... TailInterpolationKernels>
struct base_vertex_property_sampler_at<
    DerivedSampler, Real, ValueType, InterpolationKernel0, InterpolationKernel1,
    TailInterpolationKernels...> {
  using value_type =
      vertex_property_sampler_view<DerivedSampler, Real, ValueType,
                                   InterpolationKernel1,
                                   TailInterpolationKernels...>;
};
//------------------------------------------------------------------------------
template <typename DerivedSampler, typename Real, typename ValueType,
          template <typename> typename InterpolationKernel>
struct base_vertex_property_sampler_at<DerivedSampler, Real, ValueType,
                                       InterpolationKernel> {
  using value_type = std::decay_t<ValueType>&;
};
//==============================================================================
template <typename DerivedSampler, typename Real, typename ValueType,
          template <typename> typename... InterpolationKernels>
using base_sampler_at_t = typename base_vertex_property_sampler_at<
    DerivedSampler, Real, ValueType, InterpolationKernels...>::value_type;
//==============================================================================
/// CRTP inheritance class for grid_vertex_property_sampler and
/// grid_vertex_property_sampler_view
template <typename DerivedSampler, typename Real, typename ValueType,
          template <typename> typename HeadInterpolationKernel,
          template <typename> typename... TailInterpolationKernels>
struct base_vertex_property_sampler {
  template <typename, typename, typename, template <typename> typename,
            template <typename> typename...>
  friend struct base_vertex_property_sampler;
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using this_t = base_vertex_property_sampler<DerivedSampler, Real, ValueType,
                                              HeadInterpolationKernel,
                                              TailInterpolationKernels...>;
  using indexing_t =
      base_sampler_at_t<this_t, Real, ValueType, HeadInterpolationKernel,
                        TailInterpolationKernels...>;
  using real_t     = Real;
  using value_type = ValueType;
  static constexpr auto current_dimension_index() {
    return DerivedSampler::current_dimension_index();
  }
  static constexpr auto num_dimensions() {
    return sizeof...(TailInterpolationKernels) + 1;
  }
  static constexpr auto num_components() {
    return tatooine::num_components<value_type>;
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived_sampler() -> DerivedSampler& {
    return static_cast<DerivedSampler&>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived_sampler() const
      -> DerivedSampler const& {
    return static_cast<DerivedSampler const&>(*this);
  }
  //============================================================================
  auto property() const -> auto const& {
    return as_derived_sampler().property();
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return as_derived_sampler().grid(); }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
  auto data_at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived_sampler().data_at(is...);
  }
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived_sampler().position_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(arithmetic auto const x) const -> decltype(auto) {
    return as_derived_sampler().template cell_index<DimensionIndex>(x);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// vertex_property_sampler_view with i as fixed index
  auto at(size_t const i) const -> decltype(auto) {
    if constexpr (num_dimensions() > 1) {
      return indexing_t{*this, i};
    } else {
      return data_at(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t i) const -> decltype(auto) { return at(i); }
  //----------------------------------------------------------------------------
 protected:
  auto diff_stencil_coefficients(size_t const vertex_index,
                                 int const    negative_offset,
                                 int const    positive_offset) const {
    return grid().diff_stencil_coefficients(
        current_dimension_index(), positive_offset - negative_offset + 1,
        -negative_offset, vertex_index);
  }
  //----------------------------------------------------------------------------
  /// Calcuates derivative from samples and differential coefficients.
  template <typename It>
  auto differentiate(std::vector<double> const& coeffs, It sample_begin,
                     It sample_end) const {
    value_type df_dx{};
    auto       sample_it = sample_begin;
    auto       coeff_it  = begin(coeffs);
    for (; sample_it != sample_end; ++sample_it, ++coeff_it) {
      if (*coeff_it != 0) {
        df_dx += *coeff_it * *sample_it;
      }
    }
    return df_dx;
  }
  //----------------------------------------------------------------------------
  template <typename CellIndexInterpolationFactorHead,
            typename... CellIndexInterpolationFactorTail>
  auto interpolate_cell_without_derivative(
      CellIndexInterpolationFactorHead const& cit_head,
      CellIndexInterpolationFactorTail const&... cit_tail) const {
    auto const [cell_index, interpolation_factor] = cit_head;
    if constexpr (num_dimensions() == 1) {
      return HeadInterpolationKernel<value_type>{
          at(cell_index), at(cell_index + 1)}(interpolation_factor);
    } else {
      return HeadInterpolationKernel<value_type>{
          at(cell_index).interpolate_cell(cit_tail...),
          at(cell_index + 1).interpolate_cell(cit_tail...)}(
          interpolation_factor);
    }
  }
  //----------------------------------------------------------------------------
  template <typename CellIndexInterpolationFactorHead,
            typename... CellIndexInterpolationFactorTail>
  auto interpolate_cell_with_one_derivative(
      CellIndexInterpolationFactorHead const& cit_head,
      CellIndexInterpolationFactorTail const&... cit_tail) const {
    auto const [cell_index, interpolation_factor] = cit_head;
    auto const  left_index                        = cell_index;
    auto const  right_index                       = cell_index + 1;
    auto const& dim = grid().template dimension<current_dimension_index()>();
    constexpr size_t targeted_stencil_size = 5;
    constexpr int    offset                = targeted_stencil_size / 2;

    auto const left_negative_offset =
        left_index < offset ? -int(left_index) : -offset;
    auto const left_positive_offset =
        std::min<int>(dim.size() - left_index - 1,
                      targeted_stencil_size + left_negative_offset - 1);

    auto const right_negative_offset =
        right_index < offset ? -int(right_index) : -offset;
    auto const right_positive_offset =
        std::min<int>(dim.size() - right_index - 1,
                      targeted_stencil_size + right_negative_offset - 1);

    // get samples for calculating derivatives
    std::vector<value_type> samples;
    samples.reserve(right_positive_offset - left_negative_offset + 2);
    // get samples
    for (size_t i = left_negative_offset + left_index;
         i <= right_positive_offset + right_index; ++i) {
      if constexpr (num_dimensions() == 1) {
        samples.push_back(at(i));
      } else {
        samples.push_back(at(i).interpolate_cell(cit_tail...));
      }
    }

    // differentiate left sample
    auto const& coeffs_left = diff_stencil_coefficients(
        left_index, left_negative_offset, left_positive_offset);
    auto const dleft_dx = differentiate(
        coeffs_left, begin(samples),
        begin(samples) + left_positive_offset - left_negative_offset + 1);

    // differentiate right sample
    auto const& coeffs_right = diff_stencil_coefficients(
        right_index, right_negative_offset, right_positive_offset);
    auto const dright_dx = differentiate(
        coeffs_right, begin(samples) + right_negative_offset + 1, end(samples));

    auto const dy = dim[right_index] - dim[left_index];
    return HeadInterpolationKernel<value_type>{
        samples[-left_negative_offset], samples[-left_negative_offset + 1],
        dleft_dx * dy, dright_dx * dy}(interpolation_factor);
  }
  //----------------------------------------------------------------------------
  /// Decides if first derivative is needed or not.
  template <typename... CellIndicesInterpolationFactors>
  constexpr auto interpolate_cell(
      CellIndicesInterpolationFactors const&... cell_indices_interpolation_factors)
      const {
    constexpr auto num_derivatives_needed =
        HeadInterpolationKernel<value_type>::num_derivatives;
    if constexpr (num_derivatives_needed == 0) {
      return interpolate_cell_without_derivative(
          cell_indices_interpolation_factors...);
    } else if constexpr (num_derivatives_needed == 1) {
      return interpolate_cell_with_one_derivative(
          cell_indices_interpolation_factors...);
    }
  }
  //------------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto sample(std::index_sequence<Is...> /*seq*/,
                        arithmetic auto const... xs) const {
    return interpolate_cell(cell_index<Is>(xs)...);
  }
};
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct vertex_property_sampler
    : base_vertex_property_sampler<
          vertex_property_sampler<GridVertexProperty, InterpolationKernels...>,
          typename GridVertexProperty::real_t,
          typename GridVertexProperty::value_type, InterpolationKernels...>,
      tatooine::field<
          vertex_property_sampler<GridVertexProperty, InterpolationKernels...>,
          typename GridVertexProperty::real_t, sizeof...(InterpolationKernels),
          typename GridVertexProperty::value_type> {
  static_assert(sizeof...(InterpolationKernels) ==
                GridVertexProperty::num_dimensions());
  using property_t = GridVertexProperty;
  using this_t = vertex_property_sampler<property_t, InterpolationKernels...>;
  using real_t = typename GridVertexProperty::real_t;
  using value_type = typename GridVertexProperty::value_type;
  using parent_type   = base_vertex_property_sampler<this_t, real_t, value_type,
                                                InterpolationKernels...>;
  using field_parent_type =
      tatooine::field<this_t, real_t, sizeof...(InterpolationKernels), value_type>;
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
  vertex_property_sampler(property_t const& prop) : m_property{prop} {}
  //----------------------------------------------------------------------------
  vertex_property_sampler(vertex_property_sampler const& other)     = default;
  vertex_property_sampler(vertex_property_sampler&& other) noexcept = default;
  //============================================================================
  auto property() const -> auto const& { return m_property; }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return m_property.grid(); }
  //----------------------------------------------------------------------------
  auto data_at(integral auto const... is) const
      -> decltype(auto) requires(sizeof...(is) ==
                                 GridVertexProperty::grid_t::num_dimensions()) {
    return m_property(is...);
  }
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return grid().position_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(arithmetic auto const x) const -> decltype(auto) {
    return grid().template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  auto evaluate(typename field_parent_type::pos_t const& x,
                typename field_parent_type::real_t const /*t*/) const ->
      typename field_parent_type::tensor_t {
    if (!grid().is_inside(x)) {
      return field_parent_type::ood_tensor();
    }
    return invoke_unpacked(
        [&](auto const... xs) {
          return this->sample(std::make_index_sequence<num_dimensions()>{},
                              xs...);
        },
        unpack(x));
  }
};
//==============================================================================
/// holds an object of type TopSampler which can either be
/// grid_vertex_property_sampler or grid_vertex_property_sampler_view and a
/// fixed index of the top grid_vertex_property_sampler
template <typename TopSampler, typename Real, typename ValueType,
          template <typename> typename... InterpolationKernels>
struct vertex_property_sampler_view
    : base_vertex_property_sampler<
          vertex_property_sampler_view<TopSampler, Real, ValueType,
                                       InterpolationKernels...>,
          Real, ValueType, InterpolationKernels...> {
  //============================================================================
  static constexpr auto data_is_changeable() {
    return TopSampler::data_is_changeable();
  }
  using this_t     = vertex_property_sampler_view<TopSampler, Real, ValueType,
                                              InterpolationKernels...>;
  using real_t     = Real;
  using value_type = ValueType;
  using parent_type   = base_vertex_property_sampler<this_t, real_t, value_type,
                                                InterpolationKernels...>;
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
  vertex_property_sampler_view(TopSampler const& top_sampler,
                               size_t const      fixed_index)
      : m_top_sampler{top_sampler}, m_fixed_index{fixed_index} {}
  //============================================================================
  constexpr auto property() const -> auto const& {
    return m_top_sampler.property();
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return m_top_sampler.grid(); }
  //------------------------------------------------------------------------------
  /// returns data of top vertex_property_sampler at
  /// m_fixed_index and index list is...
  constexpr auto data_at(integral auto const... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler.data_at(m_fixed_index, is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  constexpr auto cell_index(arithmetic auto const x) const -> decltype(auto) {
    return m_top_sampler.template cell_index<DimensionIndex>(x);
  }
};
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct differentiated_sampler {
  static constexpr auto num_dimensions() {
    return GridVertexProperty::num_dimensions();
  }

 private:
  vertex_property_sampler<GridVertexProperty, InterpolationKernels...> const&
      m_sampler;

 public:
  differentiated_sampler(vertex_property_sampler<GridVertexProperty,
                                                 InterpolationKernels...> const&
                             vertex_property_sampler)
      : m_sampler{vertex_property_sampler} {}
  //----------------------------------------------------------------------------
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    constexpr auto                    eps = 1e-9;
    vec<TensorReal, num_dimensions()> fw = x, bw = x;

    using value_type = typename std::decay_t<decltype(m_sampler)>::value_type;
    if constexpr (is_arithmetic<value_type>) {
      auto gradient = vec<value_type, num_dimensions()>::zeros();
      for (size_t i = 0; i < num_dimensions(); ++i) {
        fw(i) += eps;
        bw(i) -= eps;
        auto dx = eps + eps;

        if (!m_sampler.grid().is_inside(fw)) {
          fw(i) = x(i);
          dx    = eps;
        }

        if (!m_sampler.grid().is_inside(bw)) {
          bw(i) = x(i);
          dx    = eps;
        }

        gradient(i) = m_sampler(fw) / dx - m_sampler(bw) / dx;
        fw(i)       = x(i);
        bw(i)       = x(i);
      }
      return gradient;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto sample(arithmetic auto const... xs) const {
    return sample(
        vec<typename GridVertexProperty::grid_t::real_t, sizeof...(xs)>{xs...});
  }
};
//==============================================================================
template <typename GridVertexProperty>
struct differentiated_sampler<GridVertexProperty, interpolation::linear,
                              interpolation::linear> {
  static constexpr auto num_dimensions() { return 2; }

 private:
  vertex_property_sampler<GridVertexProperty, interpolation::linear,
                          interpolation::linear> const& m_sampler;

 public:
  differentiated_sampler(
      vertex_property_sampler<GridVertexProperty, interpolation::linear,
                              interpolation::linear> const&
          vertex_property_sampler)
      : m_sampler{vertex_property_sampler} {}
  //----------------------------------------------------------------------------
 public:
  constexpr auto sample(arithmetic auto x, arithmetic auto y) const {
    auto const [ix, u] = m_sampler.template cell_index<0>(x);
    auto const [iy, v] = m_sampler.template cell_index<1>(y);
    decltype(auto) a   = m_sampler.data_at(ix, iy);
    decltype(auto) b   = m_sampler.data_at(ix + 1, iy);
    decltype(auto) c   = m_sampler.data_at(ix, iy + 1);
    decltype(auto) d   = m_sampler.data_at(ix + 1, iy + 1);

    auto const k     = d - c - b + a;
    auto const dx    = k * v + b - a;
    auto const dy    = k * u + c - a;
    using value_type = typename std::decay_t<decltype(m_sampler)>::value_type;
    if constexpr (is_arithmetic<value_type>) {
      return vec{dx, dy};
    } else if constexpr (is_vec<value_type>) {
      return mat{{dx(0), dy(0)}, {dx(1), dy(1)}};
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
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
  vertex_property_sampler<GridVertexProperty, interpolation::linear,
                          interpolation::linear, interpolation::linear> const&
      m_sampler;

 public:
  differentiated_sampler(
      vertex_property_sampler<
          GridVertexProperty, interpolation::linear, interpolation::linear,
          interpolation::linear> const& vertex_property_sampler)
      : m_sampler{vertex_property_sampler} {}
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
    } else if constexpr (is_vec<value_type>) {
      return mat{
          {dx(0), dy(0), dz(0)}, {dx(1), dy(1), dz(1)}, {dx(2), dy(2), dz(2)}};
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, typename TensorReal>
  constexpr auto sample(
      base_tensor<Tensor, TensorReal, num_dimensions()> const& x) const {
    return invoke_unpacked(unpack(x),
                           [this](auto const... xs) { return sample(xs...); });
  }
};
//==============================================================================
template <typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
auto diff(
    vertex_property_sampler<GridVertexProperty, InterpolationKernels...> const&
        vertex_property_sampler) {
  return differentiated_sampler<GridVertexProperty, InterpolationKernels...>{
      vertex_property_sampler};
}
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
