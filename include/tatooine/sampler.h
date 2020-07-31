#ifndef TATOOINE_SAMPLER_H
#define TATOOINE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/exceptions.h>
#include <tatooine/multidim_property.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container,
          template <typename> typename... InterpolationKernels>
struct sampler;
//==============================================================================
template <typename TopSampler,
          template <typename> typename... InterpolationKernels>
struct sampler_view;
//==============================================================================
template <typename Sampler,
          template <typename> typename... InterpolationKernels>
struct base_sampler_at;
//------------------------------------------------------------------------------
template <typename Sampler, template <typename> typename InterpolationKernel0,
          template <typename> typename InterpolationKernel1,
          template <typename> typename... TailInterpolationKernels>
struct base_sampler_at<Sampler, InterpolationKernel0, InterpolationKernel1,
                       TailInterpolationKernels...> {
  using value_type =
      sampler_view<Sampler, InterpolationKernel1, TailInterpolationKernels...>;
  using const_value_type = sampler_view<const Sampler, InterpolationKernel1,
                                        TailInterpolationKernels...>;
};
//------------------------------------------------------------------------------
template <typename Sampler, template <typename> typename InterpolationKernel>
struct base_sampler_at<Sampler, InterpolationKernel> {
  using value_type       = std::decay_t<typename Sampler::value_type>&;
  using const_value_type = std::decay_t<typename Sampler::value_type>;
};
//==============================================================================
template <typename Sampler,
          template <typename> typename... InterpolationKernels>
using base_sampler_at_t =
    typename base_sampler_at<Sampler, InterpolationKernels...>::value_type;
//==============================================================================
template <typename Sampler,
          template <typename> typename... InterpolationKernels>
using base_sampler_at_ct =
    typename base_sampler_at<Sampler,
                             InterpolationKernels...>::const_value_type;
//==============================================================================
/// CRTP inheritance class for sampler and sampler_view
template <typename Sampler, typename T,
          template <typename> typename HeadInterpolationKernel,
          template <typename> typename... TailInterpolationKernels>
struct base_sampler : crtp<Sampler> {
  template <typename, typename, template <typename> typename,
            template <typename> typename...>
  friend struct base_sampler;
  //----------------------------------------------------------------------------
  static constexpr auto current_dimension_index() {
    return Sampler::current_dimension_index();
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() {
    return Sampler::num_dimensions();
  }
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using value_type = T;
  static constexpr auto num_components() {
    return num_components_v<value_type>;
  }
  using this_t     = base_sampler<Sampler, value_type, HeadInterpolationKernel,
                              TailInterpolationKernels...>;
  using indexing_t = base_sampler_at_t<this_t, HeadInterpolationKernel,
                                       TailInterpolationKernels...>;
  using const_indexing_t = base_sampler_at_ct<this_t, HeadInterpolationKernel,
                                              TailInterpolationKernels...>;
  using crtp<Sampler>::as_derived;
  //============================================================================
  auto container() -> auto& {
    return as_derived().container();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& {
    return as_derived().container();
  }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
  auto data_at(integral auto... is) const -> value_type const& {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().data_at(is...);
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& {
    return as_derived().grid();
  }
  ////----------------------------------------------------------------------------
  // auto stencil_coefficients(size_t const dim_index, size_t const i) const {
  //  return as_derived().stencil_coefficients(dim_index, i);
  //}
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
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
      return const_indexing_t{this, i};
    } else {
      return data_at(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t i) const -> decltype(auto) {
    return at(i);
  }
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
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
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
  auto sample_cit_no_first_derivative(CITHead const& cit_head,
                                      CITTail const&... cit_tail) const {
    auto const& cell_index           = cit_head.first;
    auto const& interpolation_factor = cit_head.second;
    if constexpr (num_dimensions() == 1) {
      return HeadInterpolationKernel<value_type>::interpolate(
          at(cell_index), at(cell_index + 1), interpolation_factor);
    } else {
      return HeadInterpolationKernel<value_type>::interpolate(
          at(cell_index).sample_cit(cit_tail...),
          at(cell_index + 1).sample_cit(cit_tail...), interpolation_factor);
    }
  }
  //----------------------------------------------------------------------------
  template <typename CITHead, typename... CITTail>
  auto sample_cit_with_first_derivative(CITHead const& cit_head,
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
      return HeadInterpolationKernel<value_type>{samples[cell_index - left_index_left],
                                     samples[cell_index - left_index_left + 1],
                                     dleft_dx * dy,
                                     dright_dx * dy}(interpolation_factor);
    } else {
      return HeadInterpolationKernel<value_type>{samples[cell_index - left_index_left],
                                     samples[cell_index - left_index_left + 1],
                                     dleft_dx * dy,
                                     dright_dx * dy}(interpolation_factor);
    }
  }
  //----------------------------------------------------------------------------
  /// Decides if first derivative is needed or not.
  template <typename... CITs>
  constexpr auto sample_cit(CITs const&... cits) const {
    if constexpr (!HeadInterpolationKernel<value_type>::needs_first_derivative) {
      return sample_cit_no_first_derivative(cits...);
    } else {
      return sample_cit_with_first_derivative(cits...);
    }
  }
  //----------------------------------------------------------------------------
  /// Recursive sampling by interpolating using interpolation kernels.
  ///
  /// Calculates cell indices and interpolation factors of every position
  /// component so that only once a binary search must be done and than calls
  /// sample_cit .
 public:
  constexpr auto sample(real_number auto const... xs) const {
    static_assert(sizeof...(xs) == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    return sample_cit(cell_index<current_dimension_index()>(xs)...);
  }
};
//==============================================================================
template <typename Grid, typename Container,
          template <typename> typename... InterpolationKernels>
struct sampler
    : base_sampler<sampler<Grid, Container, InterpolationKernels...>,
                   typename Container::value_type, InterpolationKernels...> {
  using grid_t      = Grid;
  using container_t = Container;
  using value_type  = typename container_t::value_type;
  using this_t      = sampler<grid_t, container_t, InterpolationKernels...>;
  using parent_t    = base_sampler<this_t, value_type, InterpolationKernels...>;
  //============================================================================
  static constexpr size_t current_dimension_index() {
    return 0;
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() {
    return sizeof...(InterpolationKernels);
  }
  //----------------------------------------------------------------------------
  static_assert(std::is_floating_point_v<internal_data_type_t<value_type>>);
  //============================================================================
 private:
  grid_t const* m_grid;
  container_t   m_container;
  //============================================================================
 public:
  template <typename... Args>
  sampler(grid_t const& g, Args&&... args)
      : m_grid{&g}, m_container{std::forward<Args>(args)...} {}
  //----------------------------------------------------------------------------
  sampler(sampler const& other) = default;
  //----------------------------------------------------------------------------
  sampler(sampler&& other) noexcept = default;
  //----------------------------------------------------------------------------
  virtual ~sampler() = default;
  //============================================================================
  auto container() -> auto& {
    return m_container;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& {
    return m_container;
  }
  //----------------------------------------------------------------------------
  auto data_at(integral auto const... is) const -> value_type const& {
    return m_container(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Is>
  auto data_at(std::array<Is, num_dimensions()> const& is) const
      -> value_type const& {
    return m_container(is);
  }
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return m_grid->position_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return m_grid->template cell_index<DimensionIndex>(x);
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
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& {
    return *m_grid;
  }
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
template <typename TopSampler,
          template <typename> typename... InterpolationKernels>
struct sampler_view
    : base_sampler<sampler_view<TopSampler, InterpolationKernels...>,
                   typename TopSampler::value_type, InterpolationKernels...> {
  //============================================================================
  static constexpr auto data_is_changeable() {
    return TopSampler::data_is_changeable();
  }
  using value_type = typename TopSampler::value_type;
  using parent_t =
      base_sampler<sampler_view<TopSampler, InterpolationKernels...>,
                   value_type, InterpolationKernels...>;
  //============================================================================
  static constexpr auto num_dimensions() {
    return TopSampler::num_dimensions() - 1;
  }
  //============================================================================
  static constexpr auto current_dimension_index() {
    return TopSampler::current_dimension_index() + 1;
  }
  //============================================================================
  TopSampler* m_top_sampler;
  size_t      m_fixed_index;
  //============================================================================
  sampler_view(TopSampler* top_sampler, size_t fixed_index)
      : m_top_sampler{top_sampler}, m_fixed_index{fixed_index} {}
  //============================================================================
  constexpr auto container() -> auto& {
    return m_top_sampler->container();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto container() const -> auto const& {
    return m_top_sampler->container();
  }
  //------------------------------------------------------------------------------
  /// returns data of top sampler at m_fixed_index and index list is...
  constexpr auto data_at(integral auto... is) const -> value_type const& {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  constexpr auto cell_index(real_number auto const x) const -> decltype(auto) {
    return m_top_sampler->template cell_index<DimensionIndex>(x);
  }
  ////----------------------------------------------------------------------------
  // template <size_t DimIndex, size_t StencilSize>
  // auto diff_at(unsigned int num_diffs, integral auto... is) const
  //    -> decltype(auto) {
  //  static_assert(sizeof...(is) == num_dimensions(),
  //                "Number of indices is not equal to number of dimensions.");
  //  return m_top_sampler->template diff_at<DimIndex, StencilSize>(
  //      num_diffs, m_fixed_index, is...);
  //}
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& {
    return m_top_sampler->grid();
  }
  ////----------------------------------------------------------------------------
  // auto stencil_coefficients(size_t const dim_index, size_t const i) const {
  //  return m_top_sampler->stencil_coefficients(dim_index, i);
  //}
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
