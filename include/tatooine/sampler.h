#ifndef TATOOINE_SAMPLER_H
#define TATOOINE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/exceptions.h>
#include <tatooine/multidim_property.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename SamplerImpl, typename Container,
          typename... InterpolationKernels>
struct sampler;
//==============================================================================
template <typename TopSampler, typename... InterpolationKernels>
struct sampler_view;
//==============================================================================
template <typename Sampler, typename... InterpolationKernels>
struct base_sampler_at;
//------------------------------------------------------------------------------
template <typename Sampler, typename InterpolationKernel0,
          typename InterpolationKernel1, typename... TailInterpolationKernels>
struct base_sampler_at<Sampler, InterpolationKernel0, InterpolationKernel1,
                       TailInterpolationKernels...> {
  using value_type =
      sampler_view<Sampler, InterpolationKernel1, TailInterpolationKernels...>;
  using const_value_type = sampler_view<const Sampler, InterpolationKernel1,
                                        TailInterpolationKernels...>;
};
//------------------------------------------------------------------------------
template <typename Sampler, typename InterpolationKernel>
struct base_sampler_at<Sampler, InterpolationKernel> {
  using value_type       = std::decay_t<typename Sampler::value_type>&;
  using const_value_type = std::decay_t<typename Sampler::value_type>;
};
//==============================================================================
template <typename Sampler, typename... InterpolationKernels>
using base_sampler_at_t =
    typename base_sampler_at<Sampler, InterpolationKernels...>::value_type;
//==============================================================================
template <typename Sampler, typename... InterpolationKernels>
using base_sampler_at_ct =
    typename base_sampler_at<Sampler,
                             InterpolationKernels...>::const_value_type;
//==============================================================================
/// CRTP inheritance class for sampler and sampler_view
template <typename Sampler, typename T, typename HeadInterpolationKernel,
          typename... TailInterpolationKernels>
struct base_sampler : crtp<Sampler> {
  //----------------------------------------------------------------------------
  static constexpr auto current_dimension_index() {
    return Sampler::current_dimension_index();
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() {
    return sizeof...(TailInterpolationKernels) + 1;
  }
  //----------------------------------------------------------------------------
  static_assert(
      (num_dimensions() > 1 &&
       !HeadInterpolationKernel::needs_first_derivative) ||
          num_dimensions() == 1,
      "Interpolation kernels that need first derivative are currently "
      "not supported. Sorry :(");
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using value_type = T;
  static constexpr auto num_components() {
    return num_components_v<value_type>;
  }
  using this_t           = base_sampler<Sampler, T, HeadInterpolationKernel,
                              TailInterpolationKernels...>;
  using indexing_t       = base_sampler_at_t<this_t, HeadInterpolationKernel,
                                       TailInterpolationKernels...>;
  using const_indexing_t = base_sampler_at_ct<this_t, HeadInterpolationKernel,
                                              TailInterpolationKernels...>;
  using crtp<Sampler>::as_derived;
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().position_at(is...);
  }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
  decltype(auto) data_at(integral auto... is) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().data_at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// data at specified indices is...
  /// CRTP-virtual method
  auto data_at(integral auto... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().data_at(is...);
  }
  //----------------------------------------------------------------------------
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// sampler_view with i as fixed index
  auto at(size_t i) -> decltype(auto) {
    if constexpr (num_dimensions() > 1) {
      return indexing_t{this, i};
    } else {
      return data_at(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator[](size_t i) -> decltype(auto) { return at(i); }
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
  auto operator[](size_t i) const -> decltype(auto) { return at(i); }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is)
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return as_derived().template diff_at<DimIndex, StencilSize>(num_diffs,
                                                                is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is) const
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return as_derived().template diff_at<DimIndex, StencilSize>(num_diffs,
                                                                is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return as_derived().template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  /// recursive sampling by interpolating using HeadInterpolationKernel
  constexpr auto sample(real_number auto const x,
                        real_number auto const... xs) const {
    static_assert(sizeof...(xs) + 1 == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    auto const [i, t] = cell_index<current_dimension_index()>(x);
    if constexpr (!HeadInterpolationKernel::needs_first_derivative) {
      if constexpr (num_dimensions() > 1) {
        return HeadInterpolationKernel::interpolate(at(i).sample(xs...),
                                                    at(i + 1).sample(xs...), t);
      } else {
        return HeadInterpolationKernel::interpolate(at(i), at(i + 1), t);
      }
    } else {
      if constexpr (num_dimensions() > 1) {
        // TODO implement
        // return HeadInterpolationKernel{at(i).sample(xs...),
        //                               at(i + 1).sample(xs...),
        //                               diff_at<current_dimension_index(),
        //                               3>(i, .....),
        //                               diff_at<current_dimension_index(),
        //                               3>(i+1, .....)}(t);
      } else {
        // auto const& dim =
        //    grid().template dimension<current_dimension_index()>();
        // return HeadInterpolationKernel{
        //    dim[i],
        //    dim[i + 1],
        //    at(i),
        //    at(i + 1),
        //    diff_at<current_dimension_index(), 3>(1, i),
        //    diff_at<current_dimension_index(), 3>(1, i + 1)}(x);
      }
    }
  }
};
//==============================================================================
template <typename SamplerImpl, typename Container,
          typename... InterpolationKernels>
struct sampler
    : base_sampler<sampler<SamplerImpl, Container, InterpolationKernels...>,
                   typename Container::value_type, InterpolationKernels...>,
      crtp<SamplerImpl> {
  //============================================================================
  static constexpr size_t current_dimension_index() { return 0; }
  //----------------------------------------------------------------------------
  using this_t = sampler<SamplerImpl, Container, InterpolationKernels...>;

  using container_t = Container;
  using value_type  = typename Container::value_type;

  using base_sampler_parent_t =
      base_sampler<this_t, value_type, InterpolationKernels...>;
  //============================================================================
  using base_sampler_parent_t::num_dimensions;
  //============================================================================
  static_assert(num_dimensions() == sizeof...(InterpolationKernels));
  //============================================================================
 private:
  container_t m_container;
  //============================================================================
 public:
  template <typename... Args>
  sampler(Args&&... args) : m_container{std::forward<Args>(args)...} {}
  //----------------------------------------------------------------------------
  sampler(sampler const& other) = default;
  //----------------------------------------------------------------------------
  sampler(sampler&& other) = default;
  //----------------------------------------------------------------------------
  virtual ~sampler() = default;
  //============================================================================
  auto as_sampler_impl() const -> decltype(auto) {
    return crtp<SamplerImpl>::as_derived();
  }
  //----------------------------------------------------------------------------
  auto container() -> auto& { return m_container; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& { return m_container; }
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_sampler_impl().position_at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return as_sampler_impl().template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  auto data_at(integral auto const... is) -> decltype(auto) {
    return m_container(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(integral auto const... is) const -> decltype(auto) {
    return m_container(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Is>
  auto data_at(std::array<Is, num_dimensions()> const& is)
      -> decltype(auto) {
    return m_container(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Is>
  auto data_at(std::array<Is, num_dimensions()> const& is) const
      -> decltype(auto) {
    return m_container(is);
  }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is)
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return as_sampler_impl().template diff_at<DimIndex, StencilSize>(num_diffs,
                                                                     is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto const... is) const
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return as_sampler_impl().template diff_at<DimIndex, StencilSize>(num_diffs,
                                                                     is...);
  }
  //----------------------------------------------------------------------------
  constexpr auto sample(real_number auto const... xs) const {
    return base_sampler_parent_t::sample(xs...);
  }
};
//==============================================================================
/// holds an object of type TopSampler which can either be
/// sampler or sampler_view and a fixed index of the top
/// sampler
template <typename TopSampler, typename... InterpolationKernels>
struct sampler_view
    : base_sampler<sampler_view<TopSampler, InterpolationKernels...>,
                   typename TopSampler::value_type, InterpolationKernels...> {
  //============================================================================
  using value_type = typename TopSampler::value_type;
  using parent_t =
      base_sampler<sampler_view<TopSampler, InterpolationKernels...>,
                   value_type, InterpolationKernels...>;
  //============================================================================
  using parent_t::num_dimensions;
  //============================================================================
  static constexpr size_t current_dimension_index() {
    return TopSampler::current_dimension_index() + 1;
  }
  //============================================================================
  TopSampler* m_top_sampler;
  size_t      m_fixed_index;
  //============================================================================
  sampler_view(TopSampler* top_sampler, size_t fixed_index)
      : m_top_sampler{top_sampler}, m_fixed_index{fixed_index} {}
  //============================================================================
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return m_top_sampler->template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  /// returns data of top sampler at m_fixed_index and index list is...
  template <typename _TopSampler                                  = TopSampler,
            std::enable_if_t<!std::is_const_v<_TopSampler>, bool> = true>
  auto data_at(integral auto... is) -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// returns data of top sampler at m_fixed_index and index list is...
  auto data_at(integral auto... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize,
            typename _TopSampler                                  = TopSampler,
            std::enable_if_t<!std::is_const_v<_TopSampler>, bool> = true>
  auto diff_at(unsigned int num_diffs, integral auto... is) -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->template diff_at<DimIndex + 1, StencilSize>(
        num_diffs, m_fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto... is) const
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->template diff_at<DimIndex + 1, StencilSize>(
        num_diffs, m_fixed_index, is...);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
