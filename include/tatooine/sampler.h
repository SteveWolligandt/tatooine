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
  auto container() -> auto& { return as_derived().container(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& { return as_derived().container(); }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& {
    return as_derived().grid();
  }
  //----------------------------------------------------------------------------
  auto out_of_domain_value() const -> auto const& {
    return as_derived().out_of_domain_value();
  }
  //----------------------------------------------------------------------------
  auto stencil_coefficients(size_t const dim_index, size_t const i) const {
    return as_derived().stencil_coefficients(dim_index, i);
  }
  //----------------------------------------------------------------------------
  auto position_at(integral auto const... is) const {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().position_at(is...);
  }
  //----------------------------------------------------------------------------
  /// data at specified indices is...
  /// CRTP-virtual method
  auto data_at(integral auto... is) -> value_type& {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return as_derived().data_at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// data at specified indices is...
  /// CRTP-virtual method
  auto data_at(integral auto... is) const -> value_type const& {
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
    auto const cit = cell_index<current_dimension_index()>(x);
    auto const& ci = cit.first;
    auto const& t = cit.second;
    if constexpr (!HeadInterpolationKernel::needs_first_derivative) {
      if constexpr (num_dimensions() == 1) {
        return HeadInterpolationKernel::interpolate(at(ci),
                                                    at(ci + 1), t);
      } else {
        return HeadInterpolationKernel::interpolate(at(ci).sample(xs...),
                                                    at(ci + 1).sample(xs...), t);
      }
    } else {
      auto const& dim = grid().template dimension<current_dimension_index()>();
      // differentiate
      value_type   dleft_dx{};
      value_type   dright_dx{};

      size_t left_index_left  = ci == 0 ? 0 : ci - 1;
      size_t right_index_left = ci == dim.size() - 1 ? dim.size() - 1 : ci + 1;
      if (left_index_left == ci) { ++right_index_left; }
      if (right_index_left == ci) { --left_index_left; }

      size_t left_index_right  = ci;
      size_t right_index_right = ci+1 == dim.size() - 1 ? dim.size() - 1 : ci + 2;
      if (right_index_right == ci) { --left_index_right; }
      size_t const leftest_sample_index = left_index_left;

      if constexpr (num_dimensions() == 1) {
        std::vector<value_type const*> samples(right_index_right -
                                               left_index_left + 1);
        {
          // get samples
          size_t j = 0;
          for (size_t i = left_index_left; i <= right_index_right; ++i, ++j) {
            samples[j] = &at(i);
          }
       }

        // modify indices so that no sample is out of domain
       if (out_of_domain_value()) {
         {
           // modify left index of left sample
           size_t i = left_index_left - leftest_sample_index;
           while (*samples[i++] == *out_of_domain_value() &&
                  left_index_left != right_index_left) {
             ++left_index_left;
             // TODO right_index_left could be increased in certain cases
           }
         }
         {
           // modify right index of left sample
           size_t i = right_index_left - leftest_sample_index;
           while (*samples[i--] == *out_of_domain_value() &&
                  left_index_left != right_index_left) {
             --right_index_left;
             // TODO left_index_left could be decreased in certain cases
           }
         }
         {
           // modify left index of right sample
           size_t i = left_index_right - leftest_sample_index;
           while (*samples[i++] == *out_of_domain_value() &&
                  left_index_right != right_index_right) {
             ++left_index_right;
           }
           // TODO right_index_right could be increased in certain cases
         }
         {
           // modify right index of right sample
           size_t i = right_index_right - leftest_sample_index;
           while (*samples[i--] == *out_of_domain_value() &&
                  left_index_right != right_index_right) {
             --right_index_right;
             // TODO left_index_right could be decreased in certain cases
           }
         }
       }

       auto const coeffs_left = [&]() {
         if (left_index_left == ci - 1 && right_index_left == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n1_0_p1(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci && right_index_left == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_0_p1_p2(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci - 2 && right_index_left == ci) {
           return as_derived().grid().diff_stencil_coefficients_n2_n1_0(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci - 1 && right_index_left == ci) {
           return as_derived().grid().diff_stencil_coefficients_n1_0(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci && right_index_left == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_0_p1(
               current_dimension_index(), ci);
         }
         return std::vector<double>{};
       }();
       auto const coeffs_right = [&]() {
         if (left_index_right == ci && right_index_right == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_n1_0_p1(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci + 1 && right_index_right == ci + 3) {
           return as_derived().grid().diff_stencil_coefficients_0_p1_p2(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci - 1 && right_index_right == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n2_n1_0(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci && right_index_right == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n1_0(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci + 1 && right_index_right == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_0_p1(
               current_dimension_index(), ci + 1);
         }
         return std::vector<double>{};
       }();

       size_t k = 0;
       for (size_t j = left_index_left; j <= right_index_left; ++j, ++k) {
         if (coeffs_left[k] != 0) { dleft_dx += coeffs_left[k] * *samples[k]; }
       }
       k = 0;
       for (size_t j = left_index_right; j <= right_index_right; ++j, ++k) {
         if (coeffs_right[k] != 0) {
           dright_dx += coeffs_right[k] * *samples[k];
         }
       }
       auto const dy = 1 / dim[ci + 1] - dim[ci];
       return HeadInterpolationKernel{*samples[ci - leftest_sample_index],
                                      *samples[ci - leftest_sample_index + 1],
                                      dleft_dx * dy,
                                      dright_dx * dy}(t);
      } else {
        std::vector<value_type> samples(right_index_right - left_index_left +
                                        1);
        {
          // get samples
          size_t j = 0;
          for (size_t i = left_index_left; i <= right_index_right; ++i, ++j) {
            samples[j] = at(i).sample(xs...);
          }
       }

        // modify indices so that no sample is out of domain
       if (out_of_domain_value()) {
         {
           // modify left index of left sample
           size_t i = left_index_left - leftest_sample_index;
           while (samples[i++] == *out_of_domain_value() &&
                  left_index_left != right_index_left) {
             ++left_index_left;
           }
         }
         {
           // modify right index of left sample
           size_t i = right_index_left - leftest_sample_index;
           while (samples[i--] == *out_of_domain_value() &&
                  left_index_left != right_index_left) {
             --right_index_left;
           }
         }
         {
           // modify left index of right sample
           size_t i = left_index_right - leftest_sample_index;
           while (samples[i++] == *out_of_domain_value() &&
                  left_index_right != right_index_right) {
             ++left_index_right;
           }
         }
         {
           // modify right index of right sample
           size_t i = right_index_right - leftest_sample_index;
           while (samples[i--] == *out_of_domain_value() &&
                  left_index_right != right_index_right) {
             --right_index_right;
           }
         }
       }

       auto const coeffs_left = [&]() {
         if (left_index_left == ci - 1 && right_index_left == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n1_0_p1(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci && right_index_left == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_0_p1_p2(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci - 2 && right_index_left == ci) {
           return as_derived().grid().diff_stencil_coefficients_n2_n1_0(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci - 1 && right_index_left == ci) {
           return as_derived().grid().diff_stencil_coefficients_n1_0(
               current_dimension_index(), ci);
         }
         if (left_index_left == ci && right_index_left == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_0_p1(
               current_dimension_index(), ci);
         }
         return std::vector<double>{};
       }();
       auto const coeffs_right = [&]() {
         if (left_index_right == ci && right_index_right == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_n1_0_p1(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci + 1 && right_index_right == ci + 3) {
           return as_derived().grid().diff_stencil_coefficients_0_p1_p2(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci - 1 && right_index_right == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n2_n1_0(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci && right_index_right == ci + 1) {
           return as_derived().grid().diff_stencil_coefficients_n1_0(
               current_dimension_index(), ci + 1);
         }
         if (left_index_right == ci + 1 && right_index_right == ci + 2) {
           return as_derived().grid().diff_stencil_coefficients_0_p1(
               current_dimension_index(), ci + 1);
         }
         return std::vector<double>{};
       }();

       size_t k = 0;
       for (size_t j = left_index_left; j <= right_index_left; ++j, ++k) {
         if (coeffs_left[k] != 0) { dleft_dx += coeffs_left[k] * samples[k]; }
       }
       k = 0;
       for (size_t j = left_index_right; j <= right_index_right; ++j, ++k) {
         if (coeffs_right[k] != 0) {
           dright_dx += coeffs_right[k] * samples[k];
         }
       }
       auto const dy = 1 / dim[ci + 1] - dim[ci];
       return HeadInterpolationKernel{samples[ci - leftest_sample_index],
                                      samples[ci - leftest_sample_index + 1],
                                      dleft_dx * dy, dright_dx * dy}(t);
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
  static constexpr auto num_dimensions() {
    return sizeof...(InterpolationKernels);
  }
  //============================================================================
  using this_t = sampler<SamplerImpl, Container, InterpolationKernels...>;

  using container_t = Container;
  using value_type  = typename Container::value_type;
  static_assert(std::is_floating_point_v<internal_data_type_t<value_type>>);

  using base_sampler_parent_t =
      base_sampler<this_t, value_type, InterpolationKernels...>;
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
  auto out_of_domain_value() const -> auto const& {
    return as_sampler_impl().out_of_domain_value();
  }
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
  auto data_at(integral auto const... is) -> value_type& {
    return m_container(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(integral auto const... is) const -> value_type const& {
    return m_container(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Is>
  auto data_at(std::array<Is, num_dimensions()> const& is) -> value_type& {
    return m_container(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral Is>
  auto data_at(std::array<Is, num_dimensions()> const& is) const
      -> value_type const& {
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
  auto grid() const -> auto const& { return as_sampler_impl().grid(); }
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto stencil_coefficients(size_t const       i,
                            unsigned int const num_diffs) const {
    return as_sampler_impl()
        .template stencil_coefficients<DimIndex, StencilSize>(i, num_diffs);
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
  auto container() -> auto& { return m_top_sampler->container(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& { return m_top_sampler->container(); }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto cell_index(real_number auto const x) const -> decltype(auto) {
    return m_top_sampler->template cell_index<DimensionIndex>(x);
  }
  //----------------------------------------------------------------------------
  auto out_of_domain_value() const -> auto const& {
    return m_top_sampler->out_of_domain_value();
  }
  //----------------------------------------------------------------------------
  /// returns data of top sampler at m_fixed_index and index list is...
  template <typename _TopSampler                                  = TopSampler,
            std::enable_if_t<!std::is_const_v<_TopSampler>, bool> = true>
  auto data_at(integral auto... is) -> value_type& {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// returns data of top sampler at m_fixed_index and index list is...
  auto data_at(integral auto... is) const -> value_type const& {
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
    return m_top_sampler->template diff_at<DimIndex, StencilSize>(
        num_diffs, m_fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t DimIndex, size_t StencilSize>
  auto diff_at(unsigned int num_diffs, integral auto... is) const
      -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->template diff_at<DimIndex, StencilSize>(
        num_diffs, m_fixed_index, is...);
  }
  //----------------------------------------------------------------------------
  auto grid() const -> auto const& {
    return m_top_sampler->grid();
  }
  //----------------------------------------------------------------------------
  auto stencil_coefficients(size_t const dim_index, size_t const i) const {
    return m_top_sampler->stencil_coefficients(dim_index, i);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
