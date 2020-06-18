#ifndef TATOOINE_SAMPLER_H
#define TATOOINE_SAMPLER_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/exceptions.h>
#include <tatooine/multidim_property.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Container, typename... InterpolationKernels>
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
  /// CRTP-virtual method
  auto grid() -> auto& { return as_derived().grid(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto grid() const -> auto const& { return as_derived().grid(); }
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
  decltype(auto) at(size_t i) {
    if constexpr (num_dimensions() > 1) {
      return indexing_t{this, i};
    } else {
      return data_at(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// sampler_view with i as fixed index
  auto at(size_t i) const {
    if constexpr (num_dimensions() > 1) {
      return const_indexing_t{this, i};
    } else {
      return data_at(i);
    }
  }
  //----------------------------------------------------------------------------
  /// recursive sampling by interpolating using HeadInterpolationKernel
  constexpr auto sample(real_number auto x, real_number auto... xs) const {
    static_assert(sizeof...(xs) + 1 == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    auto const [i, t] =
        grid().template cell_index<current_dimension_index()>(x);
    if constexpr (!HeadInterpolationKernel::needs_first_derivative) {
      if constexpr (num_dimensions() > 1) {
        return HeadInterpolationKernel::interpolate(at(i).sample(xs...),
                                                    at(i + 1).sample(xs...), t);
      } else {
        return HeadInterpolationKernel::interpolate(at(i), at(i + 1), t);
      }
    } else {

    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <real_number Real>
  auto sample(const std::array<Real, num_dimensions()>& pos) const {
    return invoke_unpacked(
        [&pos, this](const auto... xs) { return sample(xs...); }, unpack(pos));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename Tensor, real_number Real>
  auto sample(const base_tensor<Tensor, Real, num_dimensions()>& pos) const {
    return invoke_unpacked(
        [&pos, this](const auto... xs) { return sample(xs...); }, unpack(pos));
  }
};
//==============================================================================
template <typename Grid, typename Container, typename... InterpolationKernels>
struct sampler
    : typed_multidim_property<Grid, typename Container::value_type>,
      base_sampler<sampler<Grid, Container, InterpolationKernels...>,
                   typename Container::value_type, InterpolationKernels...> {
  //============================================================================
  using this_t = sampler<Grid, Container, InterpolationKernels...>;

  using property_parent_t =
      typed_multidim_property<Grid, typename Container::value_type>;
  using property_base_t = typename property_parent_t::parent_t;
  using property_parent_t::data_at;

  using container_t = Container;
  using value_type  = typename Container::value_type;

  using base_sampler_parent_t =
      base_sampler<this_t, value_type, InterpolationKernels...>;
  //============================================================================
  static constexpr auto num_dimensions() {
    return property_parent_t::num_dimensions();
  }
  //------------------------------------------------------------------------------
  static constexpr size_t current_dimension_index() { return 0; }
  //============================================================================
  static_assert(num_dimensions() == sizeof...(InterpolationKernels));
  //============================================================================
 private:
  container_t m_container;
  //============================================================================
 public:
  template <typename... Args>
  sampler(Grid const& grid, Args&&... args)
      : property_parent_t{grid}, m_container{std::forward<Args>(args)...} {}
  //----------------------------------------------------------------------------
  sampler(sampler const& other) = default;
  //----------------------------------------------------------------------------
  sampler(sampler&& other) = default;
  //----------------------------------------------------------------------------
  virtual ~sampler() = default;
  //============================================================================
  auto container() -> auto& { return m_container; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& { return m_container; }
  //----------------------------------------------------------------------------
  /// CRTP-virtual method
  auto grid() -> auto& { return property_parent_t::grid(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto grid() const -> auto const& { return property_parent_t::grid(); }
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<property_base_t> override {
    return std::unique_ptr<this_t>{new this_t{*this}};
  }
  //----------------------------------------------------------------------------
 private:
  template <std::size_t... Seq>
  auto data_at(std::array<std::size_t, num_dimensions()> const& is,
               std::index_sequence<Seq...>) -> decltype(auto) {
    return m_container(is[Seq]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t... Seq>
  auto data_at(std::array<std::size_t, num_dimensions()> const& is,
               std::index_sequence<Seq...>) const -> decltype(auto) {
    return m_container(is[Seq]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto data_at(std::array<std::size_t, num_dimensions()> const& is)
      -> value_type& override {
    return data_at(is, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(std::array<std::size_t, num_dimensions()> const& is) const
      -> value_type const& override {
    return data_at(is, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto sample(typename Grid::pos_t const& x) const -> value_type override {
    return base_sampler_parent_t::sample(x);
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
  using value_type = typename TopSampler::value_type;
  static constexpr auto num_dimensions() {
    return TopSampler::num_dimensions() - 1;
  }
  using parent_t =
      base_sampler<sampler_view<TopSampler, InterpolationKernels...>,
                   value_type, InterpolationKernels...>;
  //------------------------------------------------------------------------------
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
  auto grid() -> auto& { return m_top_sampler->grid(); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto grid() const -> auto const& { return m_top_sampler->grid(); }
  //----------------------------------------------------------------------------
  /// returns data of top grid at m_fixed_index and index list is...
  template <typename _TopSampler                                  = TopSampler,
            std::enable_if_t<!std::is_const_v<_TopSampler>, bool> = true>
  auto data_at(integral auto... is) -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// returns data of top grid at m_fixed_index and index list is...
  auto data_at(integral auto... is) const -> decltype(auto) {
    static_assert(sizeof...(is) == num_dimensions(),
                  "Number of indices is not equal to number of dimensions.");
    return m_top_sampler->data_at(m_fixed_index, is...);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
