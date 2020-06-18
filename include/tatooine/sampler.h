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
struct sampler_iterator;
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
  using iterator         = sampler_iterator<this_t>;
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
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// sampler_view with i as fixed index
  decltype(auto) operator[](size_t i) { return at(i); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// indexing of data.
  /// if num_dimensions() == 1 returns actual data otherwise returns a
  /// sampler_view with i as fixed index
  auto operator[](size_t i) const { return at(i); }
  //----------------------------------------------------------------------------
  /// sampling by interpolating using HeadInterpolationKernel and
  /// iterators
  auto sample(real_number auto x, real_number auto... xs) const {
    static_assert(sizeof...(xs) + 1 == num_dimensions(),
                  "Number of coordinates does not match number of dimensions.");
    auto const [i, t] =
        grid().template cell_index<current_dimension_index()>(x);
    // if (begin() + i + 1 == end()) {
    //  if constexpr (HeadInterpolationKernel::needs_first_derivative) {
    //    // return HeadInterpolationKernel::from_iterators(begin() + i, begin()
    //    + i,
    //    // begin(), end(), t, xs...);
    //  } else {
    //    return HeadInterpolationKernel::from_iterators(begin() + i, begin() +
    //    i, t,
    //                                            xs...);
    //  }
    //}
    // if constexpr (HeadInterpolationKernel::needs_first_derivative) {
    //  // return HeadInterpolationKernel::from_iterators(begin() + i, begin() +
    //  i + 1,
    //  // begin(), end(), t, xs...);
    //} else {
    return HeadInterpolationKernel::from_iterators(begin() + i, begin() + i + 1,
                                                   t, xs...);
    //}
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
  //----------------------------------------------------------------------------
  // auto domain_to_global(real_number auto x, size_t i) const {
  // auto converted = (x - dimension(i).front()) /
  //                 (dimension(i).back() - dimension(i).front());
  // if (converted < 0 || converted > 1) { throw out_of_domain{}; }
  // return converted * (dimension(i).size() - 1);
  //}
  //----------------------------------------------------------------------------
   auto begin() const { return iterator{this, 0}; }
   auto end() const {
     return iterator{this, grid().template size<current_dimension_index()>()};
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
/// holds an object of type Sampler which either can be
/// sampler or sampler_view and an index of that grid
template <typename Sampler, typename... TailInterpolationKernels>
struct sampler_iterator {
  using this_t = sampler_iterator<Sampler, TailInterpolationKernels...>;
  //----------------------------------------------------------------------------
  const Sampler* m_sampler;
  size_t         m_index;
  //----------------------------------------------------------------------------
  auto operator*() const { return m_sampler->at(m_index); }

  auto& operator++() {
    ++m_index;
    return *this;
  }

  auto& operator--() {
    --m_index;
    return *this;
  }

  bool operator==(const this_t& other) const {
    return m_sampler == other.m_sampler && m_index == other.m_index;
  }

  bool operator!=(const this_t& other) const {
    return m_sampler != other.m_sampler || m_index != other.m_index;
  }
  bool operator<(const this_t& other) const { return m_index < other.m_index; }
  bool operator>(const this_t& other) const { return m_index > other.m_index; }
  bool operator<=(const this_t& other) const {
    return m_index <= other.m_index;
  }
  bool operator>=(const this_t& other) const {
    return m_index >= other.m_index;
  }

  auto operator+(size_t rhs) { return this_t{m_sampler, m_index + rhs}; }
  auto operator-(size_t rhs) { return this_t{m_sampler, m_index - rhs}; }

  auto& operator+=(size_t rhs) {
    m_index += rhs;
    return *this;
  }
  auto& operator-=(size_t rhs) {
    m_index -= rhs;
    return *this;
  }
};

//==============================================================================
/// next specification for sampler_iterator
template <typename Sampler, typename... TailInterpolationKernels>
auto next(const sampler_iterator<Sampler, TailInterpolationKernels...>& it,
          size_t                                                        x = 1) {
  return sampler_iterator<Sampler, TailInterpolationKernels...>{it.m_sampler,
                                                                it.m_index + x};
}
//------------------------------------------------------------------------------
/// prev specification for sampler_iterator
template <typename Sampler, typename... TailInterpolationKernels>
auto prev(const sampler_iterator<Sampler, TailInterpolationKernels...>& it,
          size_t                                                        x = 1) {
  return sampler_iterator<Sampler, TailInterpolationKernels...>{it.m_sampler,
                                                                it.m_index - x};
}
//==============================================================================
/// resamples a time step of a field
// template <typename... InterpolationKernels, typename Field, typename
// FieldReal,
//          size_t N, size_t... TensorDims, typename GridReal, typename
//          TimeReal>
// auto resample(const field<Field, FieldReal, N, TensorDims...>& f,
//              const grid<GridReal, N>& g, TimeReal t) {
//  static_assert(sizeof...(InterpolationKernels) > 0, "please specify
//  interpolators"); static_assert(N > 0, "number of dimensions must be greater
//  than 0"); static_assert(sizeof...(InterpolationKernels) == N,
//                "number of interpolators does not match number of
//                dimensions");
//  using real_t   = promote_t<FieldReal, GridReal>;
//  using tensor_t = typename field<Field, real_t, N, TensorDims...>::tensor_t;
//
//  sampled_field<sampler<real_t, N, typename Field::tensor_t,
//  InterpolationKernels...>,
//                real_t, N, TensorDims...>
//      resampled{g};
//
//  auto& data = resampled.sampler().data();
//
//  for (auto v : g.vertices()) {
//    auto is = v.indices();
//    try {
//      data(is) = f(v.position(), t);
//    } catch (std::exception& [>e<]) {
//      if constexpr (std::is_arithmetic_v<tensor_t>) {
//        data(is) = 0.0 / 0.0;
//      } else {
//        data(is) = tensor_t{tag::fill{0.0 / 0.0}};
//      }
//    }
//  }
//  return resampled;
//}
//
////==============================================================================
///// resamples multiple time steps of a field
// template <template <typename> typename... InterpolationKernels, typename
// Field,
//          typename FieldReal, size_t N, typename GridReal, typename TimeReal,
//          size_t... TensorDims>
// auto resample(const field<Field, FieldReal, N, TensorDims...>& f,
//              const grid<GridReal, N>& g, const linspace<TimeReal>& ts) {
//  static_assert(N > 0, "number of dimensions must be greater than 0");
//  static_assert(sizeof...(InterpolationKernels) == N + 1,
//                "number of interpolators does not match number of
//                dimensions");
//  assert(ts.size() > 0);
//  using real_t   = promote_t<FieldReal, GridReal>;
//  using tensor_t = typename field<Field, real_t, N, TensorDims...>::tensor_t;
//
//  sampled_field<
//      sampler<real_t, N + 1, tensor<real_t, TensorDims...>,
//      InterpolationKernels...>, real_t, N, TensorDims...>
//        resampled{g + ts};
//  auto& data = resampled.sampler().data();
//
//  vec<size_t, N + 1> is{tag::zeros};
//  for (auto v : g.vertices()) {
//    for (size_t i = 0; i < N; ++i) { is(i) = v[i].i(); }
//    for (auto t : ts) {
//      try {
//        data(is) = f(v.position(), t);
//      } catch (std::exception& [>e<]) {
//        if constexpr (std::is_arithmetic_v<tensor_t>) {
//          data(is) = 0.0 / 0.0;
//        } else {
//          data(is) = tensor_t{tag::fill{0.0 / 0.0}};
//        }
//      }
//      ++is(N);
//    }
//    is(N) = 0;
//  }
//
//  return resampled;
//}
//==============================================================================
// template <typename Real, size_t N,
//          template <typename> typename... InterpolationKernels>
// void write_png(sampler<Real, 2, Real, InterpolationKernels...> const&
// sampler,
//               std::string const&                              path) {
//  sampler.write_png(path);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
