#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/write_png.h>
#include <tatooine/interpolation.h>
#include <tatooine/sampler.h>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t N, template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct default_multidim_property_sampler {
  template <template <typename> typename... _InterpolationKernels>
  using vertex_property_t =
      sampler<GridVertexProperty, _InterpolationKernels...>;
  using type = typename default_multidim_property_sampler<
      N - 1, DefaultInterpolationKernel, GridVertexProperty,
      InterpolationKernels..., DefaultInterpolationKernel>::type;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct default_multidim_property_sampler<0, DefaultInterpolationKernel,
                                         GridVertexProperty,
                                         InterpolationKernels...> {
  using type = sampler<GridVertexProperty, InterpolationKernels...>;
};
template <size_t N, template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
using default_multidim_property_sampler_t =
    typename default_multidim_property_sampler<N, DefaultInterpolationKernel,
                                               GridVertexProperty,
                                               InterpolationKernels...>::type;
//==============================================================================
template <typename Grid>
struct multidim_property {
  //============================================================================
  using this_t = multidim_property<Grid>;
  //============================================================================
  static constexpr auto num_dimensions() {
    return Grid::num_dimensions();
  }
  //============================================================================
 private:
  Grid const* m_grid;
  //============================================================================
 public:
  multidim_property(Grid const& grid) : m_grid{&grid} {}
  multidim_property(multidim_property const& other)     = default;
  multidim_property(multidim_property&& other) noexcept = default;
  //----------------------------------------------------------------------------
  /// Destructor.
  virtual ~multidim_property() {}
  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual auto type() const -> std::type_info const& = 0;
  virtual auto container_type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_t> = 0;
  //----------------------------------------------------------------------------
  auto grid() -> auto& {
    return *m_grid;
  }
  auto grid() const -> auto const& {
    return *m_grid;
  }
  auto set_grid(Grid const& g) { m_grid = &g; }
};
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
struct typed_multidim_property : multidim_property<Grid> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = typed_multidim_property<Grid, ValueType, HasNonConstReference>;
  using parent_t        = multidim_property<Grid>;
  using value_type      = ValueType;
  using const_reference = ValueType const&;
  using reference =
      std::conditional_t<HasNonConstReference, ValueType&, const_reference>;
  using grid_t        = Grid;
  using parent_t::num_dimensions;
  using parent_t::grid;

  //============================================================================
  // ctors
  //============================================================================
  explicit typed_multidim_property(Grid const& grid)
      : parent_t{grid} {}
  typed_multidim_property(typed_multidim_property const&)     = default;
  typed_multidim_property(typed_multidim_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property() override = default;
  //============================================================================
  // methods
  //============================================================================
  auto type() const -> std::type_info const& override {
    return typeid(value_type);
  }
  //----------------------------------------------------------------------------
 private:
  template <template <typename> typename InterpolationKernel>
  auto sampler_() const {
    using sampler_t =
        default_multidim_property_sampler_t<num_dimensions(),
                                            InterpolationKernel, this_t>;
    grid().update_diff_stencil_coefficients();
    return sampler_t{*this};
  }
  //----------------------------------------------------------------------------
 public:
  template <template <typename> typename... InterpolationKernels>
  auto sampler() const {
    if (!grid().diff_stencil_coefficients_created_once()) {
      grid().update_diff_stencil_coefficients();
    }
    static_assert(
        sizeof...(InterpolationKernels) == 0 ||
        sizeof...(InterpolationKernels) == 1 ||
            sizeof...(InterpolationKernels) == num_dimensions(),
        "Number of interpolation kernels does not match number of dimensions.");

    if constexpr (sizeof...(InterpolationKernels) == 0) {
      using sampler_t =
          default_multidim_property_sampler_t<num_dimensions(),
                                              interpolation::cubic, this_t>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    } else if constexpr (sizeof...(InterpolationKernels) == 1) {
      return sampler_<InterpolationKernels...>();
    } else {
      using sampler_t = tatooine::sampler<this_t, InterpolationKernels...>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    }
  }
  //----------------------------------------------------------------------------
  auto linear_sampler() const -> decltype(auto) {
    return sampler<interpolation::linear>();
  }
  //----------------------------------------------------------------------------
  // data access
  //----------------------------------------------------------------------------
  constexpr auto operator()(
      std::array<size_t, num_dimensions()> const& is) const -> decltype(auto) {
    return at(is);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator()(std::array<size_t, num_dimensions()> const& is)
      -> decltype(auto) {
    return at(is);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  constexpr auto operator()(Is const... is) const -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  constexpr auto operator()(Is const... is) -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  auto at(Is const... is) const -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  auto at(Is const... is) -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto at(std::array<size_t, num_dimensions()> const& is) const
      -> const_reference = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto at(std::array<size_t, num_dimensions()> const& is)
      -> reference = 0;
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
  requires(sizeof...(Size) == Grid::num_dimensions())
#else
  template <typename... Size, enable_if<is_integral<Size...>> = true,
            enable_if<(sizeof...(Size) == Grid::num_dimensions())> = true>
#endif
  auto resize(Size const... size) -> decltype(auto) {
    return resize(std::array{static_cast<size_t>(size)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto resize(std::array<size_t, num_dimensions()> const& size)
      -> void = 0;
  //----------------------------------------------------------------------------
#ifdef TATOOINE_HAS_PNG_SUPPORT
#ifdef __cpp_concepts
  template <typename = void>
      requires(num_dimensions() == 2) &&
      (is_vec<ValueType>)
#else
  template <size_t _N                                   = num_dimensions(),
            enable_if<(_N == 2) && (is_vec<ValueType>)> = true>
#endif
          auto write_png(filesystem::path const& path) const -> void {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if (std::isnan(d(0))) {
          for (auto& c : d) {
            c = 0;
          }
        } else {
          for (auto& c : d) {
            c = std::max<typename ValueType::value_type>(
                0, std::min<typename ValueType::value_type>(1, c));
          }
        }
        image[image.get_height() - 1 - y][x].red   = d(0) * 255;
        image[image.get_height() - 1 - y][x].green = d(1) * 255;
        image[image.get_height() - 1 - y][x].blue  = d(2) * 255;
      }
    }
    image.write(path.string());
  }
#ifdef __cpp_concepts
  template <typename = void>
      requires(num_dimensions() == 2) &&
      (is_floating_point<ValueType>)
#else
  template <size_t _N                                   = num_dimensions(),
            enable_if<(_N == 2) && (is_floating_point<ValueType>> = true>
#endif
  auto write_png(filesystem::path const& path, ValueType const min = 0,
                 ValueType const max = 1) const -> void {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if constexpr (is_floating_point<ValueType>) {
          if (std::isnan(d)) {
            d = 0;
          } else {
            d = std::max<ValueType>(min, std::min<ValueType>(max, d));
            d -= min;
            d /= max - min;
          }
          image[image.get_height() - 1 - y][x].red =
          image[image.get_height() - 1 - y][x].green =
          image[image.get_height() - 1 - y][x].blue = d * 255;
        } else if constexpr (is_vec<ValueType>) {
          if (std::isnan(d(0))) {
            for (auto& c : d) {
              c = 0;
            }
          } else {
            for (auto& c : d) {
              c = std::max<typename ValueType::value_type>(
                  min, std::min<typename ValueType::value_type>(max, c));
              c -= min;
              c /= max - min;
            }
          }
          image[image.get_height() - 1 - y][x].red   = d(0) * 255;
          image[image.get_height() - 1 - y][x].green = d(1) * 255;
          image[image.get_height() - 1 - y][x].blue  = d(2) * 255;
        }
      }
    }
    image.write(path.string());
  }
#ifdef __cpp_concepts
  template <typename ColorScale>
      requires(num_dimensions() == 2) &&
      (is_floating_point<ValueType>)
#else
  template <size_t _N = num_dimensions(), typename ColorScale,
            enable_if<(_N == 2) && (is_floating_point<ValueType>)> = true>
#endif
          auto write_png(filesystem::path const& path, ColorScale&& color_scale,
                         ValueType const min = 0, ValueType const max = 1) const
      -> void {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if (std::isnan(d)) {
          d = 0;
        } else {
          d = std::max<ValueType>(min, std::min<ValueType>(max, d));
          d -= min;
          d /= max - min;
        }
        auto const col = color_scale(d) * 255;
        image[image.get_height() - 1 - y][x].red   = col(0);
        image[image.get_height() - 1 - y][x].green = col(1);
        image[image.get_height() - 1 - y][x].blue  = col(2);
      }
    }
    image.write(path.string());
  }
#endif
};
//==============================================================================
#if TATOOINE_HAS_PNG_SUPPORT
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(
    typed_multidim_property<Grid, ValueType, HasNonConstReference> const& prop,
    filesystem::path const& path) -> void {
  prop.write_png(path);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(
    filesystem::path const&                                          path,
    typed_multidim_property<Grid, ValueType, HasNonConstReference> const& prop)
    -> void {
  prop.write_png(path);
}
#endif
//==============================================================================
template <typename Grid, typename ValueType, typename Container>
struct typed_multidim_property_impl
    : typed_multidim_property<
          Grid, ValueType,
          std::is_convertible_v<
              decltype(std::declval<Container&>().at(
                  std::declval<std::array<
                      size_t, Grid::num_dimensions()>>())),
              ValueType&>>,
      Container {
  static_assert(std::is_same_v<ValueType, typename Container::value_type>);
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = typed_multidim_property_impl<Grid, ValueType, Container>;
  static constexpr bool has_non_const_reference = std::is_convertible_v<
      decltype(std::declval<Container&>().at(
          std::declval<std::array<size_t, Grid::num_dimensions()>>())),
      ValueType&>;
  using prop_parent_t =
      typed_multidim_property<Grid, ValueType, has_non_const_reference>;
  using cont_parent_t = Container;
  using value_type = typename prop_parent_t::value_type;
  using reference = typename prop_parent_t::reference;
  using const_reference = typename prop_parent_t::const_reference;
  using grid_t        = Grid;
  using prop_parent_t::num_dimensions;
  //============================================================================
  // ctors
  //============================================================================
  template <typename... Args>
  explicit typed_multidim_property_impl(Grid const& grid, Args&&... args)
      : prop_parent_t{grid}, cont_parent_t{std::forward<Args>(args)...} {}
  typed_multidim_property_impl(typed_multidim_property_impl const&) = default;
  typed_multidim_property_impl(typed_multidim_property_impl&&) noexcept =
      default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property_impl() override = default;
  //============================================================================
  // methods
  //============================================================================
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto container_type() const -> std::type_info const& override {
    return typeid(Container);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true, 
  enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  constexpr auto operator()(Is const... is) const -> decltype(auto) {
    return Container::at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true, 
  enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  constexpr auto operator()(Is const... is) -> decltype(auto) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true, 
  enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  auto at(Is const... is) const -> decltype(auto) {
    return Container::at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true, 
  enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  auto at(Is const... is) -> decltype(auto) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  template <size_t... Is>
  auto at(std::array<size_t, num_dimensions()> const& size,
          std::index_sequence<Is...> /*seq*/) const -> const_reference {
    return Container::at(size[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size) const
      -> const_reference override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  auto at(std::array<size_t, num_dimensions()> const& size,
          std::index_sequence<Is...> /*seq*/) -> reference {
    return Container::at(size[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size)
      -> reference override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto resize(std::array<size_t, num_dimensions()> const& size)
      -> void override {
    Container::resize(size);
  }
};
////==============================================================================
//template <typename Grid, typename F>
//struct typed_multidim_property_lambda
//    : typed_multidim_property<
//          Grid, typename Grid::template invoke_result_with_indices<F>, false> {
//  //============================================================================
//  // typedefs
//  //============================================================================
//  using this_t = typed_multidim_property_lambda<Grid, F>;
//  using value_type = typename Grid::template invoke_result_with_indices<F>;
//  static constexpr bool has_non_const_reference = false;
//  using prop_parent_t =
//      typed_multidim_property<Grid, value_type, has_non_const_reference>;
//  using grid_t        = Grid;
//  using prop_parent_t::num_dimensions;
//  using reference = typename prop_parent_t::reference;
//  using const_reference = typename prop_parent_t::const_reference;
//
//  //============================================================================
//  // members
//  //============================================================================
// private:
//  F m_f;
//
//  //============================================================================
//  // ctors
//  //============================================================================
// public:
//  template <typename _F>
//  typed_multidim_property_lambda(Grid const& grid, _F&& f)
//      : prop_parent_t{grid}, m_f{std::forward<_F>(f)} {}
//  typed_multidim_property_lambda(typed_multidim_property_lambda const&) =
//      default;
//  typed_multidim_property_lambda(typed_multidim_property_lambda&&) noexcept =
//      default;
//  //----------------------------------------------------------------------------
//  ~typed_multidim_property_lambda() override = default;
//  //============================================================================
//  // methods
//  //============================================================================
//  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
//    return std::unique_ptr<this_t>(new this_t{*this});
//  }
//  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <integral... Is>
//  requires(sizeof...(Is) == num_dimensions())
//#else
//  template <typename... Is, enable_if<is_integral<Is...>> = true,
//  enable_if<(sizeof...(Is) == num_dimensions())> = true>
//#endif
//  constexpr auto operator()(Is const... is) const -> decltype(auto) {
//    return m_f(is...);
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//  template <integral... Is>
//  requires(sizeof...(Is) == num_dimensions())
//#else
//  template <typename... Is, enable_if<is_integral<Is...>> = true,
//  enable_if<(sizeof...(Is) == num_dimensions())> = true>
//#endif
//  constexpr auto operator()(Is const... is) -> decltype(auto) {
//    return m_f(is...);
//  }
//  //----------------------------------------------------------------------------
//#ifdef __cpp_concepts
//  template <integral... Is>
//  requires(sizeof...(Is) == Grid::num_dimensions())
//#else
//  template <typename... Is, enable_if<is_integral<Is...>> = true,
//  enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
//#endif
//  auto at(Is const... is) const -> const_reference {
//    return m_f(is...);
//  }
//  //----------------------------------------------------------------------------
//  template <size_t... Is>
//  auto at(std::array<size_t, num_dimensions()> const& size,
//          std::index_sequence<Is...> [>seq<]) const -> const_reference {
//    return m_f(size[Is]...);
//  }
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  auto at(std::array<size_t, num_dimensions()> const& size) const
//      -> const_reference override {
//    return at(size, std::make_index_sequence<num_dimensions()>{});
//  }
//  //----------------------------------------------------------------------------
//  auto resize(std::array<size_t, num_dimensions()> const & [>size<])
//      -> void override {}
//  auto container_type() const -> std::type_info const& override {
//    return typeid(m_f);
//  }
//};
//==============================================================================
template <typename Grid, typename ValueType>
struct multidim_property_derived_type_impl;
template <typename Grid, typename ValueType>
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
using multidim_property_derived_type =
    typename multidim_property_derived_type_impl<Grid, ValueType>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct multidim_property_derived_type_impl<Grid, float> {
  using type = vec<float, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct multidim_property_derived_type_impl<Grid, double> {
  using type = vec<double, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct multidim_property_derived_type_impl<Grid, long double> {
  using type = vec<long double, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, size_t N>
struct multidim_property_derived_type_impl<Grid, vec<float, N>> {
  using type = mat<float, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, size_t N>
struct multidim_property_derived_type_impl<Grid, vec<double, N>> {
  using type = mat<double, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, size_t N>
struct multidim_property_derived_type_impl<Grid, vec<long double, N>> {
  using type = mat<long double, N, Grid::num_dimensions()>;
};
//==============================================================================
template <typename Grid, typename PropValueType, bool PropHasNonConstReference>
struct derived_typed_multidim_property {
  using this_t = derived_typed_multidim_property<Grid, PropValueType,
                                                 PropHasNonConstReference>;
  using prop_t =
      typed_multidim_property<Grid, PropValueType, PropHasNonConstReference>;
  using value_type = multidim_property_derived_type<Grid, PropValueType>;
  using grid_t     = Grid;
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
  //----------------------------------------------------------------------------
  prop_t const& m_prop;
  auto          grid() const -> auto const& { return m_prop.grid(); }
  //----------------------------------------------------------------------------
  // data access
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is>
#endif
  constexpr auto operator()(Is const... is) const -> value_type {
#ifndef __cpp_concepts
    static_assert(sizeof...(Is) == Grid::num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    static_assert(is_integral<Is...>, "Not all index types are integral.");
#endif
        return at(is...);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true,
            enable_if<(sizeof...(Is) == Grid::num_dimensions())> = true>
#endif
  auto at(Is const... is) const -> value_type {
    return at(std::make_index_sequence<Grid::num_dimensions()>{}, is...);
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Seq, typename... Is>
  auto at(std::index_sequence<Seq...> /*seq*/, Is const... is) const
      -> value_type {
    auto const indices = std::array{static_cast<size_t>(is)...};
    if constexpr (is_vec<value_type>) {
      return value_type{[&](auto const dim, auto const index) {
        if (index == 0) {
          auto const coeffs =
              grid().diff_stencil_coefficients_0_p1_p2(dim, index);
          auto p1 = indices;
          auto p2 = indices;
          p1[dim] += 1;
          p2[dim] += 2;
          return m_prop(indices) * coeffs[0] +
                 m_prop(p1) * coeffs[1] +
                 m_prop(p2) * coeffs[2];
        } else if (index == grid().size(dim) - 1) {
          auto const coeffs =
              grid().diff_stencil_coefficients_n2_n1_0(dim, index);
          auto n1 = indices;
          auto n2 = indices;
          n1[dim] -= 1;
          n2[dim] -= 2;
          return m_prop(n2) * coeffs[0] +
                 m_prop(n1) * coeffs[1] +
                 m_prop(indices) * coeffs[2];
        } else {
          auto const coeffs =
              grid().diff_stencil_coefficients_n1_0_p1(dim, index);
          auto n1 = indices;
          auto p1 = indices;
          n1[dim] -= 1;
          p1[dim] += 1;
          return m_prop(n1) * coeffs[0] +
                 m_prop(indices) * coeffs[1] +
                 m_prop(p1) * coeffs[2];
        }
      }(Seq, is)...};
    } else if constexpr (is_mat<value_type>) {
      auto derivative = value_type{};

      ([&](auto const dim, auto const index) {
        if (index == 0) {
          auto const coeffs =
              grid().diff_stencil_coefficients_0_p1_p2(dim, index);
          auto p1 = indices;
          auto p2 = indices;
          p1[dim] += 1;
          p2[dim] += 2;
          derivative.col(dim) = m_prop(indices) * coeffs[0] +
                                m_prop(p1) * coeffs[1] + m_prop(p2) * coeffs[2];
        } else if (index == grid().size(dim) - 1) {
          auto const coeffs =
              grid().diff_stencil_coefficients_n2_n1_0(dim, index);
          auto n1 = indices;
          auto n2 = indices;
          n1[dim] -= 1;
          n2[dim] -= 2;
          derivative.col(dim) = m_prop(n2) * coeffs[0] + m_prop(n1) * coeffs[1] +
                                m_prop(indices) * coeffs[2];
        } else {
          auto const coeffs =
              grid().diff_stencil_coefficients_n1_0_p1(dim, index);
          auto n1 = indices;
          auto p1 = indices;
          n1[dim] -= 1;
          p1[dim] += 1;
          derivative.col(dim) = m_prop(n1) * coeffs[0] +
                                m_prop(indices) * coeffs[1] +
                                m_prop(p1) * coeffs[2];
        }
      }(Seq, is), ...);

      return derivative;
    } else {
      return value_type{};
    }
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel>
  auto sampler_() const {
    using sampler_t =
        default_multidim_property_sampler_t<num_dimensions(),
                                            InterpolationKernel, this_t>;
    grid().update_diff_stencil_coefficients();
    return sampler_t{*this};
  }
  //----------------------------------------------------------------------------
 public:
  template <template <typename> typename... InterpolationKernels>
  auto sampler() const {
    if (!grid().diff_stencil_coefficients_created_once()) {
      grid().update_diff_stencil_coefficients();
    }
    static_assert(
        sizeof...(InterpolationKernels) == 0 ||
        sizeof...(InterpolationKernels) == 1 ||
            sizeof...(InterpolationKernels) == num_dimensions(),
        "Number of interpolation kernels does not match number of dimensions.");

    if constexpr (sizeof...(InterpolationKernels) == 0) {
      using sampler_t =
          default_multidim_property_sampler_t<num_dimensions(),
                                              interpolation::cubic, this_t>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    } else if constexpr (sizeof...(InterpolationKernels) == 1) {
      return sampler_<InterpolationKernels...>();
    } else {
      using sampler_t = tatooine::sampler<this_t, InterpolationKernels...>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    }
  }
  //----------------------------------------------------------------------------
  auto linear_sampler() const -> decltype(auto) {
    return sampler<interpolation::linear>();
  }
  //----------------------------------------------------------------------------
  auto cubic_sampler() const -> decltype(auto) {
    return sampler<interpolation::cubic>();
  }
};
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto diff(typed_multidim_property<Grid, ValueType, HasNonConstReference> const&
              prop) {
  prop.grid().update_diff_stencil_coefficients();
  return derived_typed_multidim_property<Grid, ValueType, HasNonConstReference>{
      prop};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
