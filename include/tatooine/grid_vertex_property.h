#ifndef TATOOINE_GRID_VERTEX_PROPERTY_PROPERTY_H
#define TATOOINE_GRID_VERTEX_PROPERTY_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/grid_vertex_property_sampler.h>
#include <tatooine/interpolation.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/type_traits.h>
#include <tatooine/write_png.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t N, template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct default_grid_vertex_sampler {
  template <template <typename> typename... _InterpolationKernels>
  using vertex_property_t =
      grid_vertex_property_sampler<GridVertexProperty,
                                   _InterpolationKernels...>;
  using type = typename default_grid_vertex_sampler<
      N - 1, DefaultInterpolationKernel, GridVertexProperty,
      InterpolationKernels..., DefaultInterpolationKernel>::type;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
struct default_grid_vertex_sampler<0, DefaultInterpolationKernel,
                                   GridVertexProperty,
                                   InterpolationKernels...> {
  using type =
      grid_vertex_property_sampler<GridVertexProperty, InterpolationKernels...>;
};
template <size_t N, template <typename> typename DefaultInterpolationKernel,
          typename GridVertexProperty,
          template <typename> typename... InterpolationKernels>
using default_grid_vertex_sampler_t =
    typename default_grid_vertex_sampler<N, DefaultInterpolationKernel,
                                         GridVertexProperty,
                                         InterpolationKernels...>::type;
//==============================================================================
template <typename Grid>
struct grid_vertex_property {
  //============================================================================
  using this_t        = grid_vertex_property<Grid>;
  using real_t        = typename Grid::real_t;
  using vertex_handle = typename Grid::vertex_handle;
  //============================================================================
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
  //============================================================================
 private:
  Grid const* m_grid;
  //============================================================================
 public:
  grid_vertex_property(Grid const& grid) : m_grid{&grid} {}
  grid_vertex_property(grid_vertex_property const& other)     = default;
  grid_vertex_property(grid_vertex_property&& other) noexcept = default;
  //----------------------------------------------------------------------------
  /// Destructor.
  virtual ~grid_vertex_property() {}
  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual auto type() const -> std::type_info const&           = 0;
  virtual auto container_type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_t> = 0;
  //----------------------------------------------------------------------------
  auto grid() -> auto& { return *m_grid; }
  auto grid() const -> auto const& { return *m_grid; }
  auto set_grid(Grid const& g) { m_grid = &g; }
};
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
struct typed_grid_vertex_property_interface : grid_vertex_property<Grid> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t          = typed_grid_vertex_property_interface<Grid, ValueType,
                                                      HasNonConstReference>;
  using parent_t        = grid_vertex_property<Grid>;
  using value_type      = ValueType;
  using const_reference = ValueType const&;
  using reference =
      std::conditional_t<HasNonConstReference, ValueType&, const_reference>;
  using grid_t = Grid;
  using parent_t::grid;
  using parent_t::num_dimensions;
  using typename parent_t::vertex_handle;

  //============================================================================
  // ctors
  //============================================================================
  explicit typed_grid_vertex_property_interface(Grid const& grid)
      : parent_t{grid} {}
  typed_grid_vertex_property_interface(
      typed_grid_vertex_property_interface const&) = default;
  typed_grid_vertex_property_interface(
      typed_grid_vertex_property_interface&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_grid_vertex_property_interface() override = default;
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
        default_grid_vertex_sampler_t<num_dimensions(), InterpolationKernel,
                                      this_t>;
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
          default_grid_vertex_sampler_t<num_dimensions(), interpolation::cubic,
                                        this_t>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    } else if constexpr (sizeof...(InterpolationKernels) == 1) {
      return sampler_<InterpolationKernels...>();
    } else {
      using sampler_t =
          tatooine::grid_vertex_property_sampler<this_t,
                                                 InterpolationKernels...>;
      if (!grid().diff_stencil_coefficients_created_once()) {
        grid().update_diff_stencil_coefficients();
      }
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
  //----------------------------------------------------------------------------
  // data access
  //----------------------------------------------------------------------------
  constexpr auto operator[](vertex_handle const& h) const -> decltype(auto) {
    return at(h);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator[](vertex_handle const& h) -> decltype(auto) {
    return at(h);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
      constexpr auto
      operator()(Is const... is) const -> decltype(auto) {
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
      constexpr auto
      operator()(Is const... is) -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  constexpr auto at(vertex_handle const& h) const -> decltype(auto) {
    return at(h, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto at(vertex_handle const& h) -> decltype(auto) {
    return at(h, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  template <size_t... Is>
  constexpr auto at(vertex_handle const& h,
                    std::index_sequence<Is...> /*seq*/) const
      -> decltype(auto) {
    return at(h.index(Is)...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr auto at(vertex_handle const& h, std::index_sequence<Is...> /*seq*/)
      -> decltype(auto) {
    return at(h.index(Is)...);
  }
  //----------------------------------------------------------------------------
 public:
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
#if TATOOINE_PNG_AVAILABLE
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
  template <size_t _N = num_dimensions(), enable_if<_N == 2> = true,
            enable_if_floating_point<ValueType> = true>
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
        auto const col                             = color_scale(d) * 255;
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
#if TATOOINE_PNG_AVAILABLE
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(typed_grid_vertex_property_interface<
                   Grid, ValueType, HasNonConstReference> const& prop,
               filesystem::path const&                           path) -> void {
  prop.write_png(path);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto write_png(filesystem::path const& path,
               typed_grid_vertex_property_interface<
                   Grid, ValueType, HasNonConstReference> const& prop) -> void {
  prop.write_png(path);
}
#endif
//==============================================================================
template <typename Grid, typename ValueType, typename Container>
struct typed_vertex_property
    : typed_grid_vertex_property_interface<
          Grid, ValueType,
          std::is_convertible_v<
              decltype(std::declval<Container&>().at(
                  std::declval<std::array<size_t, Grid::num_dimensions()>>())),
              ValueType&>>,
      Container {
  static_assert(std::is_same_v<ValueType, typename Container::value_type>);
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = typed_vertex_property<Grid, ValueType, Container>;
  static constexpr bool has_non_const_reference = std::is_convertible_v<
      decltype(std::declval<Container&>().at(
          std::declval<std::array<size_t, Grid::num_dimensions()>>())),
      ValueType&>;
  using prop_parent_t =
      typed_grid_vertex_property_interface<Grid, ValueType,
                                           has_non_const_reference>;
  using cont_parent_t   = Container;
  using value_type      = typename prop_parent_t::value_type;
  using reference       = typename prop_parent_t::reference;
  using const_reference = typename prop_parent_t::const_reference;
  using grid_t          = Grid;
  using prop_parent_t::num_dimensions;
  //============================================================================
  // ctors
  //============================================================================
  template <typename... Args>
  explicit typed_vertex_property(Grid const& grid, Args&&... args)
      : prop_parent_t{grid}, cont_parent_t{std::forward<Args>(args)...} {}
  typed_vertex_property(typed_vertex_property const&)     = default;
  typed_vertex_property(typed_vertex_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_vertex_property() override = default;
  //============================================================================
  // methods
  //============================================================================
  auto clone() const -> std::unique_ptr<grid_vertex_property<Grid>> override {
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
      constexpr auto
      operator()(Is const... is) const -> decltype(auto) {
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
      constexpr auto
      operator()(Is const... is) -> decltype(auto) {
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
// template <typename Grid, typename F>
// struct typed_grid_vertex_lambda
//    : typed_grid_vertex_property_interface<
//          Grid, typename Grid::template invoke_result_with_indices<F>, false>
//          {
//  //============================================================================
//  // typedefs
//  //============================================================================
//  using this_t = typed_grid_vertex_lambda<Grid, F>;
//  using value_type = typename Grid::template invoke_result_with_indices<F>;
//  static constexpr bool has_non_const_reference = false;
//  using prop_parent_t =
//      typed_grid_vertex_property_interface<Grid, value_type,
//      has_non_const_reference>;
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
//  typed_grid_vertex_lambda(Grid const& grid, _F&& f)
//      : prop_parent_t{grid}, m_f{std::forward<_F>(f)} {}
//  typed_grid_vertex_lambda(typed_grid_vertex_lambda const&) =
//      default;
//  typed_grid_vertex_lambda(typed_grid_vertex_lambda&&) noexcept =
//      default;
//  //----------------------------------------------------------------------------
//  ~typed_grid_vertex_lambda() override = default;
//  //============================================================================
//  // methods
//  //============================================================================
//  auto clone() const -> std::unique_ptr<grid_vertex_property<Grid>> override {
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
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  -
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
//  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  - auto at(std::array<size_t, num_dimensions()> const& size) const
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
struct grid_vertex_property_differentiated_type_impl;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, typename ValueType>
using grid_vertex_property_differentiated_type =
    typename grid_vertex_property_differentiated_type_impl<Grid,
                                                           ValueType>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct grid_vertex_property_differentiated_type_impl<Grid, float> {
  using type = vec<float, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct grid_vertex_property_differentiated_type_impl<Grid, double> {
  using type = vec<double, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid>
struct grid_vertex_property_differentiated_type_impl<Grid, long double> {
  using type = vec<long double, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, typename T, size_t N>
struct grid_vertex_property_differentiated_type_impl<Grid, vec<T, N>> {
  using type = mat<T, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, typename T, size_t M, size_t N>
struct grid_vertex_property_differentiated_type_impl<Grid, mat<T, M, N>> {
  using type = tensor<T, M, N, Grid::num_dimensions()>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Grid, typename T, size_t... Dims>
struct grid_vertex_property_differentiated_type_impl<Grid, tensor<T, Dims...>> {
  using type = tensor<T, Dims..., Grid::num_dimensions()>;
};
//==============================================================================
template <typename Grid, typename PropValueType, bool PropHasNonConstReference>
struct differentiated_typed_grid_vertex_property {
  using this_t =
      differentiated_typed_grid_vertex_property<Grid, PropValueType,
                                                PropHasNonConstReference>;
  using prop_t = typed_grid_vertex_property_interface<Grid, PropValueType,
                                                      PropHasNonConstReference>;
  using value_type =
      grid_vertex_property_differentiated_type<Grid, PropValueType>;
  using grid_t = Grid;
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
      constexpr auto
      operator()(Is const... is) const -> value_type {
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
    value_type d{};
    (
        [&](auto const dim, auto const index) {
          constexpr size_t targeted_stencil_size = 7;
          constexpr int    offset                = targeted_stencil_size / 2;

          auto       indices         = std::array{static_cast<size_t>(is)...};
          auto const negative_offset = std::max<int>(-indices[dim], -offset);
          auto const positive_offset =
              std::min<int>(grid().size(dim) - index - 1,
                            targeted_stencil_size + negative_offset - 1);
          auto const& coeffs = grid().diff_stencil_coefficients(
              dim, positive_offset - negative_offset + 1, -negative_offset,
              index);
          indices[dim] += negative_offset;
          for (size_t i = 0; i < size_t(positive_offset - negative_offset + 1);
               ++i, ++indices[dim]) {
            d.template slice<value_type::rank() - 1>(dim) +=
                m_prop(indices) * coeffs[i];
          }
        }(Seq, is),
        ...);
    return d;
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename InterpolationKernel>
  auto sampler_() const {
    using sampler_t =
        default_grid_vertex_sampler_t<num_dimensions(), InterpolationKernel,
                                      this_t>;
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
          default_grid_vertex_sampler_t<num_dimensions(), interpolation::cubic,
                                        this_t>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    } else if constexpr (sizeof...(InterpolationKernels) == 1) {
      return sampler_<InterpolationKernels...>();
    } else {
      using sampler_t =
          tatooine::grid_vertex_property_sampler<this_t,
                                                 InterpolationKernels...>;
      grid().update_diff_stencil_coefficients();
      return sampler_t{*this};
    }
  }
};
//==============================================================================
template <typename Grid, typename ValueType, bool HasNonConstReference>
auto diff(typed_grid_vertex_property_interface<
          Grid, ValueType, HasNonConstReference> const& prop) {
  if (!prop.grid().diff_stencil_coefficients_created_once()) {
    prop.grid().update_diff_stencil_coefficients();
  }
  return differentiated_typed_grid_vertex_property<Grid, ValueType,
                                                   HasNonConstReference>{prop};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
