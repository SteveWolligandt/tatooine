#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/write_png.h>
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
  Grid const& m_grid;
  //============================================================================
 public:
  multidim_property(Grid const& grid) : m_grid{grid} {}
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
    return m_grid;
  }
  auto grid() const -> auto const& {
    return m_grid;
  }
};
//==============================================================================
template <typename Grid, typename ValueType>
struct typed_multidim_property : multidim_property<Grid> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t        = typed_multidim_property<Grid, ValueType>;
  using parent_t      = multidim_property<Grid>;
  using value_type    = ValueType;
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
  template <template <typename> typename InterpolationKernel>
  auto sampler_() const {
    using sampler_t =
        default_multidim_property_sampler_t<num_dimensions(),
                                            InterpolationKernel, this_t>;
    grid().update_diff_stencil_coefficients();
    return sampler_t{*this};
  }
  //----------------------------------------------------------------------------
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
  // data access
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
  virtual auto at(std::array<size_t, num_dimensions()> const& size) const
      -> ValueType const& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto at(std::array<size_t, num_dimensions()> const& size)
      -> ValueType& = 0;
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
#if TATOOINE_HAS_PNG_SUPPORT
#ifdef __cpp_concepts
  template <typename = void>
  requires (num_dimensions() == 2) &&
           (is_floating_point_v<ValueType> || is_vec_v<ValueType>)
#else
  template <size_t _N                                   = num_dimensions(),
            enable_if<(_N == 2) && (is_floating_point<ValueType> ||
                                    is_vec<ValueType>)> = true>
#endif
  auto write_png(std::filesystem::path const& path) const -> void {
    png::image<png::rgb_pixel> image{
        static_cast<png::uint_32>(this->grid().size(0)),
        static_cast<png::uint_32>(this->grid().size(1))};
    for (unsigned int y = 0; y < image.get_height(); ++y) {
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto d = at(x, y);
        if constexpr (is_floating_point_v<ValueType>) {
          if (std::isnan(d)) {
            d = 0;
          } else {
            d = std::max<ValueType>(0, std::min<ValueType>(1, d));
          }
          image[image.get_height() - 1 - y][x].red =
          image[image.get_height() - 1 - y][x].green =
          image[image.get_height() - 1 - y][x].blue = d * 255;
        } else if constexpr (is_floating_point_v<ValueType>) {
          if (std::isnan(d(0))) {
            for (auto& c : d) {
              c = 0;
            }
          } else {
            for (auto& c : d) {
              c = std::max<ValueType>(0, std::min<ValueType>(1, c));
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
#endif
};
//==============================================================================
#if TATOOINE_HAS_PNG_SUPPORT
template <typename Grid, typename ValueType>
auto write_png(typed_multidim_property<Grid, ValueType> const& prop,
               std::filesystem::path const&                    path) -> void {
  prop.write_png(path);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Grid, typename ValueType>
auto write_png(std::filesystem::path const&                    path,
               typed_multidim_property<Grid, ValueType> const& prop) -> void {
  prop.write_png(path);
}
#endif
//==============================================================================
template <typename Grid, typename ValueType, typename Container>
struct typed_multidim_property_impl : typed_multidim_property<Grid, ValueType>,
                                      Container {
  static_assert(std::is_same_v<ValueType, typename Container::value_type>);
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = typed_multidim_property_impl<Grid, ValueType, Container>;
  using prop_parent_t = typed_multidim_property<Grid, ValueType>;
  using cont_parent_t = Container;
  using value_type    = typename prop_parent_t::value_type;
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
          std::index_sequence<Is...> /*seq*/) const -> ValueType const& {
    return Container::at(size[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size) const
      -> ValueType const& override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  auto at(std::array<size_t, num_dimensions()> const& size,
          std::index_sequence<Is...> /*seq*/) -> ValueType& {
    return Container::at(size[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size)
      -> ValueType& override {
    return at(size, std::make_index_sequence<num_dimensions()>{});
  }
  //----------------------------------------------------------------------------
  auto resize(std::array<size_t, num_dimensions()> const& size)
      -> void override {
    Container::resize(size);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
