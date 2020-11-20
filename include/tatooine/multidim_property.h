#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/finite_differences_coefficients.h>
#include <tatooine/write_png.h>
#include <tatooine/sampler.h>
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
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  constexpr auto operator()(Is const... is) const -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  constexpr auto operator()(Is const... is) -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  auto at(Is const... is) const -> decltype(auto) {
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
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
  template <integral... Ss>
  requires(sizeof...(Ss) == num_dimensions())
  auto resize(Ss const... ss) -> decltype(auto) {
    return resize(std::array{static_cast<size_t>(ss)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto resize(std::array<size_t, num_dimensions()> const& size)
      -> void = 0;
};
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
  typed_multidim_property_impl(typed_multidim_property_impl const&)     = default;
  typed_multidim_property_impl(typed_multidim_property_impl&&) noexcept = default;
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
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  constexpr auto operator()(Is const... is) const -> decltype(auto) {
    return Container::at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  constexpr auto operator()(Is const... is) -> decltype(auto) {
    return Container::at(is...);
  }
  //----------------------------------------------------------------------------
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  auto at(Is const... is) const -> decltype(auto) {
    return Container::at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <integral... Is>
  requires(sizeof...(Is) == Grid::num_dimensions())
  auto at(Is const... is) -> decltype(auto) {
    return Container::at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size) const
      -> ValueType const& override {
    return Container::at(size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<size_t, num_dimensions()> const& size)
      -> ValueType& override {
    return Container::at(size);
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
