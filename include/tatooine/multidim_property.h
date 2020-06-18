#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid>
struct multidim_property {
  //============================================================================
  using this_t = multidim_property<Grid>;
  //============================================================================
  static constexpr auto num_dimensions() { return Grid::num_dimensions(); }
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
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_t> = 0;
  //----------------------------------------------------------------------------
  auto grid() -> auto& { return m_grid; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto grid() const -> auto const& { return m_grid; }
};
//==============================================================================
template <typename Grid, typename T>
struct typed_multidim_property : multidim_property<Grid> {
  //============================================================================
  using this_t     = typed_multidim_property<Grid, T>;
  using parent_t   = multidim_property<Grid>;
  using value_type = T;
  using parent_t::num_dimensions;
  //============================================================================
  typed_multidim_property(Grid const& grid) : parent_t{grid} {}
  typed_multidim_property(typed_multidim_property const&)     = default;
  typed_multidim_property(typed_multidim_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property() override = default;
  //----------------------------------------------------------------------------
  const std::type_info& type() const override { return typeid(T); }
  //----------------------------------------------------------------------------
  virtual auto data_at(std::array<size_t, Grid::num_dimensions()> const& is)
      -> T& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto data_at(std::array<size_t, Grid::num_dimensions()> const& is)
      const -> T const& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(integral auto... is) -> T& {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return data_at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(integral auto... is) const -> T const& {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return data_at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  virtual auto sample(typename Grid::pos_t const& x) const -> T = 0;
  auto         sample(real_number auto... xs) const -> T {
    static_assert(
        sizeof...(xs) == Grid::num_dimensions(),
        "Number of spatial components does not match number of dimensions.");
    return sample(typename Grid::pos_t{xs...});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
