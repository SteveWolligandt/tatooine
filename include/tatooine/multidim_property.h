#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/write_png.h>
#include <tatooine/finite_differences_coefficients.h>
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
  auto grid()       -> auto& { return m_grid; }
  auto grid() const -> auto const& { return m_grid; }
  //----------------------------------------------------------------------------
  template <size_t DimensionIndex>
  auto size() const {
    return m_grid.template size<DimensionIndex>();
  }
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
  std::optional<value_type> m_out_of_domain_value;
  //============================================================================
  typed_multidim_property(Grid const& grid) : parent_t{grid} {}
  typed_multidim_property(typed_multidim_property const&)     = default;
  typed_multidim_property(typed_multidim_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property() override = default;
  //----------------------------------------------------------------------------
  const std::type_info& type() const override { return typeid(value_type); }
  //----------------------------------------------------------------------------
  virtual auto data_at(std::array<size_t, Grid::num_dimensions()> const& is)
      const -> value_type const& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto data_at(integral auto... is) const -> value_type const& {
    static_assert(sizeof...(is) == Grid::num_dimensions(),
                  "Number of indices does not match number of dimensions.");
    return data_at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  auto out_of_domain_value() const -> auto const& {
    return m_out_of_domain_value;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void set_out_of_domain_value(value_type const& value) {
    m_out_of_domain_value = value;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void unset_out_of_domain_value() { m_out_of_domain_value = {}; }
  //----------------------------------------------------------------------------
  virtual auto sample(typename Grid::pos_t const& x) const -> value_type = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto         sample(real_number auto... xs) const -> value_type {
    static_assert(
        sizeof...(xs) == Grid::num_dimensions(),
        "Number of spatial components does not match number of dimensions.");
    return sample(typename Grid::pos_t{xs...});
  }
  //----------------------------------------------------------------------------
  //template <floating_point Real, size_t DimIndex, size_t StencilSize>
  //auto stencil_coefficients(linspace<Real> const& dim,
  //                          integral auto const        i,
  //                          integral auto const        num_diffs) {
  //}
  //----------------------------------------------------------------------------
  template <size_t DimIndex, size_t StencilSize>
  auto stencil_coefficients(indexable_space auto const& dim, size_t const i,
            unsigned int const num_diffs) const {
    assert(num_diffs < StencilSize &&
           "Number of differentiations must be smaller thant stencil size");
    assert(StencilSize <= dim.size());
    size_t left_index  = std::max<long>(0, i - StencilSize / 2);
    left_index         = std::min<long>(left_index, dim.size() - StencilSize);

    vec<double, StencilSize> stencil;
    for (size_t j = 0; j < StencilSize; ++j) {
      stencil(j) = dim[left_index + j] - dim[i];
    }
    return std::pair{left_index,
                     finite_differences_coefficients(num_diffs, stencil)};
  }
  ////----------------------------------------------------------------------------
  //template <size_t DimIndex, size_t StencilSize>
  //auto stencil_coefficients(size_t const i,
  //                          unsigned int const num_diffs) const {
  //  return stencil_coefficients<DimIndex, StencilSize>(
  //      this->grid().template dimension<DimIndex>(), i, num_diffs);
  //}
  ////----------------------------------------------------------------------------
  //template <size_t DimIndex, size_t StencilSize>
  //auto diff_at(unsigned int const                   num_diffs,
  //          std::array<size_t, num_dimensions()> is) const -> value_type {
  //  static_assert(DimIndex < num_dimensions());
  //  auto const [first_idx, coeffs] =
  //      stencil_coefficients<DimIndex, StencilSize>(is[DimIndex], num_diffs);
  //  value_type d{};
  //  is[DimIndex] = first_idx;
  //  for (size_t i = 0; i < StencilSize; ++i, ++is[DimIndex]) {
  //    if (coeffs(i) != 0) { d += coeffs(i) * data_at(is); }
  //  }
  //  return d;
  //}
  ////----------------------------------------------------------------------------
  //template <size_t DimIndex, size_t StencilSize>
  //auto diff_at(unsigned int const num_diffs, integral auto... is) const {
  //  static_assert(DimIndex < num_dimensions());
  //  static_assert(sizeof...(is) == num_dimensions(),
  //                "Number of indices does not match number of dimensions.");
  //  return diff_at<DimIndex, StencilSize>(num_diffs,
  //                                     std::array{static_cast<size_t>(is)...});
  //}
};
//==============================================================================
template <real_number T, typename Grid>
void write_png(std::string const& filepath, typed_multidim_property<Grid, T> const& data, size_t width, size_t height) {
  static_assert(Grid::num_dimensions() == 2);
  static_assert(Grid::is_regular);
  png::image<png::rgb_pixel> image(width, height);
  for (unsigned int y = 0; y < image.get_height(); ++y) {
    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
      auto d = data.data_at(x, y);
      if (std::isnan(d)) {
        d = 0;
      } else {
        d = std::max<T>(0, std::min<T>(1, d));
      }
      image[image.get_height() - 1 - y][x].red =
          image[image.get_height() - 1 - y][x].green =
              image[image.get_height() - 1 - y][x].blue = d * 255;
    }
  }
  image.write(filepath);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
