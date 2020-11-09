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
template <typename Grid, typename Container>
struct typed_multidim_property : multidim_property<Grid>, Container {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t        = typed_multidim_property<Grid, Container>;
  using prop_parent_t = multidim_property<Grid>;
  using cont_parent_t = Container;
  using value_type    = typename Container::value_type;
  using grid_t        = Grid;
  using prop_parent_t::grid;
  using prop_parent_t::num_dimensions;
  //============================================================================
  // ctors
  //============================================================================
  template <typename... Args>
  explicit typed_multidim_property(Grid const& grid, Args&&... args)
      : prop_parent_t{grid}, cont_parent_t{std::forward<Args>(args)...} {}
  typed_multidim_property(typed_multidim_property const&)     = default;
  typed_multidim_property(typed_multidim_property&&) noexcept = default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property() override = default;
  //============================================================================
  // methods
  //============================================================================
  // overrides
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<multidim_property<Grid>> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto type() const -> std::type_info const& override {
    return typeid(value_type);
  }
  //----------------------------------------------------------------------------
  auto container_type() const -> std::type_info const& override {
    return typeid(Container);
  }
  template <template <typename> typename... InterpolationKernels>
  auto sampler() const {
    return tatooine::sampler<this_t, InterpolationKernels...>{*this};
  }
  ////----------------------------------------------------------------------------
  //template <size_t N = num_dimensions()>
  //requires(N == 1) || (N == 2) || (N == 3)
  //void write_vtk(std::string const& file_path,
  //               std::string const& description = "tatooine grid") const {
  //  auto writer = [this, &file_path, &description] {
  //    vtk::legacy_file_writer writer{file_path, vtk::RECTILINEAR_GRID};
  //    writer.set_title(description);
  //    writer.write_header();
  //    if constexpr (Grid::is_regular) {
  //      if constexpr (num_dimensions() == 1) {
  //        writer.write_dimensions(grid().template size<0>(), 1, 1);
  //        writer.write_origin(grid().template front<0>(), 0, 0);
  //        writer.write_spacing(grid().template dimension<0>().spacing(), 0, 0);
  //      } else if constexpr (num_dimensions() == 2) {
  //        writer.write_dimensions(grid().template size<0>(),
  //                                grid().template size<1>(), 1);
  //        writer.write_origin(grid().template front<0>(),
  //                            grid().template front<1>(), 0);
  //        writer.write_spacing(grid().template dimension<0>().spacing(),
  //                             grid().template dimension<1>().spacing(), 0);
  //      } else if constexpr (num_dimensions() == 3) {
  //        writer.write_dimensions(grid().template size<0>(),
  //                                grid().template size<1>(),
  //                                grid().template size<2>());
  //        writer.write_origin(grid().template front<0>(),
  //                            grid().template front<1>(),
  //                            grid().template front<2>());
  //        writer.write_spacing(grid().template dimension<0>().spacing(),
  //                             grid().template dimension<1>().spacing(),
  //                             grid().template dimension<2>().spacing());
  //      }
  //      return writer;
  //    } else {
  //      if constexpr (num_dimensions() == 1) {
  //        writer.write_dimensions(grid().template size<0>(), 1, 1);
  //        writer.write_x_coordinates(
  //            std::vector<double>(begin(grid().template dimension<0>()),
  //                                end(grid().template dimension<0>())));
  //        writer.write_y_coordinates(std::vector<double>{0});
  //        writer.write_z_coordinates(std::vector<double>{0});
  //      } else if constexpr (num_dimensions() == 2) {
  //        writer.write_dimensions(grid().template size<0>(),
  //                                grid().template size<1>(), 1);
  //        writer.write_x_coordinates(
  //            std::vector<double>(begin(grid().template dimension<0>()),
  //                                end(grid().template dimension<0>())));
  //        writer.write_y_coordinates(
  //            std::vector<double>(begin(grid().template dimension<1>()),
  //                                end(grid().template dimension<1>())));
  //        writer.write_z_coordinates(std::vector<double>{0});
  //      } else if constexpr (num_dimensions() == 3) {
  //        writer.write_dimensions(grid().template size<0>(),
  //                                grid().template size<1>(),
  //                                grid().template size<2>());
  //        writer.write_x_coordinates(
  //            std::vector<double>(begin(grid().template dimension<0>()),
  //                                end(grid().template dimension<0>())));
  //        writer.write_y_coordinates(
  //            std::vector<double>(begin(grid().template dimension<1>()),
  //                                end(grid().template dimension<1>())));
  //        writer.write_z_coordinates(
  //            std::vector<double>(begin(grid().template dimension<2>()),
  //                                end(grid().template dimension<2>())));
  //      }
  //      return writer;
  //    }
  //  }();
  //  // write vertex data
  //  writer.write_point_data(grid().num_vertices());
  //  std::vector<typename Container::value_type> data;
  //  grid().loop_over_vertex_indices(
  //      [&](auto const... is) { data.push_back(at(is...)); });
  //  writer.write_scalars("data", data);
  //}
  ////----------------------------------------------------------------------------
  //template <size_t N = num_dimensions()>
  //requires (N == 2) && (std::is_floating_point_v<typename Container::value_type>)
  //void write_png(std::string const& file_path) const {
  //  std::vector<typename Container::value_type> data;
  //  grid().loop_over_vertex_indices(
  //      [&](auto const... is) { data.push_back(at(is...)); });
  //  tatooine::write_png(file_path, data,
  //                      grid().template size<0>(),
  //                      grid().template size<1>());
  //}
};
//==============================================================================
//template <real_number T, typename Grid>
//void write_png(std::string const&                      filepath,
//               typed_multidim_property<Grid, T> const& data, size_t width,
//               size_t height) {
//  static_assert(Grid::num_dimensions() == 2);
//  static_assert(Grid::is_regular);
//  png::image<png::rgb_pixel> image(width, height);
//  for (unsigned int y = 0; y < image.get_height(); ++y) {
//    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
//      auto d = data.at(x, y);
//      if (std::isnan(d)) {
//        d = 0;
//      } else {
//        d = std::max<T>(0, std::min<T>(1, d));
//      }
//      image[image.get_height() - 1 - y][x].red =
//          image[image.get_height() - 1 - y][x].green =
//              image[image.get_height() - 1 - y][x].blue = d * 255;
//    }
//  }
//  image.write(filepath);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <real_number T, typename Grid>
//void write_png(std::string const&                              filepath,
//               typed_multidim_property<Grid, vec<T, 3>> const& data,
//               size_t width, size_t height) {
//  static_assert(Grid::num_dimensions() == 2);
//  static_assert(Grid::is_regular);
//  png::image<png::rgb_pixel> image(width, height);
//  for (unsigned int y = 0; y < image.get_height(); ++y) {
//    for (png::uint_32 x = 0; x < image.get_width(); ++x) {
//      auto d = data.at(x, y);
//      if (std::isnan(d(0))) {
//        image[image.get_height() - 1 - y][x].red = 0;
//      } else {
//        image[image.get_height() - 1 - y][x].red =
//            std::max<T>(0, std::min<T>(1, d(0))) * 255;
//      }
//      if (std::isnan(d(1))) {
//        image[image.get_height() - 1 - y][x].green = 0;
//      } else {
//        image[image.get_height() - 1 - y][x].green =
//            std::max<T>(0, std::min<T>(1, d(1))) * 255;
//      }
//      if (std::isnan(d(2))) {
//        image[image.get_height() - 1 - y][x].blue = 0;
//      } else {
//        image[image.get_height() - 1 - y][x].blue =
//            std::max<T>(0, std::min<T>(1, d(2))) * 255;
//      }
//    }
//  }
//  image.write(filepath);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
