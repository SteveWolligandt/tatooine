#ifndef TATOOINE_VTK_LEGACY_H
#define TATOOINE_VTK_LEGACY_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/concepts.h>
#include <tatooine/parse.h>
#include <tatooine/swap_endianess.h>
#include <tatooine/type_to_str.h>
#include <tatooine/type_traits.h>

#include <cassert>
#include <filesystem>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <istream>
#include <map>
#include <sstream>
#include <vector>
//=============================================================================
namespace tatooine::vtk {
//=============================================================================
/// reads until terminator_char was found. buffer will containg the
/// terminator_char
auto read_until(std::istream &stream, char const terminator_char, char *buffer)
    -> std::string;
//-----------------------------------------------------------------------------
/// reads stream until a linebreak was found. buffer will not contain the break
auto read_binaryline(std::istream &stream, char *buffer) -> std::string;
//-----------------------------------------------------------------------------
/// reads stream until a whitespace was found and consumes any whitespace until
/// another character was found. buffer will not contain any whitespace
auto read_word(std::istream &stream, char *buffer) -> std::string;
//-----------------------------------------------------------------------------
auto write_binary(std::ostream &stream, std::string const &str) -> void;
//-----------------------------------------------------------------------------
auto write_binary(std::ostream &stream, char const c) -> void;
//------------------------------------------------------------------------------
enum class reader_data { point_data, cell_data, unknown };
//------------------------------------------------------------------------------
enum class dataset_type {
  structured_points,
  structured_grid,
  unstructured_grid,
  polydata,
  rectilinear_grid,
  field,
  unknown
};
//------------------------------------------------------------------------------
constexpr auto to_string_view(dataset_type type) -> std::string_view {
  switch (type) {
    case dataset_type::structured_points:
      return "STRUCTURED_POINTS";
    case dataset_type::structured_grid:
      return "STRUCTURED_GRID";
    case dataset_type::unstructured_grid:
      return "UNSTRUCTURED_GRID";
    case dataset_type::polydata:
      return "POLYDATA";
    case dataset_type::rectilinear_grid:
      return "RECTILINEAR_GRID";
    case dataset_type::field:
      return "FIELD";
    default:
    case dataset_type::unknown:
      return "UNKNOWN";
  }
}
//------------------------------------------------------------------------------
auto parse_dataset_type(std::string const &) -> dataset_type;
//==============================================================================
enum class format { ascii, binary, unknown };
//------------------------------------------------------------------------------
constexpr auto to_string_view(format f) -> std::string_view {
  switch (f) {
    case format::ascii:
      return "ASCII";
    case format::binary:
      return "BINARY";
    default:
    case format::unknown:
      return "UNKNOWN";
  }
}
//------------------------------------------------------------------------------
auto parse_format(std::string const &) -> format;
//==============================================================================
enum class cell_type : int {
  vertex            = 1,
  poly_vertex       = 2,
  line              = 3,
  poly_line         = 4,
  triangle          = 5,
  triangle_strip    = 6,
  polygon           = 7,
  pixel             = 8,
  quad              = 9,
  tetra             = 10,
  voxel             = 11,
  hexahedron        = 12,
  wedge             = 13,
  pyramid           = 14,
  unknown_cell_type = 0,
};
//-----------------------------------------------------------------------------
constexpr auto to_string_view(cell_type ct) -> std::string_view {
  switch (ct) {
    case cell_type::vertex:
      return "VERTEX";
    case cell_type::poly_vertex:
      return "POLY_VERTEX";
    case cell_type::line:
      return "LINE";
    case cell_type::poly_line:
      return "POLY_LINE";
    case cell_type::triangle:
      return "TRIANGLE";
    case cell_type::triangle_strip:
      return "TRIANGLE_STRIP";
    case cell_type::polygon:
      return "POLYGON";
    case cell_type::pixel:
      return "PIXEL";
    case cell_type::quad:
      return "QUAD";
    case cell_type::tetra:
      return "TETRA";
    case cell_type::voxel:
      return "VOXEL";
    case cell_type::hexahedron:
      return "HEXAHEDRON";
    case cell_type::wedge:
      return "WEDGE";
    case cell_type::pyramid:
      return "PYRAMID";
    default:
    case cell_type::unknown_cell_type:
      return "UNKNOWN";
  }
}
//-----------------------------------------------------------------------------
auto parse_cell_type(std::string const &) -> cell_type;
//------------------------------------------------------------------------------
struct legacy_file_listener {
  // header data
  virtual auto on_version(unsigned short /*major*/, unsigned short /*minor*/)
      -> void {}
  virtual auto on_title(std::string const &) -> void {}
  virtual auto on_format(format) -> void {}
  virtual auto on_dataset_type(dataset_type) -> void {}

  // coordinate data
  virtual auto on_points(std::vector<std::array<float, 3>> const &) -> void {}
  virtual auto on_points(std::vector<std::array<double, 3>> const &) -> void {}
  virtual auto on_origin(double /*x*/, double /*y*/, double /*z*/) -> void {}
  virtual auto on_spacing(double /*x*/, double /*y*/, double /*z*/) -> void {}
  virtual auto on_dimensions(size_t /*x*/, size_t /*y*/, size_t /*z*/) -> void {
  }
  virtual auto on_x_coordinates(std::vector<float> const & /*xs*/) -> void {}
  virtual auto on_x_coordinates(std::vector<double> const & /*xs*/) -> void {}
  virtual auto on_y_coordinates(std::vector<float> const & /*ys*/) -> void {}
  virtual auto on_y_coordinates(std::vector<double> const & /*ys*/) -> void {}
  virtual auto on_z_coordinates(std::vector<float> const & /*zs*/) -> void {}
  virtual auto on_z_coordinates(std::vector<double> const & /*zs*/) -> void {}

  // index data
  virtual auto on_cells(std::vector<int> const &) -> void {}
  virtual auto on_cell_types(std::vector<cell_type> const &) -> void {}
  virtual auto on_vertices(std::vector<int> const &) -> void {}
  virtual auto on_lines(std::vector<int> const &) -> void {}
  virtual auto on_polygons(std::vector<int> const &) -> void {}
  virtual auto on_triangle_strips(std::vector<int> const &) -> void {}

  // cell- / pointdata
  virtual auto on_vectors(std::string const & /*name*/,
                          std::vector<std::array<float, 3>> const & /*vectors*/,
                          reader_data) -> void {}
  virtual auto on_vectors(
      std::string const & /*name*/,
      std::vector<std::array<double, 3>> const & /*vectors*/, reader_data)
      -> void {}
  virtual auto on_normals(std::string const & /*name*/,
                          std::vector<std::array<float, 3>> const & /*normals*/,
                          reader_data) -> void {}
  virtual auto on_normals(
      std::string const & /*name*/,
      std::vector<std::array<double, 3>> const & /*normals*/, reader_data)
      -> void {}
  virtual auto on_texture_coordinates(
      std::string const & /*name*/,
      std::vector<std::array<float, 2>> const & /*texture_coordinates*/,
      reader_data) -> void {}
  virtual auto on_texture_coordinates(
      std::string const & /*name*/,
      std::vector<std::array<double, 2>> const & /*texture_coordinates*/,
      reader_data) -> void {}
  virtual auto on_tensors(std::string const & /*name*/,
                          std::vector<std::array<float, 9>> const & /*tensors*/,
                          reader_data) -> void {}
  virtual auto on_tensors(
      std::string const & /*name*/,
      std::vector<std::array<double, 9>> const & /*tensors*/, reader_data)
      -> void {}

  virtual auto on_scalars(std::string const & /*data_name*/,
                          std::string const & /*lookup_table_name*/,
                          size_t const /*num_comps*/,
                          std::vector<float> const & /*scalars*/, reader_data)
      -> void {}
  virtual auto on_scalars(std::string const & /*data_name*/,
                          std::string const & /*lookup_table_name*/,
                          size_t const /*num_comps*/,
                          std::vector<double> const & /*scalars*/, reader_data)
      -> void {}
  virtual auto on_point_data(size_t) -> void {}
  virtual auto on_cell_data(size_t) -> void {}
  virtual auto on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<int> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/)
      -> void {}
  virtual auto on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<float> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/)
      -> void {}
  virtual auto on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<double> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/
                              ) -> void {}
};
//------------------------------------------------------------------------------
class legacy_file {
  std::vector<legacy_file_listener *> m_listeners;

  std::filesystem::path m_path;
  format                m_format;
  reader_data           m_data = reader_data::unknown;
  size_t                m_data_size;  // cell_data or point_data size
  long                  m_begin_of_data;
  char                  buffer[256];

 public:
  auto add_listener(legacy_file_listener &listener) -> void;
  //---------------------------------------------------------------------------
  legacy_file(std::filesystem::path const &path);
  //---------------------------------------------------------------------------
  auto read() -> void;
  //---------------------------------------------------------------------------
  auto set_path(std::filesystem::path const &path) -> void { m_path = path; }
  auto set_path(std::filesystem::path &&path) -> void {
    m_path = std::move(path);
  }
  auto path() const -> auto const & { return m_path; }
  //---------------------------------------------------------------------------
 private:
  auto read_header() -> void;
  auto read_data() -> void;

  auto read_spacing(std::ifstream &file) -> void;
  auto read_dimensions(std::ifstream &file) -> void;
  auto read_origin(std::ifstream &file) -> void;

  auto read_points(std::ifstream &file) -> void;
  template <typename Real>
  auto read_points_ascii(std::ifstream &file, size_t const n) -> void;
  template <typename Real>
  auto read_points_binary(std::ifstream &file, size_t const n) -> void;

  auto read_cell_types(std::ifstream &file) -> void;
  auto read_cell_types_ascii(std::ifstream &file, size_t const n) -> void;
  auto read_cell_types_binary(std::ifstream &file, size_t const n) -> void;

  auto read_indices(std::ifstream &file) -> std::vector<int>;
  auto read_indices_ascii(std::ifstream &file, size_t const size)
      -> std::vector<int>;
  auto read_indices_binary(std::ifstream &file, size_t const size)
      -> std::vector<int>;

  auto read_scalars_header(std::ifstream &file);
  auto read_scalars(std::ifstream &file) -> void;
  template <typename Real>
  auto read_scalars_ascii(std::ifstream &file, std::string const &name,
                          std::string const &lookup_table,
                          size_t const       num_comps) -> void;
  template <typename Real>
  auto read_scalars_binary(std::ifstream &file, std::string const &name,
                           std::string const &lookup_table,
                           size_t const       num_comps) -> void;

  auto read_data_header(std::ifstream &file);
  template <typename Real, size_t N>
  auto read_data(std::ifstream &file) -> std::vector<std::array<Real, N>>;
  template <typename Real, size_t N>
  auto read_data_ascii(std::ifstream &file) -> std::vector<std::array<Real, N>>;
  template <typename Real, size_t N>
  auto read_data_binary(std::ifstream &file)
      -> std::vector<std::array<Real, N>>;
  //----------------------------------------------------------------------------
  // coordinates
  auto read_coordinates_header(std::ifstream &file);
  template <typename Real>
  auto read_coordinates(std::ifstream &file, size_t n);
  template <typename Real>
  auto read_coordinates_ascii(std::ifstream &file, size_t n)
      -> std::vector<Real>;
  template <typename Real>
  auto read_coordinates_binary(std::ifstream &file, size_t n)
      -> std::vector<Real>;
  //----------------------------------------------------------------------------
  auto read_x_coordinates(std::ifstream &file) -> void;
  auto read_y_coordinates(std::ifstream &file) -> void;
  auto read_z_coordinates(std::ifstream &file) -> void;
  //----------------------------------------------------------------------------
  // index data
  auto read_cells(std::ifstream &file) -> void;
  auto read_vertices(std::ifstream &file) -> void;
  auto read_lines(std::ifstream &file) -> void;
  auto read_polygons(std::ifstream &file) -> void;
  auto read_triangle_strips(std::ifstream &file) -> void;
  //----------------------------------------------------------------------------
  // fixed size data
  auto read_vectors(std::ifstream &file) -> void;
  auto read_normals(std::ifstream &file) -> void;
  auto read_texture_coordinates(std::ifstream &file) -> void;
  auto read_tensors(std::ifstream &file) -> void;
  auto read_field_header(std::ifstream &file) -> std::pair<std::string, size_t>;
  //----------------------------------------------------------------------------
  // field data
  auto read_field_array_header(std::ifstream &file)
      -> std::tuple<std::string, size_t, size_t, std::string>;
  auto read_field(std::ifstream &file) -> void;

  template <typename Real>
  auto read_field_array_binary(std::ifstream &file, size_t num_comps,
                               size_t num_tuples) -> std::vector<Real>;
  template <typename Real>
  auto read_field_array_ascii(std::ifstream &file, size_t num_comps,
                              size_t num_tuples) -> std::vector<Real>;

  auto consume_trailing_break(std::ifstream &file) -> void;
};
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_points_ascii(std::ifstream &file, size_t const n)
    -> void {
  std::vector<std::array<Real, 3>> points;
  for (size_t i = 0; i < n; i++)
    points.push_back(
        {{static_cast<Real>(parse<Real>(vtk::read_word(file, buffer))),
          static_cast<Real>(parse<Real>(vtk::read_word(file, buffer))),
          static_cast<Real>(parse<Real>(vtk::read_word(file, buffer)))}});

  for (auto l : m_listeners)
    l->on_points(points);
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_points_binary(std::ifstream &file, size_t const n)
    -> void {
  std::vector<std::array<Real, 3>> points(n);
  if (n > 0) {
    file.read((char *)points.data(), sizeof(Real) * 3 * n);
    swap_endianess(reinterpret_cast<Real *>(points.data()), n * 3);
    for (auto l : m_listeners)
      l->on_points(points);
    consume_trailing_break(file);
  }
  // file.ignore(sizeof(Real) * 3 * n + 1);
}
//------------------------------------------------------------------------------
template <typename Real, size_t n>
auto legacy_file::read_data(std::ifstream &file)
    -> std::vector<std::array<Real, n>> {
  if (m_format == format::ascii)
    return read_data_ascii<Real, n>(file);
  else
    return read_data_binary<Real, n>(file);
}
//-----------------------------------------------------------------------------
template <typename Real, size_t n>
auto legacy_file::read_data_ascii(std::ifstream &file)
    -> std::vector<std::array<Real, n>> {
  std::vector<std::array<Real, n>> data(m_data_size);
  for (size_t i = 0; i < m_data_size; i++)
    for (size_t j = 0; j < n; j++)
      data[i][n] = parse<Real>(vtk::read_word(file, buffer));
  return data;
}
//-----------------------------------------------------------------------------
template <typename Real, size_t n>
auto legacy_file::read_data_binary(std::ifstream &file)
    -> std::vector<std::array<Real, n>> {
  std::vector<std::array<Real, n>> data(m_data_size);
  file.read((char *)data.data(), sizeof(Real) * m_data_size * n);
  swap_endianess(reinterpret_cast<Real *>(data.data()), n * m_data_size);
  consume_trailing_break(file);
  return data;
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_coordinates(std::ifstream &file, size_t n) {
  if (m_format == format::ascii)
    return read_coordinates_ascii<Real>(file, n);

  else /*if (m_format == format::binary)*/
    return read_coordinates_binary<Real>(file, n);
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_coordinates_ascii(std::ifstream &file, size_t const n)
    -> std::vector<Real> {
  std::vector<Real> coordinates(n);
  for (size_t i = 0; i < n; i++)
    coordinates[i] = parse<Real>(vtk::read_word(file, buffer));
  return coordinates;
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_coordinates_binary(std::ifstream &file, size_t const n)
    -> std::vector<Real> {
  std::vector<Real> coordinates(n);
  file.read((char *)coordinates.data(), sizeof(Real) * n);
  swap_endianess(coordinates);
  consume_trailing_break(file);
  return coordinates;
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_field_array_binary(std::ifstream &file, size_t num_comps,
                                          size_t num_tuples)
    -> std::vector<Real> {
  std::vector<Real> data(num_comps * num_tuples);
  file.read((char *)data.data(), sizeof(Real) * num_comps * num_tuples);
  swap_endianess(data);

  // consume trailing \n
  char consumer;
  file.read(&consumer, sizeof(char));

  return data;
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_field_array_ascii(std::ifstream &file, size_t num_comps,
                                         size_t num_tuples)
    -> std::vector<Real> {
  std::vector<Real> data;
  data.reserve(num_comps * num_tuples);
  for (size_t i = 0; i < num_comps * num_tuples; i++)
    data.push_back(parse<Real>(vtk::read_word(file, buffer)));

  return data;
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_scalars_ascii(std::ifstream &    file,
                                     std::string const &name,
                                     std::string const &lookup_table,
                                     size_t const       num_comps) -> void {
  std::vector<Real> scalars;
  scalars.reserve(m_data_size * num_comps);
  std::string val_str;
  for (size_t i = 0; i < m_data_size * num_comps; i++) {
    scalars.push_back(parse<Real>(vtk::read_word(file, buffer)));
  }
  for (auto l : m_listeners) {
    l->on_scalars(name, lookup_table, num_comps, scalars, m_data);
  }
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_scalars_binary(std::ifstream &    file,
                                      std::string const &name,
                                      std::string const &lookup_table,
                                      size_t const       num_comps) -> void {
  if (m_data_size > 0) {
    std::vector<Real> data(m_data_size * num_comps);
    file.read((char *)data.data(), sizeof(Real) * m_data_size * num_comps);
    swap_endianess(data);

    consume_trailing_break(file);
    for (auto l : m_listeners) {
      l->on_scalars(name, lookup_table, num_comps, data, m_data);
    }
  }
}
//------------------------------------------------------------------------------
class legacy_file_writer {
 private:
  std::ofstream  m_file;
  unsigned short m_major_version;
  unsigned short m_minor_version;
  dataset_type   m_dataset_type;
  std::string    m_title;

 public:
  legacy_file_writer(std::filesystem::path const &path, dataset_type type,
                     unsigned short major = 2, unsigned short minor = 0,
                     std::string const &title = "");
  auto is_open() -> bool;
  auto close() -> void;
  //---------------------------------------------------------------------------
 private:
  auto write_indices(std::string const &                     keyword,
                     std::vector<std::vector<size_t>> const &indices) -> void;
  template <size_t N>
  auto write_indices(std::string const &                       keyword,
                     std::vector<std::array<size_t, N>> const &indices) -> void;
  template <typename Real, size_t N>
  auto write_data(std::string const &keyword, std::string const &name,
                  std::vector<std::array<Real, N>> const &data) -> void;

 public:
  auto write_header() -> void;
  template <typename Real>
  auto write_points(std::vector<std::array<Real, 2>> const &points) -> void;
  template <typename Real>
  auto write_points(std::vector<std::array<Real, 3>> const &points) -> void;
  template <typename Real>
  auto write_points(std::vector<vec<Real, 2>> const &points) -> void;
  template <typename Real>
  auto write_points(std::vector<vec<Real, 3>> const &points) -> void;
  auto write_cells(std::vector<std::vector<size_t>> const &cells) -> void;
  auto write_cell_types(std::vector<cell_type> const &cell_types) -> void;

  auto write_vertices(std::vector<std::vector<size_t>> const &vertices) -> void;
  auto write_lines(std::vector<std::vector<size_t>> const &lines) -> void;
  auto write_polygons(std::vector<std::vector<size_t>> const &polygons) -> void;
  template <size_t N>
  auto write_polygons(std::vector<std::array<size_t, N>> const &polygons)
      -> void;
  auto write_triangle_strips(
      std::vector<std::vector<size_t>> const &triangle_strips) -> void;

#ifdef __cpp_concepts
  template <arithmetic T>
#else
  template <typename T, enable_if_arithmetic<T> = true>
#endif
  auto write_x_coordinates(std::vector<T> const &x_coordinates) -> void {
    std::stringstream ss;
    ss << "\nX_COORDINATES " << ' ' << x_coordinates.size() << ' '
       << type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : x_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
#ifdef __cpp_concepts
  template <arithmetic T>
#else
  template <typename T, enable_if_arithmetic<T> = true>
#endif
  auto write_y_coordinates(std::vector<T> const &y_coordinates) -> void {
    std::stringstream ss;
    ss << "\nY_COORDINATES " << ' ' << y_coordinates.size() << ' '
       << type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : y_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
#ifdef __cpp_concepts
  template <arithmetic T>
#else
  template <typename T, enable_if_arithmetic<T> = true>
#endif
  auto write_z_coordinates(std::vector<T> const &z_coordinates) -> void {
    std::stringstream ss;
    ss << "\nZ_COORDINATES " << ' ' << z_coordinates.size() << ' '
       << type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : z_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
  auto write_point_data(size_t i) -> void;
  auto write_cell_data(size_t i) -> void;
  template <typename Real>
  auto write_normals(std::string const &               name,
                     std::vector<std::array<Real, 3>> &normals) -> void;
  //----------------------------------------------------------------------------
  template <typename Real>
  auto write_vectors(std::string const &               name,
                     std::vector<std::array<Real, 3>> &vectors) -> void;
  //----------------------------------------------------------------------------
  template <typename Real>
  auto write_texture_coordinates(
      std::string const &               name,
      std::vector<std::array<Real, 2>> &texture_coordinates) -> void;
  //----------------------------------------------------------------------------
  template <typename Real>
  auto write_tensors(std::string const &               name,
                     std::vector<std::array<Real, 9>> &tensors) -> void;
  //----------------------------------------------------------------------------
  template <typename Data>
      requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
      (std::is_same_v<Data, int>)auto write_scalars(
          std::string const &name, std::vector<Data> const &data,
          std::string const &lookup_table_name = "default") -> void;
  //----------------------------------------------------------------------------
  template <typename Data>
      requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
      (std::is_same_v<Data, int>)auto write_scalars(
          std::string const &name, std::vector<std::vector<Data>> const &data,
          std::string const &lookup_table_name = "default") -> void;
  //----------------------------------------------------------------------------
  template <typename Data, size_t N>
      requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
      (std::is_same_v<Data, int>)auto write_scalars(
          std::string const &name, std::vector<std::array<Data, N>> const &data,
          std::string const &lookup_table_name = "default") -> void {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << type_to_str<Data>() << ' ' << N
       << '\n';
    vtk::write_binary(m_file, ss.str());
    vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
    for (auto const &arr : data)
      for (auto &comp : arr) {
        comp = swap_endianess(comp);
        m_file.write((char *)(&comp), sizeof(Data));
      }
  }
  //----------------------------------------------------------------------------
  template <typename Data, size_t N>
      requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
      (std::is_same_v<Data, int>)auto write_scalars(
          std::string const &name, std::vector<vec<Data, N>> const &data,
          std::string const &lookup_table_name = "default") -> void {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << type_to_str<Data>() << ' ' << N
       << '\n';
    vtk::write_binary(m_file, ss.str());
    vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
    Data d;
    for (auto const &v : data)
      for (size_t i = 0; i < N; ++i) {
        d = swap_endianess(v(i));
        m_file.write((char *)(&d), sizeof(Data));
      }
  }
  //----------------------------------------------------------------------------
  template <typename Real, size_t N>
      requires(std::is_same_v<Real, double>) || (std::is_same_v<Real, float>) ||
      (std::is_same_v<Real, int>)auto write_scalars(
          std::string const &name, std::vector<tensor<Real, N>> const &data,
          std::string const &lookup_table_name = "default") -> void {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << type_to_str<Real>() << ' ' << N
       << '\n';
    vtk::write_binary(m_file, ss.str());
    vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
    Real d;
    for (auto const &v : data)
      for (size_t i = 0; i < N; ++i) {
        d = swap_endianess(v(i));
        m_file.write((char *)(&d), sizeof(Real));
      }
  }
  //----------------------------------------------------------------------------
  auto write_dimensions(size_t const dimx, size_t const dimy, size_t const dimz)
      -> void;
  //----------------------------------------------------------------------------
  auto write_origin(double const orgx, double const orgy, double const orgz)
      -> void;
  //----------------------------------------------------------------------------
  auto write_spacing(double const spax, double const spay, double const spaz)
      -> void;
  //---------------------------------------------------------------------------
  auto set_version(unsigned short const major_version,
                   unsigned short const minor_version) -> void;
  //---------------------------------------------------------------------------
  auto major_version() const -> auto const & { return m_major_version; }
  auto set_major_version(unsigned short const major_version) -> void {
    m_major_version = major_version;
  }
  //---------------------------------------------------------------------------
  auto minor_version() const -> auto const & { return m_minor_version; }
  auto set_minor_version(unsigned short const minor_version) -> void {
    m_minor_version = minor_version;
  }
  //---------------------------------------------------------------------------
  auto type() const -> auto const & { return m_dataset_type; }
  auto set_type(dataset_type const type) -> void { m_dataset_type = type; }
  auto set_type(std::string const &type_str) -> void {
    m_dataset_type = parse_dataset_type(type_str);
  }
  //---------------------------------------------------------------------------
  auto title() const -> auto const & { return m_title; }
  auto set_title(std::string const &title) -> void { m_title = title; }
  auto set_title(std::string &&title) -> void { m_title = std::move(title); }
};
//=============================================================================
template <typename Real>
auto legacy_file_writer::write_points(
    std::vector<std::array<Real, 2>> const &points2) -> void {
  std::vector<std::array<Real, 3>> points3;
  points3.reserve(points2.size());
  for (auto const &x : points2) {
    points3.push_back({x[0], x[1], Real(0)});
  }
  write_points(points3);
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_points(
    std::vector<std::array<Real, 3>> const &points) -> void {
  std::stringstream ss;
  ss << "\nPOINTS " << points.size() << ' ' << type_to_str<Real>() << '\n';
  vtk::write_binary(m_file, ss.str());
  std::vector<std::array<Real, 3>> points_swapped(points);
  swap_endianess(reinterpret_cast<Real *>(points_swapped.data()),
                 3 * points.size());
  for (auto const &p : points_swapped) {
    for (auto c : p) {
      m_file.write((char *)(&c), sizeof(Real));
    }
  }
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_points(std::vector<vec<Real, 2>> const &points2)
    -> void {
  std::vector<std::array<Real, 3>> points3;
  points3.reserve(points2.size());
  for (auto const &x : points2) {
    points3.push_back({x(0), x(1), Real(0)});
  }
  write_points(points3);
}
//------------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_points(std::vector<vec<Real, 3>> const &points)
    -> void {
  std::stringstream ss;
  ss << "\nPOINTS " << points.size() << ' ' << type_to_str<Real>() << '\n';
  vtk::write_binary(m_file, ss.str());
  auto points_swapped = points;
  swap_endianess(reinterpret_cast<Real *>(points_swapped.data()),
                 3 * points.size());
  for (auto const &p : points_swapped) {
    for (size_t i = 0; i < 3; ++i) {
      m_file.write((char *)(&p[i]), sizeof(Real));
    }
  }
}
//------------------------------------------------------------------------------
template <size_t N>
auto legacy_file_writer::write_indices(
    std::string const &                       keyword,
    std::vector<std::array<size_t, N>> const &indices) -> void {
  size_t total_number = 0;
  for (auto const &is : indices)
    total_number += is.size() + 1;
  vtk::write_binary(m_file, "\n" + keyword + " " +
                                std::to_string(indices.size()) + ' ' +
                                std::to_string(total_number) + '\n');
  for (auto const &p : indices) {
    int size = (int)p.size();
    size     = swap_endianess(size);
    m_file.write((char *)(&size), sizeof(int));
    for (int i : p) {
      i = swap_endianess(i);
      m_file.write((char *)(&i), sizeof(int));
    }
  }
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto legacy_file_writer::write_data(
    std::string const &keyword, std::string const &name,
    std::vector<std::array<Real, N>> const &data) -> void {
  std::stringstream ss;
  ss << "\n" << keyword << ' ' << name << ' ' << type_to_str<Real>() << '\n';
  vtk::write_binary(m_file, ss.str());
  for (auto const &vec : data)
    for (auto comp : vec) {
      comp = swap_endianess(comp);
      m_file.write((char *)(&comp), sizeof(Real));
    }
}
//-----------------------------------------------------------------------------
template <size_t N>
auto legacy_file_writer::write_polygons(
    std::vector<std::array<size_t, N>> const &polygons) -> void {
  write_indices("POLYGONS", polygons);
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_normals(
    std::string const &name, std::vector<std::array<Real, 3>> &normals)
    -> void {
  write_data<3>("NORMALS", name, normals);
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_vectors(
    std::string const &name, std::vector<std::array<Real, 3>> &vectors)
    -> void {
  write_data<3>("VECTORS", name, vectors);
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_texture_coordinates(
    std::string const &               name,
    std::vector<std::array<Real, 2>> &texture_coordinates) -> void {
  write_data<2>("TEXTURE_COORDINATES", name, texture_coordinates);
}
//-----------------------------------------------------------------------------
template <typename Real>
auto legacy_file_writer::write_tensors(
    std::string const &name, std::vector<std::array<Real, 9>> &tensors)
    -> void {
  write_data<9>("TENSORS", name, tensors);
}
template <typename Data>
    requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
    (std::is_same_v<Data, int>)auto legacy_file_writer::write_scalars(
        std::string const &name, std::vector<Data> const &data,
        std::string const &lookup_table_name) -> void {
  std::stringstream ss;
  ss << "\nSCALARS " << name << ' ' << type_to_str<Data>() << " 1\n";
  vtk::write_binary(m_file, ss.str());
  vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
  for (auto comp : data) {
    comp = swap_endianess(comp);
    m_file.write((char *)(&comp), sizeof(Data));
  }
}
template <typename Data>
    requires(std::is_same_v<Data, double>) || (std::is_same_v<Data, float>) ||
    (std::is_same_v<Data, int>)auto legacy_file_writer::write_scalars(
        std::string const &name, std::vector<std::vector<Data>> const &data,
        std::string const &lookup_table_name) -> void {
  std::stringstream ss;
  ss << "\nSCALARS " << name << ' ' << type_to_str<Data>()
     << std::to_string(data.front().size()) + '\n';
  vtk::write_binary(m_file, ss.str());
  vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
  for (auto const &vec : data)
    for (auto comp : vec) {
      comp = swap_endianess(comp);
      m_file.write((char *)(&comp), sizeof(Data));
    }
}
//=============================================================================
}  // namespace tatooine::vtk
//=============================================================================
#endif
