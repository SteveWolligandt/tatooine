#ifndef TATOOINE_VTK_LEGACY_H
#define TATOOINE_VTK_LEGACY_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/string_conversion.h>
#include <tatooine/swap_endianess.h>
#include <tatooine/tensor.h>
#include <tatooine/type_to_str.h>
#include <tatooine/type_traits.h>

#include <cassert>
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
enum Format { UNKNOWN_FORMAT, ASCII, BINARY };
//------------------------------------------------------------------------------
enum ReaderData { POINT_DATA, CELL_DATA, UNSPECIFIED_DATA };
//------------------------------------------------------------------------------
enum DatasetType {
  UNKNOWN_TYPE,
  STRUCTURED_POINTS,
  STRUCTURED_GRID,
  UNSTRUCTURED_GRID,
  POLYDATA,
  RECTILINEAR_GRID,
  FIELD
};
//------------------------------------------------------------------------------
enum DataType {
  UNKNOWN_DATA_TYPE,
  UNSIGNED_CHAR,
  CHAR,
  UNSIGNED_SHORT,
  SHORT,
  UNSIGNED_INT,
  INT,
  UNSIGNED_LONG,
  LONG,
  FLOAT,
  DOUBLE
};
//------------------------------------------------------------------------------
enum CellType {
  UNKNOWN_CELL_TYPE = 0,
  VERTEX            = 1,
  POLY_VERTEX       = 2,
  LINE              = 3,
  POLY_LINE         = 4,
  TRIANGLE          = 5,
  TRIANGLE_STRIP    = 6,
  POLYGON           = 7,
  PIXEL             = 8,
  QUAD              = 9,
  TETRA             = 10,
  VOXEL             = 11,
  HEXAHEDRON        = 12,
  WEDGE             = 13,
  PYRAMID           = 14
};
//------------------------------------------------------------------------------
constexpr auto type_to_str(DatasetType type) -> std::string_view {
  switch (type) {
    default:
    case UNKNOWN_TYPE:
      return "UNKNOWN_TYPE";
    case STRUCTURED_POINTS:
      return "STRUCTURED_POINTS";
    case STRUCTURED_GRID:
      return "STRUCTURED_GRID";
    case UNSTRUCTURED_GRID:
      return "UNSTRUCTURED_GRID";
    case POLYDATA:
      return "POLYDATA";
    case RECTILINEAR_GRID:
      return "RECTILINEAR_GRID";
    case FIELD:
      return "FIELD";
  }
}
//-----------------------------------------------------------------------------
auto str_to_type(std::string const &type) -> DatasetType;
//-----------------------------------------------------------------------------
constexpr auto format_to_str(Format format) -> std::string_view {
  switch (format) {
    default:
    case UNKNOWN_FORMAT:
      return "UNKNOWN_FORMAT";
    case ASCII:
      return "ASCII";
    case BINARY:
      return "BINARY";
  }
}
//-----------------------------------------------------------------------------
auto str_to_format(std::string const &format) -> Format;
//-----------------------------------------------------------------------------
auto constexpr cell_type_to_str(CellType cell_type) -> std::string_view {
  switch (cell_type) {
    default:
    case UNKNOWN_CELL_TYPE:
      return "UNKNOWN_CELL_TYPE";
    case VERTEX:
      return "VERTEX";
    case POLY_VERTEX:
      return "POLY_VERTEX";
    case LINE:
      return "LINE";
    case POLY_LINE:
      return "POLY_LINE";
    case TRIANGLE:
      return "TRIANGLE";
    case TRIANGLE_STRIP:
      return "TRIANGLE_STRIP";
    case POLYGON:
      return "POLYGON";
    case PIXEL:
      return "PIXEL";
    case QUAD:
      return "QUAD";
    case TETRA:
      return "TETRA";
    case VOXEL:
      return "VOXEL";
    case HEXAHEDRON:
      return "HEXAHEDRON";
    case WEDGE:
      return "WEDGE";
    case PYRAMID:
      return "PYRAMID";
  }
}
//-----------------------------------------------------------------------------
auto str_to_cell_type(std::string const &cell_type) -> CellType;
//------------------------------------------------------------------------------
struct legacy_file_listener {
  // header data
  virtual void on_version(unsigned short /*major*/, unsigned short /*minor*/) {}
  virtual void on_title(std::string const &) {}
  virtual void on_format(Format) {}
  virtual void on_dataset_type(DatasetType) {}

  // coordinate data
  virtual void on_points(std::vector<std::array<float, 3>> const &) {}
  virtual void on_points(std::vector<std::array<double, 3>> const &) {}
  virtual void on_origin(double /*x*/, double /*y*/, double /*z*/) {}
  virtual void on_spacing(double /*x*/, double /*y*/, double /*z*/) {}
  virtual void on_dimensions(size_t /*x*/, size_t /*y*/, size_t /*z*/) {}
  virtual void on_x_coordinates(std::vector<float> const & /*xs*/) {}
  virtual void on_x_coordinates(std::vector<double> const & /*xs*/) {}
  virtual void on_y_coordinates(std::vector<float> const & /*ys*/) {}
  virtual void on_y_coordinates(std::vector<double> const & /*ys*/) {}
  virtual void on_z_coordinates(std::vector<float> const & /*zs*/) {}
  virtual void on_z_coordinates(std::vector<double> const & /*zs*/) {}

  // index data
  virtual void on_cells(std::vector<int> const &) {}
  virtual void on_cell_types(std::vector<CellType> const &) {}
  virtual void on_vertices(std::vector<int> const &) {}
  virtual void on_lines(std::vector<int> const &) {}
  virtual void on_polygons(std::vector<int> const &) {}
  virtual void on_triangle_strips(std::vector<int> const &) {}

  // cell- / pointdata
  virtual void on_vectors(std::string const & /*name*/,
                          std::vector<std::array<float, 3>> const & /*vectors*/,
                          ReaderData) {}
  virtual void on_vectors(
      std::string const & /*name*/,
      std::vector<std::array<double, 3>> const & /*vectors*/, ReaderData) {}
  virtual void on_normals(std::string const & /*name*/,
                          std::vector<std::array<float, 3>> const & /*normals*/,
                          ReaderData) {}
  virtual void on_normals(
      std::string const & /*name*/,
      std::vector<std::array<double, 3>> const & /*normals*/, ReaderData) {}
  virtual void on_texture_coordinates(
      std::string const & /*name*/,
      std::vector<std::array<float, 2>> const & /*texture_coordinates*/,
      ReaderData) {}
  virtual void on_texture_coordinates(
      std::string const & /*name*/,
      std::vector<std::array<double, 2>> const & /*texture_coordinates*/,
      ReaderData) {}
  virtual void on_tensors(std::string const & /*name*/,
                          std::vector<std::array<float, 9>> const & /*tensors*/,
                          ReaderData) {}
  virtual void on_tensors(
      std::string const & /*name*/,
      std::vector<std::array<double, 9>> const & /*tensors*/, ReaderData) {}

  virtual void on_scalars(std::string const & /*data_name*/,
                          std::string const & /*lookup_table_name*/,
                          size_t const /*num_comps*/,
                          std::vector<float> const & /*scalars*/, ReaderData) {}
  virtual void on_scalars(std::string const & /*data_name*/,
                          std::string const & /*lookup_table_name*/,
                          size_t const /*num_comps*/,
                          std::vector<double> const & /*scalars*/, ReaderData) {
  }
  virtual void on_point_data(size_t) {}
  virtual void on_cell_data(size_t) {}
  virtual void on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<int> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/) {}
  virtual void on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<float> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/) {}
  virtual void on_field_array(std::string const /*field_name*/,
                              std::string const /*field_array_name*/,
                              std::vector<double> const & /*data*/,
                              size_t /*num_comps*/, size_t /*num_tuples*/
  ) {}
};
//------------------------------------------------------------------------------
class legacy_file {
  std::vector<legacy_file_listener *> m_listeners;

  std::string m_path;
  Format      m_format;
  ReaderData  m_data = UNSPECIFIED_DATA;
  size_t      m_data_size;  // cell_data or point_data size
  long        m_begin_of_data;
  char        buffer[256];

 public:
  void add_listener(legacy_file_listener &listener);
  //---------------------------------------------------------------------------
  legacy_file(std::string const &path);
  //---------------------------------------------------------------------------
  void read();
  //---------------------------------------------------------------------------
  void        set_path(std::string const &path) { m_path = path; }
  void        set_path(std::string &&path) { m_path = std::move(path); }
  auto const &path() const { return m_path; }
  //---------------------------------------------------------------------------
 private:
  inline void read_header();
  inline void read_data();

  inline void read_spacing(std::ifstream &file);
  inline void read_dimensions(std::ifstream &file);
  inline void read_origin(std::ifstream &file);

  inline void read_points(std::ifstream &file);
  template <typename Real>
  inline void read_points_ascii(std::ifstream &file, size_t const n);
  template <typename Real>
  inline void read_points_binary(std::ifstream &file, size_t const n);

  inline void read_cell_types(std::ifstream &file);
  inline void read_cell_types_ascii(std::ifstream &file, size_t const n);
  inline void read_cell_types_binary(std::ifstream &file, size_t const n);

  inline std::vector<int> read_indices(std::ifstream &file);
  inline std::vector<int> read_indices_ascii(std::ifstream &file,
                                             size_t const   size);
  inline std::vector<int> read_indices_binary(std::ifstream &file,
                                              size_t const   size);

  auto        read_scalars_header(std::ifstream &file);
  inline void read_scalars(std::ifstream &file);
  template <typename Real>
  inline void read_scalars_ascii(std::ifstream &file, std::string const &name,
                                 std::string const &lookup_table,
                                 size_t const       num_comps);
  template <typename Real>
  inline void read_scalars_binary(std::ifstream &file, std::string const &name,
                                  std::string const &lookup_table,
                                  size_t const       num_comps);

  auto read_data_header(std::ifstream &file);
  template <typename Real, size_t N>
  inline std::vector<std::array<Real, N>> read_data(std::ifstream &file);
  template <typename Real, size_t N>
  inline std::vector<std::array<Real, N>> read_data_ascii(std::ifstream &file);
  template <typename Real, size_t N>
  inline std::vector<std::array<Real, N>> read_data_binary(std::ifstream &file);
  //----------------------------------------------------------------------------
  // coordinates
  auto read_coordinates_header(std::ifstream &file);
  template <typename Real>
  auto read_coordinates(std::ifstream &file, size_t n);
  template <typename Real>
  inline std::vector<Real> read_coordinates_ascii(std::ifstream &file,
                                                  size_t         n);
  template <typename Real>
  inline std::vector<Real> read_coordinates_binary(std::ifstream &file,
                                                   size_t         n);
  //----------------------------------------------------------------------------
  void read_x_coordinates(std::ifstream &file);
  void read_y_coordinates(std::ifstream &file);
  void read_z_coordinates(std::ifstream &file);
  //----------------------------------------------------------------------------
  // index data
  void read_cells(std::ifstream &file);
  void read_vertices(std::ifstream &file);
  void read_lines(std::ifstream &file);
  void read_polygons(std::ifstream &file);
  void read_triangle_strips(std::ifstream &file);
  //----------------------------------------------------------------------------
  // fixed size data
  void read_vectors(std::ifstream &file);
  void read_normals(std::ifstream &file);
  void read_texture_coordinates(std::ifstream &file);
  void read_tensors(std::ifstream &file);
  auto read_field_header(std::ifstream &file);
  //----------------------------------------------------------------------------
  // field data
  auto read_field_array_header(std::ifstream &file);
  void read_field(std::ifstream &file);

  template <typename Real>
  std::vector<Real> read_field_array_binary(std::ifstream &file,
                                            size_t         num_comps,
                                            size_t         num_tuples);
  template <typename Real>
  std::vector<Real> read_field_array_ascii(std::ifstream &file,
                                           size_t num_comps, size_t num_tuples);

  void consume_trailing_break(std::ifstream &file);
};
//------------------------------------------------------------------------------
template <typename Real>
void legacy_file::read_points_ascii(std::ifstream &file, size_t const n) {
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
void legacy_file::read_points_binary(std::ifstream &file, size_t const n) {
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
//-----------------------------------------------------------------------------------------------
template <typename Real, size_t n>
std::vector<std::array<Real, n>> legacy_file::read_data(std::ifstream &file) {
  if (m_format == ASCII)
    return read_data_ascii<Real, n>(file);
  else
    return read_data_binary<Real, n>(file);
}
//-----------------------------------------------------------------------------
template <typename Real, size_t n>
std::vector<std::array<Real, n>> legacy_file::read_data_ascii(
    std::ifstream &file) {
  std::vector<std::array<Real, n>> data(m_data_size);
  for (size_t i = 0; i < m_data_size; i++)
    for (size_t j = 0; j < n; j++)
      data[i][n] = parse<Real>(vtk::read_word(file, buffer));
  return data;
}
//-----------------------------------------------------------------------------
template <typename Real, size_t n>
std::vector<std::array<Real, n>> legacy_file::read_data_binary(
    std::ifstream &file) {
  std::vector<std::array<Real, n>> data(m_data_size);
  file.read((char *)data.data(), sizeof(Real) * m_data_size * n);
  swap_endianess(reinterpret_cast<Real *>(data.data()), n * m_data_size);
  consume_trailing_break(file);
  return data;
}
//-----------------------------------------------------------------------------------------------
template <typename Real>
auto legacy_file::read_coordinates(std::ifstream &file, size_t n) {
  if (m_format == ASCII)
    return read_coordinates_ascii<Real>(file, n);

  else /*if (m_format == BINARY)*/
    return read_coordinates_binary<Real>(file, n);
}
//-----------------------------------------------------------------------------------------------
template <typename Real>
std::vector<Real> legacy_file::read_coordinates_ascii(std::ifstream &file,
                                                      size_t const   n) {
  std::vector<Real> coordinates(n);
  for (size_t i = 0; i < n; i++)
    coordinates[i] = parse<Real>(vtk::read_word(file, buffer));
  return coordinates;
}
//-----------------------------------------------------------------------------
template <typename Real>
std::vector<Real> legacy_file::read_coordinates_binary(std::ifstream &file,
                                                       size_t const   n) {
  std::vector<Real> coordinates(n);
  file.read((char *)coordinates.data(), sizeof(Real) * n);
  swap_endianess(coordinates);
  consume_trailing_break(file);
  return coordinates;
}
//------------------------------------------------------------------------------
template <typename Real>
std::vector<Real> legacy_file::read_field_array_binary(std::ifstream &file,
                                                       size_t         num_comps,
                                                       size_t num_tuples) {
  std::vector<Real> data(num_comps * num_tuples);
  file.read((char *)data.data(), sizeof(Real) * num_comps * num_tuples);

  // consume trailing \n
  char consumer;
  file.read(&consumer, sizeof(char));

  return data;
}
//------------------------------------------------------------------------------
template <typename Real>
std::vector<Real> legacy_file::read_field_array_ascii(std::ifstream &file,
                                                      size_t         num_comps,
                                                      size_t num_tuples) {
  std::vector<Real> data;
  data.reserve(num_comps * num_tuples);
  for (size_t i = 0; i < num_comps * num_tuples; i++)
    data.push_back(parse<Real>(vtk::read_word(file, buffer)));

  return data;
}
//-----------------------------------------------------------------------------
template <typename Real>
void legacy_file::read_scalars_ascii(std::ifstream &    file,
                                     std::string const &name,
                                     std::string const &lookup_table,
                                     size_t const       num_comps) {
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
void legacy_file::read_scalars_binary(std::ifstream &    file,
                                      std::string const &name,
                                      std::string const &lookup_table,
                                      size_t const       num_comps) {
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
  DatasetType    m_dataset_type;
  std::string    m_title;

 public:
  legacy_file_writer(std::string const &path, DatasetType type,
                     unsigned short major = 2, unsigned short minor = 0,
                     std::string const &title = "");
  bool is_open();
  auto close() -> void;
  //---------------------------------------------------------------------------
 private:
  void write_indices(std::string const &                     keyword,
                     std::vector<std::vector<size_t>> const &indices);
  template <size_t N>
  inline void write_indices(std::string const &                       keyword,
                            std::vector<std::array<size_t, N>> const &indices);
  template <typename Real, size_t N>
  void write_data(std::string const &keyword, std::string const &name,
                  std::vector<std::array<Real, N>> const &data);

 public:
  void write_header();
  template <typename Real>
  void write_points(std::vector<std::array<Real, 3>> const &points);
  template <typename Real>
  void write_points(std::vector<vec<Real, 3>> const &points);
  void write_cells(std::vector<std::vector<size_t>> const &cells);
  void write_cell_types(std::vector<CellType> const &cell_types);

  void write_vertices(std::vector<std::vector<size_t>> const &vertices);
  void write_lines(std::vector<std::vector<size_t>> const &lines);
  void write_polygons(std::vector<std::vector<size_t>> const &polygons);
  template <size_t N>
  void write_polygons(std::vector<std::array<size_t, N>> const &polygons);
  void write_triangle_strips(
      std::vector<std::vector<size_t>> const &triangle_strips);

  template <real_number T>
  void write_x_coordinates(std::vector<T> const &x_coordinates) {
    std::stringstream ss;
    ss << "\nX_COORDINATES " << ' ' << x_coordinates.size() << ' '
       << tatooine::type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : x_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
  template <real_number T>
  void write_y_coordinates(std::vector<T> const &y_coordinates) {
    std::stringstream ss;
    ss << "\nY_COORDINATES " << ' ' << y_coordinates.size() << ' '
       << tatooine::type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : y_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
  template <real_number T>
  void write_z_coordinates(std::vector<T> const &z_coordinates) {
    std::stringstream ss;
    ss << "\nZ_COORDINATES " << ' ' << z_coordinates.size() << ' '
       << tatooine::type_to_str<T>() << '\n';
    vtk::write_binary(m_file, ss.str());
    T d;
    for (auto const &c : z_coordinates) {
      d = swap_endianess(c);
      m_file.write((char *)(&d), sizeof(T));
    }
  }
  void write_point_data(size_t i);
  void write_cell_data(size_t i);
  template <typename Real>
  void write_normals(std::string const &               name,
                     std::vector<std::array<Real, 3>> &normals);
  //----------------------------------------------------------------------------
  template <typename Real>
  void write_vectors(std::string const &               name,
                     std::vector<std::array<Real, 3>> &vectors);
  //----------------------------------------------------------------------------
  template <typename Real>
  void write_texture_coordinates(
      std::string const &               name,
      std::vector<std::array<Real, 2>> &texture_coordinates);
  //----------------------------------------------------------------------------
  template <typename Real>
  void write_tensors(std::string const &               name,
                     std::vector<std::array<Real, 9>> &tensors);
  //----------------------------------------------------------------------------
  template <typename Data>
  requires (std::is_same_v<Data, double>) ||
           (std::is_same_v<Data, float>) ||
           (std::is_same_v<Data, int>)
  void write_scalars(std::string const &name, std::vector<Data> const &data,
                     std::string const &lookup_table_name = "default");
  //----------------------------------------------------------------------------
  template <typename Data>
  requires (std::is_same_v<Data, double>) ||
           (std::is_same_v<Data, float>) ||
           (std::is_same_v<Data, int>)
  void write_scalars(std::string const &                   name,
                     std::vector<std::vector<Data>> const &data,
                     std::string const &lookup_table_name = "default");
  //----------------------------------------------------------------------------
  template <typename Data, size_t N>
  requires (std::is_same_v<Data, double>) ||
           (std::is_same_v<Data, float>) ||
           (std::is_same_v<Data, int>)
  void write_scalars(std::string const &                     name,
                     std::vector<std::array<Data, N>> const &data,
                     std::string const &lookup_table_name = "default") {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << tatooine::type_to_str<Data>() << ' '
       << N << '\n';
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
  requires (std::is_same_v<Data, double>) ||
           (std::is_same_v<Data, float>) ||
           (std::is_same_v<Data, int>)
  void write_scalars(std::string const &              name,
                     std::vector<vec<Data, N>> const &data,
                     std::string const &lookup_table_name = "default") {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << tatooine::type_to_str<Data>() << ' '
       << N << '\n';
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
  requires (std::is_same_v<Real, double>) ||
           (std::is_same_v<Real, float>) ||
           (std::is_same_v<Real, int>)
  void write_scalars(std::string const &                 name,
                     std::vector<tensor<Real, N>> const &data,
                     std::string const &lookup_table_name = "default") {
    std::stringstream ss;
    ss << "\nSCALARS " << name << ' ' << tatooine::type_to_str<Real>() << ' '
       << N << '\n';
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
  void write_dimensions(size_t dimx, size_t dimy, size_t dimz);
  //----------------------------------------------------------------------------
  void write_origin(double orgx, double orgy, double orgz);
  //----------------------------------------------------------------------------
  void write_spacing(double spax, double spay, double spaz);
  //---------------------------------------------------------------------------
  void set_version(unsigned short const major_version,
                   unsigned short const minor_version);
  //---------------------------------------------------------------------------
  auto const &major_version() const { return m_major_version; }
  void        set_major_version(unsigned short const major_version) {
    m_major_version = major_version;
  }
  //---------------------------------------------------------------------------
  auto const &minor_version() const { return m_minor_version; }
  void        set_minor_version(unsigned short const minor_version) {
    m_minor_version = minor_version;
  }
  //---------------------------------------------------------------------------
  auto const &type() const { return m_dataset_type; }
  auto        type_str() const { return type_to_str(m_dataset_type); }
  void        set_type(DatasetType const type) { m_dataset_type = type; }
  void        set_type(std::string const &type_str) {
    m_dataset_type = str_to_type(type_str);
  }
  //---------------------------------------------------------------------------
  auto const &title() const { return m_title; }
  void        set_title(std::string const &title) { m_title = title; }
  void        set_title(std::string &&title) { m_title = std::move(title); }
};
//=============================================================================
template <typename Real>
void legacy_file_writer::write_points(
    std::vector<std::array<Real, 3>> const &points) {
  std::stringstream ss;
  ss << "\nPOINTS " << points.size() << ' ' << tatooine::type_to_str<Real>()
     << '\n';
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
void legacy_file_writer::write_points(std::vector<vec<Real, 3>> const &points) {
  std::stringstream ss;
  ss << "\nPOINTS " << points.size() << ' ' << tatooine::type_to_str<Real>()
     << '\n';
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
void legacy_file_writer::write_indices(
    std::string const &                       keyword,
    std::vector<std::array<size_t, N>> const &indices) {
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
void legacy_file_writer::write_data(
    std::string const &keyword, std::string const &name,
    std::vector<std::array<Real, N>> const &data) {
  std::stringstream ss;
  ss << "\n"
     << keyword << ' ' << name << ' ' << tatooine::type_to_str<Real>() << '\n';
  vtk::write_binary(m_file, ss.str());
  for (auto const &vec : data)
    for (auto comp : vec) {
      comp = swap_endianess(comp);
      m_file.write((char *)(&comp), sizeof(Real));
    }
}
//-----------------------------------------------------------------------------
template <size_t N>
void legacy_file_writer::write_polygons(
    std::vector<std::array<size_t, N>> const &polygons) {
  write_indices("POLYGONS", polygons);
}
//-----------------------------------------------------------------------------
template <typename Real>
void legacy_file_writer::write_normals(
    std::string const &name, std::vector<std::array<Real, 3>> &normals) {
  write_data<3>("NORMALS", name, normals);
}
//-----------------------------------------------------------------------------
template <typename Real>
void legacy_file_writer::write_vectors(
    std::string const &name, std::vector<std::array<Real, 3>> &vectors) {
  write_data<3>("VECTORS", name, vectors);
}
//-----------------------------------------------------------------------------
template <typename Real>
void legacy_file_writer::write_texture_coordinates(
    std::string const &               name,
    std::vector<std::array<Real, 2>> &texture_coordinates) {
  write_data<2>("TEXTURE_COORDINATES", name, texture_coordinates);
}
//-----------------------------------------------------------------------------
template <typename Real>
void legacy_file_writer::write_tensors(
    std::string const &name, std::vector<std::array<Real, 9>> &tensors) {
  write_data<9>("TENSORS", name, tensors);
}
template <typename Data>
requires (std::is_same_v<Data, double>) ||
         (std::is_same_v<Data, float>) ||
         (std::is_same_v<Data, int>)
void legacy_file_writer::write_scalars(std::string const &      name,
                                       std::vector<Data> const &data,
                                       std::string const &lookup_table_name) {
  std::stringstream ss;
  ss << "\nSCALARS " << name << ' ' << tatooine::type_to_str<Data>() << " 1\n";
  vtk::write_binary(m_file, ss.str());
  vtk::write_binary(m_file, "\nLOOKUP_TABLE " + lookup_table_name + '\n');
  for (auto comp : data) {
    comp = swap_endianess(comp);
    m_file.write((char *)(&comp), sizeof(Data));
  }
}
template <typename Data>
requires (std::is_same_v<Data, double>) ||
         (std::is_same_v<Data, float>) ||
         (std::is_same_v<Data, int>)
void legacy_file_writer::write_scalars(
    std::string const &name, std::vector<std::vector<Data>> const &data,
    std::string const &lookup_table_name) {
  std::stringstream ss;
  ss << "\nSCALARS " << name << ' ' << tatooine::type_to_str<Data>()
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
