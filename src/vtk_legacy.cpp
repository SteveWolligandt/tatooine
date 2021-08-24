#include <tatooine/vtk_legacy.h>

#include <tatooine/concepts.h>
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
#include <utility>
#include <vector>
//=============================================================================
namespace tatooine::vtk {
//=============================================================================
/// reads until terminator_char was found. buffer will containg the
/// terminator_char
auto read_until(std::istream &stream, char const terminator_char, char *buffer)
    -> std::string {
  size_t idx = 0;
  do {
    if (!stream.eof()) { stream.read(&buffer[idx], sizeof(char));
}
    idx++;
  } while (buffer[idx - 1] != terminator_char && !stream.eof());
  buffer[idx] = '\0';
  return buffer;
}
//-----------------------------------------------------------------------------
/// reads stream until a linebreak was found. buffer will not contain the break
auto read_binaryline(std::istream &stream, char *buffer) -> std::string {
  auto str = read_until(stream, '\n', buffer);
  return str.erase(str.size() - 1, 1);
}
//-----------------------------------------------------------------------------
/// reads stream until a whitespace was found and consumes any whitespace until
/// another character was found. buffer will not contain any whitespace
auto read_word(std::istream &stream, char *buffer) -> std::string {
  size_t idx = 0;
  do {
    if (!stream.eof()) { stream.read(&buffer[idx], sizeof(char));
}

    idx++;
  } while ((buffer[idx - 1] != ' ') && (buffer[idx - 1] != '\n') &&
           (buffer[idx - 1] != '\t') && !stream.eof());
  buffer[idx - 1] = '\0';
  std::string word(buffer);

  do {
    if (!stream.eof()) { stream.read(buffer, sizeof(char));
}
  } while ((buffer[0] == ' ' || buffer[0] == '\n' || buffer[0] == '\t') &&
           !stream.eof());
  if (!stream.eof()) {
    stream.seekg(stream.tellg() - std::streampos(1), stream.beg);
}
  return word;
}
//-----------------------------------------------------------------------------
auto write_binary(std::ostream &stream, std::string const &str) -> void {
  stream.write(str.c_str(), long(sizeof(char)) * long(str.size()));
}
//-----------------------------------------------------------------------------
auto write_binary(std::ostream &stream, char const c) -> void {
  stream.write(&c, long(sizeof(char)));
}
//-----------------------------------------------------------------------------
auto parse_dataset_type(std::string const &type) -> dataset_type {
  if (type == "STRUCTURED_POINTS") {
    return dataset_type::structured_points;
  } if (type == "STRUCTURED_GRID") {
    return dataset_type::structured_grid;
  } if (type == "UNSTRUCTURED_GRID") {
    return dataset_type::unstructured_grid;
  } else if (type == "POLYDATA") {
    return dataset_type::polydata;
  } else if (type == "RECTILINEAR_GRID") {
    return dataset_type::rectilinear_grid;
  } else if (type == "FIELD") {
    return dataset_type::field;
  }
  return dataset_type::unknown;
}
//-----------------------------------------------------------------------------
auto parse_format(std::string const &f) -> format {
  if (f == to_string_view(format::ascii)) {
    return format::ascii;
  } if (f == to_string_view(format::binary)) {
    return format::binary;
  }
  return format::unknown;
}
//-----------------------------------------------------------------------------
auto parse_cell_type(std::string const &ct) -> cell_type {
  if (ct == "VERTEX") {
    return cell_type::vertex;
  } if (ct == "POLY_VERTEX") {
    return cell_type::poly_vertex;
  } if (ct == "LINE") {
    return cell_type::line;
  } else if (ct == "POLY_LINE") {
    return cell_type::poly_line;
  } else if (ct == "TRIANGLE") {
    return cell_type::triangle;
  } else if (ct == "TRIANGLE_STRIP") {
    return cell_type::triangle_strip;
  } else if (ct == "POLYGON") {
    return cell_type::polygon;
  } else if (ct == "PIXEL") {
    return cell_type::pixel;
  } else if (ct == "QUAD") {
    return cell_type::quad;
  } else if (ct == "TETRA") {
    return cell_type::tetra;
  } else if (ct == "VOXEL") {
    return cell_type::voxel;
  } else if (ct == "HEXAHEDRON") {
    return cell_type::hexahedron;
  } else if (ct == "WEDGE") {
    return cell_type::wedge;
  } else if (ct == "PYRAMID") {
    return cell_type::pyramid;
  }
  return cell_type::unknown_cell_type;
}
//---------------------------------------------------------------------------
auto legacy_file::add_listener(legacy_file_listener &listener) -> void {
  m_listeners.push_back(&listener);
}
//===========================================================================
legacy_file::legacy_file(filesystem::path path) : m_path(std::move(path)) {}
//---------------------------------------------------------------------------
auto legacy_file::read() -> void {
  read_header();
  read_data();
}
//---------------------------------------------------------------------------
auto legacy_file::read_scalars_header(std::ifstream &file) {
  std::string       scalar_params = vtk::read_binaryline(file, buffer);
  std::stringstream scalar_params_stream(scalar_params);

  auto data_name    = vtk::read_word(scalar_params_stream, buffer);
  auto data_type    = vtk::read_word(scalar_params_stream, buffer);
  auto num_comp_str = vtk::read_word(scalar_params_stream, buffer);
  consume_trailing_break(file);
  // number of components is optional
  size_t num_comps = 1;
  if (num_comp_str.empty()) {
    vtk::read_word(file, buffer);  // consume empty space
  } else if (num_comp_str != "LOOKUP_TABLE") {
    num_comps = parse<size_t>(num_comp_str);
    vtk::read_word(file, buffer);  // consume LOOKUP_TABLE keyword
  }
  auto lookup_table_name = vtk::read_word(file, buffer);
  return std::make_tuple(data_name, data_type, num_comps, lookup_table_name);
}
//----------------------------------------------------------------------------
auto legacy_file::read_data_header(std::ifstream &file) {
  return std::make_pair(vtk::read_word(file, buffer),
                        vtk::read_word(file, buffer));
}
//----------------------------------------------------------------------------
// coordinates
auto legacy_file::read_coordinates_header(std::ifstream &file) {
  return std::make_pair(parse<size_t>(vtk::read_word(file, buffer)),
                        vtk::read_word(file, buffer));
}
//----------------------------------------------------------------------------
auto legacy_file::read_x_coordinates(std::ifstream &file) -> void {
  auto const  header = read_coordinates_header(file);
  auto const &n      = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto c = read_coordinates<float>(file, n);
    for (auto *l : m_listeners) {
      l->on_x_coordinates(c);
}
  } else if (type == "double") {
    auto c = read_coordinates<double>(file, n);
    for (auto *l : m_listeners) {
      l->on_x_coordinates(c);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_y_coordinates(std::ifstream &file) -> void {
  auto const  header = read_coordinates_header(file);
  auto const &n      = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto c = read_coordinates<float>(file, n);
    for (auto *l : m_listeners) {
      l->on_y_coordinates(c);
}
  } else if (type == "double") {
    auto c = read_coordinates<double>(file, n);
    for (auto *l : m_listeners) {
      l->on_y_coordinates(c);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_z_coordinates(std::ifstream &file) -> void {
  auto const  header = read_coordinates_header(file);
  auto const &n      = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto c = read_coordinates<float>(file, n);
    for (auto *l : m_listeners) {
      l->on_z_coordinates(c);
}
  } else if (type == "double") {
    auto c = read_coordinates<double>(file, n);
    for (auto *l : m_listeners) {
      l->on_z_coordinates(c);
}
  }
}
//----------------------------------------------------------------------------
// index data
auto legacy_file::read_cells(std::ifstream &file) -> void {
  auto i = read_indices(file);
  for (auto *l : m_listeners) {
    l->on_cells(i);
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_vertices(std::ifstream &file) -> void {
  auto const i = read_indices(file);
  for (auto *l : m_listeners) {
    l->on_vertices(i);
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_lines(std::ifstream &file) -> void {
  auto const i = read_indices(file);
  for (auto *l : m_listeners) {
    l->on_lines(i);
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_polygons(std::ifstream &file) -> void {
  auto const i = read_indices(file);
  for (auto *l : m_listeners) {
    l->on_polygons(i);
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_triangle_strips(std::ifstream &file) -> void {
  auto const i = read_indices(file);
  for (auto *l : m_listeners) {
    l->on_triangle_strips(i);
  }
}
//----------------------------------------------------------------------------
// fixed size data
auto legacy_file::read_vectors(std::ifstream &file) -> void {
  auto const  header = read_data_header(file);
  auto const &name   = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto data = read_data<float, 3>(file);
    for (auto *l : m_listeners) {
      l->on_vectors(name, data, m_data);
}
  } else if (type == "double") {
    auto data = read_data<double, 3>(file);
    for (auto *l : m_listeners) {
      l->on_vectors(name, data, m_data);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_normals(std::ifstream &file) -> void {
  auto const  header = read_data_header(file);
  auto const &name   = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto data = read_data<float, 3>(file);
    for (auto *l : m_listeners) {
      l->on_normals(name, data, m_data);
}
  } else if (type == "double") {
    auto data = read_data<double, 3>(file);
    for (auto *l : m_listeners) {
      l->on_normals(name, data, m_data);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_texture_coordinates(std::ifstream &file) -> void {
  auto const  header = read_data_header(file);
  auto const &name   = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto data = read_data<float, 2>(file);
    for (auto *l : m_listeners) {
      l->on_texture_coordinates(name, data, m_data);
}
  } else if (type == "double") {
    auto data = read_data<double, 2>(file);
    for (auto *l : m_listeners) {
      l->on_texture_coordinates(name, data, m_data);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_tensors(std::ifstream &file) -> void {
  auto const  header = read_data_header(file);
  auto const &name   = header.first;
  auto const &type   = header.second;
  if (type == "float") {
    auto data = read_data<float, 9>(file);
    for (auto *l : m_listeners) {
      l->on_tensors(name, data, m_data);
}
  } else if (type == "double") {
    auto data = read_data<double, 9>(file);
    for (auto *l : m_listeners) {
      l->on_tensors(name, data, m_data);
}
  }
}
//----------------------------------------------------------------------------
auto legacy_file::read_field_header(std::ifstream &file)
    -> std::pair<std::string, size_t> {
  std::string       field_params = vtk::read_binaryline(file, buffer);
  std::stringstream field_params_stream(field_params);

  return {vtk::read_word(field_params_stream, buffer),
          parse<size_t>(vtk::read_word(field_params_stream, buffer))};
}
//----------------------------------------------------------------------------
// field data
auto legacy_file::read_field_array_header(std::ifstream &file)
    -> std::tuple<std::string, size_t, size_t, std::string> {
  std::string       field_array_params = vtk::read_binaryline(file, buffer);
  std::stringstream field_array_params_stream(field_array_params);
  std::string       array_name;
  field_array_params_stream >> array_name;
  while (array_name == "METADATA" || array_name == "INFORMATION" ||
         field_array_params.empty()) {
    field_array_params        = vtk::read_binaryline(file, buffer);
    field_array_params_stream = std::stringstream{field_array_params};
    if (!field_array_params.empty()) {
      field_array_params_stream >> array_name;
    }
  }
  size_t num_comps = 0, num_tuples = 0;
  std::string datatype_str;
  field_array_params_stream >> num_comps >> num_tuples >> datatype_str;
  return {array_name, num_comps, num_tuples, datatype_str};
}
//----------------------------------------------------------------------------
auto legacy_file::read_field(std::ifstream &file) -> void {
  auto const  header     = read_field_header(file);
  auto const &field_name = header.first;
  auto const &num_arrays = header.second;
  for (size_t i = 0; i < num_arrays; ++i) {
    auto const [field_array_name, num_comps, num_tuples, datatype_str] =
        read_field_array_header(file);

    if (m_format == format::ascii) {
      if (datatype_str == "int") {
        auto data = read_field_array_ascii<int>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      } else if (datatype_str == "float") {
        auto data = read_field_array_ascii<float>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      } else if (datatype_str == "double") {
        auto data = read_field_array_ascii<double>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      }
    } else if (m_format == format::binary) {
      if (datatype_str == "int") {
        auto data = read_field_array_binary<int>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      } else if (datatype_str == "float") {
        auto data = read_field_array_binary<float>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      } else if (datatype_str == "double") {
        auto data =
            read_field_array_binary<double>(file, num_comps, num_tuples);
        for (auto *l : m_listeners) {
          l->on_field_array(field_name, field_array_name, data, num_comps,
                            num_tuples);
        }
      }
    }
  }
}
//------------------------------------------------------------------------------
auto legacy_file::consume_trailing_break(std::ifstream &file) -> void {
  char consumer = 0;
  file.read(&consumer, sizeof(char));
}
//-----------------------------------------------------------------------------------------------
auto legacy_file::read_header() -> void {
#ifdef TATOOINE_STD_FILESYSTEM_AVAILABLE
  std::ifstream file(m_path, std::ifstream::binary);
#else
  std::ifstream file(m_path.string(), std::ifstream::binary);
#endif
  if (file.is_open()) {
    // read part1 # vtk DataFile Version x.x
    std::string part1 = vtk::read_binaryline(file, buffer);
    for (auto *listener : m_listeners) {
      listener->on_version((unsigned short)(atoi(&buffer[23])),
                           (unsigned short)(atoi(&buffer[25])));
}

    // read part2 maximal 256 characters
    std::string part2 = vtk::read_binaryline(file, buffer);
    for (auto *listener : m_listeners) {
      listener->on_title(part2);
    }

    // read part3 ASCII | BINARY
    std::string part3 = vtk::read_binaryline(file, buffer);
    if (part3 == "ASCII" || part3 == "ascii") {
      m_format = format::ascii;
    } else if (part3 == "BINARY" || part3 == "binary") {
      m_format = format::binary;
    } else {
      m_format = format::unknown;
    }

    for (auto *listener : m_listeners) {
      listener->on_format(m_format);
}

    // read part4 STRUCTURED_POINTS | STRUCTURED_GRID | UNSTRUCTURED_GRID |
    // POLYDATA | RECTILINEAR_GRID | FIELD
    file.read(buffer, sizeof(char) * 8);  // consume "DATASET "
    auto part4 = parse_dataset_type(vtk::read_binaryline(file, buffer));
    for (auto *listener : m_listeners) {
      listener->on_dataset_type(part4);
    }

    m_begin_of_data = file.tellg();
    file.close();
  } else {
    throw std::runtime_error{"[vtk::legacy_file] could not open file " +
                             m_path.string()};
  }
}
//-----------------------------------------------------------------------------
auto legacy_file::read_data() -> void {
#ifdef TATOOINE_STD_FILESYSTEM_AVAILABLE
  std::ifstream file(m_path, std::ifstream::binary);
#else
  std::ifstream file(m_path.string(), std::ifstream::binary);
#endif
  if (file.is_open()) {
    file.seekg(m_begin_of_data, file.beg);
    std::string keyword;

    while (!file.eof()) {
      keyword = vtk::read_word(file, buffer);
      if (!keyword.empty()) {
        if (keyword == "POINTS") {
          read_points(file);
        } else if (keyword == "LINES") {
          read_lines(file);
        } else if (keyword == "VERTICES") {
          read_vertices(file);
        } else if (keyword == "POLYGONS") {
          read_polygons(file);
        } else if (keyword == "CELLS") {
          read_cells(file);
        } else if (keyword == "CELL_TYPES") {
          read_cell_types(file);
        } else if (keyword == "DIMENSIONS") {
          read_dimensions(file);
        } else if (keyword == "ORIGIN") {
          read_origin(file);
        } else if (keyword == "SPACING") {
          read_spacing(file);
        } else if (keyword == "X_COORDINATES") {
          read_x_coordinates(file);
        } else if (keyword == "Y_COORDINATES") {
          read_y_coordinates(file);
        } else if (keyword == "Z_COORDINATES") {
          read_z_coordinates(file);
        } else if (keyword == "POINT_DATA") {
          auto const word = vtk::read_word(file, buffer);
          m_data_size     = size_t(parse<int>(word));
          m_data          = reader_data::point_data;
          for (auto *l : m_listeners) {
            l->on_point_data(m_data_size);
          }
        } else if (keyword == "CELL_DATA") {
          m_data_size = size_t(parse<int>(vtk::read_word(file, buffer)));
          m_data      = reader_data::point_data;
          for (auto *l : m_listeners) {
            l->on_cell_data(m_data_size);
}
        } else if (keyword == "SCALARS") {
          read_scalars(file);
        } else if (keyword == "VECTORS") {
          read_vectors(file);
        } else if (keyword == "NORMALS") {
          read_normals(file);
        } else if (keyword == "TEXTURE_COORDINATES") {
          read_texture_coordinates(file);
        } else if (keyword == "FIELD") {
          read_field(file);
        } else {
            //std::cerr << "[tatooine::vtk::legacy_file] unknown keyword: "
          //          << keyword << '\n';
        }
      }
    }
  } else {
    throw std::runtime_error{
        "[tatooine::vtk::legacy_file] could not open file " + m_path.string()};
};
}
//------------------------------------------------------------------------------
auto legacy_file::read_spacing(std::ifstream &file) -> void {
  std::array<double, 3> spacing{parse<double>(vtk::read_word(file, buffer)),
                                parse<double>(vtk::read_word(file, buffer)),
                                parse<double>(vtk::read_word(file, buffer))};
  for (auto *l : m_listeners) {
    l->on_spacing(spacing[0], spacing[1], spacing[2]);
  }
}
//------------------------------------------------------------------------------
auto legacy_file::read_dimensions(std::ifstream &file) -> void {
  std::array<size_t, 3> dims{parse<size_t>(vtk::read_word(file, buffer)),
                             parse<size_t>(vtk::read_word(file, buffer)),
                             parse<size_t>(vtk::read_word(file, buffer))};
  for (auto *l : m_listeners) {
    l->on_dimensions(dims[0], dims[1], dims[2]);
  }
}
//------------------------------------------------------------------------------
auto legacy_file::read_origin(std::ifstream &file) -> void {
  std::array<double, 3> origin{parse<double>(vtk::read_word(file, buffer)),
                               parse<double>(vtk::read_word(file, buffer)),
                               parse<double>(vtk::read_word(file, buffer))};
  for (auto *l : m_listeners) {
    l->on_origin(origin[0], origin[1], origin[2]);
  }
}
//------------------------------------------------------------------------------
auto legacy_file::read_points(std::ifstream &file) -> void {
  auto const num_points_str = vtk::read_word(file, buffer);
  auto const n              = parse<size_t>(num_points_str);
  auto const datatype_str   = vtk::read_word(file, buffer);

  if (m_format == format::ascii) {
    if (datatype_str == "float") {
      read_points_ascii<float>(file, n);
    } else if (datatype_str == "double") {
      read_points_ascii<double>(file, n);
}
  } else if (m_format == format::binary) {
    if (datatype_str == "float") {
      read_points_binary<float>(file, n);
    } else if (datatype_str == "double") {
      read_points_binary<double>(file, n);
}
  }
}
//-----------------------------------------------------------------------------------------------
auto legacy_file::read_cell_types(std::ifstream &file) -> void {
  auto num_cell_types_str = vtk::read_word(file, buffer);
  auto num_cell_types     = parse<size_t>(num_cell_types_str);

  if (m_format == format::ascii) {
    read_cell_types_ascii(file, num_cell_types);
  } else if (m_format == format::binary) {
    read_cell_types_binary(file, num_cell_types);
}
}
//-----------------------------------------------------------------------------------------------
auto legacy_file::read_cell_types_ascii(std::ifstream &file,
                                        size_t const   num_cell_types) -> void {
  std::vector<cell_type> cell_types;
  std::string            cell_type_str;
  for (size_t i = 0; i < num_cell_types; i++) {
    cell_type_str = vtk::read_word(file, buffer);
    cell_types.push_back((cell_type)parse<int>(cell_type_str));
  }
  for (auto *listener : m_listeners) {
    listener->on_cell_types(cell_types);
}
}
//-----------------------------------------------------------------------------
auto legacy_file::read_cell_types_binary(std::ifstream &file,
                                         size_t const num_cell_types) -> void {
  std::vector<cell_type> cell_types(num_cell_types);
  file.read((char *)cell_types.data(), sizeof(int) * num_cell_types);
  swap_endianess(cell_types);
  for (auto *listener : m_listeners) {
    listener->on_cell_types(cell_types);
}
  consume_trailing_break(file);
}
//-----------------------------------------------------------------------------------------------
auto legacy_file::read_indices(std::ifstream &file) -> std::vector<int> {
  auto const num_indices_str = vtk::read_word(file, buffer);
  auto const size_str        = vtk::read_word(file, buffer);

  [[maybe_unused]] auto const num_indices = parse<size_t>(num_indices_str);
  auto const                  size        = parse<size_t>(size_str);

  if (m_format == format::ascii) {
    return read_indices_ascii(file, size);
  }
  /* if (m_format == format::binary) */     return read_indices_binary(file, size);

}
//-----------------------------------------------------------------------------
auto legacy_file::read_indices_ascii(std::ifstream & /*file*/,
                                     size_t const /*size*/)
    -> std::vector<int> {
  std::vector<int> indices;
  // indices.reserve(size);
  // std::string val_str;
  // for (size_t i = 0; i < size; i++)
  //  indices.push_back(parse<size_t>(vtk::read_word(file, buffer)));
  return indices;
}
//-----------------------------------------------------------------------------
auto legacy_file::read_indices_binary(std::ifstream &file, size_t const size)
    -> std::vector<int> {
  std::vector<int> data(size);
  if (size > 0) {
    file.read((char *)data.data(), sizeof(int) * size);
    swap_endianess(data);
    consume_trailing_break(file);
  }
  return data;
}
//------------------------------------------------------------------------------
auto legacy_file::read_scalars(std::ifstream &file) -> void {
  auto const &[name, type, num_comps, lookup_table] = read_scalars_header(file);
  if (m_format == format::ascii) {
    if (type == "float") {
      read_scalars_ascii<float>(file, name, lookup_table, num_comps);
    } else if (type == "double") {
      read_scalars_ascii<double>(file, name, lookup_table, num_comps);
    }
  } else if (m_format == format::binary) {
    if (type == "float") {
      read_scalars_binary<float>(file, name, lookup_table, num_comps);
    } else if (type == "double") {
      read_scalars_binary<double>(file, name, lookup_table, num_comps);
    }
  }
}
//------------------------------------------------------------------------------
legacy_file_writer::legacy_file_writer(filesystem::path const &path,
                                       dataset_type type, unsigned short major,
                                       unsigned short     minor,
                                       std::string title)
#ifdef TATOOINE_STD_FILESYSTEM_AVAILABLE
    : m_file{path, std::ofstream::binary},
#else
    : m_file{path.string(), std::ofstream::binary},
#endif
      m_major_version{major},
      m_minor_version{minor},
      m_dataset_type{type},
      m_title{std::move(title)} {}

auto legacy_file_writer::is_open() -> bool { return m_file.is_open(); }
auto legacy_file_writer::close() -> void { m_file.close(); }
//---------------------------------------------------------------------------
auto legacy_file_writer::write_dimensions(size_t const dimx, size_t const dimy,
                                          size_t const dimz) -> void {
  vtk::write_binary(m_file, "\nDIMENSIONS " + std::to_string(dimx) + ' ' +
                                std::to_string(dimy) + ' ' +
                                std::to_string(dimz));
}
//----------------------------------------------------------------------------
auto legacy_file_writer::write_origin(double const orgx, double const orgy,
                                      double const orgz) -> void {
  vtk::write_binary(m_file, "\nORIGIN " + std::to_string(orgx) + ' ' +
                                std::to_string(orgy) + ' ' +
                                std::to_string(orgz));
}
//----------------------------------------------------------------------------
auto legacy_file_writer::write_spacing(double const spax, double const spay,
                                       double const spaz) -> void {
  vtk::write_binary(m_file, "\nSPACING " + std::to_string(spax) + ' ' +
                                std::to_string(spay) + ' ' +
                                std::to_string(spaz));
}
//---------------------------------------------------------------------------
auto legacy_file_writer::set_version(unsigned short const major_version,
                                     unsigned short const minor_version)
    -> void {
  set_major_version(major_version);
  set_minor_version(minor_version);
}
//---------------------------------------------------------------------------
auto legacy_file_writer::write_header() -> void {
  // write opener
  vtk::write_binary(m_file, "# vtk DataFile Version " +
                                std::to_string(m_major_version) + "." +
                                std::to_string(m_minor_version) + '\n');

  // write title
  vtk::write_binary(m_file, m_title + '\n');

  // write part3 ASCII | BINARY
  vtk::write_binary(m_file, std::string{to_string_view(format::binary)} + '\n');

  // write part4 STRUCTURED_POINTS | STRUCTURED_GRID | UNSTRUCTURED_GRID |
  // POLYDATA | RECTILINEAR_GRID | FIELD
  std::stringstream ss;
  ss << "DATASET " << to_string_view(m_dataset_type);
  vtk::write_binary(m_file, ss.str());
}
//------------------------------------------------------------------------------
auto legacy_file_writer::write_indices(
    std::string const &keyword, std::vector<std::vector<size_t>> const &indices)
    -> void {
  size_t total_number = 0;
  for (auto const &is : indices) {
    total_number += is.size() + 1;
  }
  vtk::write_binary(m_file, "\n" + keyword + " " +
                                std::to_string(indices.size()) + ' ' +
                                std::to_string(total_number) + '\n');
  for (auto const &p : indices) {
    auto size = static_cast<int>(p.size());
    size     = swap_endianess(size);
    m_file.write((char *)(&size), sizeof(int));
    for (int i : p) {
      i = swap_endianess(i);
      m_file.write((char *)(&i), sizeof(int));
    }
  }
}
//------------------------------------------------------------------------------
auto legacy_file_writer::write_cells(
    std::vector<std::vector<size_t>> const &cells) -> void {
  write_indices("CELLS", cells);
}
//------------------------------------------------------------------------------
auto legacy_file_writer::write_cell_types(
    std::vector<cell_type> const &cell_types) -> void {
  vtk::write_binary(m_file,
                    "\nCELL_TYPES " + std::to_string(cell_types.size()) + '\n');
  int ict = 0;
  for (auto ct : cell_types) {
    ict = swap_endianess(static_cast<int>(ct));
    m_file.write((char *)(&ict), sizeof(int));
  }
}
//------------------------------------------------------------------------------
auto legacy_file_writer::write_point_data(size_t i) -> void {
  vtk::write_binary(m_file, "\nPOINT_DATA " + std::to_string(i));
}
//-----------------------------------------------------------------------------
auto legacy_file_writer::write_cell_data(size_t i) -> void {
  vtk::write_binary(m_file, "\nCELL_DATA " + std::to_string(i));
}
//-----------------------------------------------------------------------------
auto legacy_file_writer::write_vertices(
    std::vector<std::vector<size_t>> const &vertices) -> void {
  write_indices("VERTICES", vertices);
}
//-----------------------------------------------------------------------------
auto legacy_file_writer::write_lines(
    std::vector<std::vector<size_t>> const &lines) -> void {
  write_indices("LINES", lines);
}
//-----------------------------------------------------------------------------
auto legacy_file_writer::write_polygons(
    std::vector<std::vector<size_t>> const &polygons) -> void {
  write_indices("POLYGONS", polygons);
}
//-----------------------------------------------------------------------------
auto legacy_file_writer::write_triangle_strips(
    std::vector<std::vector<size_t>> const &lines) -> void {
  write_indices("TRIANGLE_STRIPS", lines);
}
//=============================================================================
}  // namespace tatooine::vtk
//=============================================================================
