#ifndef TATOOINE_VTK_XML_READER_H
#define TATOOINE_VTK_XML_READER_H
//==============================================================================
#include <tatooine/filesystem.h>
#include <tatooine/parse.h>
#include <tatooine/vtk/xml/byte_order.h>
#include <tatooine/vtk/xml/image_data.h>
#include <tatooine/vtk/xml/poly_data.h>
#include <tatooine/vtk/xml/rectilinear_grid.h>
#include <tatooine/vtk/xml/structured_grid.h>
#include <tatooine/vtk/xml/unstructured_grid.h>
#include <tatooine/vtk/xml/vtk_type.h>

#include <array>
#include <cstring>
#include <fstream>
#include <optional>
#include <rapidxml.hpp>
#include <sstream>
#include <string>
#include <vector>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
struct reader {
  //==============================================================================
  // MEMBERS
  //==============================================================================
 private:
  vtk_type                 m_type        = vtk_type::unknown;
  xml::byte_order          m_byte_order  = byte_order::unknown;
  xml::data_type           m_header_type = data_type::unknown;
  std::string              m_version     = "";
  filesystem::path         m_path        = {};
  mutable std::ifstream    m_file;
  rapidxml::xml_document<> m_doc;
  std::size_t              m_begin_appended_data = {};

  std::optional<xml::image_data>        m_image_data;
  std::optional<xml::rectilinear_grid>  m_rectilinear_grid;
  std::optional<xml::structured_grid>   m_structured_grid;
  std::optional<xml::poly_data>         m_poly_data;
  std::optional<xml::unstructured_grid> m_unstructured_grid;

 public:
  auto type() const { return m_type; }
  auto byte_order() const { return m_byte_order; }
  auto version() const -> auto const& { return m_version; }
  auto image_data() const -> auto const& { return m_image_data; }
  auto rectilinear_grid() const -> auto const& { return m_rectilinear_grid; }
  auto structured_grid() const -> auto const& { return m_structured_grid; }
  auto poly_data() const -> auto const& { return m_poly_data; }
  auto unstructured_grid() const -> auto const& { return m_unstructured_grid; }
  auto file() const -> auto const& { return m_file; }
  auto file() -> auto& { return m_file; }
  auto xml_document() const -> auto const& { return m_doc; }
  auto xml_document() -> auto& { return m_doc; }

  /// Returns number of bytes of the following data block.
  auto read_appended_data_size(std::size_t offset) const -> std::size_t;
  /// Reads the data block that follows the size of the actual data block.
  /// num_bytes needs to be retrieved from read_appended_data_size method.
  auto read_appended_data(char* data, std::size_t num_bytes,
                          std::size_t offset) const -> void;

  //==============================================================================
  // CTORS
  //==============================================================================
  reader(filesystem::path const& path);
  ~reader() { m_file.close(); }
  //==============================================================================
  // STATIC METHODS
  //==============================================================================
  /// The binary appended data part needs to be extracted in order to make
  /// rapidxml read the file.
  auto extract_appended_data(std::string& content) -> void;
  //==============================================================================
  // METHODS
  //==============================================================================
  auto read_meta() -> void;
};
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
#endif
