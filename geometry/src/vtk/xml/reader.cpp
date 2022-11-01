#include <tatooine/vtk/xml/reader.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
reader::reader(filesystem::path const& path) : m_path{path}, m_file{m_path} {
  using namespace rapidxml;
  if (!m_file.is_open()) {
    return;
  }
  auto content = [&] {
    auto buffer = std::stringstream{};
    buffer << m_file.rdbuf();
    return buffer.str();
  }();
  extract_appended_data(content);
  // start parsing
  m_doc.parse<0>(content.data());
  read_meta();
}
//------------------------------------------------------------------------------
auto reader::extract_appended_data(std::string& content) -> void {
  static constexpr std::string_view opening_appended_data = "<AppendedData";
  static constexpr std::string_view closing_appended_data = "</AppendedData>";
  m_begin_appended_data = content.find(opening_appended_data);
  if (m_begin_appended_data != std::string::npos) {
    m_begin_appended_data = content.find('>', m_begin_appended_data);
    m_begin_appended_data = content.find('_', m_begin_appended_data) + 1;
  } else {
    return;
  }
  auto end_appended_data =
      m_begin_appended_data == std::string::npos
          ? std::string::npos
          : content.find(closing_appended_data, m_begin_appended_data);
  if (end_appended_data != std::string::npos) {
    end_appended_data = content.rfind('\n', end_appended_data);
  }
  content.erase(m_begin_appended_data, end_appended_data - m_begin_appended_data);
}
//==============================================================================
auto reader::read_meta() -> void {
  auto* root = m_doc.first_node();
  if (std::strcmp(root->name(), "VTKFile") != 0) {
    throw std::runtime_error{"File is not a VTK file."};
  }
  m_type       = parse_vtk_type(root->first_attribute("type")->value());
  m_version    = root->first_attribute("version")->value();
  m_byte_order = parse_byte_order(root->first_attribute("byte_order")->value());
  auto header_type_attr = root->first_attribute("header_type");
  if (header_type_attr != nullptr) {
    m_header_type = parse_data_type(header_type_attr->value());
  }
  auto n = root->first_node(std::string(to_string(m_type)).c_str());
  switch (m_type) {
    case vtk_type::image_data:
      m_image_data = xml::image_data{*this, n};
      break;
    case vtk_type::rectilinear_grid:
      m_rectilinear_grid = xml::rectilinear_grid{*this, n};
      break;
    case vtk_type::structured_grid:
      m_structured_grid = xml::structured_grid{*this, n};
      break;
    case vtk_type::poly_data:
      m_poly_data = xml::poly_data{*this, n};
      break;
    case vtk_type::unstructured_grid:
      m_unstructured_grid = xml::unstructured_grid{*this, n};
      break;
    case vtk_type::unknown:
    default:
      break;
  }
}
//==============================================================================
auto reader::read_appended_data_size(std::size_t offset) const -> std::size_t {
  m_file.seekg(m_begin_appended_data, m_file.beg);
  m_file.seekg(offset, m_file.cur);
  auto size = std::size_t{};
  visit(m_header_type, [&](unsigned_integral auto i) {
    m_file.read(reinterpret_cast<char*>(&i), sizeof(decltype(i)));
    size = static_cast<std::size_t>(i);
  });
  return size;
}
//==============================================================================
auto reader::read_appended_data(char* data, std::size_t num_bytes,
                                std::size_t offset) const -> void {
  m_file.seekg(m_begin_appended_data, m_file.beg);
  m_file.seekg(offset, m_file.cur);
  m_file.seekg(xml::size(m_header_type), m_file.cur);
  m_file.read(data, num_bytes);
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
