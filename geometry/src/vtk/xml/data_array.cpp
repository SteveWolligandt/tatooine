#include <tatooine/vtk/xml/data_array.h>
#include <tatooine/vtk/xml/reader.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
data_array::data_array(reader& r, rapidxml::xml_node<>* node)
    : m_type{parse_data_type(node->first_attribute("type")->value())},
      m_format{parse_format(node->first_attribute("format")->value())},
      m_reader{&r},
      m_node{node} {
  auto const name_attr = node->first_attribute("Name");
  if (name_attr != nullptr) {
    m_name = name_attr->value();
  }
  if (m_format == xml::format::appended) {
    m_offset = std::stoul(node->first_attribute("offset")->value());
  }
  if (auto* n = node->first_attribute("NumberOfComponents"); n != nullptr) {
    m_num_components = std::stoul(n->value());
  }
}
//==============================================================================
auto data_array::read_appended_data(char* data, std::size_t num_bytes) const
    -> void {
  m_reader->read_appended_data(data, num_bytes, m_offset);
}
//------------------------------------------------------------------------------
auto data_array::read_appended_data_size() const -> std::size_t {
  return m_reader->read_appended_data_size(m_offset);
}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
