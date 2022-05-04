#include <tatooine/vtk/xml.h>
//==============================================================================
namespace tatooine::vtk::xml {
//==============================================================================
auto reader::extract_appended_data(std::string& content)
    -> std::vector<std::uint8_t> {
  static constexpr std::string_view opening_appended_data = "<AppendedData";
  static constexpr std::string_view closing_appended_data = "</AppendedData>";
  auto begin_appended_data = content.find(opening_appended_data);
  if (begin_appended_data != std::string::npos) {
    begin_appended_data = content.find('\n', begin_appended_data);
    begin_appended_data = content.find('_', begin_appended_data);
    ++begin_appended_data;
  }
  auto end_appended_data =
      begin_appended_data == std::string::npos
          ? std::string::npos
          : content.find(closing_appended_data, begin_appended_data);
  if (end_appended_data != std::string::npos) {
    end_appended_data = content.rfind('\n', end_appended_data);
  }
  std::vector<std::uint8_t> appended_data;
  if (begin_appended_data != std::string::npos) {
    appended_data.resize((end_appended_data - begin_appended_data) *
                         sizeof(std::string::value_type) /
                         sizeof(std::uint8_t));
    std::copy(next(begin(content), begin_appended_data),
              next(begin(content), end_appended_data),
              reinterpret_cast<std::string::value_type*>(appended_data.data()));
  }
  content.erase(begin_appended_data, end_appended_data - begin_appended_data);
  return appended_data;
}
//==============================================================================
auto reader::extract_extents(rapidxml::xml_attribute<> const* attr)
    -> std::array<std::pair<std::size_t, std::size_t>, 3> {
  std::array<std::pair<std::size_t, std::size_t>, 3> extent;
  auto extent_stream = std::stringstream{attr->value()};
  extent_stream >> extent[0].first >> extent[0].second >> extent[1].first >>
      extent[1].second >> extent[2].first >> extent[2].second;
  return extent;
}
//==============================================================================
auto reader::read() -> void {
  using namespace rapidxml;
  auto file = std::ifstream{m_path};
  if (file.is_open()) {
    auto content = [&] {
      auto buffer = std::stringstream{};
      buffer
          << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      buffer << file.rdbuf();
      return buffer.str();
    }();
    file.close();

    m_appended_data = extract_appended_data(content);

    // start parsing
    auto doc = xml_document<>{};
    doc.parse<0>(content.data());
    auto*       root = doc.first_node();
    auto* const name = root->name();
    if (std::strcmp(name, "VTKFile") != 0) {
      throw std::runtime_error{"File is not a VTK file."};
    }
    auto* const type = root->first_attribute("type")->value();
    for (auto l : m_listeners) {
      l->on_vtk_type(type);
    }
    auto* const version = root->first_attribute("version")->value();
    for (auto l : m_listeners) {
      l->on_vtk_version(version);
    }
    auto const byte_order =
        to_byte_order(root->first_attribute("byte_order")->value());
    for (auto l : m_listeners) {
      l->on_vtk_byte_order(byte_order);
    }
    if (std::strcmp(type, "StructuredGrid") == 0) {
      read_structured_grid(root->first_node(type));
    }
  }
}
//==============================================================================
auto reader::read_points(rapidxml::xml_node<>* node) -> void {
  auto* data_array_node = node->first_node("DataArray");
  auto  meta            = data_array{data_array_node};
  if (meta.format() == data_array::format_t::appended) {
    auto* data_begin = &m_appended_data[meta.offset()];
    read_appended_data(data_begin, meta, [&](auto* const data) {
      for (auto l : m_listeners) {
        l->on_points(data);
      }
    });
  }
}
//==============================================================================
auto reader::read_structured_grid(
    rapidxml::xml_node<> const* structured_grid_node) -> void {
  auto whole_extent =
      extract_extents(structured_grid_node->first_attribute("WholeExtent"));
  for (auto l : m_listeners) {
    l->on_structured_grid(whole_extent);
  }

  for (auto* piece_node                  = structured_grid_node->first_node();
       piece_node != nullptr; piece_node = piece_node->next_sibling()) {
    auto extent = extract_extents(piece_node->first_attribute("Extent"));
    for (auto l : m_listeners) {
      l->on_structured_grid_piece(extent);
    }

    read_points(piece_node->first_node("Points"));
    read_point_data(piece_node->first_node("PointData"));
    read_cell_data(piece_node->first_node("CellData"));
  }
}
//==============================================================================
auto reader::read_point_data(rapidxml::xml_node<>* node) -> void {
  std::vector<std::tuple<char const*, char const*, std::size_t>>
                                                      appended_pointset_data;
  std::vector<std::pair<data_attribute, char const*>> attributes;
  for (auto* attr_attr = node->first_attribute(); attr_attr != nullptr;
       attr_attr       = attr_attr->next_attribute()) {
    attributes.emplace_back(to_data_attribute(attr_attr->name()),
                            attr_attr->value());
  }
  for (auto* data_array_node = node->first_node(); data_array_node != nullptr;
       data_array_node       = data_array_node->next_sibling()) {
    auto meta = data_array{data_array_node};
    if (meta.format() == data_array::format_t::appended) {
      auto* data_begin = &m_appended_data[meta.offset()];
      read_appended_data(data_begin, meta, [&](auto* const data) {
        for (auto l : m_listeners) {
          l->on_point_data(meta.name(), data);
        }
      });
    }
  }
}
//==============================================================================
auto reader::read_cell_data(rapidxml::xml_node<>* /*node*/) -> void {}
//==============================================================================
}  // namespace tatooine::vtk::xml
//==============================================================================
