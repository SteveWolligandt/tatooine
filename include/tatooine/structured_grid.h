#ifndef TATOOINE_STRUCTURED_GRID_H
#define TATOOINE_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/multidim_size.h>
#include <tatooine/pointset.h>
#include <rapidxml.hpp>
#include <rapidxml.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename IndexOrder = x_fastest>
struct structured_grid : pointset<Real, NumDimensions>,
                         dynamic_multidim_size<IndexOrder> {
  struct linear_cell_sampler;
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_t                 = structured_grid;
  using pointset_parent_t      = pointset<Real, NumDimensions>;
  using multidim_size_parent_t = dynamic_multidim_size<IndexOrder>;
  using typename pointset_parent_t::vertex_handle;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static auto constexpr num_dimensions() { return NumDimensions; }
  //============================================================================
  // CTORS
  //============================================================================
  structured_grid()                           = default;
  structured_grid(structured_grid const&)     = default;
  structured_grid(structured_grid&&) noexcept = default;
  auto operator=(structured_grid const&) -> structured_grid& = default;
  auto operator=(structured_grid&&) noexcept -> structured_grid& = default;
  //----------------------------------------------------------------------------
  structured_grid(filesystem::path const& path) { read(path); }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if_integral<Size...> = true>
#endif
  structured_grid(Size const... size) {
    static auto constexpr num_indices = sizeof...(Size);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    resize(size...);
  }
  //============================================================================
  // METHODS
  //============================================================================
#ifdef __cpp_concepts
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == NumDimensions)
#else
  template <typename... Ts, enable_if<is_arithmetic<Ts...> > = true,
            enable_if<sizeof...(Ts) == NumDimensions> = true>
#endif
      auto insert_vertex(Ts const... ts) = delete;
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto vertex_at(Is const... is) const -> auto const& {
    static auto constexpr num_indices = sizeof...(Is);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_t::vertex_at(multidim_size_parent_t::plain_index(is...));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto vertex_at(Is const... is) -> auto& {
    static auto constexpr num_indices = sizeof...(Is);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_t::vertex_at(multidim_size_parent_t::plain_index(is...));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if_integral<Size...> = true>
#endif
  auto resize(Size const... sizes) {
    static auto constexpr num_indices = sizeof...(Size);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    pointset_parent_t::resize((sizes * ...));
    multidim_size_parent_t::resize(sizes...);
  }
  //----------------------------------------------------------------------------
  auto read(filesystem::path const& path) -> void;
  auto read_vts(filesystem::path const& path) -> void;
};
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::read(
    filesystem::path const& path) -> void {
  if (path.extension() == ".vts") {
    read_vts(path);
  } else {
    throw std::runtime_error{"File extension \"" + path.extension().string() +
                             "\" not recognized by structured grid."};
  }
}
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::read_vts(
    filesystem::path const& path) -> void {
  using namespace rapidxml;
  auto file = std::ifstream{path};
  if (file.is_open()) {
    auto buffer = std::stringstream{};
    buffer << file.rdbuf();
    file.close();
    auto content =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>" +
        buffer.str();


    //extract appended data section if present
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
      end_appended_data = content.rfind("\n", end_appended_data);
    }
    std::vector<std::uint8_t> appended_data;
    if (begin_appended_data != std::string::npos) {
      appended_data.resize((end_appended_data - begin_appended_data) *
                           sizeof(std::string::value_type) /
                           sizeof(std::uint8_t));
      std::copy(
          begin(content) + begin_appended_data,
          begin(content) + end_appended_data,
          reinterpret_cast<std::string::value_type*>(appended_data.data()));
    }
    content.erase(begin_appended_data, end_appended_data - begin_appended_data);

    // start parsing
    auto doc = xml_document<>{}; 
    doc.parse<0>(content.data());
    auto root = doc.first_node();
    auto const name = root->name();
    if (std::strcmp(name, "VTKFile") != 0) {
      throw std::runtime_error{"File is not a VTK file."};
    }
    if (std::strcmp(root->first_attribute("type")->value(), "StructuredGrid") !=
        0) {
      throw std::runtime_error{"VTK file is not a structured grid."};
    }

    auto structured_grid_node = root->first_node("StructuredGrid");
    std::array<std::pair<size_t, size_t>, 3> whole_extents;
    {
      auto whole_extent_stream = std::stringstream{
          structured_grid_node->first_attribute("WholeExtent")->value()};
      whole_extent_stream >> whole_extents[0].first >> whole_extents[0].second
                          >> whole_extents[1].first >> whole_extents[1].second
                          >> whole_extents[2].first >> whole_extents[2].second;
    }
    std::array<size_t, 3> resolution{
        whole_extents[0].second - whole_extents[0].first + 1,
        whole_extents[1].second - whole_extents[1].first + 1,
        whole_extents[2].second - whole_extents[2].first + 1};
    resize(resolution[0], resolution[1], resolution[2]);
    for (auto piece_node                   = structured_grid_node->first_node();
         piece_node != nullptr; piece_node = piece_node->next_sibling()) {
      std::array<std::pair<size_t, size_t>, 3> extents;
      {
        auto extents_stream = std::stringstream{
            piece_node->first_attribute("Extent")->value()};
        extents_stream >> extents[0].first >> extents[0].second >>
            extents[1].first >> extents[1].second >> extents[2].first >>
            extents[2].second;
      }
      std::array<size_t, 3> piece_origins{
          extents[0].first - whole_extents[0].first,
          extents[1].first - whole_extents[1].first,
          extents[2].first - whole_extents[2].first};
      std::array<size_t, 3> piece_resolution{
          extents[0].second - extents[0].first,
          extents[1].second - extents[1].first,
          extents[2].second - extents[2].first};

      [[maybe_unused]] auto point_data_node =
          piece_node->first_node("PointData");
      [[maybe_unused]] auto cell_data_node = piece_node->first_node("CellData");
      auto                  points_node    = piece_node->first_node("Points");
      auto points_data_array_node = points_node->first_node("DataArray");

      auto points_offset =
          std::stoul(points_data_array_node->first_attribute("offset")->value());
      auto points_type =
          points_data_array_node->first_attribute("type")->value();
      auto points_data_number_of_components = std::stoul(
          points_data_array_node->first_attribute("NumberOfComponents")->value());

      std::vector<std::tuple<char const*, char const*, std::size_t>>
          appended_pointset_data;
      for (auto data_array_node = point_data_node->first_node();
           data_array_node != nullptr;
           data_array_node = data_array_node->next_sibling()) {
        auto  n         = data_array_node->name();
        auto* type_attr = data_array_node->first_attribute("type");
        auto* name_attr = data_array_node->first_attribute("Name");
        auto  name      = name_attr->value();
        auto  type      = type_attr->value();
        auto offset =
            std::stoul(data_array_node->first_attribute("offset")->value());
        appended_pointset_data.emplace_back(name, type, offset);
      }

      auto read_and_proceed = [&](auto const& type, auto& read, auto& write) {
        using write_type = std::decay_t<decltype(write)>;
        if (std::strcmp(type, "Int8") == 0) {
          using type = std::int8_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "UInt8") == 0) {
          using type = std::uint8_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "Int16") == 0) {
          using type = std::int16_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "UInt16") == 0) {
          using type = std::uint16_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "Int32") == 0) {
          using type = std::int32_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "UInt32") == 0) {
          using type = std::uint32_t;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "Float32") == 0) {
          using type = float;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        } else if (std::strcmp(type, "Float64") == 0) {
          using type = double;
          write      = static_cast<write_type>(*reinterpret_cast<type*>(read));
          read += sizeof(type);
        }
      };
      auto iterate_piece = [&](auto&& f) {
        for_loop(std::forward<decltype(f)>(f),
                 std::pair{piece_origins[0], piece_resolution[0]},
                 std::pair{piece_origins[1], piece_resolution[1]},
                 std::pair{piece_origins[2], piece_resolution[2]});
      };
      {
        auto extract_points = [&, data_ptr = &appended_data[points_offset]](
                                  auto const... is) mutable {
          auto& x = vertex_at(is...);
          for (size_t i = 0; i < num_dimensions(); ++i) {
            read_and_proceed(points_type, data_ptr, x(i));
          }
        };
        iterate_piece(extract_points);
      }

      for (auto& prop_data : appended_pointset_data) {
        auto& prop = this->scalar_vertex_property(std::get<0>(prop_data));
        iterate_piece([&, data_ptr = &appended_data[std::get<2>(prop_data)]](
                          auto const... is) mutable {
          auto& p =
              prop[vertex_handle{multidim_size_parent_t::plain_index(is...)}];
          read_and_proceed(std::get<1>(prop_data), data_ptr, p);
        });
      }
    }
  }
}
//==============================================================================
template <std::size_t NumDimensions>
using StructuredGrid   = structured_grid<real_t, NumDimensions>;
using structured_grid2 = StructuredGrid<2>;
using structured_grid3 = StructuredGrid<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
struct structured_grid<Real, NumDimensions, IndexOrder>::linear_cell_sampler {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
