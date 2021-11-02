which: no null in (./^/dev)
Autojump is not installed. Please install it first from https://github.com/wting/autojump#installation
#ifndef TATOOINE_STRUCTURED_GRID_H
#define TATOOINE_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/multidim_size.h>
#include <tatooine/pointset.h>
#include <rapidxml.hpp>
#include <tatooine/uniform_tree_hierarchy.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename IndexOrder>
struct structured_grid;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
struct structured_grid_cell_hierarchy
    : base_uniform_tree_hierarchy<
          Real, NumDimensions,
          structured_grid_cell_hierarchy<Real, NumDimensions, IndexOrder>> {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_t   = structured_grid_cell_hierarchy;
  using real_t        = Real;
  using index_order_t = IndexOrder;
  using grid_t =
      structured_grid_cell_hierarchy<Real, NumDimensions, IndexOrder>;
  using parent_t = base_uniform_tree_hierarchy<Real, NumDimensions, this_t>;
  //============================================================================
  // INHERITED METHODS
  //============================================================================
  using parent_t::center;
  using parent_t::is_inside;
  using parent_t::is_simplex_inside;
  using parent_t::max;
  using parent_t::min;
  using parent_t::is_at_max_depth;
  using parent_t::is_splitted;
  using parent_t::split_and_distribute;
  using parent_t::children;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
  // MEMBERS
  //============================================================================
  grid_t const*                                       m_grid = nullptr;
  std::vector<std::array<std::size_t, NumDimensions>> m_vertex_handles;
  std::vector<std::array<std::size_t, NumDimensions>> m_cell_handles;
  //============================================================================
  // CTORS
  //============================================================================
  structured_grid_cell_hierarchy(grid_t const& grid) : m_grid{&grid} {}
  structured_grid_cell_hierarchy()                                  = default;
  structured_grid_cell_hierarchy(structured_grid_cell_hierarchy const&)     = default;
  structured_grid_cell_hierarchy(structured_grid_cell_hierarchy&&) noexcept = default;
  auto operator=(structured_grid_cell_hierarchy const&)
      -> structured_grid_cell_hierarchy&                            = default;
  auto operator=(structured_grid_cell_hierarchy&&) noexcept
      -> structured_grid_cell_hierarchy&                            = default;
  virtual ~structured_grid_cell_hierarchy()                         = default;
  explicit structured_grid_cell_hierarchy(
      grid_t const& grid, size_t const max_depth = parent_t::default_max_depth)
      : parent_t{vec<Real, NumDims>::zeros(), vec<Real, NumDims>::zeros(), 1,
                 max_depth},
        m_grid{&grid} {
    parent_t::operator=(grid.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  structured_grid_cell_hierarchy(vec_t const& min, vec_t const& max, grid_t const& grid,
                         size_t const max_depth = parent_t::default_max_depth)
      : parent_t{min, max, 1, max_depth}, m_grid{&grid} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  structured_grid_cell_hierarchy(vec_t const& min, vec_t const& max, size_t const level,
                         size_t const max_depth, grid_t const& grid)
      : parent_t{min, max, level, max_depth}, m_grid{&grid} {}
  //============================================================================
  // METHODS
  //============================================================================
 public:
  auto grid() const -> auto const& { return *m_grid; }
  auto constexpr holds_vertices() const { return !m_vertex_handles.empty(); }
  auto constexpr holds_cells() const { return !m_cell_handles.empty(); }
  //----------------------------------------------------------------------------
  auto num_vertex_handles() const { return size(m_vertex_handles); }
  auto num_cell_handles() const { return size(m_cell_handles); }
  //----------------------------------------------------------------------------
  auto insert_vertex(vertex_handle const v) -> bool {
    if (!is_inside(grid().vertex_at(v))) {
      return false;
    }
    if (holds_vertices()) {
      if (is_at_max_depth()) {
        m_vertex_handles.push_back(v);
      } else {
        split_and_distribute();
        distribute_vertex(v);
      }
    } else {
      if (is_splitted()) {
        distribute_vertex(v);
      } else {
        m_vertex_handles.push_back(v);
      }
    }
    return true;
  }
  //------------------------------------------------------------------------------
 private:
  template <size_t... Is>
  auto insert_cell(cell_handle const c, std::index_sequence<Is...> /*seq*/)
      -> bool {
    auto const vs = grid()[c];
    if (!is_simplex_inside(grid()[std::get<Is>(vs)]...)) {
      return false;
    }
    if (holds_cells()) {
      if (is_at_max_depth()) {
        m_cell_handles.push_back(c);
      } else {
        split_and_distribute();
        distribute_cell(c);
      }
    } else {
      if (is_splitted()) {
        distribute_cell(c);
      } else {
        m_cell_handles.push_back(c);
      }
    }
    return true;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto insert_cell(cell_handle const c) -> bool {
    return insert_cell(
        c, std::make_index_sequence<grid_t::num_vertices_per_simplex()>{});
  }
  //----------------------------------------------------------------------------
  auto distribute() {
    if (!m_vertex_handles.empty()) {
      distribute_vertex(m_vertex_handles.front());
      m_vertex_handles.clear();
    }
    if (!m_cell_handles.empty()) {
      distribute_cell(m_cell_handles.front());
      m_cell_handles.clear();
    }
  }
  //------------------------------------------------------------------------------
  auto construct(vec_t const& min, vec_t const& max, size_t const level,
                 size_t const max_depth) const {
    return std::unique_ptr<this_t>{
        new this_t{min, max, level, max_depth, grid()}};
  }
  //----------------------------------------------------------------------------
  auto distribute_vertex(vertex_handle const v) {
    for (auto& child : children()) {
      child->insert_vertex(v);
    }
  }
  //----------------------------------------------------------------------------
  auto distribute_cell(cell_handle const c) {
    for (auto& child : children()) {
      child->insert_cell(c);
    }
  }
  //============================================================================
  auto collect_possible_intersections(
      ray<Real, NumDims> const& r,
      std::set<cell_handle>&               possible_collisions) const -> void {
    if (parent_t::check_intersection(r)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_possible_intersections(r, possible_collisions);
        }
      } else {
        std::copy(begin(m_cell_handles), end(m_cell_handles),
                  std::inserter(possible_collisions, end(possible_collisions)));
      }
    }
  }
  //----------------------------------------------------------------------------
  auto collect_possible_intersections(ray<Real, NumDims> const& r) const {
    std::set<cell_handle> possible_collisions;
    collect_possible_intersections(r, possible_collisions);
    return possible_collisions;
  }
  //----------------------------------------------------------------------------
  auto collect_nearby_cells(vec<Real, NumDims> const& pos,
                            std::set<cell_handle>& cells) const -> void {
    if (is_inside(pos)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_nearby_cells(pos, cells);
        }
      } else {
        if (!m_cell_handles.empty()) {
          std::copy(begin(m_cell_handles), end(m_cell_handles),
                    std::inserter(cells, end(cells)));
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_cells(vec<Real, NumDims> const& pos) const {
    std::set<cell_handle> cells;
    collect_nearby_cells(pos, cells);
    return cells;
  }
};
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
