#ifndef TATOOINE_STRUCTURED_GRID_H
#define TATOOINE_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/multidim_size.h>
#include <tatooine/pointset.h>
#include <tatooine/uniform_tree_hierarchy.h>

#include <rapidxml.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
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
  using this_t        = structured_grid_cell_hierarchy;
  using real_t        = Real;
  using index_order_t = IndexOrder;
  using grid_t        = structured_grid<Real, NumDimensions, IndexOrder>;
  using parent_t = base_uniform_tree_hierarchy<Real, NumDimensions, this_t>;
  //============================================================================
  // INHERITED TYPES
  //============================================================================
  using typename parent_t::pos_t;
  using typename parent_t::vec_t;
  //============================================================================
  // INHERITED METHODS
  //============================================================================
  using parent_t::center;
  using parent_t::children;
  using parent_t::extents;
  using parent_t::is_at_max_depth;
  using parent_t::is_inside;
  using parent_t::is_simplex_inside;
  using parent_t::is_splitted;
  using parent_t::max;
  using parent_t::min;
  using parent_t::split_and_distribute;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static constexpr auto num_dimensions() { return NumDimensions; }
  //============================================================================
  // MEMBERS
  //============================================================================
  grid_t const*                                       m_grid = nullptr;
  std::vector<std::array<std::size_t, NumDimensions>> m_cell_handles;
  //============================================================================
  // CTORS
  //============================================================================
  structured_grid_cell_hierarchy() = default;
  structured_grid_cell_hierarchy(structured_grid_cell_hierarchy const&) =
      default;
  structured_grid_cell_hierarchy(structured_grid_cell_hierarchy&&) noexcept =
      default;
  auto operator=(structured_grid_cell_hierarchy const&)
      -> structured_grid_cell_hierarchy& = default;
  auto operator=(structured_grid_cell_hierarchy&&) noexcept
      -> structured_grid_cell_hierarchy&    = default;
  virtual ~structured_grid_cell_hierarchy() = default;

  explicit structured_grid_cell_hierarchy(grid_t const& grid) : m_grid{&grid} {}
  explicit structured_grid_cell_hierarchy(
      grid_t const& grid, size_t const max_depth = parent_t::default_max_depth)
      : parent_t{pos_t::zeros(), pos_t::zeros(), 1, max_depth}, m_grid{&grid} {
    parent_t::operator=(grid.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  structured_grid_cell_hierarchy(
      vec_t const& min, vec_t const& max, grid_t const& grid,
      size_t const max_depth = parent_t::default_max_depth)
      : parent_t{min, max, 1, max_depth}, m_grid{&grid} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  structured_grid_cell_hierarchy(vec_t const& min, vec_t const& max,
                                 size_t const level, size_t const max_depth,
                                 grid_t const& grid)
      : parent_t{min, max, level, max_depth}, m_grid{&grid} {}
  //============================================================================
  // METHODS
  //============================================================================
 public:
  auto grid() const -> auto const& { return *m_grid; }
  auto constexpr holds_cells() const { return !m_cell_handles.empty(); }
  //----------------------------------------------------------------------------
  auto num_cell_handles() const { return size(m_cell_handles); }
  //----------------------------------------------------------------------------
#ifndef __cpp_concepts
  template <typename... Indices, enable_if_integral<Indices...> = true>
#else
  template <integral... Indices>
#endif
  constexpr auto is_cell_inside(Indices const... is) const {
    if constexpr (NumDimensions == 2) {
      return is_cell_inside_2(is...);
    } else if constexpr (NumDimensions == 3) {
      return is_cell_inside_3(is...);
    }
  }
  //----------------------------------------------------------------------------
 private:
#ifndef __cpp_concepts
  template <size_t _NumDimensions            = NumDimensions,
            enable_if<(_NumDimensions == 2)> = true, typename... Indices,
            enable_if_integral<Indices...>   = true>
#else
  template <integral... Indices>
#endif
  constexpr auto is_cell_inside_2(Indices const... is) const
#ifdef __cpp_concepts
      requires(NumDimensions == 2)
#endif
  {
    return false;
  }
  //----------------------------------------------------------------------------
#ifndef __cpp_concepts
  template <size_t _NumDimensions            = NumDimensions,
            enable_if<(_NumDimensions == 3)> = true>
#endif
  constexpr auto is_cell_inside_3(std::size_t const ix, std::size_t const iy,
                                  std::size_t const iz) const
#ifdef __cpp_concepts
      requires(NumDimensions == 3)
#endif
  {
    auto const c = center();
    auto const e = extents() / 2;

    // vertices
    auto xs = std::array{grid().vertex_at(ix, iy, iz) - c,
                         grid().vertex_at(ix + 1, iy, iz) - c,
                         grid().vertex_at(ix, iy + 1, iz) - c,
                         grid().vertex_at(ix + 1, iy + 1, iz) - c,
                         grid().vertex_at(ix, iy, iz + 1) - c,
                         grid().vertex_at(ix + 1, iy, iz + 1) - c,
                         grid().vertex_at(ix, iy + 1, iz + 1) - c,
                         grid().vertex_at(ix + 1, iy + 1, iz + 1) - c};

    // edges
    auto const es =
        std::array{xs[1] - xs[0], xs[3] - xs[1], xs[2] - xs[3], xs[0] - xs[2],
                   xs[5] - xs[4], xs[7] - xs[5], xs[6] - xs[7], xs[4] - xs[6],
                   xs[4] - xs[0], xs[5] - xs[1], xs[6] - xs[2], xs[7] - xs[3]};
    // faces
    auto const fs = std::array{cross(es[0], es[1]),  cross(es[9], es[5]),
                               cross(es[4], -es[5]), cross(es[8], -es[7]),
                               cross(es[11], es[2]), cross(es[0], -es[9])};

    auto constexpr us =
        std::array{vec_t{1, 0, 0}, vec_t{0, 1, 0}, vec_t{0, 0, 1}};

    auto is_separating_axis = [&](auto const& axis) {
      auto const dots =
          std::array{dot(xs[0], axis), dot(xs[1], axis), dot(xs[2], axis),
                     dot(xs[3], axis), dot(xs[4], axis), dot(xs[5], axis),
                     dot(xs[6], axis), dot(xs[7], axis)};
      auto r = e.x() * std::abs(dot(us[0], axis)) +
               e.y() * std::abs(dot(us[1], axis)) +
               e.z() * std::abs(dot(us[2], axis));
      return tatooine::max(-tatooine::max(dots[0], dots[1], dots[2], dots[3],
                                          dots[4], dots[5], dots[6], dots[7]),
                           tatooine::min(dots[0], dots[1], dots[2], dots[3],
                                         dots[4], dots[5], dots[6], dots[7])) >
             r;
    };

    for (auto const& u : us) {
      if (is_separating_axis(u)) {
        return false;
      }
    }
    for (auto const& u : us) {
      for (auto const& e : es) {
        if (is_separating_axis(cross(u, e))) {
          return false;
        }
      }
    }
    for (auto const& f : fs) {
      if (is_separating_axis(f)) {
        return false;
      }
    }
    return true;
  }

  //------------------------------------------------------------------------------
  template <typename... Indices>
  auto insert_cell(Indices const... is) -> bool {
    if (!is_cell_inside(is...)) {
      return false;
    }
    if (holds_cells()) {
      if (is_at_max_depth()) {
        m_cell_handles.push_back(std::array{static_cast<std::size_t>(is)...});
      } else {
        split_and_distribute();
        distribute_cell(is...);
      }
    } else {
      if (is_splitted()) {
        distribute_cell(is...);
      } else {
        m_cell_handles.push_back(std::array{static_cast<std::size_t>(is)...});
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto distribute() {
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
#ifndef __cpp_concepts
  template <typename... Indices, enable_if_integral<Indices...> = true>
#else
  template <integral... Indices>
#endif
  auto distribute_cell(Indices const... is) {
    for (auto& child : children()) {
      child->insert_cell(is...);
    }
  }
  //============================================================================
  auto collect_nearby_cells(
      vec_t const&                                      pos,
      std::set<std::array<std::size_t, NumDimensions>>& cells) const -> void {
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
  auto nearby_cells(pos_t const& pos) const {
    std::set<std::array<std::size_t, NumDimensions>> cells;
    collect_nearby_cells(pos, cells);
    return cells;
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename IndexOrder = x_fastest>
struct structured_grid : pointset<Real, NumDimensions>,
                         dynamic_multidim_size<IndexOrder> {
  template <typename T>
  struct linear_cell_sampler;
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_t                 = structured_grid;
  using pointset_parent_t      = pointset<Real, NumDimensions>;
  using multidim_size_parent_t = dynamic_multidim_size<IndexOrder>;
  using hierarchy_t =
      structured_grid_cell_hierarchy<Real, NumDimensions, IndexOrder>;
  using typename pointset_parent_t::pos_t;
  using typename pointset_parent_t::vec_t;
  using typename pointset_parent_t::vertex_handle;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static auto constexpr num_dimensions() { return NumDimensions; }
  //============================================================================
  // MEMBERS
  //============================================================================
  mutable std::unique_ptr<hierarchy_t> m_hierarchy;
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
  auto hierarchy() const -> auto const& {return m_hierarchy;}
  //----------------------------------------------------------------------------
  auto update_hierarchy() const {
    if (m_hierarchy != nullptr) {
      m_hierarchy.reset();
    }
    auto const aabb = this->axis_aligned_bounding_box();
    m_hierarchy =
        std::make_unique<hierarchy_t>(aabb.min(), aabb.max(), *this, 4);
    auto       it = [&](auto const... is) { m_hierarchy->insert_cell(is...); };
    auto const s  = m_grid->size();
    if constexpr (NumDimensions == 2) {
      for_loop(it, s[0] - 1, s[1] - 1);
    } else if constexpr (NumDimensions == 3) {
      for_loop(it, s[0] - 1, s[1] - 1, s[2] - 1);
    }
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <arithmetic... Ts>
  requires(sizeof...(Ts) == NumDimensions)
#else
  template <typename... Ts, enable_if<is_arithmetic<Ts...>> = true,
            enable_if<sizeof...(Ts) == NumDimensions> = true>
#endif
      auto insert_vertex(Ts const... ts) = delete;
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if_integral<Indices...> = true>
#endif
  auto vertex_at(Indices const... is) const -> auto const& {
    static auto constexpr num_indices = sizeof...(Indices);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_t::vertex_at(
        multidim_size_parent_t::plain_index(is...));
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Indices>
#else
  template <typename... Indices, enable_if_integral<Indices...> = true>
#endif
  auto vertex_at(Indices const... is) -> auto& {
    static auto constexpr num_indices = sizeof...(Indices);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_t::vertex_at(
        multidim_size_parent_t::plain_index(is...));
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
  //----------------------------------------------------------------------------
  template <typename... Indices, enable_if_integral<Indices...> = true>
  auto local_cell_coordinates(pos_t const x, Indices const... is) const
      -> pos_t;
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(std::string const& name) const {
    if (m_hierarchy == nullptr) {
      update_hierarchy();
    }
    return linear_vertex_property_sampler<T>{this, &vertex_property<T>(name)};
  }
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

    // extract appended data section if present
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
    auto       root = doc.first_node();
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
      whole_extent_stream >> whole_extents[0].first >>
          whole_extents[0].second >> whole_extents[1].first >>
          whole_extents[1].second >> whole_extents[2].first >>
          whole_extents[2].second;
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
        auto extents_stream =
            std::stringstream{piece_node->first_attribute("Extent")->value()};
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

      auto points_offset = std::stoul(
          points_data_array_node->first_attribute("offset")->value());
      auto points_type =
          points_data_array_node->first_attribute("type")->value();
      auto points_data_number_of_components = std::stoul(
          points_data_array_node->first_attribute("NumberOfComponents")
              ->value());

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
        auto  offset =
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
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
template <typename... Indices, enable_if_integral<Indices...>>
auto structured_grid<Real, NumDimensions, IndexOrder>::local_cell_coordinates(
    pos_t const x, Indices const... is) const -> pos_t {
  static auto constexpr num_indices = sizeof...(Indices);
  static_assert(num_indices == num_dimensions(),
                "Number of Indices does not match number of dimensions");
  auto const cell_indices = std::array{static_cast<std::size_t>(is)...};
  if constexpr (NumDimensions == 2) {
    auto const& A         = vertex_at(cell_indices[0], cell_indices[1]);
    auto const& B         = vertex_at(cell_indices[0] + 1, cell_indices[1]);
    auto const& C         = vertex_at(cell_indices[0], cell_indices[1] + 1);
    auto const& D         = vertex_at(cell_indices[0] + 1, cell_indices[1] + 1);
    auto const  sqrt_disc = std::sqrt(
        (D.x() * D.x() + (-2 * C.x() - 2 * B.x() + 2 * A.x()) * D.x() +
         C.x() * C.x() + (2 * B.x() - 2 * A.x()) * C.x() + B.x() * B.x() -
         2 * A.x() * B.x() + A.x() * A.x()) *
            x.y() * x.y() +
        (((-2 * D.x() + 2 * C.x() + 2 * B.x() - 2 * A.x()) * x.x() +
          2 * A.x() * D.x() + (2 * A.x() - 4 * B.x()) * C.x() +
          2 * A.x() * B.x() - 2 * A.x() * A.x()) *
             D.y() +
         ((2 * D.x() - 2 * C.x() - 2 * B.x() + 2 * A.x()) * x.x() +
          (2 * B.x() - 4 * A.x()) * D.x() + 2 * B.x() * C.x() -
          2 * B.x() * B.x() + 2 * A.x() * B.x()) *
             C.y() +
         ((2 * D.x() - 2 * C.x() - 2 * B.x() + 2 * A.x()) * x.x() +
          (2 * C.x() - 4 * A.x()) * D.x() - 2 * C.x() * C.x() +
          (2 * B.x() + 2 * A.x()) * C.x()) *
             B.y() +
         ((-2 * D.x() + 2 * C.x() + 2 * B.x() - 2 * A.x()) * x.x() -
          2 * D.x() * D.x() + (2 * C.x() + 2 * B.x() + 2 * A.x()) * D.x() -
          4 * B.x() * C.x()) *
             A.y()) *
            x.y() +
        (x.x() * x.x() - 2 * A.x() * x.x() + A.x() * A.x()) * D.y() * D.y() +
        ((-2 * x.x() * x.x() + (2 * B.x() + 2 * A.x()) * x.x() -
          2 * A.x() * B.x()) *
             C.y() +
         (-2 * x.x() * x.x() + (2 * C.x() + 2 * A.x()) * x.x() -
          2 * A.x() * C.x()) *
             B.y() +
         (2 * x.x() * x.x() +
          (2 * D.x() - 4 * C.x() - 4 * B.x() + 2 * A.x()) * x.x() -
          2 * A.x() * D.x() + 4 * B.x() * C.x()) *
             A.y()) *
            D.y() +
        (x.x() * x.x() - 2 * B.x() * x.x() + B.x() * B.x()) * C.y() * C.y() +
        ((2 * x.x() * x.x() +
          (-4 * D.x() + 2 * C.x() + 2 * B.x() - 4 * A.x()) * x.x() +
          4 * A.x() * D.x() - 2 * B.x() * C.x()) *
             B.y() +
         (-2 * x.x() * x.x() + (2 * D.x() + 2 * B.x()) * x.x() -
          2 * B.x() * D.x()) *
             A.y()) *
            C.y() +
        (x.x() * x.x() - 2 * C.x() * x.x() + C.x() * C.x()) * B.y() * B.y() +
        (-2 * x.x() * x.x() + (2 * D.x() + 2 * C.x()) * x.x() -
         2 * C.x() * D.x()) *
            A.y() * B.y() +
        (x.x() * x.x() - 2 * D.x() * x.x() + D.x() * D.x()) * A.y() * A.y());
    auto sol1 = vec{
        (sqrt_disc + (-D.x() + C.x() + B.x() - A.x()) * x.y() +
         (x.x() - A.x()) * D.y() + (-x.x() - B.x() + 2 * A.x()) * C.y() +
         (C.x() - x.x()) * B.y() + (x.x() + D.x() - 2 * C.x()) * A.y()) /
            ((2 * B.x() - 2 * A.x()) * D.y() + (2 * A.x() - 2 * B.x()) * C.y() +
             (2 * C.x() - 2 * D.x()) * B.y() + (2 * D.x() - 2 * C.x()) * A.y()),
        -(sqrt_disc + (D.x() - C.x() - B.x() + A.x()) * x.y() +
          (A.x() - x.x()) * D.y() + (x.x() - B.x()) * C.y() +
          (x.x() + C.x() - 2 * A.x()) * B.y() +
          (-x.x() - D.x() + 2 * B.x()) * A.y()) /
            ((2 * C.x() - 2 * A.x()) * D.y() + (2 * B.x() - 2 * D.x()) * C.y() +
             (2 * A.x() - 2 * C.x()) * B.y() +
             (2 * D.x() - 2 * B.x()) * A.y())};
    return sol1;
  } else if constexpr (NumDimensions == 3) {
  }
}
//==============================================================================
template <std::size_t NumDimensions>
using StructuredGrid   = structured_grid<real_t, NumDimensions>;
using structured_grid2 = StructuredGrid<2>;
using structured_grid3 = StructuredGrid<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
template <typename T>
struct structured_grid<Real, NumDimensions, IndexOrder>::linear_cell_sampler
    : field<structured_grid<Real, NumDimensions,
                            IndexOrder>::linear_cell_sampler<T>,
            Real, NumDimensions, T> {
  using this_t     = linear_cell_sampler;
  using parent_t   = field<this_t, Real, NumDimensions, T>;
  using grid_t     = structured_grid<Real, NumDimensions, IndexOrder>;
  using property_t = typename grid_t::template vertex_property<T>;
  using vec_t      = typename grid_t::vec_t;
  using pos_t      = typename grid_t::pos_t;
  using typename parent_t::tensor_t;

  grid_t const*     m_grid;
  property_t const* m_property;

  auto evaluate(pos_t const& x, real_t const /*t*/) const -> tensor_t {
    auto possible_cells = m_grid->m_hierarchy()->nearby_cells(x);
    auto cell_it = end(possible_cells);

    for (auto it = begin(possible_cells); it!=end(possible_cells); ++it) {
      auto const local_pos = [&] {
        if constexpr (NumDimensions == 2) {
          local_pos = m_grid->local_cell_coordinates(x, it->at(0), it->at(1));
        }
        else if constexpr (NumDimensions == 3) {
          local_pos = m_grid->local_cell_coordinates(x, it->at(0), it->at(1),
                                                     it->at(2));
        }
      }();
      bool all_components_between_zero_and_one = true;
      for (size_t i = 0; i < NumDimensions; ++i) {
        if (local_pos(i) < 0 || local_pos(i) > 1) {
          all_components_between_zero_and_one = false;
          break;
        }
      }
      if (all_components_between_zero_and_one) {
        cell_it = it;
        break;
      }
    }

    if (cell_it == end(possible_cells)) {
      return tensor_t{0.0 / 0.0};
    }
    return [&](){
        if constexpr (NumDimensions == 2) {
          return ;
        }
        else if constexpr (NumDimensions == 3) {
          return ;
        }
    }();

  }
};
//==============================================================================
}  // namespace tatooine
#endif
