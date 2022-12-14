#ifndef TATOOINE_STRUCTURED_GRID_H
#define TATOOINE_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/dynamic_multidim_size.h>
#include <tatooine/pointset.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/vtk/xml.h>
// #include <tatooine/detail/structured_grid/vertex_container.h>
#include <tatooine/detail/structured_grid/vts_writer.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions,
          typename IndexOrder = x_fastest>
struct structured_grid : pointset<Real, NumDimensions>,
                         dynamic_multidim_size<IndexOrder> {
  //============================================================================
  // INTERNAL TYPES
  //============================================================================
  template <typename T>
  struct linear_cell_sampler_type;
  struct hierarchy_type;
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_type              = structured_grid;
  using pointset_parent_type   = pointset<Real, NumDimensions>;
  using multidim_size_parent_t = dynamic_multidim_size<IndexOrder>;
  using typename pointset_parent_type::vertex_property_type;
  template <typename T>
  using typed_vertex_property_type =
      pointset_parent_type::template typed_vertex_property_type<T>;
  using typename pointset_parent_type::pos_type;
  using typename pointset_parent_type::vec_type;
  using typename pointset_parent_type::vertex_handle;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static auto constexpr num_dimensions() { return NumDimensions; }
  //============================================================================
  // MEMBERS
  //============================================================================
  mutable std::unique_ptr<hierarchy_type> m_hierarchy;
  //============================================================================
  // CTORS
  //============================================================================
  structured_grid()                                              = default;
  structured_grid(structured_grid const&)                        = default;
  structured_grid(structured_grid&&) noexcept                    = default;
  auto operator=(structured_grid const&) -> structured_grid&     = default;
  auto operator=(structured_grid&&) noexcept -> structured_grid& = default;
  //----------------------------------------------------------------------------
  structured_grid(filesystem::path const& path) { read(path); }
  //----------------------------------------------------------------------------
  structured_grid(integral auto const... size) {
    static auto constexpr num_indices = sizeof...(size);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    resize(size...);
  }
  //============================================================================
  // METHODS
  //============================================================================
  auto hierarchy() const -> auto const& { return m_hierarchy; }
  //----------------------------------------------------------------------------
  auto update_hierarchy() const {
    if (m_hierarchy != nullptr) {
      m_hierarchy.reset();
    }
    auto const aabb = this->axis_aligned_bounding_box();
    m_hierarchy =
        std::make_unique<hierarchy_type>(aabb.min(), aabb.max(), *this, 4);
    auto       it = [&](auto const... is) { m_hierarchy->insert_cell(is...); };
    auto const s  = this->size();
    if constexpr (NumDimensions == 2) {
      for_loop(it, s[0] - 1, s[1] - 1);
    } else if constexpr (NumDimensions == 3) {
      for_loop(it, s[0] - 1, s[1] - 1, s[2] - 1);
    }
  }
  //----------------------------------------------------------------------------
  auto insert_vertex(arithmetic auto const... ts) = delete;
  //============================================================================
  auto vertex_at(integral auto const... is) const -> auto const& {
    static auto constexpr num_indices = sizeof...(is);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_type::vertex_at(
        multidim_size_parent_t::plain_index(is...));
  }
  //----------------------------------------------------------------------------
  auto vertex_at(integral auto const... is) -> auto& {
    static auto constexpr num_indices = sizeof...(is);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return pointset_parent_type::vertex_at(
        multidim_size_parent_t::plain_index(is...));
  }
  //----------------------------------------------------------------------------
  auto resize(integral auto const... sizes) {
    static auto constexpr num_indices = sizeof...(sizes);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    this->vertices().resize((sizes * ...));
    multidim_size_parent_t::resize(sizes...);
  }
  //----------------------------------------------------------------------------
  auto write(filesystem::path const& path) -> void;
  template <typename HeaderType = std::uint64_t>
  auto write_vts(filesystem::path const& path) -> void;
  //----------------------------------------------------------------------------
  auto read(filesystem::path const& path) -> void;
  auto read_vts(filesystem::path const& path) -> void;
  //----------------------------------------------------------------------------
  auto local_cell_coordinates(pos_type const x, integral auto const... is) const
      -> pos_type {
    static auto constexpr num_indices = sizeof...(is);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return local_cell_coordinates(x,
                                  std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  auto local_cell_coordinates(
      pos_type const                                x,
      std::array<std::size_t, NumDimensions> const& cell) const -> pos_type;
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(
      typed_vertex_property_type<T> const& prop) const {
    if (m_hierarchy == nullptr) {
      update_hierarchy();
    }
    return linear_cell_sampler_type<T>{*this, prop};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(std::string const& name) const {
    return linear_vertex_property_sampler(
        this->template vertex_property<T>(name));
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::write(
    filesystem::path const& path) -> void {
  if (path.extension() == ".vts") {
    write_vts(path);
  } else {
    throw std::runtime_error{"File extension \"" + path.extension().string() +
                             "\" not recognized by structured grid."};
  }
}
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
template <typename HeaderType>
auto structured_grid<Real, NumDimensions, IndexOrder>::write_vts(
    filesystem::path const& path) -> void {
  detail::structured_grid::vts_writer<this_type, HeaderType>{*this}.write(path);
}
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
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::read_vts(
    filesystem::path const& path) -> void {
  // TODO write binary data arrays with number of bytes at the beginning of each
  // array
  auto        reader = vtk::xml::reader{path};
  auto const& g      = *reader.structured_grid();

  resize(g.whole_extent2[0] - g.whole_extent1[0] + 1,
         g.whole_extent2[1] - g.whole_extent1[1] + 1,
         g.whole_extent2[2] - g.whole_extent1[2] + 1);
  for (auto const& p : g.pieces()) {
    auto const cur_piece_origin =
        std::array{p.extent1->at(0) - g.whole_extent1.at(0),
                   p.extent1->at(1) - g.whole_extent1.at(1),
                   p.extent1->at(2) - g.whole_extent1.at(2)};
    auto const cur_piece_resolution = std::array{
        p.extent2[0] - p.extent1[0] + 1, p.extent2[1] - p.extent1[1] + 1,
        p.extent2[2] - p.extent1[2] + 1};

    p.points.visit_data([&](auto const& point_data) {
      // always 3 components in vtk data array
      for (std::size_t i = 0; i < point_data.size(); i += 3) {
        if constexpr (num_dimensions() == 2) {
          // just omit third component when reading to a 3d line
          vertex_at(i / 3) = {point_data[i], point_data[i + 1]};
        } else if constexpr (num_dimensions() == 3) {
          vertex_at(i / 3) = {point_data[i], point_data[i + 1],
                              point_data[i + 2]};
        }
      }
    });
  }
  // auto on_point_data(std::string const& name, float const* v)->void override
  // {
  // auto& prop = template vertex_property<float>(name);
  // for_loop(
  //    [&](auto const... is) mutable {
  //      auto& p = prop[vertex_handle{grid.plain_index(is...)}];
  //      p       = *v++;
  //    },
  //    std::pair{cur_piece_origin[0], cur_piece_resolution[0]},
  //    std::pair{cur_piece_origin[1], cur_piece_resolution[1]},
  //    std::pair{cur_piece_origin[2], cur_piece_resolution[2]});
  //}
}
//------------------------------------------------------------------------------

template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::local_cell_coordinates(
    pos_type const                                x,
    std::array<std::size_t, NumDimensions> const& cell_indices) const
    -> pos_type {
  auto              bary = pos_type::fill(Real(0.5));  // initial
  auto              dx   = pos_type::fill(Real(0.1));
  auto              i    = std::size_t(0);
  auto const        tol  = Real(1e-12);
  auto              Dff  = mat<Real, NumDimensions, NumDimensions>{};
  static auto const max_num_iterations = std::size_t(20);
  if constexpr (NumDimensions == 2) {
    auto const&       v0 = vertex_at(cell_indices[0], cell_indices[1]);
    auto const&       v1 = vertex_at(cell_indices[0] + 1, cell_indices[1]);
    auto const&       v2 = vertex_at(cell_indices[0], cell_indices[1] + 1);
    auto const&       v3 = vertex_at(cell_indices[0] + 1, cell_indices[1] + 1);
    auto              ff = vec_type{};
    static auto const max_num_iterations = std::size_t(20);
    for (; i < max_num_iterations && squared_euclidean_length(dx) > tol; ++i) {
      // apply Newton-Raphson method to solve f(x,y)=0
      ff = (1 - bary.x()) * (1 - bary.y()) * v0 +
           bary.x() * (1 - bary.y()) * v1 + (1 - bary.x()) * bary.y() * v2 +
           bary.x() * bary.y() * v3 - x;
      Dff(0, 0) = bary.y() * v3.x() - bary.y() * v2.x() +
                  (1 - bary.y()) * v1.x() - (1 - bary.y()) * v0.x();
      Dff(0, 1) = bary.x() * v3.x() + (1 - bary.x()) * v2.x() -
                  bary.x() * v1.x() - (1 - bary.x()) * v0.x();
      Dff(1, 0) = bary.y() * v3.y() - bary.y() * v2.y() +
                  (1 - bary.y()) * v1.y() - (1 - bary.y()) * v0.y();
      Dff(1, 1) = bary.x() * v3.y() + (1 - bary.x()) * v2.y() -
                  bary.x() * v1.y() - (1 - bary.x()) * v0.y();
      dx = *solve(Dff, -ff);
      bary += dx;
      if (squared_euclidean_length(bary) > 100) {
        i = max_num_iterations;  // non convergent: just to save time
      }
    }
    if (i < max_num_iterations) {
      return bary;
    }
  } else if constexpr (NumDimensions == 3) {
    auto const& v0 =
        vertex_at(cell_indices[0], cell_indices[1], cell_indices[2]);
    auto const& v1 =
        vertex_at(cell_indices[0] + 1, cell_indices[1], cell_indices[2]);
    auto const& v2 =
        vertex_at(cell_indices[0], cell_indices[1] + 1, cell_indices[2]);
    auto const& v3 =
        vertex_at(cell_indices[0] + 1, cell_indices[1] + 1, cell_indices[2]);
    auto const& v4 =
        vertex_at(cell_indices[0], cell_indices[1], cell_indices[2] + 1);
    auto const& v5 =
        vertex_at(cell_indices[0] + 1, cell_indices[1], cell_indices[2] + 1);
    auto const& v6 =
        vertex_at(cell_indices[0], cell_indices[1] + 1, cell_indices[2] + 1);
    auto const& v7 = vertex_at(cell_indices[0] + 1, cell_indices[1] + 1,
                               cell_indices[2] + 1);
    auto const  x0 = v0.x();
    auto const  y0 = v0.y();
    auto const  z0 = v0.z();
    auto const  x1 = v1.x();
    auto const  y1 = v1.y();
    auto const  z1 = v1.z();
    auto const  x2 = v2.x();
    auto const  y2 = v2.y();
    auto const  z2 = v2.z();
    auto const  x3 = v3.x();
    auto const  y3 = v3.y();
    auto const  z3 = v3.z();
    auto const  x4 = v4.x();
    auto const  y4 = v4.y();
    auto const  z4 = v4.z();
    auto const  x5 = v5.x();
    auto const  y5 = v5.y();
    auto const  z5 = v5.z();
    auto const  x6 = v6.x();
    auto const  y6 = v6.y();
    auto const  z6 = v6.z();
    auto const  x7 = v7.x();
    auto const  y7 = v7.y();
    auto const  z7 = v7.z();

    auto ff = vec_type{};
    for (; i < max_num_iterations && squared_euclidean_length(dx) > tol; ++i) {
      auto const a = bary.x();
      auto const b = bary.y();
      auto const c = bary.z();
      // apply Newton-Raphson method to solve ff(x,y)=x
      ff = (1 - a) * (1 - b) * (1 - c) * v0;
      ff += a * (1 - b) * (1 - c) * v1;
      ff += (1 - a) * b * (1 - c) * v2;
      ff += a * b * (1 - c) * v3;
      ff += (1 - a) * (1 - b) * c * v4;
      ff += a * (1 - b) * c * v5;
      ff += (1 - a) * b * c * v6;
      ff += a * b * c * v7;
      ff -= x;

      Dff(0, 0) = b * c * x7 - b * c * x6 + (1 - b) * c * x5 +
                  (b - 1) * c * x4 + (b - b * c) * x3 + (b * c - b) * x2 +
                  ((b - 1) * c - b + 1) * x1 + ((1 - b) * c + b - 1) * x0;
      Dff(0, 1) = a * c * x7 + (1 - a) * c * x6 - a * c * x5 +
                  (a - 1) * c * x4 + (a - a * c) * x3 +
                  ((a - 1) * c - a + 1) * x2 + (a * c - a) * x1 +
                  ((1 - a) * c + a - 1) * x0;
      Dff(0, 2) = a * b * x7 + (1 - a) * b * x6 + (a - a * b) * x5 +
                  ((a - 1) * b - a + 1) * x4 - a * b * x3 + (a - 1) * b * x2 +
                  (a * b - a) * x1 + ((1 - a) * b + a - 1) * x0;
      Dff(1, 0) = b * c * y7 - b * c * y6 + (1 - b) * c * y5 +
                  (b - 1) * c * y4 + (b - b * c) * y3 + (b * c - b) * y2 +
                  ((b - 1) * c - b + 1) * y1 + ((1 - b) * c + b - 1) * y0;
      Dff(1, 1) = a * c * y7 + (1 - a) * c * y6 - a * c * y5 +
                  (a - 1) * c * y4 + (a - a * c) * y3 +
                  ((a - 1) * c - a + 1) * y2 + (a * c - a) * y1 +
                  ((1 - a) * c + a - 1) * y0;
      Dff(1, 2) = a * b * y7 + (1 - a) * b * y6 + (a - a * b) * y5 +
                  ((a - 1) * b - a + 1) * y4 - a * b * y3 + (a - 1) * b * y2 +
                  (a * b - a) * y1 + ((1 - a) * b + a - 1) * y0;
      Dff(2, 0) = b * c * z7 - b * c * z6 + (1 - b) * c * z5 +
                  (b - 1) * c * z4 + (b - b * c) * z3 + (b * c - b) * z2 +
                  ((b - 1) * c - b + 1) * z1 + ((1 - b) * c + b - 1) * z0;
      Dff(2, 1) = a * c * z7 + (1 - a) * c * z6 - a * c * z5 +
                  (a - 1) * c * z4 + (a - a * c) * z3 +
                  ((a - 1) * c - a + 1) * z2 + (a * c - a) * z1 +
                  ((1 - a) * c + a - 1) * z0;
      Dff(2, 2) = a * b * z7 + (1 - a) * b * z6 + (a - a * b) * z5 +
                  ((a - 1) * b - a + 1) * z4 - a * b * z3 + (a - 1) * b * z2 +
                  (a * b - a) * z1 + ((1 - a) * b + a - 1) * z0;

      dx = *solve(Dff, -ff);
      bary += dx;
      if (squared_euclidean_length(bary) > 100) {
        i = max_num_iterations;  // non convergent: just to save time
      }
    }
    if (i < max_num_iterations) {
      return bary;
    }
  }
  return pos_type::fill(Real(0) / Real(0));
}
//==============================================================================
template <std::size_t NumDimensions>
using StructuredGrid   = structured_grid<real_number, NumDimensions>;
using structured_grid2 = StructuredGrid<2>;
using structured_grid3 = StructuredGrid<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
template <typename T>
struct structured_grid<Real, NumDimensions,
                       IndexOrder>::linear_cell_sampler_type
    : field<structured_grid<Real, NumDimensions,
                            IndexOrder>::linear_cell_sampler_type<T>,
            Real, NumDimensions, T> {
  using this_type     = linear_cell_sampler_type;
  using real_type     = Real;
  using parent_type   = field<this_type, Real, NumDimensions, T>;
  using grid_type     = structured_grid<Real, NumDimensions, IndexOrder>;
  using property_type = typename grid_type::template typed_vertex_property_type<T>;
  using vec_type      = typename grid_type::vec_type;
  using pos_type      = typename grid_type::pos_type;
  using typename parent_type::tensor_type;

 private:
  grid_type const*     m_grid;
  property_type const* m_property;

 public:
  linear_cell_sampler_type(grid_type const& grid, property_type const& prop)
      : m_grid{&grid}, m_property{&prop} {}

  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return *m_grid; }
  auto property() const -> auto const& { return *m_property; }
  //----------------------------------------------------------------------------
  auto evaluate(pos_type const& x, real_type const /*t*/) const -> tensor_type {
    auto possible_cells = grid().hierarchy()->nearby_cells(x);

    for (auto const& cell : possible_cells) {
      auto const c = grid().local_cell_coordinates(x, cell);
      if (std::isnan(c(0))) {
        continue;
      }
      auto is_inside = true;
      for (size_t i = 0; i < NumDimensions; ++i) {
        if (c(i) < -1e-10 || c(i) > 1 + 1e-10) {
          is_inside = false;
          break;
        }
      }
      if (!is_inside) {
        continue;
      }
      if constexpr (NumDimensions == 2) {
        return (1 - c(0)) * (1 - c(1)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1])}] +
               c(0) * (1 - c(1)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0] + 1, cell[1])}] +
               (1 - c(0)) * c(1) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1] + 1)}] +
               c(0) * c(1) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0] + 1, cell[1] + 1)}];
      } else if constexpr (NumDimensions == 3) {
        return (1 - c(0)) * (1 - c(1)) * (1 - c(2)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1], cell[2])}] +
               c(0) * (1 - c(1)) * (1 - c(2)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0] + 1, cell[1], cell[2])}] +
               (1 - c(0)) * c(1) * (1 - c(2)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1] + 1, cell[2])}] +
               c(0) * c(1) * (1 - c(2)) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0] + 1, cell[1] + 1, cell[2])}] +
               (1 - c(0)) * (1 - c(1)) * c(2) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1], cell[2] + 1)}] +
               c(0) * (1 - c(1)) * c(2) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0] + 1, cell[1], cell[2] + 1)}] +
               (1 - c(0)) * c(1) * c(2) *
                   property()[vertex_handle{
                       grid().plain_index(cell[0], cell[1] + 1, cell[2] + 1)}] +
               c(0) * c(1) * c(2) *
                   property()[vertex_handle{grid().plain_index(
                       cell[0] + 1, cell[1] + 1, cell[2] + 1)}];
      }
    }
    return this->ood_tensor();
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
struct structured_grid<Real, NumDimensions, IndexOrder>::hierarchy_type
    : base_uniform_tree_hierarchy<Real, NumDimensions, hierarchy_type> {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_type     = hierarchy_type;
  using real_type     = Real;
  using index_order_t = IndexOrder;
  using grid_type     = structured_grid<Real, NumDimensions, IndexOrder>;
  using parent_type =
      base_uniform_tree_hierarchy<Real, NumDimensions, this_type>;
  using cell_t = std::array<std::size_t, NumDimensions>;
  //============================================================================
  // INHERITED TYPES
  //============================================================================
  using typename parent_type::pos_type;
  using typename parent_type::vec_type;
  //============================================================================
  // INHERITED METHODS
  //============================================================================
  using parent_type::center;
  using parent_type::children;
  using parent_type::extents;
  using parent_type::is_at_max_depth;
  using parent_type::is_inside;
  using parent_type::is_simplex_inside;
  using parent_type::is_splitted;
  using parent_type::max;
  using parent_type::min;
  using parent_type::split_and_distribute;
  //============================================================================
  // STATIC METHODS
  //============================================================================
  static constexpr auto num_dimensions() -> std::size_t {
    return NumDimensions;
  }
  //============================================================================
  // MEMBERS
  //============================================================================
  grid_type const*    m_grid = nullptr;
  std::vector<cell_t> m_cell_handles;
  //============================================================================
  // CTORS
  //============================================================================
  hierarchy_type()                                             = default;
  hierarchy_type(hierarchy_type const&)                        = default;
  hierarchy_type(hierarchy_type&&) noexcept                    = default;
  auto operator=(hierarchy_type const&) -> hierarchy_type&     = default;
  auto operator=(hierarchy_type&&) noexcept -> hierarchy_type& = default;
  virtual ~hierarchy_type()                                    = default;

  explicit hierarchy_type(grid_type const& grid) : m_grid{&grid} {}
  explicit hierarchy_type(
      grid_type const& grid,
      size_t const     max_depth = parent_type::default_max_depth)
      : parent_type{pos_type::zeros(), pos_type::zeros(), 1, max_depth},
        m_grid{&grid} {
    parent_type::operator=(grid.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  hierarchy_type(vec_type const& min, vec_type const& max,
                 grid_type const& grid,
                 size_t const     max_depth = parent_type::default_max_depth)
      : parent_type{min, max, 1, max_depth}, m_grid{&grid} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  hierarchy_type(vec_type const& min, vec_type const& max, size_t const level,
                 size_t const max_depth, grid_type const& grid)
      : parent_type{min, max, level, max_depth}, m_grid{&grid} {}
  //============================================================================
  // METHODS
  //============================================================================
 public:
  auto grid() const -> auto const& { return *m_grid; }
  auto constexpr holds_cells() const { return !m_cell_handles.empty(); }
  //----------------------------------------------------------------------------
  auto num_cell_handles() const { return size(m_cell_handles); }
  //----------------------------------------------------------------------------
  constexpr auto is_cell_inside(integral auto const... is) const {
    if constexpr (NumDimensions == 2) {
      return is_cell_inside_2(is...);
    } else if constexpr (NumDimensions == 3) {
      return is_cell_inside_3(is...);
    }
  }
  //----------------------------------------------------------------------------
 private:
  constexpr auto is_cell_inside_2(std::size_t const ix,
                                  std::size_t const iy) const
  requires(NumDimensions == 2)
  {
    auto const c  = center();
    auto const e  = extents() / 2;
    auto const us = std::array{vec_type{1, 0}, vec_type{0, 1}};
    auto const xs = std::array{
        grid().vertex_at(ix, iy) - c, grid().vertex_at(ix + 1, iy) - c,
        grid().vertex_at(ix + 1, iy + 1) - c, grid().vertex_at(ix, iy + 1) - c};
    auto is_separating_axis = [&](auto const& axis) {
      auto const ps = std::array{dot(xs[0], axis), dot(xs[1], axis),
                                 dot(xs[2], axis), dot(xs[3], axis)};
      auto       r  = e.x() * std::abs(dot(us[0], axis)) +
               e.y() * std::abs(dot(us[1], axis));
      return tatooine::max(-tatooine::max(ps), tatooine::min(ps)) > r;
    };
    for (auto const& u : us) {
      if (is_separating_axis(u)) {
        return false;
      }
    }
    for (size_t i = 0; i < size(xs); ++i) {
      auto const j = i == size(xs) - 1 ? 0 : i + 1;
      if (is_separating_axis(
              vec_type{xs[i].y() - xs[j].y(), xs[j].x() - xs[i].x()})) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  constexpr auto is_cell_inside_3(std::size_t const ix, std::size_t const iy,
                                  std::size_t const iz) const
  requires(NumDimensions == 3)
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
        std::array{vec_type{1, 0, 0}, vec_type{0, 1, 0}, vec_type{0, 0, 1}};

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

 public:
  //------------------------------------------------------------------------------
  template <typename... Indices>
  auto insert_cell(Indices const... is) -> bool {
    if (!is_cell_inside(is...)) {
      return false;
    }
    if (holds_cells()) {
      if (is_at_max_depth()) {
        m_cell_handles.push_back(cell_t{static_cast<std::size_t>(is)...});
      } else {
        split_and_distribute();
        distribute_cell(is...);
      }
    } else {
      if (is_splitted()) {
        distribute_cell(is...);
      } else {
        m_cell_handles.push_back(cell_t{static_cast<std::size_t>(is)...});
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto distribute() {
    if (holds_cells()) {
      distribute_cell(m_cell_handles.front());
      m_cell_handles.clear();
    }
  }
  //------------------------------------------------------------------------------
  auto construct(vec_type const& min, vec_type const& max, size_t const level,
                 size_t const max_depth) const {
    return std::unique_ptr<this_type>{
        new this_type{min, max, level, max_depth, grid()}};
  }
  //----------------------------------------------------------------------------
  auto distribute_cell(integral auto const... is) {
    for (auto& child : children()) {
      child->insert_cell(is...);
    }
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  auto distribute_cell(std::array<std::size_t, NumDimensions> const& is,
                       std::index_sequence<Is...> /*seq*/) {
    distribute_cell(is[Is]...);
  }
  //----------------------------------------------------------------------------
  auto distribute_cell(std::array<std::size_t, NumDimensions> const& is) {
    distribute_cell(is, std::make_index_sequence<NumDimensions>{});
  }
  //============================================================================
  auto collect_nearby_cells(vec_type const& pos, std::set<cell_t>& cells) const
      -> void {
    if (is_inside(pos)) {
      if (is_splitted()) {
        for (auto const& child : children()) {
          child->collect_nearby_cells(pos, cells);
        }
      } else {
        if (holds_cells()) {
          boost::copy(m_cell_handles, std::inserter(cells, end(cells)));
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  auto nearby_cells(pos_type const& pos) const {
    std::set<cell_t> cells;
    collect_nearby_cells(pos, cells);
    return cells;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
