#ifndef TATOOINE_STRUCTURED_GRID_H
#define TATOOINE_STRUCTURED_GRID_H
//==============================================================================
#include <tatooine/multidim_size.h>
#include <tatooine/pointset.h>
#include <tatooine/uniform_tree_hierarchy.h>
#include <tatooine/vtk/xml.h>
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
  struct linear_cell_sampler_t;
  struct hierarchy_t;
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_t                 = structured_grid;
  using pointset_parent_t      = pointset<Real, NumDimensions>;
  using multidim_size_parent_t = dynamic_multidim_size<IndexOrder>;
  using typename pointset_parent_t::pos_t;
  using typename pointset_parent_t::vec_t;
  using typename pointset_parent_t::vertex_handle;
  template <typename T>
  using vertex_property_t =
      typename pointset_parent_t::template vertex_property_t<T>;
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
  auto hierarchy() const -> auto const& { return m_hierarchy; }
  //----------------------------------------------------------------------------
  auto update_hierarchy() const {
    if (m_hierarchy != nullptr) {
      m_hierarchy.reset();
    }
    auto const aabb = this->axis_aligned_bounding_box();
    m_hierarchy =
        std::make_unique<hierarchy_t>(aabb.min(), aabb.max(), *this, 4);
    auto       it = [&](auto const... is) { m_hierarchy->insert_cell(is...); };
    auto const s  = this->size();
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
  template <typename... Ts, enable_if<is_arithmetic<Ts...> > = true,
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
      -> pos_t {
    static auto constexpr num_indices = sizeof...(Indices);
    static_assert(num_indices == num_dimensions(),
                  "Number of Indices does not match number of dimensions");
    return local_cell_coordinates(x,
                                  std::array{static_cast<std::size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  auto local_cell_coordinates(
      pos_t const x, std::array<std::size_t, NumDimensions> const& cell) const
      -> pos_t;
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(vertex_property_t<T> const& name) const {
    if (m_hierarchy == nullptr) {
      update_hierarchy();
      std::cout << "updating done!\n";
    }
    return linear_cell_sampler_t<T>{this, &vertex_property<T>(name)};
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto linear_vertex_property_sampler(std::string const& name) const {
    if (m_hierarchy == nullptr) {
      update_hierarchy();
    }
    return linear_cell_sampler_t<T>{*this,
                                    this->template vertex_property<T>(name)};
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
  struct listener_t : vtk::xml::listener {
    this_t& grid;
    listener_t(this_t& g) : grid{g} {}

    std::array<std::pair<std::size_t, std::size_t>, 3> whole_extent;
    std::array<std::size_t, 3>                         resolution;
    std::array<size_t, 3>                              cur_piece_origin;
    std::array<size_t, 3>                              cur_piece_resolution;

    auto on_structured_grid(
        std::array<std::pair<std::size_t, std::size_t>, 3> const& d)
        -> void override {
      whole_extent = d;

      resolution =
          std::array{whole_extent[0].second - whole_extent[0].first + 1,
                     whole_extent[1].second - whole_extent[1].first + 1,
                     whole_extent[2].second - whole_extent[2].first + 1};

      grid.resize(resolution[0], resolution[1], resolution[2]);
    }
    auto on_structured_grid_piece(
        std::array<std::pair<std::size_t, std::size_t>, 3> const& extents)
        -> void override {
      cur_piece_origin = std::array{extents[0].first - whole_extent[0].first,
                                    extents[1].first - whole_extent[1].first,
                                    extents[2].first - whole_extent[2].first};
      cur_piece_resolution = std::array{extents[0].second - extents[0].first,
                                        extents[1].second - extents[1].first,
                                        extents[2].second - extents[2].first};
    }
    auto on_points(std::array<double, 3> const* v) -> void override {
      auto extract_points = [&](auto const... is) mutable {
        auto& x = grid.vertex_at(is...);
        for (size_t i = 0; i < num_dimensions(); ++i) {
          x(i) = v->at(i);
        }
        ++v;
      };
      for_loop(extract_points,
               std::pair{cur_piece_origin[0], cur_piece_resolution[0]},
               std::pair{cur_piece_origin[1], cur_piece_resolution[1]},
               std::pair{cur_piece_origin[2], cur_piece_resolution[2]});
    }
    auto on_point_data(std::string const& name, float const* v)
        -> void override {
      auto& prop = grid.template vertex_property<float>(name);
      for_loop(
          [&](auto const... is) mutable {
            auto& p = prop[vertex_handle{grid.plain_index(is...)}];
            p       = *v++;
          },
          std::pair{cur_piece_origin[0], cur_piece_resolution[0]},
          std::pair{cur_piece_origin[1], cur_piece_resolution[1]},
          std::pair{cur_piece_origin[2], cur_piece_resolution[2]});
    }
  } listener{*this};
  auto reader = vtk::xml::reader{path};
  reader.listen(listener);
  reader.read();
}
//------------------------------------------------------------------------------

template <typename Real, std::size_t NumDimensions, typename IndexOrder>
auto structured_grid<Real, NumDimensions, IndexOrder>::local_cell_coordinates(
    pos_t const                                   x,
    std::array<std::size_t, NumDimensions> const& cell_indices) const -> pos_t {
  if constexpr (NumDimensions == 2) {
    auto const& v0 = vertex_at(cell_indices[0], cell_indices[1]);
    auto const& v1 = vertex_at(cell_indices[0] + 1, cell_indices[1]);
    auto const& v2 = vertex_at(cell_indices[0], cell_indices[1] + 1);
    auto const& v3 = vertex_at(cell_indices[0] + 1, cell_indices[1] + 1);
    auto const  a  = v0;
    auto const  b  = v1 - v0;
    auto const  c  = v2 - v0;
    auto const  d  = v0 - v1 - v2 + v3;

    auto              bary = pos_t{Real(0.5), Real(0.5)};  // initial
    auto              dx   = Real(0.1) * pos_t::ones();
    auto              i    = std::size_t(0);
    auto const        tol  = 1e-12;
    auto              Df   = mat<Real, 2, 2>{};
    static auto const max_num_iterations = std::size_t(20);
    for (; i < max_num_iterations && euclidean_length(dx) > tol; ++i) {
      // apply Newton-Raphson method to solve f(x,y)=0
      auto f = a + b * bary.x() + c * bary.y() + d * bary.x() * bary.y() - x;
      // Newton: x_{n+1} = x_n - (Df^-1)*f
      // or equivalently denoting dx = x_{n+1}-x_n
      // Newton: Df*dx=-f
      Df.col(0) = (b + d * bary.y());  // df/dx
      Df.col(1) = (c + d * bary.x());  // df/dy
      dx        = solve(Df, -f);
      bary += dx;
      if (euclidean_length(bary) > 10) {
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
    auto const  a  = v0;
    auto const  b  = v1 - v0;
    auto const  c  = v2 - v0;
    auto const  d  = v4 - v0;
    auto const  e  = v3 - v2 - v1 + v0;
    auto const  f  = v5 - v4 - v1 + v0;
    auto const  g  = v6 - v4 - v2 + v0;
    auto const  h  = v7 - v6 - v5 + v4 - v3 + v2 + v1 - v0;

    auto              bary = pos_t{Real(0.5), Real(0.5), Real(0.5)};  // initial
    auto              dx   = Real(0.1) * pos_t::ones();
    auto              i    = std::size_t(0);
    auto const        tol  = 1e-12;
    auto              Df   = mat<Real, 3, 3>{};
    static auto const max_num_iterations = std::size_t(20);
    for (; i < max_num_iterations && euclidean_length(dx) > tol; ++i) {
      // apply Newton-Raphson method to solve f(x,y)=0
      auto const ff = a + b * bary.x() + c * bary.y() + d * bary.z() +
                      e * bary.x() * bary.y() + f * bary.x() * bary.z() +
                      g * bary.y() * bary.z() +
                      h * bary.x() * bary.y() * bary.z() - x;
      Df.col(0) =
          b + e * bary.y() + f * bary.z() + h * bary.y() * bary.z();  // df/dx
      Df.col(1) =
          c + e * bary.x() + g * bary.z() + h * bary.x() * bary.z();  // df/dy
      Df.col(2) =
          d + f * bary.x() + g * bary.y() + h * bary.x() * bary.y();  // df/dz
      dx = solve(Df, -ff);
      bary += dx;
      if (euclidean_length(bary) > 10) {
        i = max_num_iterations;  // non convergent: just to save time
      }
    }
    if (i < max_num_iterations) {
      return bary;
    }
  }
  return pos_t{tag::fill{Real(0) / Real(0)}};
}
//==============================================================================
template <std::size_t NumDimensions>
using StructuredGrid   = structured_grid<real_t, NumDimensions>;
using structured_grid2 = StructuredGrid<2>;
using structured_grid3 = StructuredGrid<3>;
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
template <typename T>
struct structured_grid<Real, NumDimensions, IndexOrder>::linear_cell_sampler_t
    : field<structured_grid<Real, NumDimensions,
                            IndexOrder>::linear_cell_sampler_t<T>,
            Real, NumDimensions, T> {
  using this_t     = linear_cell_sampler_t;
  using parent_t   = field<this_t, Real, NumDimensions, T>;
  using grid_t     = structured_grid<Real, NumDimensions, IndexOrder>;
  using property_t = typename grid_t::template vertex_property_t<T>;
  using vec_t      = typename grid_t::vec_t;
  using pos_t      = typename grid_t::pos_t;
  using typename parent_t::tensor_t;

 private:
  grid_t const*     m_grid;
  property_t const* m_property;

 public:
  linear_cell_sampler_t(grid_t const& grid, property_t const& prop)
      : m_grid{&grid}, m_property{&prop} {}

  //----------------------------------------------------------------------------
  auto grid() const -> auto const& { return *m_grid; }
  auto property() const -> auto const& { return *m_property; }
  //----------------------------------------------------------------------------
  auto evaluate(pos_t const& x, real_t const /*t*/) const -> tensor_t {
    auto possible_cells = grid().hierarchy()->nearby_cells(x);

    for (auto const& cell : possible_cells) {
      auto const c         = grid().local_cell_coordinates(x, cell);
      auto       is_inside = true;
      for (size_t i = 0; i < NumDimensions; ++i) {
        if (c(i) < 0 || c(i) > 1) {
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
      }
    }
    return this->ood_tensor();
  }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename IndexOrder>
struct structured_grid<Real, NumDimensions, IndexOrder>::hierarchy_t
    : base_uniform_tree_hierarchy<Real, NumDimensions, hierarchy_t> {
  //============================================================================
  // TYPEDEFS
  //============================================================================
  using this_t        = hierarchy_t;
  using real_t        = Real;
  using index_order_t = IndexOrder;
  using grid_t        = structured_grid<Real, NumDimensions, IndexOrder>;
  using parent_t = base_uniform_tree_hierarchy<Real, NumDimensions, this_t>;
  using cell_t   = std::array<std::size_t, NumDimensions>;
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
  grid_t const*       m_grid = nullptr;
  std::vector<cell_t> m_cell_handles;
  //============================================================================
  // CTORS
  //============================================================================
  hierarchy_t()                       = default;
  hierarchy_t(hierarchy_t const&)     = default;
  hierarchy_t(hierarchy_t&&) noexcept = default;
  auto operator=(hierarchy_t const&) -> hierarchy_t& = default;
  auto operator=(hierarchy_t&&) noexcept -> hierarchy_t& = default;
  virtual ~hierarchy_t()                                 = default;

  explicit hierarchy_t(grid_t const& grid) : m_grid{&grid} {}
  explicit hierarchy_t(grid_t const& grid,
                       size_t const  max_depth = parent_t::default_max_depth)
      : parent_t{pos_t::zeros(), pos_t::zeros(), 1, max_depth}, m_grid{&grid} {
    parent_t::operator=(grid.bounding_box());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  hierarchy_t(vec_t const& min, vec_t const& max, grid_t const& grid,
              size_t const max_depth = parent_t::default_max_depth)
      : parent_t{min, max, 1, max_depth}, m_grid{&grid} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 private:
  hierarchy_t(vec_t const& min, vec_t const& max, size_t const level,
              size_t const max_depth, grid_t const& grid)
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
  constexpr auto is_cell_inside_2(std::size_t const ix,
                                  std::size_t const iy) const
#ifdef __cpp_concepts
      requires(NumDimensions == 2)
#endif
  {
    auto const c  = center();
    auto const e  = extents() / 2;
    auto const us = std::array{vec_t{1, 0}, vec_t{0, 1}};
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
              vec_t{xs[i].y() - xs[j].y(), xs[j].x() - xs[i].x()})) {
        return false;
      }
    }
    return true;
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

 public :
     //------------------------------------------------------------------------------
     template <typename... Indices>
     auto
     insert_cell(Indices const... is) -> bool {
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
  auto collect_nearby_cells(vec_t const& pos, std::set<cell_t>& cells) const
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
  auto nearby_cells(pos_t const& pos) const {
    std::set<cell_t> cells;
    collect_nearby_cells(pos, cells);
    return cells;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
