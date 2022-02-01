#ifndef TATOOINE_DETAIL_POINTSET_VERTEX_CONTAINER_H
#define TATOOINE_DETAIL_POINTSET_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct vertex_container {
  using pointset_t      = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle_t = typename pointset_t::vertex_handle;
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    iterator() = default;
    iterator(vertex_handle_t const vh, pointset_t const* ps)
        : m_vh{vh}, m_ps{ps} {}
    iterator(iterator const& other) : m_vh{other.m_vh}, m_ps{other.m_ps} {}

   private:
    vertex_handle_t   m_vh{};
    pointset_t const* m_ps = nullptr;

   public:
    constexpr auto increment() {
      do {
        ++m_vh;
      } while (!m_ps->is_valid(m_vh));
    }
    constexpr auto decrement() {
      do {
        --m_vh;
      } while (!m_ps->is_valid(m_vh));
    }

    [[nodiscard]] constexpr auto equal(iterator const& other) const {
      return m_vh == other.m_vh;
    }
    [[nodiscard]] auto dereference() const { return m_vh; }

    constexpr auto at_end() const {
      return m_vh.index() == m_ps->vertex_position_data().size();
    }
  };
  //==========================================================================
 private:
  pointset_t const* m_pointset;

 public:
  vertex_container(pointset_t const* ps) : m_pointset{ps} {}
  vertex_container(vertex_container const&)     = default;
  vertex_container(vertex_container&&) noexcept = default;
  auto operator=(vertex_container const&) -> vertex_container& = default;
  auto operator=(vertex_container&&) noexcept -> vertex_container& = default;
  ~vertex_container()                                              = default;
  //==========================================================================
  auto begin() const {
    iterator vi{vertex_handle_t{0}, m_pointset};
    if (!m_pointset->is_valid(*vi)) {
      ++vi;
    }
    return vi;
  }
  //--------------------------------------------------------------------------
  static constexpr auto end() { return typename iterator::sentinel_type{}; }
  //--------------------------------------------------------------------------
  auto size() const {
    return m_pointset->vertex_position_data().size() -
           m_pointset->invalid_vertices().size();
  }
  auto data_container() const -> auto const& {
    return m_pointset->vertex_position_data();
  }
  auto data() const { return data_container().data(); }
  auto operator[](std::size_t const i) const {
    return m_pointset->at(vertex_handle_t{i});
  }
  auto operator[](std::size_t const i) {
    return m_pointset->at(vertex_handle_t{i});
  }
  auto operator[](vertex_handle_t const i) const { return m_pointset->at(i); }
  auto operator[](vertex_handle_t const i) { return m_pointset->at(i); }
  auto at(std::size_t const i) const {
    return m_pointset->at(vertex_handle_t{i});
  }
  auto at(std::size_t const i) { return m_pointset->at(vertex_handle_t{i}); }
  auto at(vertex_handle_t const i) const { return m_pointset->at(i); }
  auto at(vertex_handle_t const i) { return m_pointset->at(i); }
};
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto begin(vertex_container<Real, NumDimensions> verts) {
  return verts.begin();
}
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
auto end(vertex_container<Real, NumDimensions> verts) {
  return verts.end();
}
//------------------------------------------------------------------------------
template <typename Real, std::size_t NumDimensions>
auto size(vertex_container<Real, NumDimensions> verts) {
  return verts.size();
}
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto vertices(pointset<Real, NumDimensions> const& ps) {
  return ps.vertices();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
template <typename Real, std::size_t NumDimensions>
inline constexpr bool std::ranges::enable_borrowed_range<
    typename tatooine::detail::pointset::vertex_container<Real,
                                                          NumDimensions>> =
    true;
#endif
