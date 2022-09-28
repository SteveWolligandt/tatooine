#ifndef TATOOINE_DETAIL_POINTSET_VERTEX_CONTAINER_H
#define TATOOINE_DETAIL_POINTSET_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct const_vertex_container {
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    iterator() = default;
    iterator(vertex_handle const vh, pointset_type const* ps)
        : m_vh{vh}, m_ps{ps} {}
    iterator(iterator const& other) : m_vh{other.m_vh}, m_ps{other.m_ps} {}

   private:
    vertex_handle m_vh{};
    pointset_type const*      m_ps = nullptr;

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
  pointset_type const* m_pointset;

 public:
  const_vertex_container(pointset_type const* ps) : m_pointset{ps} {}
  const_vertex_container(const_vertex_container const&)     = default;
  const_vertex_container(const_vertex_container&&) noexcept = default;
  auto operator=(const_vertex_container const&)
      -> const_vertex_container& = default;
  auto operator=(const_vertex_container&&) noexcept
      -> const_vertex_container& = default;
  ~const_vertex_container()      = default;
  //==========================================================================
  auto begin() const {
    auto vi = iterator{vertex_handle{0}, m_pointset};
    while (!m_pointset->is_valid(*vi)) {
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
    return m_pointset->at(vertex_handle{i});
  }
  auto operator[](std::size_t const i) {
    return m_pointset->at(vertex_handle{i});
  }
  auto operator[](vertex_handle const i) const { return m_pointset->at(i); }
  auto operator[](vertex_handle const i) { return m_pointset->at(i); }
  auto at(std::size_t const i) const {
    return m_pointset->at(vertex_handle{i});
  }
  auto at(std::size_t const i) { return m_pointset->at(vertex_handle{i}); }
  auto at(vertex_handle const i) const { return m_pointset->at(i); }
  auto at(vertex_handle const i) { return m_pointset->at(i); }
};
//==============================================================================
static_assert(std::ranges::range<const_vertex_container<real_number, 2>>);
//static_assert(std::input_iterator<std::ranges::iterator_t<const_vertex_container<real_number, 2>>>);
//static_assert(std::ranges::forward_range<const_vertex_container<real_number, 2>>);
//static_assert(std::ranges::forward_range<const_vertex_container<real_number, 3>>);
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct vertex_container {
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    iterator() = default;
    iterator(vertex_handle const vh, pointset_type* ps) : m_vh{vh}, m_ps{ps} {}
    iterator(iterator const& other) : m_vh{other.m_vh}, m_ps{other.m_ps} {}

   private:
    vertex_handle  m_vh{};
    pointset_type* m_ps = nullptr;

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
  pointset_type* m_pointset;

 public:
  vertex_container(pointset_type* ps) : m_pointset{ps} {}
  vertex_container(vertex_container const&)                        = default;
  vertex_container(vertex_container&&) noexcept                    = default;
  auto operator=(vertex_container const&) -> vertex_container&     = default;
  auto operator=(vertex_container&&) noexcept -> vertex_container& = default;
  ~vertex_container()                                              = default;
  //==========================================================================
  auto begin() const {
    iterator vi{vertex_handle{0}, m_pointset};
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
    return m_pointset->at(vertex_handle{i});
  }
  auto operator[](std::size_t const i) {
    return m_pointset->at(vertex_handle{i});
  }
  auto operator[](vertex_handle const i) const { return m_pointset->at(i); }
  auto operator[](vertex_handle const i) { return m_pointset->at(i); }
  auto at(std::size_t const i) const {
    return m_pointset->at(vertex_handle{i});
  }
  auto at(std::size_t const i) { return m_pointset->at(vertex_handle{i}); }
  auto at(vertex_handle const i) const { return m_pointset->at(i); }
  auto at(vertex_handle const i) { return m_pointset->at(i); }
  auto resize(std::size_t const n) {
    m_pointset->m_vertex_position_data.resize(n);
    for (auto& [key, prop] : m_pointset->vertex_properties()) {
      prop->resize(n);
    }
  }
  auto reserve(std::size_t const n) {
    m_pointset->m_vertex_position_data.reserve(n);
    for (auto& [key, prop] : m_pointset->vertex_properties()) {
      prop->reserve(n);
    }
  }
};
//==============================================================================
//static_assert(std::ranges::forward_range<vertex_container<real_number, 2>>);
//static_assert(std::ranges::forward_range<vertex_container<real_number, 3>>);
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
inline constexpr auto std::ranges::enable_borrowed_range<
    typename tatooine::detail::pointset::vertex_container<Real,
                                                          NumDimensions>> =
    true;
#endif
