#ifndef TATOOINE_DYNAMIC_MULTIDIM_SIZE_H
#define TATOOINE_DYNAMIC_MULTIDIM_SIZE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/functional.h>
#include <tatooine/index_order.h>
#include <tatooine/static_multidim_size.h>
#include <tatooine/template_helper.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename IndexOrder>
class dynamic_multidim_size {
 public:
  using this_type        = dynamic_multidim_size<IndexOrder>;
  using index_order_type = IndexOrder;

 private:
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  std::vector<std::size_t> m_size = {};

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  dynamic_multidim_size() = default;
  //----------------------------------------------------------------------------
  dynamic_multidim_size(dynamic_multidim_size const& other)     = default;
  dynamic_multidim_size(dynamic_multidim_size&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator                 =(dynamic_multidim_size const& other)
      -> dynamic_multidim_size& = default;
  auto operator                 =(dynamic_multidim_size&& other) noexcept
      -> dynamic_multidim_size& = default;
  //----------------------------------------------------------------------------
  ~dynamic_multidim_size() = default;
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  explicit dynamic_multidim_size(
      dynamic_multidim_size<OtherIndexing> const& other)
      : m_size{other.size()} {}

  template <typename OtherIndexing>
  explicit dynamic_multidim_size(dynamic_multidim_size<OtherIndexing>&& other)
      : m_size{std::move(other.m_size)} {}

  template <typename OtherIndexing>
  auto operator=(dynamic_multidim_size<OtherIndexing> const& other)
      -> dynamic_multidim_size& {
    m_size = other.m_size;
    return *this;
  }
  template <typename OtherIndexing>
  auto operator=(dynamic_multidim_size<OtherIndexing>&& other)
      -> dynamic_multidim_size& {
    m_size = std::move(other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(integral auto const... size)
      : m_size{static_cast<std::size_t>(size)...} {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(std::vector<std::size_t>&& size)
      : m_size(std::move(size)) {}
  //----------------------------------------------------------------------------
  explicit dynamic_multidim_size(integral_range auto const& size)
      : m_size(begin(size), end(size)) {}
  //----------------------------------------------------------------------------
  // comparisons
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator==(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() != other.num_dimensions()) {
      return false;
    }
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] != other.size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator!=(dynamic_multidim_size<OtherIndexing> const& other) const {
    if (num_dimensions() == other.num_dimensions()) {
      return false;
    }
    for (std::size_t i = 0; i < num_dimensions(); ++i) {
      if (m_size[i] == other.size(i)) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  auto num_dimensions() const { return m_size.size(); }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto size() const -> auto const& { return m_size; }
  /// \return size of dimensions i
  auto size(std::size_t const i) const { return m_size[i]; }
  //----------------------------------------------------------------------------
  auto num_components() const {
    return std::accumulate(begin(m_size), end(m_size), std::size_t(1),
                           std::multiplies<std::size_t>{});
  }
  //----------------------------------------------------------------------------
  auto resize(integral auto const... size) -> void {
    m_size = {static_cast<std::size_t>(size)...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(integral_range auto const& size) -> void {
    m_size = std::vector<std::size_t>(begin(size), end(size));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(std::vector<std::size_t>&& size) -> void {
    m_size = std::move(size);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto resize(std::vector<std::size_t> const& size) -> void { m_size = size; }
  //----------------------------------------------------------------------------
  auto constexpr in_range(integral auto const... indices) const {
    if (sizeof...(indices) != num_dimensions()) {
      return false;
    }
    return in_range(std::array{static_cast<std::size_t>(indices)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr in_range(integral_range auto const& indices) const {
    if (indices.size() != num_dimensions()) {
      return false;
    }
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= size(i)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto constexpr plain_index(integral auto const... indices) const {
    assert(sizeof...(indices) == num_dimensions());
    assert(in_range(indices...));
    return IndexOrder::plain_index(m_size, indices...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr plain_index(integral_range auto const& indices) const {
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return IndexOrder::plain_index(m_size, indices);
  }
  //----------------------------------------------------------------------------
  auto constexpr multi_index(std::size_t const gi) const {
    return IndexOrder::multi_index(m_size, gi);
  }
  //============================================================================
  struct indices_iterator {
    //--------------------------------------------------------------------------
    this_type const*         m_multidim_size = nullptr;
    std::vector<std::size_t> m_status        = {};
    //--------------------------------------------------------------------------
    indices_iterator(this_type const& c, std::vector<std::size_t> status)
        : m_multidim_size{&c}, m_status{std::move(status)} {}
    //--------------------------------------------------------------------------
    indices_iterator(indices_iterator const& other)     = default;
    indices_iterator(indices_iterator&& other) noexcept = default;
    //--------------------------------------------------------------------------
    auto operator            =(indices_iterator const& other)
        -> indices_iterator& = default;
    auto operator            =(indices_iterator&& other) noexcept
        -> indices_iterator& = default;
    //--------------------------------------------------------------------------
    ~indices_iterator() = default;
    //--------------------------------------------------------------------------
    auto operator++() {
      ++m_status.front();
      auto size_it   = m_multidim_size->size().begin();
      auto status_it = m_status.begin();
      for (; size_it != prev(m_multidim_size->size().end());
           ++status_it, ++size_it) {
        if (*size_it <= *status_it) {
          *status_it = 0;
          ++(*next(status_it));
        }
      }
    }
    //--------------------------------------------------------------------------
    auto operator==(indices_iterator const& other) const {
      if (m_multidim_size != other.m_multidim_size) {
        return false;
      }
      for (std::size_t i = 0; i < m_multidim_size->num_dimensions(); ++i) {
        if (m_status[i] != other.m_status[i]) {
          return false;
        }
      }
      return true;
    }
    //--------------------------------------------------------------------------
    auto operator!=(indices_iterator const& other) const {
      return !operator==(other);
    }
    //--------------------------------------------------------------------------
    auto operator*() const -> auto const& { return m_status; }
  };
  //----------------------------------------------------------------------------
  auto begin_indices() const {
    return indices_iterator{*this,
                            std::vector<std::size_t>(num_dimensions(), 0)};
  }
  //----------------------------------------------------------------------------
  auto end_indices() const {
    auto v   = std::vector<std::size_t>(num_dimensions(), 0);
    v.back() = size().back();
    return indices_iterator{*this, std::move(v)};
  }
  //------------------------------------------------------------------------------
  struct index_range {
   private:
    this_type const* m_multidim_size;

   public:
    explicit index_range(this_type const* multidim_size)
        : m_multidim_size{multidim_size} {}
    auto begin() const { return m_multidim_size->begin_indices(); }
    auto end() const { return m_multidim_size->end_indices(); }
  };
  auto indices() const { return index_range{this}; }
};
//==============================================================================
// deduction guides
//==============================================================================
dynamic_multidim_size()->dynamic_multidim_size<x_fastest>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder> const&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder> &&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename... Resolution>
dynamic_multidim_size(Resolution...) -> dynamic_multidim_size<x_fastest>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
