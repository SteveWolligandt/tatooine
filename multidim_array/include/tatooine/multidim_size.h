#ifndef TATOOINE_MULTIDIM_SIZE_H
#define TATOOINE_MULTIDIM_SIZE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/for_loop.h>
#include <tatooine/functional.h>
#include <tatooine/index_order.h>
#include <tatooine/multidim.h>
#include <tatooine/template_helper.h>
#include <tatooine/type_traits.h>
#include <tatooine/utility.h>
#include <tatooine/static_multidim_size.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename IndexOrder>
class dynamic_multidim_size {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  std::vector<std::size_t> m_size;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  dynamic_multidim_size() = default;
  //----------------------------------------------------------------------------
  dynamic_multidim_size(dynamic_multidim_size const& other)     = default;
  dynamic_multidim_size(dynamic_multidim_size&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(dynamic_multidim_size const& other)
      -> dynamic_multidim_size& = default;
  auto operator=(dynamic_multidim_size&& other) noexcept
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
 public:
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
    assert(sizeof...(indices) == num_dimensions());
    return in_range(std::array{static_cast<std::size_t>(indices)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto constexpr in_range(integral_range auto const& indices) const {
    assert(indices.size() == num_dimensions());
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
  //----------------------------------------------------------------------------
  auto constexpr indices() const { return dynamic_multidim{m_size}; }
};
//==============================================================================
// deduction guides
//==============================================================================
dynamic_multidim_size()->dynamic_multidim_size<x_fastest>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder> const&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename IndexOrder>
dynamic_multidim_size(dynamic_multidim_size<IndexOrder>&&)
    -> dynamic_multidim_size<IndexOrder>;
template <typename... Resolution>
dynamic_multidim_size(Resolution...) -> dynamic_multidim_size<x_fastest>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
