#ifndef TATOOINE_NON_OWNING_MULTIDIM_ARRAY_H
#define TATOOINE_NON_OWNING_MULTIDIM_ARRAY_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename IndexOrder = x_fastest>
class non_owning_multidim_array : public dynamic_multidim_size<IndexOrder> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using value_type = T;
  using this_type     = non_owning_multidim_array<T, IndexOrder>;
  using parent_type   = dynamic_multidim_size<IndexOrder>;
  using parent_type::in_range;
  using parent_type::indices;
  using parent_type::num_components;
  using parent_type::num_dimensions;
  using parent_type::plain_index;
  using parent_type::size;
  //============================================================================
  // members
  //============================================================================
  T const* m_data;
  //============================================================================
  // ctors
  //============================================================================
 public:
  non_owning_multidim_array(non_owning_multidim_array const& other) = default;
  //----------------------------------------------------------------------------
  auto operator                     =(non_owning_multidim_array const& other)
      -> non_owning_multidim_array& = default;
  //----------------------------------------------------------------------------
  ~non_owning_multidim_array() = default;
  //----------------------------------------------------------------------------
  template <typename OtherIndexing>
  auto operator=(non_owning_multidim_array<T, OtherIndexing> const& other)
      -> non_owning_multidim_array& {
    if (parent_type::operator!=(other)) {
      parent_type::resize(other.size());
    }
    parent_type::operator=(other);
    return *this;
  }
  //============================================================================
  explicit non_owning_multidim_array(T const* data, integral auto const... size)
      : parent_type{size...}, m_data(std::move(data)) {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <unsigned_integral UInt>
  non_owning_multidim_array(T const* data, std::vector<UInt> const& size)
      : parent_type{size}, m_data(std::move(data)) {
  }
  //============================================================================
  // methods
  //============================================================================
  auto at(integral auto const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(range auto const& indices) const -> auto const& {
    using Indices = std::decay_t<decltype(indices)>;
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data[plain_index(indices)];
  }
  //------------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
#ifndef NDEBUG
    if (!in_range(is...)) {
      std::cerr << "will now crash because indices [ ";
      ((std::cerr << is << ' '), ...);
      std::cerr << "] are not in range\n";
    }
#endif
    assert(in_range(is...));
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(range auto const& indices) const -> auto const& {
    using Indices = std::decay_t<decltype(indices)>;
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return at(indices);
  }
  //----------------------------------------------------------------------------
  auto operator[](size_t i) const -> auto const& { return m_data[i]; }
  auto operator[](size_t i) -> auto& { return m_data[i]; }
  //----------------------------------------------------------------------------
  constexpr auto data() const -> auto { return m_data; }
  //----------------------------------------------------------------------------
  auto change_data(T const* new_data) { m_data = new_data; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
