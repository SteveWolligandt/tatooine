#ifndef TATOOINE_NON_OWNING_MULTIDIM_ARRAY_H
#define TATOOINE_NON_OWNING_MULTIDIM_ARRAY_H
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
  using this_t     = non_owning_multidim_array<T, IndexOrder>;
  using parent_t   = dynamic_multidim_size<IndexOrder>;
  using parent_t::in_range;
  using parent_t::indices;
  using parent_t::num_components;
  using parent_t::num_dimensions;
  using parent_t::plain_index;
  using parent_t::size;
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
    if (parent_t::operator!=(other)) {
      parent_t::resize(other.size());
    }
    parent_t::operator=(other);
    return *this;
  }
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if_integral<Size...> = true>
#endif
  explicit non_owning_multidim_array(T const* data, Size... size)
      : parent_t{size...}, m_data(std::move(data)) {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <unsigned_integral UInt>
#else
  template <typename UInt, enable_if<is_unsigned_integral<UInt> > = true>
#endif
  non_owning_multidim_array(T const* data, std::vector<UInt> const& size)
      : parent_t{size}, m_data(std::move(data)) {
  }
  //============================================================================
  // methods
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto at(Is const... is) const -> auto const& {
    assert(sizeof...(is) == num_dimensions());
    assert(in_range(is...));
    return m_data[plain_index(is...)];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices> > = true>
#endif
  auto at(Indices const& indices) const -> auto const& {
    static_assert(is_integral<typename Indices::value_type>,
                  "index range must hold integral type");
    assert(indices.size() == num_dimensions());
    assert(in_range(indices));
    return m_data[plain_index(indices)];
  }
  //------------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  auto operator()(Is const... is) const -> auto const& {
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
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices> > = true>
#endif
  auto operator()(Indices const& indices) const -> auto const& {
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
