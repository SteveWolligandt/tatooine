#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t N>
struct multidim_property {
  using this_t = multidim_property<N>;
  //============================================================================
  multidim_property()                                   = default;
  multidim_property(multidim_property const& other)     = default;
  multidim_property(multidim_property&& other) noexcept = default;
  //============================================================================
  /// Destructor.
  virtual ~multidim_property() {}
  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual auto type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_t> = 0;
};
//==============================================================================
template <typename T, size_t N>
struct typed_multidim_property : multidim_property<N> {
  //----------------------------------------------------------------------------
  ~typed_multidim_property() override = default;
  //----------------------------------------------------------------------------
  const std::type_info& type() const override { return typeid(T); }
  //----------------------------------------------------------------------------
  virtual auto at(std::array<size_t, N> const& is) -> T& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  virtual auto at(std::array<size_t, N> const& is) const -> T const& = 0;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto... is) -> T& {
    static_assert(sizeof...(is) == N,
                  "Number of indices does not match number of dimensions.");
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(integral auto... is) const -> T const& {
    static_assert(sizeof...(is) == N,
                  "Number of indices does not match number of dimensions.");
    return at(std::array{static_cast<size_t>(is)...});
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto... is) -> T& {
    static_assert(sizeof...(is) == N,
                  "Number of indices does not match number of dimensions.");
    return at(std::array{static_cast<size_t>(is)...});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto... is) const -> T const& {
    static_assert(sizeof...(is) == N,
                  "Number of indices does not match number of dimensions.");
    return at(std::array{static_cast<size_t>(is)...});
  }
};
//==============================================================================
template <typename T, size_t N, typename Container>
struct typed_multidim_property_impl : typed_multidim_property<T, N> {
  using this_t      = typed_multidim_property_impl<T, N, Container>;
  using parent_t    = typed_multidim_property<T, N>;
  using container_t = Container;
  //============================================================================
 private:
  container_t m_container;
  //============================================================================
 public:
  template <typename... Args>
  typed_multidim_property_impl(Args&&... args)
      : m_container{std::forward<Args>(args)...} {}
  //----------------------------------------------------------------------------
  typed_multidim_property_impl(typed_multidim_property_impl const& other) =
      default;
  //----------------------------------------------------------------------------
  typed_multidim_property_impl(typed_multidim_property_impl&& other) = default;
  //----------------------------------------------------------------------------
  ~typed_multidim_property_impl() override = default;
  //============================================================================
  auto container() -> auto& { return m_container; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto container() const -> auto const& { return m_container; }
  //----------------------------------------------------------------------------
  virtual std::unique_ptr<multidim_property<N>> clone() const override {
    return std::unique_ptr<this_t>{new this_t{*this}};
  }

 private:
  template <std::size_t... Seq>
  auto at(std::array<std::size_t, N> const& is, std::index_sequence<Seq...>)
      -> T& {
    return m_container(is[Seq]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t... Seq>
  auto at(std::array<std::size_t, N> const& is,
          std::index_sequence<Seq...>) const -> T const& {
    return m_container(is[Seq]...);
  }

 public:
  auto at(std::array<std::size_t, N> const& is) -> T & override {
    return at(is, std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<std::size_t, N> const& is) const -> T const& override {
    return at(is, std::make_index_sequence<N>{});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
