#ifndef TATOOINE_MULTIDIM_PROPERTY_H
#define TATOOINE_MULTIDIM_PROPERTY_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Owner, size_t N>
struct multidim_property {
  //============================================================================
  using this_t = multidim_property<Owner, N>;
  //============================================================================
  Owner const& m_owner;
  //============================================================================
  multidim_property(Owner const& owner)
      : m_owner{owner} {}
  multidim_property(multidim_property const& other)     = default;
  multidim_property(multidim_property&& other) noexcept = default;
  //----------------------------------------------------------------------------
  /// Destructor.
  virtual ~multidim_property() {}
  //----------------------------------------------------------------------------
  /// for identifying type.
  virtual auto type() const -> std::type_info const& = 0;
  //----------------------------------------------------------------------------
  virtual auto clone() const -> std::unique_ptr<this_t> = 0;
};
//==============================================================================
template <typename Owner, typename T, size_t N,
          template <typename> typename... InterpolationKernels>
struct typed_multidim_property : multidim_property<Owner, N> {
  //============================================================================
  using this_t   = typed_multidim_property<Owner, T, N, InterpolationKernels...>;
  using parent_t = multidim_property<Owner, N>;
  //============================================================================
  typed_multidim_property(Owner const& owner) : parent_t{owner} {}
  typed_multidim_property(typed_multidim_property const&)     = default;
  typed_multidim_property(typed_multidim_property&&) noexcept = default;
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
template <typename Owner, typename T, size_t N, typename Container,
          template <typename> typename... InterpolationKernels>
struct typed_multidim_property_impl
    : typed_multidim_property<Owner, T, N, InterpolationKernels...> {
  //============================================================================
  using this_t = typed_multidim_property_impl<Owner, T, N, Container,
                                              InterpolationKernels...>;
  using parent_t =
      typed_multidim_property<Owner, T, N, InterpolationKernels...>;
  using property_base_t = typename parent_t::parent_t;
  using container_t = Container;
  //============================================================================
 private:
  container_t m_container;
  //============================================================================
 public:
  template <typename... Args>
  typed_multidim_property_impl(Owner const& owner, Args&&... args)
      : parent_t{owner}, m_container{std::forward<Args>(args)...} {}
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
  virtual std::unique_ptr<property_base_t> clone() const override {
    return std::unique_ptr<this_t>{new this_t{*this}};
  }
  //----------------------------------------------------------------------------
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  auto at(std::array<std::size_t, N> const& is) -> T & override {
    return at(is, std::make_index_sequence<N>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(std::array<std::size_t, N> const& is) const -> T const& override {
    return at(is, std::make_index_sequence<N>{});
  }
  //----------------------------------------------------------------------------
  auto sample(real_number auto... xs) const -> T {
    static_assert(
        sizeof...(xs) == N,
        "Number of spatial components does not match number of dimensions.");
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
