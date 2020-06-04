#ifndef TATOOINE_CRTP_H
#define TATOOINE_CRTP_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Derived>
struct crtp {
  using derived_t = Derived;
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  constexpr auto as_derived() -> Derived & {
    return static_cast<Derived &>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto operator*() -> Derived & {
    return as_derived();
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator->() -> Derived * {
    return &as_derived();
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived() const -> Derived const & {
    return static_cast<Derived const &>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto operator*() const -> Derived const & {
    return as_derived();
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto operator->() const -> Derived const * {
    return &as_derived();
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
