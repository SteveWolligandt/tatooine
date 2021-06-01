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
  template <typename D = Derived>
  [[nodiscard]] constexpr auto as_derived() -> D& {
    return static_cast<D&>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  template <typename D = Derived>
  [[nodiscard]] constexpr auto as_derived() const -> D const & {
    return static_cast<D const &>(*this);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
