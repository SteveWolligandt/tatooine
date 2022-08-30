#ifndef TATOOINE_CRTP_H
#define TATOOINE_CRTP_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Derived>
struct crtp {
  using derived_type = Derived;
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived() -> derived_type& {
    return static_cast<derived_type&>(*this);
  }
  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  [[nodiscard]] constexpr auto as_derived() const -> derived_type const& {
    return static_cast<derived_type const&>(*this);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
