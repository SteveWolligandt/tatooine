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
  constexpr auto &as_derived() { return static_cast<derived_t &>(*this); }

  //----------------------------------------------------------------------------
  /// returns casted as_derived data
  constexpr auto &as_derived() const {
    return static_cast<const derived_t &>(*this);
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
