#ifndef TATOOINE_HANDLE_H
#define TATOOINE_HANDLE_H

#include <cstdint>

//============================================================================
namespace tatooine {
//============================================================================
struct handle {
  static constexpr std::size_t invalid_idx =
      std::numeric_limits<std::size_t>::max();
  //==========================================================================
  std::size_t i;
  //==========================================================================
  handle() : i{invalid_idx} {}
  handle(std::size_t _i) : i{_i} {}
  handle(const handle&) = default;
  handle(handle&&)      = default;
  handle& operator=(const handle&) = default;
  handle& operator=(handle&&) = default;
  //==========================================================================
  auto& operator++() {
    ++this->i;
    return *this;
  }
  //--------------------------------------------------------------------------
  auto& operator--() {
    --this->i;
    return *this;
  }
  //--------------------------------------------------------------------------
  auto& operator=(std::size_t i) {
    this->i = i;
    return *this;
  }
};
//============================================================================
}  // namespace tatooine
//============================================================================
#endif
