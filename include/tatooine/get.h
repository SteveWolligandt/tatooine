#ifndef TATOOINE_GET_H
#define TATOOINE_GET_H
//==============================================================================
#include <cstdint>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Container, std::size_t I>
struct get_impl;
template <typename Container, std::size_t I>
using get = typename get_impl<Container, I>::type;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
