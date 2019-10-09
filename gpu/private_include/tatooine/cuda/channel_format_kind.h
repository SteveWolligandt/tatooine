#ifndef TATOOINE_GPU_CUDA_CHANNEL_FORMAT_KIND_H
#define TATOOINE_GPU_CUDA_CHANNEL_FORMAT_KIND_H

#include <cstdint>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
template <typename T>
struct channel_format_kind_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<float> {
  static constexpr auto value = cudaChannelFormatKindFloat;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<double> {
  static constexpr auto value = cudaChannelFormatKindFloat;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::uint8> {
  static constexpr auto value = cudaChannelFormatKindUnsigned;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::uint16> {
  static constexpr auto value = cudaChannelFormatKindUnsigned;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::uint32> {
  static constexpr auto value = cudaChannelFormatKindUnsigned;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::int8> {
  static constexpr auto value = cudaChannelFormatKindSigned;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::int16> {
  static constexpr auto value = cudaChannelFormatKindSigned;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct channel_format_kind_impl<std::int32> {
  static constexpr auto value = cudaChannelFormatKindSigned;
};
//------------------------------------------------------------------------------
template <typename T>
constexpr auto channel_format_kind() {
  return channel_format_kind_impl<T>::value();
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
