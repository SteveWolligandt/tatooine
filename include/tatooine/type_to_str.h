#ifndef TATOOINE_TYPE_TO_STR_H
#define TATOOINE_TYPE_TO_STR_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename>
struct type_to_str_false_type : std::false_type {};
//------------------------------------------------------------------------------
template <typename Data>
constexpr auto type_to_str() -> std::string_view {
  static_assert(type_to_str_false_type<Data>::value, "unknown type");
  return "";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<double>() -> std::string_view {
  return "double";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<long double>() -> std::string_view {
  return "long double";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<float>() -> std::string_view {
  return "float";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<int>() -> std::string_view {
  return "int";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<unsigned int>() -> std::string_view {
  return "unsigned int";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<long>() -> std::string_view {
  return "long";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<unsigned long>() -> std::string_view {
  return "unsigned long";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<long long>() -> std::string_view {
  return "long long";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<unsigned long long>() -> std::string_view {
  return "unsigned long long";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<char>() -> std::string_view {
  return "char";
}
//------------------------------------------------------------------------------
template <>
constexpr auto type_to_str<unsigned char>() -> std::string_view {
  return "unsigned char";
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
