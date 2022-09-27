#ifndef TATOOINE_GL_VERTEXBUFFER_H
#define TATOOINE_GL_VERTEXBUFFER_H
//==============================================================================
#include <tatooine/gl/buffer.h>
#include <tatooine/gl/errorcheck.h>
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/utility.h>
#include <tatooine/gl/vbohelpers.h>
#include <tatooine/tensor.h>
#include <tatooine/tuple.h>

#include <initializer_list>
#include <iostream>
#include <vector>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
template <typename... Ts>
class vertexbuffer
    : public buffer<GL_ARRAY_BUFFER,
                    std::conditional_t<sizeof...(Ts) == 1, head_t<Ts...>,
                                       tuple<Ts...> > > {
 public:
  using parent_type = buffer<
      GL_ARRAY_BUFFER,
      std::conditional_t<sizeof...(Ts) == 1, head_t<Ts...>, tuple<Ts...> > >;
  using this_type = vertexbuffer<Ts...>;
  using typename parent_type::value_type;

  static constexpr auto data_size = parent_type::data_size;

  static buffer_usage const default_usage = buffer_usage::STATIC_DRAW;

  static constexpr auto num_attributes = sizeof...(Ts);
  static constexpr std::array<GLsizei, num_attributes> num_components{
      static_cast<GLsizei>(tatooine::tensor_num_components<Ts>)...};
  static constexpr std::array<GLenum, num_attributes> types{
      value_type_v<Ts>...};
  static constexpr std::array<std::size_t, num_attributes> offsets =
      attr_offset<num_attributes, Ts...>::gen(0, 0);

  //----------------------------------------------------------------------------
  vertexbuffer(buffer_usage usage = default_usage) : parent_type{usage} {}
  vertexbuffer(vertexbuffer const& other) : parent_type{other} {}
  vertexbuffer(vertexbuffer&& other) noexcept : parent_type{other} {}

  auto operator=(this_type const& other) -> auto& {
    parent_type::operator=(other);
    return *this;
  }

  auto operator=(this_type&& other) noexcept -> auto& {
    parent_type::operator=(std::move(other));
    return *this;
  }

  vertexbuffer(std::size_t n, buffer_usage usage = default_usage)
      : parent_type(n, usage) {}
  vertexbuffer(std::size_t n, value_type const& initial,
               buffer_usage usage = default_usage)
      : parent_type(n, initial, usage) {}
  vertexbuffer(std::vector<value_type> const& data, buffer_usage usage = default_usage)
      : parent_type(data, usage) {}
  vertexbuffer(std::initializer_list<value_type>&& list)
      : parent_type(std::move(list), default_usage) {}

  void push_back(Ts const&... ts) {
    if constexpr (num_attributes == 1) {
      parent_type::push_back(ts...);
    } else {
      parent_type::push_back(tuple{ts...});
    }
  }

  //============================================================================
  static constexpr void activate_attributes() {
    for (unsigned int i = 0; i < num_attributes; i++) {
      gl::enable_vertex_attrib_array(i);
      gl::vertex_attrib_pointer(i, num_components[i], types[i], GL_FALSE,
                                data_size, (void*)offsets[i]);
    }
  }
  //----------------------------------------------------------------------------
 private:
  template <std::convertible_to<GLboolean>... Normalized, GLsizei... Is>
  static constexpr auto activate_attributes(
      std::integer_sequence<GLsizei, Is...>, Normalized... normalized) -> void {
    static_assert(sizeof...(Normalized) == sizeof...(Is));
    static_assert(sizeof...(Normalized) == num_attributes);
    (
        [&](GLsizei i, GLboolean normalized) {
          gl::enable_vertex_attrib_array(i);
          gl::vertex_attrib_pointer(i, num_components[i], types[i], normalized,
                                    data_size, (void*)offsets[i]);
        }(Is, static_cast<GLboolean>(normalized)),
        ...);
  }
  //----------------------------------------------------------------------------
 public:
  template <std::convertible_to<GLboolean>... Normalized>
  static constexpr void activate_attributes(Normalized... normalized) {
    activate_attributes(std::make_integer_sequence<GLsizei, num_attributes>{},
                        normalized...);
  }
  //----------------------------------------------------------------------------
  static constexpr void deactivate_attributes() {
    for (unsigned int i = 0; i < num_attributes; i++)
      gl::disable_vertex_attrib_array(i);
  }
};

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================

#endif
