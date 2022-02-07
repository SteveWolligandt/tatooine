#ifndef TATOOINE_GL_VERTEXBUFFER_H
#define TATOOINE_GL_VERTEXBUFFER_H
//==============================================================================
#include <tatooine/gl/buffer.h>
#include <tatooine/gl/errorcheck.h>
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/utility.h>
#include <tatooine/gl/vbohelpers.h>
#include <tatooine/num_components.h>
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
  using this_t = vertexbuffer<Ts...>;
  using data_t = typename parent_type::data_t;

  static constexpr auto data_size = parent_type::data_size;

  static usage_t const default_usage = usage_t::STATIC_DRAW;

  static constexpr auto num_attributes = sizeof...(Ts);
  static constexpr std::array<std::size_t, num_attributes> num_components{
      tatooine::num_components<Ts>...};
  static constexpr std::array<GLenum, num_attributes> types{
      value_type_v<Ts>...};
  static constexpr std::array<std::size_t, num_attributes> offsets =
      attr_offset<num_attributes, Ts...>::gen(0, 0);

  //----------------------------------------------------------------------------
  vertexbuffer(usage_t usage = default_usage) : parent_type{usage} {}
  vertexbuffer(vertexbuffer const& other) : parent_type{other} {}
  vertexbuffer(vertexbuffer&& other) noexcept : parent_type{other} {}

  auto operator=(this_t const& other) -> auto& {
    parent_type::operator=(other);
    return *this;
  }

  auto operator=(this_t&& other) noexcept -> auto& {
    parent_type::operator=(std::move(other));
    return *this;
  }

  vertexbuffer(std::size_t n, usage_t usage = default_usage)
      : parent_type(n, usage) {}
  vertexbuffer(std::size_t n, data_t const& initial,
               usage_t usage = default_usage)
      : parent_type(n, initial, usage) {}
  vertexbuffer(std::vector<data_t> const& data, usage_t usage = default_usage)
      : parent_type(data, usage) {}
  vertexbuffer(std::initializer_list<data_t>&& list)
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
  template <typename... Normalized, std::size_t... Is>
  static constexpr void activate_attributes(std::index_sequence<Is...>,
                                            Normalized... normalized) {
    static_assert(sizeof...(Normalized) == sizeof...(Is));
    static_assert(sizeof...(Normalized) == num_attributes);
    // static_assert((std::is_same_v<GLboolean, std::decay_t<Normalized>> &&
    // ...));
    (
        [&](auto i, auto normalized) {
          gl::enable_vertex_attrib_array(i);
          gl::vertex_attrib_pointer(i, num_components[i], types[i], normalized,
                                    data_size, (void*)offsets[i]);
        }(Is, normalized),
        ...);
  }
  //----------------------------------------------------------------------------
 public:
  template <typename... Normalized>
  static constexpr void activate_attributes(Normalized... normalized) {
    activate_attributes(std::make_index_sequence<num_attributes>{},
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
