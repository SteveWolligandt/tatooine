#ifndef TATOOINE_GL_TEXTURE_H
#define TATOOINE_GL_TEXTURE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/gl/errorcheck.h>
#include <tatooine/vec.h>
#include <tatooine/gl/glfunctions.h>
#include <tatooine/gl/glwrapper.h>
#include <tatooine/gl/idholder.h>
#include <tatooine/gl/pixelunpackbuffer.h>
#include <tatooine/gl/shader.h>
#include <tatooine/gl/texcomponents.h>
#include <tatooine/gl/texpng.h>
#include <tatooine/gl/texsettings.h>
#include <tatooine/gl/textarget.h>
#include <tatooine/gl/type.h>

#include <iostream>
#include <type_traits>
#include <utility>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
template <typename ValueType>
concept texture_value =
    arithmetic<ValueType> || same_as<ValueType, tex::depth24>;
//==============================================================================
enum class wrap_mode : GLint {
  clamp_to_border = GL_CLAMP_TO_BORDER,
  clamp_to_edge   = GL_CLAMP_TO_EDGE,
  repeat          = GL_REPEAT,
  mirrored_repeat = GL_MIRRORED_REPEAT
};
//==============================================================================
enum class interpolation_mode : GLint {
  nearest                = GL_NEAREST,
  linear                 = GL_LINEAR,
  nearest_mipmap_nearest = GL_NEAREST_MIPMAP_NEAREST,
  linear_mipmap_nearest  = GL_LINEAR_MIPMAP_NEAREST,
  nearest_mipmap_linear  = GL_NEAREST_MIPMAP_LINEAR,
  linear_mipmap_linear   = GL_LINEAR_MIPMAP_LINEAR
};
//==============================================================================
enum class compare_func : GLint {
  never    = GL_NEVER,
  less     = GL_LESS,
  lequal   = GL_LEQUAL,
  greater  = GL_GREATER,
  notequal = GL_NOTEQUAL,
  gequal   = GL_GEQUAL,
  always   = GL_ALWAYS
};
//==============================================================================
enum class compare_mode : GLint {
  compare_r_to_texture = GL_COMPARE_R_TO_TEXTURE,
  none                 = GL_NONE
};
//==============================================================================
template <std::size_t NumDimensions, texture_value ValueType,
          texture_component Components>
class texture : public id_holder<GLuint> {
  static_assert(NumDimensions >= 1 && NumDimensions <= 3,
                "number of dimensions must be between 1 and 3");

 public:
  //============================================================================
  using value_type                            = ValueType;
  using components_type                       = Components;
  static constexpr auto target                = tex::target<NumDimensions>;
  static constexpr auto binding               = tex::binding<NumDimensions>;
  static constexpr auto default_interpolation = interpolation_mode::linear;
  static constexpr auto default_wrap_mode     = wrap_mode::repeat;
  static constexpr auto num_components() {
    return components_type::num_components;
  }
  static constexpr auto num_dimensions() { return NumDimensions; }

  static constexpr auto gl_internal_format =
      tex::settings<ValueType, Components>::internal_format;
  static constexpr auto gl_format =
      tex::settings<ValueType, Components>::format;
  static constexpr auto gl_type = tex::settings<ValueType, Components>::type;
  static constexpr std::array<GLenum, 3> wrapmode_indices{
      GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R};

  static constexpr bool is_readable_from_png =
      (NumDimensions == 2 && is_same<Components, R>) ||
      (NumDimensions == 2 && is_same<Components, RGB>) ||
      (NumDimensions == 2 && is_same<Components, RGBA>) ||
      (NumDimensions == 2 && is_same<Components, BGR>) ||
      (NumDimensions == 2 && is_same<Components, BGRA>);
  static constexpr bool is_writable_to_png =
      (NumDimensions == 2 && is_same<Components, R>) ||
      (NumDimensions == 2 && is_same<Components, RG>) ||
      (NumDimensions == 2 && is_same<Components, RGB>) ||
      (NumDimensions == 2 && is_same<Components, RGBA>) ||
      (NumDimensions == 2 && is_same<Components, BGR>) ||
      (NumDimensions == 2 && is_same<Components, BGRA>);

 protected:
  //============================================================================
  std::array<std::size_t, NumDimensions> m_size;

 public:
  //============================================================================
  texture() : texture{std::make_index_sequence<NumDimensions>{}} {}
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  template <std::size_t... Is>
  texture(std::index_sequence<Is...>) : m_size{((void)Is, 0)...} {
    create_id();
    set_wrap_mode(default_wrap_mode);
    set_interpolation_mode(default_interpolation);
  }
  //----------------------------------------------------------------------------
  /// TODO: copy wrap and interpolation modes
  texture(texture const& other) : texture{} { copy_data(other); }
  //----------------------------------------------------------------------------
  texture(texture&& other)
      : id_holder{std::move(other)}, m_size{std::move(other.m_size)} {}
  //----------------------------------------------------------------------------
  auto& operator=(texture const& other) {
    copy_data(other);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(texture&& other) {
    id_holder::operator=(std::move(other));
    m_size             = std::move(other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
  ~texture() {
    if constexpr (NumDimensions == 1) {
      if (bound_texture1d() == id()) {
        unbind();
      }
    } else if constexpr (NumDimensions == 2) {
      if (bound_texture2d() == id()) {
        unbind();
      }
    } else if constexpr (NumDimensions == 3) {
      if (bound_texture3d() == id()) {
        unbind();
      }
    }
    if (id()) {
      gl::delete_textures(1, &id_ref());
    }
  }
  //----------------------------------------------------------------------------
  texture(integral auto const... sizes)
  requires(sizeof...(sizes) == NumDimensions)
      : m_size{static_cast<std::size_t>(sizes)...} {
    create_id();
    set_wrap_mode(default_wrap_mode);
    set_interpolation_mode(default_interpolation);
    resize(sizes...);
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
      texture(OtherType const* const data, integral auto const... sizes)
  requires(sizeof...(sizes) == NumDimensions) 
      : m_size{static_cast<std::size_t>(sizes)...} {
    upload_data(
        std::vector<ValueType>(data, data + num_components() * (sizes * ...)));
  }
  //----------------------------------------------------------------------------
  texture(ValueType const* const data, integral auto const... sizes)
  requires(sizeof...(sizes) == NumDimensions) 
      : m_size{static_cast<std::size_t>(sizes)...} {
    create_id();
    set_interpolation_mode(default_interpolation);
    set_wrap_mode(default_wrap_mode);
    m_size = std::array<std::size_t, NumDimensions>{
        static_cast<std::size_t>(sizes)...};
    upload_data(data);
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  texture(std::vector<OtherType> const& data, integral auto const... sizes)
  requires (sizeof...(sizes) == NumDimensions) &&
           (num_components() == 1)
      : m_size{static_cast<std::size_t>(sizes)...} {
    create_id();
    set_interpolation_mode(default_interpolation);
    set_wrap_mode(default_wrap_mode);
    upload_data(data.data());
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  texture(std::vector<vec<OtherType, num_components()>> const& data,
          integral auto const... sizes)
  requires (sizeof...(sizes) == NumDimensions) &&
           (num_components() > 1)
      : m_size{static_cast<std::size_t>(sizes)...} {
    create_id();
    set_interpolation_mode(default_interpolation);
    set_wrap_mode(default_wrap_mode);
    upload_data(data.data());
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  requires(num_components() == 1)
  texture(dynamic_multidim_array<OtherType> const& data) : texture{} {
    auto const converted_data = std::vector<value_type>(
        begin(data.internal_container()), end(data.internal_container()));
    std::ranges::copy(data.size(), begin(m_size));
    upload_data(converted_data.data());
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  requires(num_components() > 1)
  texture(dynamic_multidim_array<vec<OtherType, num_components()>> const& data)
    : texture{} {
    auto const converted_data = std::vector<vec<value_type, num_components()>>(
        begin(data.internal_container()), end(data.internal_container()));
    std::ranges::copy(data.size(), begin(m_size));
    upload_data(converted_data.data());
  }
  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  texture(interpolation_mode interp_mode, wrap_mode wrap_mode,
          std::index_sequence<Is...>)
      : m_size{((void)Is, 0)...} {
    create_id();
    set_interpolation_mode(interp_mode);
    set_wrap_mode(wrap_mode);
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  texture(interpolation_mode interp_mode, wrap_mode wrap_mode)
      : texture{interp_mode, wrap_mode,
                std::make_index_sequence<NumDimensions>{}} {}

  //----------------------------------------------------------------------------
  template <integral... Sizes>
  requires(sizeof...(Sizes) == NumDimensions) 
      texture(interpolation_mode interp_mode, wrap_mode wrap_mode,
              Sizes const... sizes)
      : m_size{sizes...} {
    create_id();
    set_interpolation_mode(interp_mode);
    set_wrap_mode(wrap_mode);
    resize(sizes...);
  }
  //----------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType, integral... Sizes>
  requires(sizeof...(Sizes) == NumDimensions)
      texture(interpolation_mode interp_mode, wrap_mode wrap_mode,
              std::vector<OtherType> const& data, Sizes const... sizes)
      : m_size{sizes...} {
    create_id();
    set_interpolation_mode(interp_mode);
    set_wrap_mode(wrap_mode);
    upload_data(data);
  }
  //----------------------------------------------------------------------------
  texture(std::string const& filepath) : texture{} { read(filepath); }
  //----------------------------------------------------------------------------
  texture(interpolation_mode interp_mode, wrap_mode wrap_mode,
          std::string const& filepath)
      : texture{interp_mode, wrap_mode} {
    read(filepath);
  }

 private:
  //----------------------------------------------------------------------------
  void create_id() { gl::create_textures(target, 1, &id_ref()); }

 public:
  //----------------------------------------------------------------------------
  auto bind(GLuint unit = 0) const {
    assert(unit < GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
    gl::active_texture(GL_TEXTURE0 + unit);
    auto last_tex = bound_texture(binding);
    gl::bind_texture(target, id());
    return last_tex;
  }

  //----------------------------------------------------------------------------
  static void unbind(GLuint unit = 0) {
    gl::active_texture(GL_TEXTURE0 + unit);
    gl::bind_texture(target, 0);
  }

  //------------------------------------------------------------------------------
  void bind_image_texture(GLuint unit) const {
    gl::bind_image_texture(unit, id(), 0, GL_FALSE, 0, GL_READ_ONLY,
                           gl_internal_format);
  }

  //------------------------------------------------------------------------------
  void bind_image_texture(GLuint unit) {
    gl::bind_image_texture(unit, id(), 0, GL_FALSE, 0, GL_READ_WRITE,
                           gl_internal_format);
  }
  //------------------------------------------------------------------------------
  void bind_image_texture_read_write(GLuint unit) { bind_image_texture(unit); }

  //------------------------------------------------------------------------------
  void bind_image_texture_write(GLuint unit) {
    gl::bind_image_texture(unit, id(), 0, GL_FALSE, 0, GL_WRITE_ONLY,
                           gl_internal_format);
  }

  //------------------------------------------------------------------------------
  void unbind_image_texture(GLuint unit) {
    gl::bind_image_texture(unit, 0, 0, GL_FALSE, 0, GL_READ_WRITE,
                           gl_internal_format);
  }

  //------------------------------------------------------------------------------

  void bind_image_texture_layer(GLuint unit, GLint layer) const
      requires(NumDimensions == 3) {
    gl::bind_image_texture(unit, id(), 0, GL_TRUE, layer, GL_READ_ONLY,
                           gl_internal_format);
  }

  //------------------------------------------------------------------------------

  void bind_image_texture_layer(GLuint unit,
                                GLint  layer) requires(NumDimensions == 3) {
    gl::bind_image_texture(unit, id(), 0, GL_TRUE, layer, GL_READ_WRITE,
                           gl_internal_format);
  }

  //------------------------------------------------------------------------------

  static void unbind_image_texture_layer(GLuint unit,
                                         GLint  layer) requires(NumDimensions ==
                                                               3) {
    gl::bind_image_texture(unit, 0, 0, GL_TRUE, layer, GL_READ_WRITE,
                           gl_internal_format);
  }

  //----------------------------------------------------------------------------
  template <std::size_t... Is>
  std::size_t num_texels(std::index_sequence<Is...>) const {
    return (m_size[Is] * ...);
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  std::size_t num_texels() const {
    return num_texels(std::make_index_sequence<NumDimensions>{});
  }

  //----------------------------------------------------------------------------
  void copy_data(texture const& other) {
    resize(other.m_size);
    if constexpr (NumDimensions == 1) {
      gl::copy_image_sub_data(other.id(), target, 0, 0, 0, 0, id(), target, 0,
                              0, 0, 0, m_size[0], 1, 1);
    } else if (NumDimensions == 2) {
      gl::copy_image_sub_data(other.id(), target, 0, 0, 0, 0, id(), target, 0,
                              0, 0, 0, m_size[0], m_size[1], 1);

    } else {
      gl::copy_image_sub_data(other.id(), target, 0, 0, 0, 0, id(), target, 0,
                              0, 0, 0, m_size[0], m_size[1], m_size[2]);
    }
  }
  //------------------------------------------------------------------------------
  template <integral Size>
  auto resize(std::array<Size, NumDimensions> const& size) -> void {
    auto last_tex = bind();
    if constexpr (is_same<std::size_t, Size>) {
      m_size = size;
    } else {
      m_size = std::array<std::size_t, NumDimensions>{begin(size), end(size)};
    }
    if constexpr (NumDimensions == 1) {
      gl::tex_image_1d(target, 0, gl_internal_format, width(), 0, gl_format,
                       gl_type, nullptr);
    } else if constexpr (NumDimensions == 2) {
      gl::tex_image_2d(target, 0, gl_internal_format, width(), height(), 0,
                       gl_format, gl_type, nullptr);
    } else if constexpr (NumDimensions == 3) {
      gl::tex_image_3d(target, 0, gl_internal_format, width(), height(),
                       depth(), 0, gl_format, gl_type, nullptr);
    }
    if (last_tex > 0) {
      gl::bind_texture(target, last_tex);
    }
  }
  //------------------------------------------------------------------------------
  template <integral... Sizes>
  requires(sizeof...(Sizes) == NumDimensions)
  auto resize(Sizes const... sizes)
      -> void {
    m_size        = std::array{static_cast<std::size_t>(sizes)...};
    auto last_tex = bind();
    if constexpr (NumDimensions == 1) {
      gl::tex_image_1d(target, 0, gl_internal_format, width(), 0, gl_format,
                       gl_type, nullptr);
    } else if constexpr (NumDimensions == 2) {
      gl::tex_image_2d(target, 0, gl_internal_format, width(), height(), 0,
                       gl_format, gl_type, nullptr);
    } else if constexpr (NumDimensions == 3) {
      gl::tex_image_3d(target, 0, gl_internal_format, width(), height(),
                       depth(), 0, gl_format, gl_type, nullptr);
    }
    if (last_tex > 0) {
      gl::bind_texture(target, last_tex);
    }
  }
  //------------------------------------------------------------------------------
  /// \{
 private:
  auto upload_data(value_type const* const data) -> void {
    auto last_tex = bind();
    if constexpr (NumDimensions == 1) {
      gl::tex_image_1d(target, 0, gl_internal_format, width(), 0, gl_format,
                       gl_type, data);
    } else if constexpr (NumDimensions == 2) {
      gl::tex_image_2d(target, 0, gl_internal_format, width(), height(), 0,
                       gl_format, gl_type, data);
    } else if constexpr (NumDimensions == 3) {
      gl::tex_image_3d(target, 0, gl_internal_format, width(), height(),
                       depth(), 0, gl_format, gl_type, data);
    }
    if (last_tex > 0) {
      gl::bind_texture(target, last_tex);
    }
  }
  //------------------------------------------------------------------------------
  auto upload_data(vec<value_type, num_components()> const* data) -> void
  requires (num_components() > 1) {
    auto last_tex = bind();
    if constexpr (NumDimensions == 1) {
      gl::tex_image_1d(target, 0, gl_internal_format, width(), 0, gl_format,
                       gl_type, data[0].data());
    } else if constexpr (NumDimensions == 2) {
      gl::tex_image_2d(target, 0, gl_internal_format, width(), height(), 0,
                       gl_format, gl_type, data[0].data());
    } else if constexpr (NumDimensions == 3) {
      gl::tex_image_3d(target, 0, gl_internal_format, width(), height(),
                       depth(), 0, gl_format, gl_type, data[0].data());
    }
    if (last_tex > 0) {
      gl::bind_texture(target, last_tex);
    }
  }

 public:
  //------------------------------------------------------------------------------
  template <integral... Sizes>
  requires (sizeof...(Sizes) == NumDimensions)
  auto upload_data(value_type const* const data, Sizes const... sizes) {
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    upload_data(data);
  }
  //------------------------------------------------------------------------------
  template <integral... Sizes>
  requires (sizeof...(Sizes) == NumDimensions) &&
           (num_components() > 1)
  auto upload_data(vec<value_type, num_components()> const* const data,
                   Sizes const... sizes) {
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  auto upload_data(dynamic_multidim_array<value_type> const& data)
  requires (num_components() == 1) {
    std::ranges::copy(data.size(), begin(m_size));
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  auto upload_data(
      dynamic_multidim_array<vec<value_type, num_components()>> const& data)
  requires(num_components() > 1) {
    std::ranges::copy(data.size(), begin(m_size));
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  requires (num_components() == 1)
  auto upload_data(
      dynamic_multidim_array<OtherType> const& data) {
    std::ranges::copy(data.size(), begin(m_size));
    std::vector<value_type>{begin(data.internal_container()),
                            end(data.internal_container())};
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType>
  requires (num_components() > 1)
  auto upload_data(
      dynamic_multidim_array<vec<OtherType, num_components()>> const& data) {
    std::ranges::copy(data.size(), begin(m_size));
    auto const converted_data = std::vector<value_type>{
        begin(data.internal_container()), end(data.internal_container())};
    upload_data(converted_data.data());
  }
  //------------------------------------------------------------------------------
  template <integral... Sizes>
  requires (num_components() == 1) &&
           (sizeof...(Sizes) == NumDimensions)
  auto upload_data(std::vector<value_type> const& data,
                   Sizes const... sizes) {
    assert((sizes * ...) = size(data));
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  template <integral... Sizes>
  requires (num_components() > 1) &&
           (sizeof...(Sizes) == NumDimensions)
  auto upload_data(std::vector<vec<value_type, num_components()>> const& data,
                   Sizes const... sizes) {
    assert((sizes * ...) = size(data));
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType, integral... Sizes>
  requires (num_components() == 1) &&
           (sizeof...(Sizes) == NumDimensions)
  auto upload_data(std::vector<OtherType> const& data, Sizes const... sizes) {
    assert((sizes * ...) = size(data));
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    auto converted_data = std::vector<value_type>(begin(data), end(data));
    upload_data(converted_data.data());
  }
  //------------------------------------------------------------------------------
  template <convertible_to<value_type> OtherType, integral... Sizes>
  requires (num_components() > 1) &&
           (sizeof...(Sizes) == NumDimensions)
  auto upload_data(std::vector<vec<OtherType, num_components()>> const& data,
                   Sizes const... sizes) {
    assert((sizes * ...) = size(data));
    m_size = std::array{static_cast<std::size_t>(sizes)...};
    auto converted_data =
        std::vector<vec<value_type, num_components()>>(begin(data), end(data));
    upload_data(converted_data.data());
  }
  /// \}
  //------------------------------------------------------------------------------
  /// \{
  auto download_data() const {
    if constexpr (num_components() == 1) {
      auto data = dynamic_multidim_array<ValueType>{m_size};
      gl::get_texture_image(
          id(), 0, gl_format, gl_type,
          num_components() * num_texels() * sizeof(value_type),
          data.data());
      return data;
    } else {
      auto data =
          dynamic_multidim_array<vec<ValueType, num_components()>>{m_size};
      gl::get_texture_image(
          id(), 0, gl_format, gl_type,
          num_components() * num_texels() * sizeof(value_type),
          data.data());
      return data;
    }
  }
  //------------------------------------------------------------------------------
  auto download_data(std::vector<value_type>& data) const -> void {
    auto const n = num_components() * num_texels();
    if (data.size() != n) {
      data.resize(n);
    }
    download_data(data.data());
  }
  //------------------------------------------------------------------------------
  auto download_data(std::vector<vec<value_type, num_components()>>& data) const
      -> void
  requires(num_components() > 1) {
    auto const n = num_texels();
    if (data.size() != n) {
      data.resize(n);
    }
    download_data(data.front().data());
  }
  //------------------------------------------------------------------------------
  auto download_data(value_type* data) const -> void {
    gl::get_texture_image(id(), 0, gl_format, gl_type,
                          num_texels() * num_components() * sizeof(value_type),
                          data);
  }
  //------------------------------------------------------------------------------
  auto& download_sub_data(GLint xoffset, GLsizei width,
                          std::vector<ValueType>& data, GLint level = 0) const
  requires (num_components() == 1) &&
           (num_dimensions() == 1) {
    if (data.size() != width * num_components()) {
      data.resize(width * num_components());
    }
    gl::get_texture_sub_image(id(), level, xoffset, 0, 0, width, 1, 1,
                              gl_format, gl_type,
                              data.size() * sizeof(value_type), data.data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& download_sub_data(GLint xoffset, GLsizei width,
                          std::vector<vec<ValueType, num_components()>>& data,
                          GLint level = 0) const
  requires (num_components() > 1) &&
           (num_dimensions() == 1) {
    if (data.size() != static_cast<std::size_t>(width)) {
      data.resize(width * num_components());
    }
    gl::get_texture_sub_image(
        id(), level, xoffset, 0, 0, width, 1, 1, gl_format, gl_type,
        data.size() * num_components() * sizeof(value_type),
        data.front().data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto download_sub_data(GLint xoffset, GLsizei width, GLint level = 0) const
  requires (NumDimensions == 1) {
    if constexpr (num_components() == 1) {
      auto data = std::vector<ValueType>(width);
      gl::get_texture_sub_image(id(), level, xoffset, 0, 0, width, 1, 1,
                                gl_format, gl_type,
                                data.size() * sizeof(value_type), data.data());
      return data;
    } else {
      auto data = std::vector<vec<ValueType, num_components()>>(width);
      gl::get_texture_sub_image(
          id(), level, xoffset, 0, 0, width, 1, 1, gl_format, gl_type,
          data.size() * num_components() * sizeof(value_type), data.front().data());
      return data;
    }
  }
  //------------------------------------------------------------------------------
  auto& download_sub_data(GLint xoffset, GLint yoffset, GLsizei width,
                          GLsizei height, std::vector<ValueType>& data,
                          GLint level = 0) const
  requires (num_components() == 1) &&
           (NumDimensions == 2) {
    if (data.size() != static_cast<std::size_t>(width * height)) {
      data.resize(width * height);
    }
    gl::get_texture_sub_image(id(), level, xoffset, yoffset, 0, width, height,
                              1, gl_format, gl_type,
                              data.size() * sizeof(value_type), data.data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& download_sub_data(GLint xoffset, GLint yoffset, GLsizei width,
                          GLsizei                                        height,
                          std::vector<vec<ValueType, num_components()>>& data,
                          GLint level = 0) const
  requires (num_components() > 1) &&
           (NumDimensions == 2) {
    if (data.size() != static_cast<std::size_t>(width * height)) {
      data.resize(width * height);
    }
    gl::get_texture_sub_image(
        id(), level, xoffset, yoffset, 0, width, height, 1, gl_format, gl_type,
        data.size() * num_components() * sizeof(value_type), data.front().data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto download_sub_data(GLint xoffset, GLint yoffset, GLsizei width,
                         GLsizei height, GLint level = 0) const
  requires (NumDimensions == 2) {
    if constexpr (num_components() == 1) {
      auto data = dynamic_multidim_array<ValueType>{width, height};
      gl::get_texture_sub_image(id(), level, xoffset, yoffset, 0, width, height,
                                1, gl_format, gl_type,
                                data.num_components() * sizeof(value_type), data.data());
      return data;
    } else {
      auto data = dynamic_multidim_array<vec<ValueType, num_components()>>{width, height};
      gl::get_texture_sub_image(
          id(), level, xoffset, yoffset, 0, width, height, 1, gl_format,
          gl_type,
          data.num_components() * num_components() * sizeof(value_type),
          data.data());
      return data;
    }
  }
  //------------------------------------------------------------------------------
  auto& download_sub_data(GLint xoffset, GLint yoffset, GLint zoffset,
                          GLsizei width, GLsizei height, GLsizei depth,
                          std::vector<ValueType>& data, GLint level = 0) const
  requires (num_components() == 1) && 
           (NumDimensions == 3) {
    if (data.size() != static_cast<std::size_t>(width * height * depth)) {
      data.resize(width * height * depth);
    }
    gl::get_texture_sub_image(id(), level, xoffset, yoffset, zoffset, width,
                              height, depth, gl_format, gl_type,
                              data.size() * sizeof(value_type), data.data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& download_sub_data(GLint xoffset, GLint yoffset, GLint zoffset,
                          GLsizei width, GLsizei height, GLsizei depth,
                          std::vector<vec<ValueType, num_components()>>& data,
                          GLint level = 0) const
  requires (num_components() > 1) &&
           (NumDimensions == 3) {
    if (data.size() != static_cast<std::size_t>(width * height * depth)) {
      data.resize(width * height * depth);
    }
    gl::get_texture_sub_image(
        id(), level, xoffset, yoffset, zoffset, width, height, depth, gl_format,
        gl_type, data.size() * num_components() * sizeof(value_type),
        data.front().data());
    return data;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto download_sub_data(GLint xoffset, GLint yoffset, GLint zoffset,
                         GLsizei width, GLsizei height, GLsizei depth,
                         GLint level = 0) const
  requires(NumDimensions == 3) {
    if constexpr (num_components() == 1) {
      auto data = dynamic_multidim_array<ValueType>{width, height, depth};
      gl::get_texture_sub_image(id(), level, xoffset, yoffset, zoffset, width,
                                height, depth, gl_format, gl_type,
                                data.num_components() * sizeof(value_type), data.data());
      return data;
    } else {
      auto data = dynamic_multidim_array<vec<ValueType, num_components()>>{
          width, height, depth};
      gl::get_texture_sub_image(id(), level, xoffset, yoffset, zoffset, width,
                                height, depth, gl_format, gl_type,
                                data.num_components() * sizeof(value_type),
                                data.internal_container().front().data());
      return data;
    }
  }
  /// \}
  //----------------------------------------------------------------------------
  template <integral... Indices>
  requires(sizeof...(Indices) == NumDimensions)
  auto operator()(Indices const... indices) const {
    return download_sub_data(indices..., ((void)indices, 1)..., 0).front();
  }
  //----------------------------------------------------------------------------
  auto width() const { return m_size[0]; }
  //----------------------------------------------------------------------------
  auto height() const requires(NumDimensions > 1) { return m_size[1]; }
  //----------------------------------------------------------------------------
  auto depth() const requires(NumDimensions > 2) { return m_size[2]; }
  //----------------------------------------------------------------------------
  /// setting all wrapmodes to same mode
  auto set_wrap_mode(wrap_mode mode) -> void {
    set_wrap_mode_s(mode);
    if constexpr (NumDimensions > 1) {
      set_wrap_mode_t(mode);
    }
    if constexpr (NumDimensions > 2) {
      set_wrap_mode_r(mode);
    }
  }

  //----------------------------------------------------------------------------
  /// setting all wrapmodes individual modes
  template <std::size_t... Is>
  auto set_wrap_mode(std::index_sequence<Is...>,
                     same_as<wrap_mode> auto const... modes) -> void
  requires(sizeof...(modes) == NumDimensions) {
    using discard = int[];
    (void)discard{((void)set_wrap_mode(Is, modes), 0)...};
  }
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  /// setting all wrapmodes individual modes
  auto set_wrap_mode(same_as<wrap_mode> auto const... modes) -> void
  requires (sizeof...(modes) == NumDimensions) {
    set_wrap_mode(std::make_index_sequence<NumDimensions>{}, modes...);
  }
  //----------------------------------------------------------------------------
  /// setting wrapmode with index i to modes
  auto set_wrap_mode(std::size_t const i, wrap_mode const mode) -> void{
    gl::texture_parameter_i(id(), wrapmode_indices[i],
                            static_cast<GLint>(mode));
  }
  //----------------------------------------------------------------------------
  /// setting all wrapmodes to repeat
  auto set_repeat() -> void { set_wrap_mode(wrap_mode::repeat); }
  //----------------------------------------------------------------------------
  auto set_wrap_mode_s(wrap_mode const mode) -> void {
    set_wrap_mode(0, mode);
  }
  //----------------------------------------------------------------------------
  auto set_wrap_mode_t(wrap_mode const mode) -> void
  requires(NumDimensions > 1) {
    set_wrap_mode(1, mode);
  }
  //----------------------------------------------------------------------------
  auto set_wrap_mode_r(wrap_mode const mode) -> void
  requires(NumDimensions > 2) {
    set_wrap_mode(2, mode);
  }
  //------------------------------------------------------------------------------
  auto set_interpolation_mode(interpolation_mode const mode) -> void{
    set_interpolation_mode_min(mode);
    set_interpolation_mode_mag(mode);
  }
  //------------------------------------------------------------------------------
  auto set_interpolation_mode_min(interpolation_mode const mode) -> void {
    gl::texture_parameter_i(id(), GL_TEXTURE_MIN_FILTER, static_cast<GLint>(mode));
  }
  //------------------------------------------------------------------------------
  auto set_interpolation_mode_mag(interpolation_mode const mode) -> void {
    gl::texture_parameter_i(id(), GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mode));
  }
  //----------------------------------------------------------------------------
  auto set_compare_func(compare_func const f) -> void
  requires is_same<components_type, Depth> {
    gl::texture_parameter_i(id(), GL_TEXTURE_COMPARE_FUNC, static_cast<GLint>(f));
  }
  //----------------------------------------------------------------------------
  auto set_compare_mode(compare_mode const m) -> void
  requires is_same<components_type, Depth> {
    gl::texture_parameter_i(id(), GL_TEXTURE_COMPARE_MODE, static_cast<GLint>(m));
  }
  //----------------------------------------------------------------------------
  auto clear(arithmetic auto const... comps)
  requires(sizeof...(comps) == num_components()) {
    clear(std::array<value_type, num_components()>{
        static_cast<value_type>(comps)...});
  }
  //------------------------------------------------------------------------------
  auto clear(std::array<value_type, num_components()> const& col) {
    gl::clear_tex_image(id(), 0, gl_format, gl_type, col.data());
  }
  //------------------------------------------------------------------------------
  auto set_data(pixelunpackbuffer<value_type> const& pbo)
  requires(NumDimensions == 2) {
    pbo.bind();
    auto last_tex = bind();
    gl::tex_sub_image_2d(GL_TEXTURE_2D, 0, 0, 0, width(), height(), gl_format,
                         gl_type, 0);
    if (last_tex > 0) {
      gl::bind_texture(target, last_tex);
    }
  }
  //------------------------------------------------------------------------------
  auto read(std::string const& filepath) {
    auto ext = filepath.substr(filepath.find_last_of('.') + 1);
    if constexpr (NumDimensions == 2 && is_readable_from_png) {
      if (ext == "png") {
        read_png(filepath);
        return;
      }
    }

    throw std::runtime_error("could not read fileformat ." + ext);
  }
  //------------------------------------------------------------------------------
  auto write(std::string const filepath) const {
    auto ext = filepath.substr(filepath.find_last_of('.') + 1);
    if constexpr (NumDimensions == 2 && is_writable_to_png) {
      if constexpr (has_png_support()) {
        if (ext == "png") {
          write_png(filepath);
          return;
        }
      }
    }
    throw std::runtime_error("could not write fileformat ." + ext);
  }
  //----------------------------------------------------------------------------
  auto read_png(std::string const& filepath) -> void
  requires (has_png_support()) &&
           (NumDimensions == 2) &&
           is_readable_from_png {
    using tex_png_t = tex_png<value_type, components_type>;
    auto image = typename tex_png_t::png_t {};
    image.read(filepath);
    m_size[0] = image.get_width();
    m_size[1] = image.get_height();

    auto data = [this]() {
      if constexpr (num_components() == 1) {
        auto data = std::vector<value_type>{};
        data.reserve(num_texels());
        return data;
      } else {
        auto data = std::vector<vec<value_type, num_components()>>{};
        data.reserve(num_texels());
        return data;
      }
    }();
    for (png::uint_32 y = 0; y < height(); ++y) {
      for (png::uint_32 x = 0; x < width(); ++x) {
        tex_png_t::load_pixel(data, image, x, y);
      }
    }
    if constexpr (is_floating_point<value_type>) {
      auto normalize = [](auto d) { return d / 255.0F; };
      std::ranges::transform(data, begin(data), normalize);
    }

    upload_data(data.data());
  }
  //------------------------------------------------------------------------------
  auto write_png(std::string const& filepath) const
  requires (has_png_support()) &&
           (NumDimensions == 2) &&
           is_writable_to_png {
    using tex_png_t = tex_png<value_type, components_type>;
    auto image = typename tex_png_t::png_t{static_cast<png::uint_32>(width()),
                                           static_cast<png::uint_32>(height())};
    auto data       = download_data();

    for (png::uint_32 y = 0; y < image.get_height(); ++y){
      for (png::uint_32 x = 0; x < image.get_width(); ++x) {
        auto const idx = x + width() * y;
        tex_png_t::save_pixel(data.internal_container(), image, x, y, idx);
      }
    }
    image.write(filepath);
  }
};
//==============================================================================
template <texture_value ValueType, texture_component Components>
using tex1 = texture<1, ValueType, Components>;
template <texture_value ValueType, texture_component Components>
using tex2 = texture<2, ValueType, Components>;
template <texture_value ValueType, texture_component Components>
using tex3 = texture<3, ValueType, Components>;

template <texture_value ValueType>
using tex1r = tex1<ValueType, R>;
template <texture_value ValueType>
using tex1rg = tex1<ValueType, RG>;
template <texture_value ValueType>
using tex1rgb = tex1<ValueType, RGB>;
template <texture_value ValueType>
using tex1rgba = tex1<ValueType, RGBA>;
template <texture_value ValueType>
using tex1bgr = tex1<ValueType, BGR>;
template <texture_value ValueType>
using tex1bgra = tex1<ValueType, BGRA>;
template <texture_value ValueType>
using tex2r = tex2<ValueType, R>;
template <texture_value ValueType>
using tex2rg = tex2<ValueType, RG>;
template <texture_value ValueType>
using tex2rgb = tex2<ValueType, RGB>;
template <texture_value ValueType>
using tex2rgba = tex2<ValueType, RGBA>;
template <texture_value ValueType>
using tex2bgr = tex2<ValueType, BGR>;
template <texture_value ValueType>
using tex2bgra = tex2<ValueType, BGRA>;
template <texture_value ValueType>
using tex2depth = tex2<ValueType, Depth>;
template <texture_value ValueType>
using tex3r = tex3<ValueType, R>;
template <texture_value ValueType>
using tex3rg = tex3<ValueType, RG>;
template <texture_value ValueType>
using tex3rgb = tex3<ValueType, RGB>;
template <texture_value ValueType>
using tex3rgba = tex3<ValueType, RGBA>;
template <texture_value ValueType>
using tex3bgr = tex3<ValueType, BGR>;
template <texture_value ValueType>
using tex3bgra = tex3<ValueType, BGRA>;

using tex1r8ui  = tex1r<GLubyte>;
using tex1r16ui = tex1r<GLushort>;
using tex1r32ui = tex1r<GLuint>;
using tex1r8i   = tex1r<GLbyte>;
using tex1r16i  = tex1r<GLshort>;
using tex1r32i  = tex1r<GLint>;
using tex1r16f  = tex1r<GLhalf>;
using tex1r32f  = tex1r<GLfloat>;

using tex1rg8ui  = tex1rg<GLubyte>;
using tex1rg16ui = tex1rg<GLushort>;
using tex1rg32ui = tex1rg<GLuint>;
using tex1rg8i   = tex1rg<GLbyte>;
using tex1rg16i  = tex1rg<GLshort>;
using tex1rg32i  = tex1rg<GLint>;
using tex1rg16f  = tex1rg<GLhalf>;
using tex1rg32f  = tex1rg<GLfloat>;

using tex1rgb8ui  = tex1rgb<GLubyte>;
using tex1rgb16ui = tex1rgb<GLushort>;
using tex1rgb32ui = tex1rgb<GLuint>;
using tex1rgb8i   = tex1rgb<GLbyte>;
using tex1rgb16i  = tex1rgb<GLshort>;
using tex1rgb32i  = tex1rgb<GLint>;
using tex1rgb16f  = tex1rgb<GLhalf>;
using tex1rgb32f  = tex1rgb<GLfloat>;

using tex1rgba8ui  = tex1rgba<GLubyte>;
using tex1rgba16ui = tex1rgba<GLushort>;
using tex1rgba32ui = tex1rgba<GLuint>;
using tex1rgba8i   = tex1rgba<GLbyte>;
using tex1rgba16i  = tex1rgba<GLshort>;
using tex1rgba32i  = tex1rgba<GLint>;
using tex1rgba16f  = tex1rgba<GLhalf>;
using tex1rgba32f  = tex1rgba<GLfloat>;

using tex2r8ui  = tex2r<GLubyte>;
using tex2r16ui = tex2r<GLushort>;
using tex2r32ui = tex2r<GLuint>;
using tex2r8i   = tex2r<GLbyte>;
using tex2r16i  = tex2r<GLshort>;
using tex2r32i  = tex2r<GLint>;
using tex2r16f  = tex2r<GLhalf>;
using tex2r32f  = tex2r<GLfloat>;

using tex2rg8ui  = tex2rg<GLubyte>;
using tex2rg16ui = tex2rg<GLushort>;
using tex2rg32ui = tex2rg<GLuint>;
using tex2rg8i   = tex2rg<GLbyte>;
using tex2rg16i  = tex2rg<GLshort>;
using tex2rg32i  = tex2rg<GLint>;
using tex2rg16f  = tex2rg<GLhalf>;
using tex2rg32f  = tex2rg<GLfloat>;

using tex2rgb8ui  = tex2rgb<GLubyte>;
using tex2rgb16ui = tex2rgb<GLushort>;
using tex2rgb32ui = tex2rgb<GLuint>;
using tex2rgb8i   = tex2rgb<GLbyte>;
using tex2rgb16i  = tex2rgb<GLshort>;
using tex2rgb32i  = tex2rgb<GLint>;
using tex2rgb16f  = tex2rgb<GLhalf>;
using tex2rgb32f  = tex2rgb<GLfloat>;

using tex2rgba8ui  = tex2rgba<GLubyte>;
using tex2rgba16ui = tex2rgba<GLushort>;
using tex2rgba32ui = tex2rgba<GLuint>;
using tex2rgba8i   = tex2rgba<GLbyte>;
using tex2rgba16i  = tex2rgba<GLshort>;
using tex2rgba32i  = tex2rgba<GLint>;
using tex2rgba16f  = tex2rgba<GLhalf>;
using tex2rgba32f  = tex2rgba<GLfloat>;

using texdepth16ui = tex2depth<GLushort>;
using texdepth24ui = tex2depth<tex::depth24>;
using texdepth32ui = tex2depth<GLuint>;
using texdepth32f  = tex2depth<GLfloat>;

template <texture_value ValueType, texture_component Components>
struct texture_format_qualifier_impl;
template <>
struct texture_format_qualifier_impl<float, R> {
  static constexpr std::string_view value = "r32f";
};
template <>
struct texture_format_qualifier_impl<float, RG> {
  static constexpr std::string_view value = "rg32f";
};
template <>
struct texture_format_qualifier_impl<float, RGB> {
  static constexpr std::string_view value = "rgba32f";
};
template <>
struct texture_format_qualifier_impl<float, RGBA> {
  static constexpr std::string_view value = "rgba32f";
};
template <>
struct texture_format_qualifier_impl<std::uint8_t, RGBA> {
  static constexpr std::string_view value = "rgba8";
};
template <texture_value ValueType, texture_component Components>
static auto constexpr texture_format_qualifier =
    texture_format_qualifier_impl<ValueType, Components>::value;
//------------------------------------------------------------------------------
template <either_of<R, RG, RGBA> Components>
auto to_2d(tex1<float, Components> const& t1, std::size_t const height,
           std::size_t const local_size_x = 32,
           std::size_t const local_size_y = 32) {
  auto t2 = tex2<float, Components>{t1.width(), height};
  auto s  = shader{};
  auto ss = std::stringstream{};
  ss << "#version 430\n"
     << "uniform int width;\n"
     << "uniform int height;\n"
     << "layout(local_size_x = " << local_size_x
     << ", local_size_y = " << local_size_y << ") in;\n"
     << "layout("
     << texture_format_qualifier<
            float, Components> << ", binding = 0) uniform image1D t1;\n"
     << "layout("
     << texture_format_qualifier<
            float, Components> << ", binding = 1) uniform image2D t2;\n"
     << "void main() {\n"
     << "  ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);\n"
     << "  if(pixel_coords.x < width && pixel_coords.y < height) {\n"
     << "    imageStore(t2, pixel_coords, imageLoad(t1, pixel_coords.x));\n"
     << "  }\n"
     << "}\n";
  s.add_stage<computeshader>(gl::shadersource{ss.str()});
  s.create();
  s.bind();
  s.set_uniform("width", static_cast<int>(t1.width()));
  s.set_uniform("height", static_cast<int>(height));
  t1.bind_image_texture(0);
  t2.bind_image_texture(1);
  dispatch_compute(
      static_cast<int>(
          std::ceil(t1.width() / static_cast<double>(local_size_x))),
      static_cast<int>(std::ceil(height / static_cast<double>(local_size_y))),
      1);
  return t2;
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
