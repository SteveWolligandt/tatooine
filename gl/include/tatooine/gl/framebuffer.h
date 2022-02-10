#ifndef TATOOINE_GL_FRAMEBUFFER_H
#define TATOOINE_GL_FRAMEBUFFER_H
//==============================================================================
#include "errorcheck.h"
#include "dllexport.h"
#include <tatooine/gl/texture.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class framebuffer : public id_holder<GLuint> {
 public:
  DLL_API framebuffer();
  DLL_API ~framebuffer();
  template <typename... Textures>
  DLL_API framebuffer(const Textures&... textures) : framebuffer{} {
    constexpr auto num_color_attachements =
        num_color_components<typename Textures::components...>();
    unsigned int i = 0;
    // attach textures one after another, incrementing i if texture is a color
    // texture
    (void)std::array{(texture_depth_component<typename Textures::components>
                          ? attach(textures)
                          : attach(textures, i++))...};

    // create array for glDrawBuffers function with all color attachements
    std::array<GLenum, num_color_attachements> colbufs;
    auto add_attachement = [j = 0, &colbufs](const auto& tex) mutable {
      using TA         = decltype(tex);
      using Texture    = typename std::decay_t<TA>;
      using Components = typename Texture::components;
      if constexpr (texture_color_component<Components>) {
        colbufs[j] = GL_COLOR_ATTACHMENT0 + j;
        return GL_COLOR_ATTACHMENT0 + j++;
      }
      return GL_NONE;
    };
    (void)std::array{add_attachement(textures)...};
    gl::named_framebuffer_draw_buffers(id(), num_color_attachements, colbufs.data());
  }

  template <typename T, typename Components>
  DLL_API GLenum attach(const tex2<T, Components>& tex, unsigned int i = 0);
  template <typename T>
  DLL_API GLenum attach(const tex2<T, Depth>& depth_tex);
 private:
  // this is necessary for constructor taking variadic parameters
  template <typename T>
  constexpr GLenum attach(const tex2<T, Depth>& depth_tex, unsigned int) {
    return attach(depth_tex);
  }

 public:
  DLL_API void        bind();
  DLL_API static void unbind();

  DLL_API void clear();

  //============================================================================
 private:
  template <typename... Cs>
  static constexpr auto num_color_components() {
    return sum((texture_color_component<Cs> ? 1 : 0)...);
  }
  //============================================================================
  template <typename... Ts>
  static constexpr auto sum(Ts... ts) {
    return (ts + ...);
  }
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
