#ifndef YAVIN_OIT_WINDOW
#define YAVIN_OIT_WINDOW

#include "atomic_counter_buffer.h"
#include "pixel_unpack_buffer.h"
#include "shader.h"
#include "shader_storage_buffer.h"
#include "texture_2d.h"
#include "vbo_helpers.h"
#include "vertex_array.h"
#include "vertex_buffer.h"
#include "window.h"
#include "gl_wrapper.h"

//==============================================================================
namespace yavin {
//==============================================================================

class OITWindow : public Window {
 public:
  //============================================================================
  class screen_quad_t {
   public:
    using vbo_t = VertexBuffer<vec2>;
    screen_quad_t()
        : m_vbo{{vec2{0, 0}}, {vec2{1, 0}}, {vec2{0, 1}}, {vec2{1, 1}}},
          m_ibo{0, 1, 2, 3} {
      m_vao.bind();
      m_vbo.bind();
      m_ibo.bind();
    }

    void draw() {
      m_vao.bind();
      m_vao.draw_triangle_strip(4);
    }

   private:
    VertexArray m_vao;
    vbo_t       m_vbo;
    IndexBuffer m_ibo;
  };
  //============================================================================
  struct linked_list_element {
    vec4         color;
    unsigned int next_index;
    float        depth;
    vec2         pad;
  };
  //============================================================================
  const static std::string vertex_shader_path;
  const static std::string fragment_shader_path;
  //----------------------------------------------------------------------------
  DLL_API OITWindow(const std::string& name, size_t width, size_t height,
                    unsigned int linked_list_size_factor,
                    unsigned int major = 4, unsigned int minor = 5);

  //----------------------------------------------------------------------------
  DLL_API void init();
  //----------------------------------------------------------------------------
  void set_render_function(std::function<void()> render_function) {
    m_oit_render_function = render_function;
  }
  void set_resize_callback(std::function<void(int, int)> resize_fun) {
    m_oit_resize_callback = resize_fun;
  }
  //----------------------------------------------------------------------------
  void set_clear_color(const glm::vec4& clear_color) {
    gl::clear_color(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    m_linked_list_render_shader.set_uniform("clear_color", clear_color);
  }

 private:
  std::function<void()>         m_oit_render_function;
  std::function<void(int, int)> m_oit_resize_callback;

  size_t m_width, m_height;
  size_t m_linked_list_size_factor;

  Shader                                   m_linked_list_render_shader;
  AtomicCounterBuffer                      m_atomic_counter;
  ShaderStorageBuffer<linked_list_element> m_linked_list;
  ShaderStorageBuffer<unsigned int>        m_linked_list_size;
  Texture2D<unsigned int, R>               m_head_indices_tex;
  PixelUnpackBuffer<unsigned int>          m_clear_buffer;
  screen_quad_t                            m_fullscreen_quad;
};

//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
