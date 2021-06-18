#include <yavin/oit_window.h>
#include <yavin/gl_wrapper.h>
#include <yavin/orthographic_camera.h>
#include <yavin/shader_include_paths.h>

//==============================================================================
namespace yavin {
//==============================================================================

const std::string OITWindow::vertex_shader_path =
    shader_dir + std::string("linked_list_render.vert");

//------------------------------------------------------------------------------

const std::string OITWindow::fragment_shader_path =
    shader_dir + std::string("linked_list_render.frag");

//------------------------------------------------------------------------------

OITWindow::OITWindow(const std::string& name, size_t width, size_t height,
                     unsigned int linked_list_size_factor, unsigned int major,
                     unsigned int minor)
    : Window(name, width, height, major, minor),
      m_oit_render_function([]() {}),
      m_width(width),
      m_height(height),
      m_linked_list_size_factor(linked_list_size_factor),
      m_atomic_counter{0},
      m_linked_list(width * height * m_linked_list_size_factor),
      m_linked_list_size(
          std::vector<unsigned int>{
              (unsigned int)(width * height * m_linked_list_size_factor)},
          ShaderStorageBuffer<unsigned int>::STATIC_DRAW),
      m_head_indices_tex(width, height) {
  disable_multisampling();
  m_linked_list_render_shader.add_shader_stage<VertexShader>(
      vertex_shader_path);
  m_linked_list_render_shader.add_shader_stage<FragmentShader>(
      fragment_shader_path);
  m_linked_list_render_shader.create();
  m_linked_list_render_shader.bind();

  OrthographicCamera cam(0, 1, 0, 1, -1, 1, width, height);
  m_linked_list_render_shader.set_uniform("projection",
                                          cam.projection_matrix());
  m_linked_list_render_shader.set_uniform("modelview", cam.view_matrix());

  m_atomic_counter.bind(5);
  m_linked_list_size.bind(8);
  m_linked_list.bind(9);

  m_head_indices_tex.bind();
  m_head_indices_tex.set_interpolation_mode(tex::NEAREST);
  m_head_indices_tex.set_wrap_mode(tex::REPEAT);
  m_head_indices_tex.unbind();
  m_head_indices_tex.bind_image_texture(7);

  m_clear_buffer.upload_data(std::vector(m_width * m_height, 0xffffffff));

  Window::set_render_function([this]() {
    clear_color_buffer();

    // clear linked list
    m_atomic_counter.to_zero();
    m_head_indices_tex.set_data(m_clear_buffer);

    // user render function
    m_oit_render_function();

    // draw linked list
    gl::viewport(0, 0, m_width, m_height);
    m_linked_list_render_shader.bind();
    m_fullscreen_quad.draw();
  });
  Window::set_resize_callback([this](int w, int h) {
    m_width  = w;
    m_height = h;

    m_clear_buffer.upload_data(
        std::vector<unsigned int>(m_width * m_height, 0xffffffff));

    m_head_indices_tex.bind();
    m_clear_buffer.unbind();
    m_head_indices_tex.resize(m_width, m_height);
    m_head_indices_tex.bind_image_texture(7);

    m_linked_list.gpu_malloc(m_width * m_height * m_linked_list_size_factor);
    m_linked_list_size.front() = m_width * m_height * m_linked_list_size_factor;
    OrthographicCamera cam(0, 1, 0, 1, -1, 1, m_width, m_height);
    m_linked_list_render_shader.bind();
    m_linked_list_render_shader.set_uniform("projection",
                                            cam.projection_matrix());
    m_linked_list_render_shader.set_uniform("modelview", cam.view_matrix());

    m_oit_resize_callback(w, h);
  });
}

//==============================================================================
}  // namespace yavin
//==============================================================================
