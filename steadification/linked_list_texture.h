#ifndef TATOOINE_STEADIFICATION_LINKED_LIST_TEXTURE_LIST
#define TATOOINE_STEADIFICATION_LINKED_LIST_TEXTURE_LIST
//==============================================================================
#include <yavin>
//==============================================================================
using namespace yavin;
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename Node>
class linked_list_texture {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  unsigned int                    m_w, m_h;
  const unsigned int              m_buffer_scaling = 2;
  atomiccounterbuffer             m_atomic_counter;
  shaderstoragebuffer<Node>       m_linked_list;
  tex2r<unsigned int>             m_head_indices_tex;
  tex2r<unsigned int>             m_list_length_tex;
  pixelunpackbuffer<unsigned int> m_head_indices_clear_buffer;
  pixelunpackbuffer<unsigned int> m_list_length_clear_buffer;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  linked_list_texture(unsigned int w, unsigned int h)
      : m_w{w},
        m_h{h},
        m_atomic_counter{0},
        m_linked_list(m_w * m_h * m_buffer_scaling,
                      {0xffffffff, 0.f, 0.f, 0.f}),
        m_head_indices_tex(w, h),
        m_head_indices_clear_buffer{std::vector(m_w * m_h, 0xffffffff)},
        m_list_length_clear_buffer{std::vector(m_w * m_h, 0)} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  void resize(unsigned int w, unsigned int h) {
    m_w = w;
    m_h = h;
    m_head_indices_clear_buffer.upload_data(std::vector(m_w * m_h, 0xffffffff));
    m_head_indices_tex.resize(m_w, m_h);
    m_linked_list.gpu_malloc(m_w * m_h * m_buffer_scaling);
  }
  //----------------------------------------------------------------------------
  void bind(unsigned int at_slot = 0, unsigned int hi_slot = 0,
            unsigned int length_slot = 1, unsigned int ll_slot = 0) {
    m_atomic_counter.bind(at_slot);
    m_head_indices_tex.bind_image_texture(hi_slot);
    m_head_indices_tex.bind_image_texture(length_slot);
    m_linked_list.bind(ll_slot);
  }
  //----------------------------------------------------------------------------
  void clear() {
    m_atomic_counter.to_zero();
    m_head_indices_tex.set_data(m_head_indices_clear_buffer);
    m_list_length_tex.set_data(m_list_length_clear_buffer);
  }
  //----------------------------------------------------------------------------
  auto width() const { return m_w; }
  auto height() const { return m_h; }
  //----------------------------------------------------------------------------
  const auto& buffer() const { return m_linked_list; }
  //----------------------------------------------------------------------------
  auto&       counter() { return m_atomic_counter; }
  const auto& counter() const { return m_atomic_counter; }
  //----------------------------------------------------------------------------
  auto buffer_size() { return m_w * m_h * m_buffer_scaling; }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
