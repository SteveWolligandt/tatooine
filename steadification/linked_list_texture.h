#ifndef LINKED_LIST_TEXTURE_LIST
#define LINKED_LIST_TEXTURE_LIST

#include <yavin>
using namespace yavin;

template <typename node_t>
class LinkedListTexture {
 public:
  LinkedListTexture(unsigned int w, unsigned int h)
      : m_w{w},
        m_h{h},
        m_atomic_counter{0},
        m_linked_list(m_w * m_h * m_buffer_scaling,
                      {0xffffffff, 0.f, 0.f, 0.f}),
        m_head_indices_tex(w, h),
        m_clear_buffer{std::vector(m_w * m_h, 0xffffffff)} {}

  void resize(unsigned int w, unsigned int h) {
    m_w = w;
    m_h = h;
    m_clear_buffer.upload_data(std::vector(m_w * m_h, 0xffffffff));
    m_head_indices_tex.resize(m_w, m_h);
    m_linked_list.gpu_malloc(m_w * m_h * m_buffer_scaling);
  }

  void bind(unsigned int at_slot = 0, unsigned int hi_slot = 0,
            unsigned int ll_slot = 0) {
    m_atomic_counter.bind(at_slot);
    m_head_indices_tex.bind_image_texture(hi_slot);
    m_linked_list.bind(ll_slot);
  }

  void clear() {
    m_atomic_counter.to_zero();
    m_head_indices_tex.set_data(m_clear_buffer);
  }
  auto        w() { return m_w; }
  auto        h() { return m_h; }
  const auto& buffer() { return m_linked_list; }
  auto& counter() { return m_atomic_counter; }
  const auto& counter() const { return m_atomic_counter; }
  auto        buffer_size() { return m_w * m_h * m_buffer_scaling; }

 private:
  unsigned int                    m_w, m_h;
  unsigned int                    m_buffer_scaling = 10;
  atomiccounterbuffer             m_atomic_counter;
  shaderstoragebuffer<node_t>     m_linked_list;
  tex2r<unsigned int>      m_head_indices_tex;
  pixelunpackbuffer<unsigned int> m_clear_buffer;
};

#endif
