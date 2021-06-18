#ifndef YAVIN_LINKED_LIST_TEXTURE_LIST
#define YAVIN_LINKED_LIST_TEXTURE_LIST
//==============================================================================
#include "atomiccounterbuffer.h"
#include "shaderstoragebuffer.h"
#include "pixelunpackbuffer.h"
#include <yavin/texture.h>
//==============================================================================
namespace yavin {
//==============================================================================
template <typename Node>
class linked_list_texture {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  size_t                          m_w, m_h, m_num_nodes;
  atomiccounterbuffer             m_atomic_counter;
  shaderstoragebuffer<Node>       m_linked_list;
  tex2r32ui                       m_head_index_tex;
  tex2r32ui                       m_list_length_tex;
  pixelunpackbuffer<unsigned int> m_head_index_clear_buffer;
  pixelunpackbuffer<unsigned int> m_list_length_clear_buffer;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  linked_list_texture(size_t w, size_t h, size_t num_nodes, const Node& initial)
      : m_w{w},
        m_h{h},
        m_num_nodes{num_nodes},
        m_atomic_counter{0},
        m_linked_list(m_num_nodes, initial),
        m_head_index_tex{w, h},
        m_list_length_tex{w, h},
        m_head_index_clear_buffer(m_w * m_h, (unsigned int)(0xffffffff)),
        m_list_length_clear_buffer(m_w * m_h, (unsigned int)(0)) {
    clear();
    for (auto i : m_head_index_tex.download_data()) {
      if (i != 0xffffffff) {std::cerr << "somethings wrong\n";}
    }
  }
  //----------------------------------------------------------------------------
  linked_list_texture(const linked_list_texture& other) = default;
  linked_list_texture(linked_list_texture&& other)      = default;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  void resize(size_t w, size_t h, size_t num_nodes) {
    m_w = w;
    m_h = h;
    m_num_nodes = num_nodes;
    m_head_index_clear_buffer.upload_data(std::vector(m_w * m_h, 0xffffffff));
    m_head_index_tex.resize(m_w, m_h);
    m_linked_list.gpu_malloc(m_num_nodes);
  }
  //----------------------------------------------------------------------------
  void bind(unsigned int at_slot = 0, unsigned int hi_slot = 0,
            unsigned int length_slot = 1, unsigned int ll_slot = 0) const {
    m_atomic_counter.bind(at_slot);
    m_head_index_tex.bind_image_texture(hi_slot);
    m_list_length_tex.bind_image_texture(length_slot);
    m_linked_list.bind(ll_slot);
  }
  //----------------------------------------------------------------------------
  void clear() {
    m_atomic_counter.to_zero();
    m_head_index_tex.set_data(m_head_index_clear_buffer);
    m_list_length_tex.set_data(m_list_length_clear_buffer);
    m_head_index_clear_buffer.unbind();
  }
  //----------------------------------------------------------------------------
  void resize_buffer(size_t size) {
    m_linked_list.resize(size);
    m_num_nodes = size;
  }
  //----------------------------------------------------------------------------
  auto width() const { return m_w; }
  auto height() const { return m_h; }
  auto buffer_size() const { return m_num_nodes; }
  //----------------------------------------------------------------------------
  const auto& buffer() const { return m_linked_list; }
  //----------------------------------------------------------------------------
  auto&       counter() { return m_atomic_counter; }
  const auto& counter() const { return m_atomic_counter; }
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
