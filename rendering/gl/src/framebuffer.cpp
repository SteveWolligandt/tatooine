#include <yavin/framebuffer.h>
//==============================================================================
namespace yavin {
//==============================================================================
framebuffer::framebuffer() {
  gl::create_framebuffers(1, &id_ref());
}
//------------------------------------------------------------------------------
framebuffer::~framebuffer() {
  gl::delete_framebuffers(1, &id_ref());
}
//------------------------------------------------------------------------------
template <typename T, typename Components>
GLenum framebuffer::attach(const tex2<T, Components>& tex, unsigned int i) {
  assert(i < GL_MAX_COLOR_ATTACHMENTS);
  gl::named_framebuffer_texture(id(), GL_COLOR_ATTACHMENT0 + i, tex.id(), 0);
  gl_framebuffer_not_complete_check(id());
  return GL_COLOR_ATTACHMENT0 + i;
}
//------------------------------------------------------------------------------
template <typename T>
GLenum framebuffer::attach(const tex2<T, Depth>& tex) {
  gl::named_framebuffer_texture(id(), GL_DEPTH_ATTACHMENT, tex.id(), 0);
  gl_framebuffer_not_complete_check(id());
  return GL_DEPTH_ATTACHMENT;
}
//------------------------------------------------------------------------------
void framebuffer::bind() {
  gl::bind_framebuffer(GL_FRAMEBUFFER, id());
}
//------------------------------------------------------------------------------
void framebuffer::unbind() {
  gl::bind_framebuffer(GL_FRAMEBUFFER, 0);
}
//------------------------------------------------------------------------------
void framebuffer::clear() {
  // glClearNamedFramebufferfi(id(), GLenum buffer, GLint drawbuffer,
  //                           GLfloat depth, GLint stencil);
}
//==============================================================================
template GLenum framebuffer::attach<float, R>(const tex2r<float>&,
                                              unsigned int);
template GLenum framebuffer::attach<int8_t, R>(const tex2r<int8_t>&,
                                               unsigned int);
template GLenum framebuffer::attach<uint8_t, R>(const tex2r<uint8_t>&,
                                                unsigned int);
template GLenum framebuffer::attach<int16_t, R>(const tex2r<int16_t>&,
                                                unsigned int);
template GLenum framebuffer::attach<uint16_t, R>(const tex2r<uint16_t>&,
                                                 unsigned int);
template GLenum framebuffer::attach<int32_t, R>(const tex2r<int32_t>&,
                                                unsigned int);
template GLenum framebuffer::attach<uint32_t, R>(const tex2r<uint32_t>&,
                                                 unsigned int);

template GLenum framebuffer::attach<float, RG>(const tex2rg<float>&,
                                               unsigned int);
template GLenum framebuffer::attach<int8_t, RG>(const tex2rg<int8_t>&,
                                                unsigned int);
template GLenum framebuffer::attach<uint8_t, RG>(const tex2rg<uint8_t>&,
                                                 unsigned int);
template GLenum framebuffer::attach<int16_t, RG>(const tex2rg<int16_t>&,
                                                 unsigned int);
template GLenum framebuffer::attach<uint16_t, RG>(const tex2rg<uint16_t>&,
                                                  unsigned int);
template GLenum framebuffer::attach<int32_t, RG>(const tex2rg<int32_t>&,
                                                 unsigned int);
template GLenum framebuffer::attach<uint32_t, RG>(const tex2rg<uint32_t>&,
                                                  unsigned int);
template GLenum framebuffer::attach<float, RGB>(const tex2rgb<float>&,
                                                unsigned int);
template GLenum framebuffer::attach<int8_t, RGB>(const tex2rgb<int8_t>&,
                                                 unsigned int);
template GLenum framebuffer::attach<uint8_t, RGB>(const tex2rgb<uint8_t>&,
                                                  unsigned int);
template GLenum framebuffer::attach<int16_t, RGB>(const tex2rgb<int16_t>&,
                                                  unsigned int);
template GLenum framebuffer::attach<uint16_t, RGB>(const tex2rgb<uint16_t>&,
                                                   unsigned int);
template GLenum framebuffer::attach<int32_t>(const tex2rgb<int32_t>&,
                                             unsigned int);
template GLenum framebuffer::attach<uint32_t>(const tex2rgb<uint32_t>&,
                                              unsigned int);
template GLenum framebuffer::attach<float, RGBA>(const tex2rgba<float>&,
                                                 unsigned int);
template GLenum framebuffer::attach<int8_t, RGBA>(const tex2rgba<int8_t>&,
                                                  unsigned int);
template GLenum framebuffer::attach<uint8_t, RGBA>(const tex2rgba<uint8_t>&,
                                                   unsigned int);
template GLenum framebuffer::attach<int16_t, RGBA>(const tex2rgba<int16_t>&,
                                                   unsigned int);
template GLenum framebuffer::attach<uint16_t, RGBA>(const tex2rgba<uint16_t>&,
                                                    unsigned int);
template GLenum framebuffer::attach<int32_t, RGBA>(const tex2rgba<int32_t>&,
                                                   unsigned int);
template GLenum framebuffer::attach<uint32_t, RGBA>(const tex2rgba<uint32_t>&,
                                                    unsigned int);
template GLenum framebuffer::attach<uint32_t>(const tex2<uint32_t, Depth>&);
template GLenum framebuffer::attach<float>(const tex2<float, Depth>&);

//==============================================================================
}  // namespace yavin
//==============================================================================
