#include <yavin/context.h>
#include <yavin/window.h>
#include <thread>
#include <chrono>
//==============================================================================
namespace yavin {
//==============================================================================
context::context() {
  m_glfw_context = std::make_unique<glfw::context>();
  make_current();
}
//------------------------------------------------------------------------------
context::context(context& parent) {
  m_glfw_context = std::make_unique<glfw::context>(parent.get());
}
//------------------------------------------------------------------------------
context::context(window& parent) {
  m_glfw_context = std::make_unique<glfw::context>(parent.get());
}
//==============================================================================
// methods
//==============================================================================
context context::create_shared_context() {
  return context{*this};
}
//------------------------------------------------------------------------------
auto context::make_current() -> void {
  m_glfw_context->make_current();
}
//------------------------------------------------------------------------------
void context::release() {
  m_glfw_context->release();
}
//==============================================================================
}  // namespace yavin
//==============================================================================
