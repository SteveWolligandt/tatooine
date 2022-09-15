#ifndef TATOOINE_GL_GL_INCLUDES_H
#define TATOOINE_GL_GL_INCLUDES_H
//==============================================================================
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>

#if defined(_WIN32) || defined(WIN32)
#include <glad/glad_wgl.h>
#else
#include <glad/glad_egl.h>
#endif
//------------------------------------------------------------------------------
#include <GLFW/glfw3.h>
//------------------------------------------------------------------------------
#include <GL/gl.h>
//------------------------------------------------------------------------------
#ifdef None
#undef None
#endif
//------------------------------------------------------------------------------
#ifdef Complex
#undef Complex
#endif
//==============================================================================
#endif
