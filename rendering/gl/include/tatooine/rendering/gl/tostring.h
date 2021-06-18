#ifndef YAVIN_TO_STRING_H
#define YAVIN_TO_STRING_H
//==============================================================================
#include <string>
#include "glincludes.h"
//==============================================================================
namespace yavin {
//==============================================================================
inline auto texparami_to_string(GLint i) -> std::string {
  switch (i) {
    // texture wrapping
    case GL_CLAMP_TO_BORDER: return "GL_CLAMP_TO_BORDER";
    case GL_CLAMP_TO_EDGE: return "GL_CLAMP_TO_EDGE";
    case GL_REPEAT: return "GL_REPEAT";
    case GL_MIRRORED_REPEAT: return "GL_MIRRORED_REPEAT";

    // texture interpolation
    case GL_NEAREST: return "GL_NEAREST";
    case GL_LINEAR: return "GL_LINEAR";
    case GL_NEAREST_MIPMAP_NEAREST: return "GL_NEAREST_MIPMAP_NEAREST";
    case GL_LINEAR_MIPMAP_NEAREST: return "GL_LINEAR_MIPMAP_NEAREST";
    case GL_NEAREST_MIPMAP_LINEAR: return "GL_NEAREST_MIPMAP_LINEAR";
    case GL_LINEAR_MIPMAP_LINEAR: return "GL_LINEAR_MIPMAP_LINEAR";
    default: return std::to_string(i);
  }
}
//------------------------------------------------------------------------------
inline auto to_string(GLenum e) -> std::string {
  switch (e) {
    // buffers
    case GL_ARRAY_BUFFER: return "GL_ARRAY_BUFFER";
    case GL_ATOMIC_COUNTER_BUFFER: return "GL_ATOMIC_COUNTER_BUFFER";
    case GL_COPY_READ_BUFFER: return "GL_COPY_READ_BUFFER";
    case GL_COPY_WRITE_BUFFER: return "GL_COPY_WRITE_BUFFER";
    case GL_DISPATCH_INDIRECT_BUFFER: return "GL_DISPATCH_INDIRECT_BUFFER";
    case GL_DRAW_INDIRECT_BUFFER: return "GL_DRAW_INDIRECT_BUFFER";
    case GL_ELEMENT_ARRAY_BUFFER: return "GL_ELEMENT_ARRAY_BUFFER";
    case GL_PIXEL_PACK_BUFFER: return "GL_PIXEL_PACK_BUFFER";
    case GL_PIXEL_UNPACK_BUFFER: return "GL_PIXEL_UNPACK_BUFFER";
    case GL_QUERY_BUFFER: return "GL_QUERY_BUFFER";
    case GL_SHADER_STORAGE_BUFFER: return "GL_SHADER_STORAGE_BUFFER";
    case GL_TEXTURE_BUFFER: return "GL_TEXTURE_BUFFER";
    case GL_TRANSFORM_FEEDBACK_BUFFER: return "GL_TRANSFORM_FEEDBACK_BUFFER";
    case GL_UNIFORM_BUFFER: return "GL_UNIFORM_BUFFER";

    // accesses
    case GL_READ_ONLY: return "GL_READ_ONLY";
    case GL_WRITE_ONLY: return "GL_WRITE_ONLY";
    case GL_READ_WRITE: return "GL_READ_WRITE";

    // usages
    case GL_STREAM_DRAW: return "GL_STREAM_DRAW";
    case GL_STREAM_READ: return "GL_STREAM_READ";
    case GL_STREAM_COPY: return "GL_STREAM_COPY";
    case GL_STATIC_DRAW: return "GL_STATIC_DRAW";
    case GL_STATIC_READ: return "GL_STATIC_READ";
    case GL_STATIC_COPY: return "GL_STATIC_COPY";
    case GL_DYNAMIC_DRAW: return "GL_DYNAMIC_DRAW";
    case GL_DYNAMIC_READ: return "GL_DYNAMIC_READ";
    case GL_DYNAMIC_COPY:
      return "GL_DYNAMIC_COPY";

      // enableable
    case GL_ALPHA_TEST: return "GL_ALPHA_TEST";
    case GL_AUTO_NORMAL: return "GL_AUTO_NORMAL";
    case GL_BLEND: return "GL_BLEND";
    case GL_CLIP_PLANE0: return "GL_CLIP_PLANE0";
    case GL_CLIP_PLANE1: return "GL_CLIP_PLANE1";
    case GL_CLIP_PLANE2: return "GL_CLIP_PLANE2";
    case GL_CLIP_PLANE3: return "GL_CLIP_PLANE3";
    case GL_CLIP_PLANE4: return "GL_CLIP_PLANE4";
    case GL_CLIP_PLANE5: return "GL_CLIP_PLANE5";
    case GL_COLOR_LOGIC_OP: return "GL_COLOR_LOGIC_OP";
    case GL_COLOR_MATERIAL: return "GL_COLOR_MATERIAL";
    case GL_COLOR_SUM: return "GL_COLOR_SUM";
    case GL_COLOR_TABLE: return "GL_COLOR_TABLE";
    case GL_CONVOLUTION_1D: return "GL_CONVOLUTION_1D";
    case GL_CONVOLUTION_2D: return "GL_CONVOLUTION_2D";
    case GL_CULL_FACE: return "GL_CULL_FACE";
    case GL_DEPTH_TEST: return "GL_DEPTH_TEST";
    case GL_DITHER: return "GL_DITHER";
    case GL_FOG: return "GL_FOG";
    case GL_HISTOGRAM: return "GL_HISTOGRAM";
    case GL_INDEX_LOGIC_OP: return "GL_INDEX_LOGIC_OP";
    case GL_LIGHT0: return "GL_LIGHT0";
    case GL_LIGHT1: return "GL_LIGHT1";
    case GL_LIGHT2: return "GL_LIGHT2";
    case GL_LIGHT3: return "GL_LIGHT3";
    case GL_LIGHT4: return "GL_LIGHT4";
    case GL_LIGHT5: return "GL_LIGHT5";
    case GL_LIGHT6: return "GL_LIGHT6";
    case GL_LIGHT7: return "GL_LIGHT7";
    case GL_LIGHTING: return "GL_LIGHTING";
    case GL_LINE_SMOOTH: return "GL_LINE_SMOOTH";
    case GL_LINE_STIPPLE: return "GL_LINE_STIPPLE";
    case GL_MAP1_COLOR_4: return "GL_MAP1_COLOR_4";
    case GL_MAP1_INDEX: return "GL_MAP1_INDEX";
    case GL_MAP1_NORMAL: return "GL_MAP1_NORMAL";
    case GL_MAP1_TEXTURE_COORD_1: return "GL_MAP1_TEXTURE_COORD_1";
    case GL_MAP1_TEXTURE_COORD_2: return "GL_MAP1_TEXTURE_COORD_2";
    case GL_MAP1_TEXTURE_COORD_3: return "GL_MAP1_TEXTURE_COORD_3";
    case GL_MAP1_TEXTURE_COORD_4: return "GL_MAP1_TEXTURE_COORD_4";
    case GL_MAP1_VERTEX_3: return "GL_MAP1_VERTEX_3";
    case GL_MAP1_VERTEX_4: return "GL_MAP1_VERTEX_4";
    case GL_MAP2_COLOR_4: return "GL_MAP2_COLOR_4";
    case GL_MAP2_INDEX: return "GL_MAP2_INDEX";
    case GL_MAP2_NORMAL: return "GL_MAP2_NORMAL";
    case GL_MAP2_TEXTURE_COORD_1: return "GL_MAP2_TEXTURE_COORD_1";
    case GL_MAP2_TEXTURE_COORD_2: return "GL_MAP2_TEXTURE_COORD_2";
    case GL_MAP2_TEXTURE_COORD_3: return "GL_MAP2_TEXTURE_COORD_3";
    case GL_MAP2_TEXTURE_COORD_4: return "GL_MAP2_TEXTURE_COORD_4";
    case GL_MAP2_VERTEX_3: return "GL_MAP2_VERTEX_3";
    case GL_MAP2_VERTEX_4: return "GL_MAP2_VERTEX_4";
    case GL_MINMAX: return "GL_MINMAX";
    case GL_MULTISAMPLE: return "GL_MULTISAMPLE";
    case GL_NORMALIZE: return "GL_NORMALIZE";
    case GL_POINT_SMOOTH: return "GL_POINT_SMOOTH";
    case GL_POINT_SPRITE: return "GL_POINT_SPRITE";
    case GL_POLYGON_OFFSET_FILL: return "GL_POLYGON_OFFSET_FILL";
    case GL_POLYGON_OFFSET_LINE: return "GL_POLYGON_OFFSET_LINE";
    case GL_POLYGON_OFFSET_POINT: return "GL_POLYGON_OFFSET_POINT";
    case GL_POLYGON_SMOOTH: return "GL_POLYGON_SMOOTH";
    case GL_POLYGON_STIPPLE: return "GL_POLYGON_STIPPLE";
    case GL_POST_COLOR_MATRIX_COLOR_TABLE:
      return "GL_POST_COLOR_MATRIX_COLOR_TABLE";
    case GL_POST_CONVOLUTION_COLOR_TABLE:
      return "GL_POST_CONVOLUTION_COLOR_TABLE";
    case GL_RESCALE_NORMAL: return "GL_RESCALE_NORMAL";
    case GL_SAMPLE_ALPHA_TO_COVERAGE: return "GL_SAMPLE_ALPHA_TO_COVERAGE";
    case GL_SAMPLE_ALPHA_TO_ONE: return "GL_SAMPLE_ALPHA_TO_ONE";
    case GL_SAMPLE_COVERAGE: return "GL_SAMPLE_COVERAGE";
    case GL_SEPARABLE_2D: return "GL_SEPARABLE_2D";
    case GL_SCISSOR_TEST: return "GL_SCISSOR_TEST";
    case GL_STENCIL_TEST: return "GL_STENCIL_TEST";
    case GL_TEXTURE_1D: return "GL_TEXTURE_1D";
    case GL_TEXTURE_2D: return "GL_TEXTURE_2D";
    case GL_TEXTURE_3D: return "GL_TEXTURE_3D";
    case GL_TEXTURE_CUBE_MAP: return "GL_TEXTURE_CUBE_MAP";
    case GL_TEXTURE_GEN_Q: return "GL_TEXTURE_GEN_Q";
    case GL_TEXTURE_GEN_R: return "GL_TEXTURE_GEN_R";
    case GL_TEXTURE_GEN_S: return "GL_TEXTURE_GEN_S";
    case GL_TEXTURE_GEN_T: return "GL_TEXTURE_GEN_T";
    case GL_VERTEX_PROGRAM_POINT_SIZE: return "GL_VERTEX_PROGRAM_POINT_SIZE";
    case GL_VERTEX_PROGRAM_TWO_SIDE: return "GL_VERTEX_PROGRAM_TWO_SIDE";

    // formats
    case GL_RED: return "GL_RED";
    case GL_RG: return "GL_RG";
    case GL_RGB: return "GL_RGB";
    case GL_BGR: return "GL_BGR";
    case GL_RGBA: return "GL_RGBA";
    case GL_BGRA: return "GL_BGRA";
    case GL_RED_INTEGER: return "GL_RED_INTEGER";
    case GL_RG_INTEGER: return "GL_RG_INTEGER";
    case GL_RGB_INTEGER: return "GL_RGB_INTEGER";
    case GL_BGR_INTEGER: return "GL_BGR_INTEGER";
    case GL_RGBA_INTEGER: return "GL_RGBA_INTEGER";
    case GL_BGRA_INTEGER: return "GL_BGRA_INTEGER";
    case GL_STENCIL_INDEX: return "GL_STENCIL_INDEX";
    case GL_DEPTH_COMPONENT: return "GL_DEPTH_COMPONENT";
    case GL_DEPTH_STENCIL: return "GL_DEPTH_STENCIL";
    case GL_R8: return "GL_R8";
    case GL_RG8: return "GL_RG8";
    case GL_RGB8: return "GL_RGB8";
    case GL_RGBA8: return "GL_RGBA8";
    case GL_R8I: return "GL_R8I";
    case GL_RG8I: return "GL_RG8I";
    case GL_RGB8I: return "GL_RGB8I";
    case GL_RGBA8I: return "GL_RGBA8I";
    case GL_R8UI: return "GL_R8UI";
    case GL_RG8UI: return "GL_RG8UI";
    case GL_RGB8UI: return "GL_RGB8UI";
    case GL_RGBA8UI: return "GL_RGBA8UI";
    case GL_R16I: return "GL_R16I";
    case GL_RG16I: return "GL_RG16I";
    case GL_RGB16I: return "GL_RGB16I";
    case GL_RGBA16I: return "GL_RGBA16I";
    case GL_R16UI: return "GL_R16UI";
    case GL_RG16UI: return "GL_RG16UI";
    case GL_RGB16UI: return "GL_RGB16UI";
    case GL_RGBA16UI: return "GL_RGBA16UI";
    case GL_R32I: return "GL_R32I";
    case GL_RG32I: return "GL_RG32I";
    case GL_RGB32I: return "GL_RGB32I";
    case GL_RGBA32I: return "GL_RGBA32I";
    case GL_R32UI: return "GL_R32UI";
    case GL_RG32UI: return "GL_RG32UI";
    case GL_RGB32UI: return "GL_RGB32UI";
    case GL_RGBA32UI: return "GL_RGBA32UI";
    case GL_R16: return "GL_R16";
    case GL_RG16: return "GL_RG16";
    case GL_RGB16: return "GL_RGB16";
    case GL_RGBA16: return "GL_RGBA16";
    case GL_R32F: return "GL_R32F";
    case GL_RG32F: return "GL_RG32F";
    case GL_RGB32F: return "GL_RGB32F";
    case GL_RGBA32F: return "GL_RGBA32F";
    case GL_R16F: return "GL_R16F";
    case GL_RG16F: return "GL_RG16F";
    case GL_RGB16F: return "GL_RGB16F";
    case GL_RGBA16F: return "GL_RGBA16F";

    case GL_UNSIGNED_BYTE: return "GL_UNSIGNED_BYTE";
    case GL_BYTE: return "GL_BYTE";
    case GL_UNSIGNED_SHORT: return "GL_UNSIGNED_SHORT";
    case GL_SHORT: return "GL_SHORT";
    case GL_UNSIGNED_INT: return "GL_UNSIGNED_INT";
    case GL_INT: return "GL_INT";
    case GL_FLOAT: return "GL_FLOAT";
    case GL_UNSIGNED_BYTE_3_3_2: return "GL_UNSIGNED_BYTE_3_3_2";
    case GL_UNSIGNED_BYTE_2_3_3_REV: return "GL_UNSIGNED_BYTE_2_3_3_REV";
    case GL_UNSIGNED_SHORT_5_6_5: return "GL_UNSIGNED_SHORT_5_6_5";
    case GL_UNSIGNED_SHORT_5_6_5_REV: return "GL_UNSIGNED_SHORT_5_6_5_REV";
    case GL_UNSIGNED_SHORT_4_4_4_4: return "GL_UNSIGNED_SHORT_4_4_4_4";
    case GL_UNSIGNED_SHORT_4_4_4_4_REV: return "GL_UNSIGNED_SHORT_4_4_4_4_REV";
    case GL_UNSIGNED_SHORT_5_5_5_1: return "GL_UNSIGNED_SHORT_5_5_5_1";
    case GL_UNSIGNED_SHORT_1_5_5_5_REV: return "GL_UNSIGNED_SHORT_1_5_5_5_REV";
    case GL_UNSIGNED_INT_8_8_8_8: return "GL_UNSIGNED_INT_8_8_8_8";
    case GL_UNSIGNED_INT_8_8_8_8_REV: return "GL_UNSIGNED_INT_8_8_8_8_REV";
    case GL_UNSIGNED_INT_10_10_10_2: return "GL_UNSIGNED_INT_10_10_10_2";
    case GL_UNSIGNED_INT_2_10_10_10_REV:
      return "GL_UNSIGNED_INT_2_10_10_10_REV";

      // framebuffer attachements
    case GL_COLOR_ATTACHMENT0: return "GL_COLOR_ATTACHEMENT0";
    case GL_COLOR_ATTACHMENT1: return "GL_COLOR_ATTACHEMENT1";
    case GL_COLOR_ATTACHMENT2: return "GL_COLOR_ATTACHEMENT2";
    case GL_COLOR_ATTACHMENT3: return "GL_COLOR_ATTACHEMENT3";
    case GL_COLOR_ATTACHMENT4: return "GL_COLOR_ATTACHEMENT4";
    case GL_COLOR_ATTACHMENT5: return "GL_COLOR_ATTACHEMENT5";
    case GL_COLOR_ATTACHMENT6: return "GL_COLOR_ATTACHEMENT6";
    case GL_COLOR_ATTACHMENT7: return "GL_COLOR_ATTACHEMENT7";
    case GL_COLOR_ATTACHMENT8: return "GL_COLOR_ATTACHEMENT8";
    case GL_COLOR_ATTACHMENT9: return "GL_COLOR_ATTACHEMENT9";
    case GL_COLOR_ATTACHMENT10: return "GL_COLOR_ATTACHEMENT10";
    case GL_COLOR_ATTACHMENT11: return "GL_COLOR_ATTACHEMENT11";
    case GL_COLOR_ATTACHMENT12: return "GL_COLOR_ATTACHEMENT12";
    case GL_COLOR_ATTACHMENT13: return "GL_COLOR_ATTACHEMENT13";
    case GL_COLOR_ATTACHMENT14: return "GL_COLOR_ATTACHEMENT14";
    case GL_COLOR_ATTACHMENT15: return "GL_COLOR_ATTACHEMENT15";
    case GL_DEPTH_ATTACHMENT: return "GL_DEPTH_ATTACHMENT";
    case GL_STENCIL_ATTACHMENT: return "GL_STENCIL_ATTACHMENT";
    case GL_DEPTH_STENCIL_ATTACHMENT:
      return "GL_DEPTH_STENCIL_ATTACHMENT";

      // primitives
    case GL_POINTS: return "GL_POINTS";
    case GL_LINE_STRIP: return "GL_LINE_STRIP";
    case GL_LINE_LOOP: return "GL_LINE_LOOP";
    case GL_LINES: return "GL_LINES";
    case GL_LINE_STRIP_ADJACENCY: return "GL_LINE_STRIP_ADJACENCY";
    case GL_LINES_ADJACENCY: return "GL_LINES_ADJACENCY";
    case GL_TRIANGLE_STRIP: return "GL_TRIANGLE_STRIP";
    case GL_TRIANGLE_FAN: return "GL_TRIANGLE_FAN";
    case GL_TRIANGLES: return "GL_TRIANGLES";
    case GL_TRIANGLE_STRIP_ADJACENCY: return "GL_TRIANGLE_STRIP_ADJACENCY";
    case GL_TRIANGLES_ADJACENCY: return "GL_TRIANGLES_ADJACENCY";
    case GL_PATCHES: return "GL_PATCHES";


    case GL_TEXTURE_WRAP_S: return "GL_TEXTURE_WRAP_S";
    case GL_TEXTURE_WRAP_T: return "GL_TEXTURE_WRAP_T";
    case GL_TEXTURE_WRAP_R: return "GL_TEXTURE_WRAP_R";

    case GL_TEXTURE_MIN_FILTER: return "GL_TEXTURE_MIN_FILTER";
    case GL_TEXTURE_MAG_FILTER: return "GL_TEXTURE_MAG_FILTER";

    default: return "GLenum(" + std::to_string(e) + ")";
  }
}
//------------------------------------------------------------------------------
inline auto map_access_to_string(GLbitfield b) -> std::string {
  std::string flags;
  auto        pipe = [&flags]() {
    if (!flags.empty()) { flags += std::string(" | "); }
  };
  if (GL_MAP_READ_BIT & b) {
    pipe();
    flags += "GL_MAP_READ_BIT";
  }
  if (GL_MAP_WRITE_BIT & b) {
    pipe();
    flags += "GL_MAP_WRITE_BIT";
  }
  if (GL_MAP_PERSISTENT_BIT & b) {
    pipe();
    flags += "GL_MAP_PERSISTENT_BIT";
  }
  if (GL_MAP_COHERENT_BIT & b) {
    pipe();
    flags += "GL_MAP_COHERENT_BIT";
  }
  if (GL_MAP_INVALIDATE_RANGE_BIT & b) {
    pipe();
    flags += "GL_MAP_INVALIDATE_RANGE_BIT";
  }
  if (GL_MAP_INVALIDATE_BUFFER_BIT & b) {
    pipe();
    flags += "GL_MAP_INVALIDATE_BUFFER_BIT";
  }
  if (GL_MAP_FLUSH_EXPLICIT_BIT & b) {
    pipe();
    flags += "GL_MAP_FLUSH_EXPLICIT_BIT";
  }
  if (GL_MAP_UNSYNCHRONIZED_BIT & b) {
    pipe();
    flags += "GL_MAP_UNSYNCHRONIZED_BIT";
  }
  if (GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT & b) {
    pipe();
    flags += "GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT";
  }
  if (GL_ELEMENT_ARRAY_BARRIER_BIT & b) {
    pipe();
    flags += "GL_ELEMENT_ARRAY_BARRIER_BIT";
  }
  if (GL_UNIFORM_BARRIER_BIT & b) {
    pipe();
    flags += "GL_UNIFORM_BARRIER_BIT";
  }
  if (GL_TEXTURE_FETCH_BARRIER_BIT & b) {
    pipe();
    flags += "GL_TEXTURE_FETCH_BARRIER_BIT";
  }
  if (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT & b) {
    pipe();
    flags += "GL_SHADER_IMAGE_ACCESS_BARRIER_BIT";
  }
  if (GL_COMMAND_BARRIER_BIT & b) {
    pipe();
    flags += "GL_COMMAND_BARRIER_BIT";
  }
  if (GL_PIXEL_BUFFER_BARRIER_BIT & b) {
    pipe();
    flags += "GL_PIXEL_BUFFER_BARRIER_BIT";
  }
  if (GL_TEXTURE_UPDATE_BARRIER_BIT & b) {
    pipe();
    flags += "GL_TEXTURE_UPDATE_BARRIER_BIT";
  }
  if (GL_BUFFER_UPDATE_BARRIER_BIT & b) {
    pipe();
    flags += "GL_BUFFER_UPDATE_BARRIER_BIT";
  } 
  if (GL_FRAMEBUFFER_BARRIER_BIT & b) {
    pipe();
    flags += "GL_FRAMEBUFFER_BARRIER_BIT";
  }
  if (GL_TRANSFORM_FEEDBACK_BARRIER_BIT & b) {
    pipe();
    flags += "GL_TRANSFORM_FEEDBACK_BARRIER_BIT";
  }
  if (GL_ATOMIC_COUNTER_BARRIER_BIT & b) {
    pipe();
    flags += "GL_ATOMIC_COUNTER_BARRIER_BIT";
  }
  if (GL_SHADER_STORAGE_BARRIER_BIT & b) {
    pipe();
    flags += "GL_SHADER_STORAGE_BARRIER_BIT";
  }
  return flags;
}
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
