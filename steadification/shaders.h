#ifndef SHADERS_H
#define SHADERS_H

//==============================================================================
#include <yavin>

//==============================================================================
struct VertFragShader : yavin::shader {
  VertFragShader(const std::string& vert, const std::string& frag);
  void set_projection(const glm::mat4& projection);
};

//==============================================================================
struct CompShader : yavin::shader {
  CompShader(const std::string& comp);
  void dispatch2d(GLuint w, GLuint h);
};

//==============================================================================
struct StreamsurfaceBaseShader : VertFragShader {
  StreamsurfaceBaseShader(const std::string& frag_path);
  void set_backward_tau_range(GLfloat min, GLfloat max);
  void set_forward_tau_range(GLfloat min, GLfloat max);
  void set_u_range(GLfloat min, GLfloat max);
};

//==============================================================================
struct ScreenSpaceBaseShader : VertFragShader {
  ScreenSpaceBaseShader(GLuint w, GLuint h,
                        const std::string& frag_shader_path);
  void resize(GLuint w, GLuint h);
};

//==============================================================================
struct StreamsurfaceToLinkedListShader : StreamsurfaceBaseShader {
  StreamsurfaceToLinkedListShader(GLuint s);
  void set_size(GLuint s);
};

//==============================================================================
struct StreamsurfaceTauShader : StreamsurfaceBaseShader {
  StreamsurfaceTauShader();
  void set_tau_range(const glm::vec2& tau_range);
  void set_color_scale_slot(int color_scale_slot);
};

//==============================================================================
struct StreamsurfaceVectorfieldShader : StreamsurfaceBaseShader {
  StreamsurfaceVectorfieldShader();
};

//==============================================================================
struct LinkedListToHeadVectorsShader : CompShader {
  LinkedListToHeadVectorsShader();
};

//==============================================================================
struct LICShader : CompShader {
  LICShader(GLuint num_steps, GLfloat min_x, GLfloat min_y, GLfloat max_x,
            GLfloat max_y);

  void set_num_steps(GLuint num_steps);
  void set_bounding_min(GLfloat x, GLfloat y);
  void set_bounding_max(GLfloat x, GLfloat y);
};

//==============================================================================
struct ColorLICShader : CompShader {
  ColorLICShader() : CompShader("color_lic.comp") {}
};

//==============================================================================
struct WeightShader : CompShader {
  WeightShader();
  void set_bw_tau(GLfloat tau);
  void set_fw_tau(GLfloat tau);
};

//==============================================================================
struct SpatialCoverageShader : CompShader {
  SpatialCoverageShader();
};

//==============================================================================
struct MeshViewerShader : VertFragShader {
  MeshViewerShader();
  void set_tex_slot(const int slot);
  void set_modelview(const glm::mat4& modelview);
};

//------------------------------------------------------------------------------
struct WeightShowShader : ScreenSpaceBaseShader {
  WeightShowShader(GLuint w, GLuint h);
  void set_t0(GLfloat t0);
  void set_bw_tau(GLfloat tau);
  void set_fw_tau(GLfloat tau);
};

//------------------------------------------------------------------------------
struct ScreenSpaceShader : ScreenSpaceBaseShader {
  ScreenSpaceShader(GLuint w, GLuint h);
  void set_resolution(GLuint w, GLuint h);
  void resize(GLuint w, GLuint h);
};

//==============================================================================
struct LineShader : VertFragShader {
  LineShader(GLfloat r = 0, GLfloat g = 0, GLfloat b = 0, GLfloat a = 1);
  void set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a);
};

#endif
