#ifndef TATOOINE_GPU_COMPUTE_SHADER_LAYOUT_LITERAL_H
#define TATOOINE_GPU_COMPUTE_SHADER_LAYOUT_LITERAL_H
#define TATOOINE_COMPUTE_SHADER_LAYOUT1D(x) \
  "layout(local_size_x = " #x ") in;"
#define TATOOINE_COMPUTE_SHADER_LAYOUT2D(x, y) \
  "layout(local_size_x = " #x ", local_size_y = " #y ") in;"
#define TATOOINE_COMPUTE_SHADER_LAYOUT3D(x, y, z) \
  "layout(local_size_x = " #x ", local_size_y = " #y ", local_size_z = " #z \
  ") in;"
#endif
