#ifndef TATOOINE_CUDA_RUNGEKUTTA4_H
#define TATOOINE_CUDA_RUNGEKUTTA4_H

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__host__ float2 rungekutta4_step_steady2(cudaTextureObject_t tex, float2 pos, float stepwidth,
                                         float2 min, float2 max, uint2 res) {
  float2 k1 =
      stepwidth *
      sample_vectorfield_steady2(
          tex, pos, min, max, res);
  const auto x2 = make_float2(pos.x + k1.x * 0.5f, pos.y + k1.y * 0.5f);
  if (x2.x < min.x || x2.y > max.x || x2.y < min.y || x2.y > max.y) {
    return make_float2(0.0f / 0.0f, 0.0f / 0.0f);
  }
  float2 k2 = stepwidth * sample_vectorfield_steady2(tex, x2, min, max, res);
  const auto x3 = make_float2(pos.x + k2.x * 0.5f, pos.y + k2.y * 0.5f);
  if (x3.x < min.x || x3.y > max.x || x3.y < min.y || x3.y > max.y) {
    return make_float2(0.0f / 0.0f, 0.0f / 0.0f);
  }
  float2 k3 = stepwidth * sample_vectorfield_steady2(tex, x3, min, max, res);
  const auto x4 = make_float2(pos.x + k3.x, pos.y + k3.y);
  if (x4.x < min.x || x4.y > max.x || x4.y < min.y || x4.y > max.y) {
    return make_float2(0.0f / 0.0f, 0.0f / 0.0f);
  }
  float2 k4 = stepwidth * sample_vectorfield_steady2(tex, x4, min, max, res);
  const auto stepped =
      make_float2(pos.x + (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6.0f,
                  pos.y + (k1.y + 2 * k2.y + 2 * k3.y + k4.y) / 6.0f);
  if (stepped.x < min.x || stepped.y > max.x || stepped.y < min.y ||
      stepped.y > max.y) {
    return make_float2(0.0f / 0.0f, 0.0f / 0.0f);
  }
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
