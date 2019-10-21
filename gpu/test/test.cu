#include <fstream>
#include <vector>

const size_t       width = 256, height = 256;
const dim3 dimblock(16, 16);

//==============================================================================
__global__ void kernel(cudaTextureObject_t tex, float* out, size_t width,
                       size_t height, float theta) {
  const size_t x   = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y   = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t idx = y * width + x;
  if (x < width && y < height) {
    // calculate normalized texture coordinates
    float u = x / float(width - 1);
    float v = y / float(height - 1);

    // transform texture coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // sample texture and assign to output array
    const auto col   = tex2D<float4>(tex, tu, tu);
    out[idx * 3]     = col.x;
    out[idx * 3 + 1] = col.y;
    out[idx * 3 + 2] = col.z;
  }
}
//==============================================================================
auto make_host_texture() {
  std::vector<float> h_original(width * height * 4, 0.0f);
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i         = x + width * y;
      h_original[i * 4] = 1;
    }
  }
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i             = x + width * y;
      h_original[i * 4 + 1] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i             = x + width * y;
      h_original[i * 4 + 2] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i             = x + width * y;
      h_original[i * 4 + 1] = 1;
      h_original[i * 4 + 2] = 1;
    }
  }
  return h_original;
}

void write_host_texture(const std::vector<float>& tex) {
  std::ofstream file{"untransformed.ppm"};
  if (file.is_open()) {
    file << "P3\n" << width << ' ' << height << "\n255\n";
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const size_t i = x + width * (height - 1 - y);
        file << static_cast<unsigned int>(tex[i * 4] * 255) << ' '
             << static_cast<unsigned int>(tex[i * 4 + 1] * 255) << ' '
             << static_cast<unsigned int>(tex[i * 4 + 2] * 255) << ' ';
      }
      file << '\n';
    }
  }
}
void write_transformed_texture(const std::vector<float>& transformed) {
  std::ofstream file{"transformed.ppm"};
  if (file.is_open()) {
    file << "P3\n" << width << ' ' << height << "\n255\n";
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const size_t i = x + width * (height - 1 - y);
        file << static_cast<unsigned int>(transformed[i * 3] * 255) << ' '
             << static_cast<unsigned int>(transformed[i * 3 + 1] * 255) << ' '
             << static_cast<unsigned int>(transformed[i * 3 + 2] * 255) << ' ';
      }
      file << '\n';
    }
  }
}

int main() {
  const auto h_original = make_host_texture();  // creates float-rgba texture with 4
                                                // differently colored areas
  write_host_texture(h_original);

  // upload texture data to cudaArray
  cudaArray_t d_arr_original;
  cudaChannelFormatDesc desc{32, 32, 32, 32, cudaChannelFormatKindFloat};
  cudaMallocArray(&d_arr_original, &desc, width, height);
  cudaMemcpy2DToArray(d_arr_original, 0, 0, static_cast<const void*>(h_original.data()),
                      width * sizeof(float), width * sizeof(float), height,
                      cudaMemcpyHostToDevice);

  // create texture from cudaArray
  cudaTextureObject_t d_tex;

  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType         = cudaResourceTypeArray;
  res_desc.res.array.array = d_arr_original;

  cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode       = cudaReadModeElementType;
  tex_desc.addressMode[0] = cudaAddressModeWrap;
  tex_desc.addressMode[1] = cudaAddressModeWrap;
  tex_desc.filterMode     = cudaFilterModeLinear;

  tex_desc.readMode         = cudaReadModeElementType;
  tex_desc.normalizedCoords = true;
  cudaCreateTextureObject(&d_tex, &res_desc, &tex_desc, nullptr);

  // create device memory for output of transformed texture
  float*             d_transformed;
  cudaMalloc(&d_transformed, width * height * 3 * sizeof(float));

  // call kernel
  const dim3 dimgrid(width / dimblock.x + 1, height / dimblock.y + 1);
  kernel<<<dimblock, dimgrid>>>(d_tex, &d_transformed[0], width, height, M_PI / 4);
  cudaDeviceSynchronize();

  // download transformed texture data and write
  std::vector<float> h_transformed(width*height*3);
  cudaMemcpy(&h_transformed[0], d_transformed, sizeof(float) * width * height * 3,
             cudaMemcpyDeviceToHost);
  write_transformed_texture(h_transformed);
}
