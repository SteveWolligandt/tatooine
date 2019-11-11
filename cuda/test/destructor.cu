#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================

class S {
  size_t n;
  float* dev_buffer;

  public:
  __host__ S(size_t _n) : n{_n} {
    cudaMalloc(&dev_buffer, sizeof(float) * n);
  }

  __host__ __device__ S(const S& other) : n{other.n} {
#   ifdef __CUDA_ARCH__
    printf("copy pointer on device\n");
    dev_buffer = other.dev_buffer;
#   else
    printf("deep copy on host\n");
    cudaMalloc(&dev_buffer, sizeof(float) * n);
    cudaMemcpy(dev_buffer, other.dev_buffer, sizeof(float) * n, cudaMemcpyDeviceToDevice);
#   endif
  }
  
  __host__ __device__ ~S() {
#   ifdef __CUDA_ARCH__
      printf("do not free on device\n");
#   else
      printf("free on host\n");
      cudaFree(dev_buffer);
      dev_buffer = nullptr;
#   endif
  }

  __host__ __device__ auto& operator[](size_t i) {
    return dev_buffer[i];
  }
};

__global__ void kernel(S s) {
  s[0] = 2;
}

TEST_CASE() {
  S s(8);
  auto s2 = s;
  kernel<<<1,1>>>(s);
  //s[1] = 3;
}

//==============================================================================
}
}
}
//==============================================================================
