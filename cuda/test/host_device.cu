#include <catch2/catch.hpp>
#include <tatooine/cuda/functions.cuh>
//==============================================================================
namespace tatooine{
namespace cuda{
namespace test{
//==============================================================================

__global__ void kernel(bool* dev_on_host, bool* dev_on_device) {
  *dev_on_host = on_host();
  *dev_on_device = on_device();
}
//==============================================================================
TEST_CASE("host_device") {
  REQUIRE(on_host() == true);
  REQUIRE(on_device() == false);
  auto dev_on_host = malloc<bool>(1);
  auto dev_on_device = malloc<bool>(1);
  kernel<<<1,1>>>(dev_on_host, dev_on_device);
  bool host_on_host, host_on_device;
  memcpy(&host_on_host, dev_on_host, sizeof(bool), cudaMemcpyDeviceToHost);
  memcpy(&host_on_device, dev_on_device, sizeof(bool), cudaMemcpyDeviceToHost);
  REQUIRE(host_on_host == false);
  REQUIRE(host_on_device == true);
}


//==============================================================================
} // namespace tatooine
} // namespace cuda
} // namespace test
//==============================================================================
