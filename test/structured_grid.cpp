#include <tatooine/structured_grid.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(structured_grid3, "unstructured_grid_1",
                 "[unstructured_grid]") {
  read("/home/steve/firetec/valley_losAlamos/output.1000.vts");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
