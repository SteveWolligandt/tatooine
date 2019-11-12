#include <Tatooine/streamsurface.h>
#include "datasets.h"

int main(int argc, char** argv) {
  FixedDoubleGyre vf;
  tatooine::Streamsurface ssf {
    vf, 100, 
      tatooine::grid_edge {
        settings_t<FixedDoubleGyre>::domain.at(3, 3),
        settings_t<FixedDoubleGyre>::domain.at(4, 4)
      }
  };

  ssf.discretize(4, 0.01, -5, 5);
}
