#include "steadification_without_graphics.h"
#include <tatooine/grid_sampler.h>

Steadification::Steadification(const tatooine::grid<real_t, 2>& _grid,
                               real_t _t0, real_t _btau, real_t _ftau,
                               size_t _seed_res, real_t _stepsize)
    : grid(_grid),
      t0{_t0},
      btau{_btau},
      ftau{_ftau},
      seed_res{_seed_res},
      stepsize{_stepsize},
      ribbon_cache(10000, tatooine::total_memory() / 2) {}
