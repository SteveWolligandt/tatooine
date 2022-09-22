#ifndef TATOOINE_SIMULATED_ANNEALING_H
#define TATOOINE_SIMULATED_ANNEALING_H
//==============================================================================
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>

#include "constants.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Energy, typename Status>
struct simulated_annealing_listener {
  using energy_type = Energy;
  using status_type = Status;
  virtual auto on_new_best_status(size_t /*i*/, Energy const & /*best_energy*/,
                                  Status const & /*best_status*/,
                                  Energy const & /*cur_energy*/,
                                  Status const & /*cur_status*/) const -> void {
  }
  virtual auto on_end_of_iteration(size_t /*i*/, Energy const & /*best_energy*/,
                                   Status const & /*best_status*/,
                                   Energy const & /*cur_energy*/,
                                   Status const & /*cur_status*/) const
      -> void {}
  virtual auto on_using_worse(size_t /*i*/, Energy const & /*best_energy*/,
                              Status const & /*best_status*/,
                              Energy const & /*cur_energy*/,
                              Status const & /*cur_status*/) const -> void {}
  virtual auto on_going_back(size_t /*i*/, Energy const & /*best_energy*/,
                             Status const & /*best_status*/,
                             Energy const & /*cur_energy*/,
                             Status const & /*cur_status*/) const -> void {}
};
//==============================================================================
template <template <typename> typename Comparator = std::less,
          typename InitialStatus, typename EnergyFunction,
          typename TemperaturFunction, typename NeighborFunction,
          typename RandomEngine>
auto simulated_annealing(
    InitialStatus &&initial_status, size_t const num_iterations,
    EnergyFunction &&energy_fun, TemperaturFunction &&temperature_fun,
    NeighborFunction &&neighbor_fun, RandomEngine &&random_engine,
    std::vector<simulated_annealing_listener<
        std::decay_t<decltype(energy_fun(std::declval<InitialStatus>()))>,
        std::decay_t<InitialStatus>> *> const &listeners = {}) {
  using Status = std::decay_t<InitialStatus>;
  using Energy = decltype(energy_fun(std::declval<Status>()));

  Comparator<Energy>                     comparator;
  std::uniform_real_distribution<double> uni01{0.0, 1.0};

  Status status = std::forward<InitialStatus>(initial_status);
  Energy energy = energy_fun(initial_status);

  auto best                        = std::pair{energy, initial_status};
  auto &[best_energy, best_status] = best;

  for (size_t i = 0; i < num_iterations; ++i) {
    auto   t          = temperature_fun(i);
    Status new_status = neighbor_fun(status, t);
    Energy new_energy = energy_fun(new_status);

    // always accept status with better energy
    if (comparator(new_energy, best_energy)) {
      status = best_status = std::move(new_status);
      energy = best_energy = std::move(new_energy);
      for (auto const l : listeners) {
        l->on_new_best_status(i, best_energy, best_status, energy, status);
      }
      for (auto const &l : listeners) {
        l->on_end_of_iteration(i, best_energy, best_status, energy, status);
      }

    } else {
      auto const energy_delta = new_energy - energy;

      if (gcem::exp(-energy_delta / t) > uni01(random_engine)) {
        // take new even if worse
        status = std::move(new_status);
        energy = std::move(new_energy);
        for (auto const & l : listeners) {
          l->on_using_worse(i, best_energy, best_status, energy, status);
        }

      } else {
        // go back to best
        status = best_status;
        energy = best_energy;
        for (auto const &l : listeners) {
          l->on_going_back(i, best_energy, best_status, energy, status);
        }
      }
      for (auto const &l : listeners) {
        l->on_end_of_iteration(i, best_energy, best_status, energy, status);
      }
    }
  }
  return best;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
