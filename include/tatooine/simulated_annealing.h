#ifndef TATOOINE_SIMULATED_ANNEALING_H
#define TATOOINE_SIMULATED_ANNEALING_H

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include "constants.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Energy, typename Status>
struct simulated_annealing_listener {
  using energy_t = Energy;
  using status_t = Status;
  virtual void on_new_best_status(size_t /*i*/, const Energy & /*best_energy*/,
                                  const Status & /*best_status*/,
                                  const Energy & /*cur_energy*/,
                                  const Status & /*cur_status*/) const {}
  virtual void on_end_of_iteration(size_t /*i*/, const Energy & /*best_energy*/,
                                   const Status & /*best_status*/,
                                   const Energy & /*cur_energy*/,
                                   const Status & /*cur_status*/) const {}
  virtual void on_using_worse(size_t /*i*/, const Energy & /*best_energy*/,
                              const Status & /*best_status*/,
                              const Energy & /*cur_energy*/,
                              const Status & /*cur_status*/) const {}
  virtual void on_going_back(size_t /*i*/, const Energy & /*best_energy*/,
                             const Status & /*best_status*/,
                             const Energy & /*cur_energy*/,
                             const Status & /*cur_status*/) const {}
};

template <template <typename> typename Comparator = std::less,
          typename InitialStatus, typename EnergyFunction,
          typename TemperaturFunction, typename NeighborFunction,
          typename RandomEngine>
auto simulated_annealing(
    InitialStatus &&initial_status, const size_t num_iterations,
    EnergyFunction &&energy_fun, TemperaturFunction &&temperature_fun,
    NeighborFunction &&neighbor_fun, RandomEngine &&random_engine,
    const std::vector<simulated_annealing_listener<
        std::decay_t<decltype(energy_fun(std::declval<InitialStatus>()))>,
        std::decay_t<InitialStatus>> *> &listeners = {}) {
  using Status = std::decay_t<InitialStatus>;
  using Energy = decltype(energy_fun(std::declval<Status>()));

  Comparator<Energy>                     comparator;
  std::uniform_real_distribution<double> uni01{0.0, 1.0};

  Status status = std::forward<InitialStatus>(initial_status);
  Energy energy = energy_fun(initial_status);

  std::pair best{energy, initial_status};
  auto &[best_energy, best_status] = best;

  for (size_t i = 0; i < num_iterations; ++i) {
    auto   t          = temperature_fun(i);
    Status new_status = neighbor_fun(status, t);
    Energy new_energy = energy_fun(new_status);

    // always accept status with better energy
    if (comparator(new_energy, best_energy)) {
      status = best_status = std::move(new_status);
      energy = best_energy = std::move(new_energy);
      for (const auto l : listeners) {
        l->on_new_best_status(i, best_energy, best_status, energy, status);
      }
      for (const auto &l : listeners) {
        l->on_end_of_iteration(i, best_energy, best_status, energy, status);
      }

    } else {
      const auto energy_delta = new_energy - energy;

      if (std::exp(-energy_delta / t) > uni01(random_engine)) {
        // take new even if worse
        status = std::move(new_status);
        energy = std::move(new_energy);
        for (const auto &l : listeners) {
          l->on_using_worse(i, best_energy, best_status, energy, status);
        }

      } else {
        // go back to best
        status = best_status;
        energy = best_energy;
        for (const auto &l : listeners) {
          l->on_going_back(i, best_energy, best_status, energy, status);
        }
      }
      for (const auto &l : listeners) {
        l->on_end_of_iteration(i, best_energy, best_status, energy, status);
      }
    }
  }
  return best;
}

// template <typename Real, typename Status>
// struct SimulatedAnnealingListener {
//   using real_type   = Real;
//   using status_t = Status;
//   virtual void on_new_best_status(size_t [>i*/, real_type /*temperature<],
//                                   real_type [>best_energy<],
//                                   const status_t & [>best_status<],
//                                   real_type [>cur_energy<],
//                                   const status_t & [>cur_status<]) const {}
//   virtual void on_end_of_iteration(size_t [>i*/, real_type /*temperature<],
//                                    real_type [>best_energy<],
//                                    const status_t & [>best_status<],
//                                    real_type [>cur_energy<],
//                                    const status_t & [>cur_status<]) const {}
//   virtual void on_using_worse(size_t [>i*/, real_type /*temperature<],
//                               real_type [>best_energy<],
//                               const status_t & [>best_status<],
//                               real_type [>cur_energy<],
//                               const status_t & [>cur_status<]) const {}
//   virtual void on_going_back(size_t [>i*/, real_type /*temperature<],
//                              real_type [>best_energy<],
//                              const status_t & [>best_status<],
//                              real_type [>cur_energy<],
//                              const status_t & [>cur_status<]) const {}
// };
//
// //==============================================================================
// //! Tries to find global minima (comparator_t = std::less) or maxima
// //! (comparator_t = std::greater).
// template <typename Real, typename Status,
//           template <typename> typename comparator_t = std::less>
// class SimulatedAnnealing {
//  public:
//   using real_type     = Real;
//   using status_t   = Status;
//   using listener_t = SimulatedAnnealingListener<real_type, status_t>;
//
//  private:
//   std::vector<const listener_t *> m_listeners;
//   //----------------------------------------------------------------------------
//  public:
//   void add_listener(const listener_t &l) { m_listeners.push_back(&l); }
//
//   //----------------------------------------------------------------------------
//   template <typename energy_f, typename temperature_f, typename next_f,
//             typename random_engine_t = std::mt19937_64>
//   auto operator()(const status_t &initial_status, const size_t c,
//                   energy_f &&energy, temperature_f &&temperature, next_f
//                   &&next, random_engine_t &&random_engine = random_engine_t{
//                       std::random_device{}()}) {
//     comparator_t<real_type> comparator;
//     status_t             cur_status = initial_status;
//     real_type               cur_energy = energy(initial_status);
//
//     status_t best_status = initial_status;
//     real_type   best_energy = cur_energy;
//
//     std::uniform_real_distribution<real_type> uni_0_1{0.0, 1.0};
//
//     for (size_t i = 0; i < c; ++i) {
//       auto     t          = temperature(i);
//       status_t new_status = next(cur_status, t, i);
//       real_type   new_energy = energy(new_status);
//
//       // always accept status with better energy
//       if (comparator(new_energy, best_energy)) {
//         best_status = new_status;
//         best_energy = new_energy;
//         cur_status  = new_status;
//         cur_energy  = new_energy;
//         for (const auto &l : m_listeners) {
//           l->on_new_best_status(i, t, best_energy, best_status, cur_energy,
//                                 cur_status);
//         }
//         for (const auto &l : m_listeners) {
//           l->on_end_of_iteration(i, t, best_energy, best_status, cur_energy,
//                                  cur_status);
//         }
//         continue;
//       }
//
//       auto energy_delta = new_energy - cur_energy;
//
//       // sometimes accept status with lower energy
//       // if (std::exp(energy_delta / t) > uni_0_1(random_engine)) {
//       if (std::exp(-energy_delta / (t * constants::boltzmann)) >
//           uni_0_1(random_engine)) {
//         cur_status = new_status;
//         cur_energy = new_energy;
//         for (const auto &l : m_listeners) {
//           l->on_using_worse(i, t, best_energy, best_status, cur_energy,
//                             cur_status);
//         }
//       } else {
//         cur_status = best_status;
//         cur_energy = best_energy;
//         for (const auto &l : m_listeners) {
//           l->on_going_back(i, t, best_energy, best_status, cur_energy,
//                            cur_status);
//         }
//       }
//
//       for (const auto &l : m_listeners) {
//         l->on_end_of_iteration(i, t, best_energy, best_status, cur_energy,
//                                cur_status);
//       }
//     }
//     return std::pair{best_energy, best_status};
//   }
// };

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
