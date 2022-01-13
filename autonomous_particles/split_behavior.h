#ifndef TATOOINE_AUTONOMOUS_PARTICLES_SPLIT_BEHAVIOR_H
#define TATOOINE_AUTONOMOUS_PARTICLES_SPLIT_BEHAVIOR_H
namespace tatooine::autonomous_particles {
enum class split_behavior_t {
  two_splits,
  three_splits,
  three_in_square_splits,
  five_splits,
  seven_splits,
  centered_four,
  unknown
};
}
#endif
