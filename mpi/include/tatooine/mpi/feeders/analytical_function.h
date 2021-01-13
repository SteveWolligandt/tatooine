#ifndef TATOOINE_MPI_FEEDERS_ANALYTICAL_FUNCTION_H
#define TATOOINE_MPI_FEEDERS_ANALYTICAL_FUNCTION_H
//==============================================================================
#include <tatooine/tensor.h>
#include <array>
#include <memory>
#include <set>
//==============================================================================
namespace tatooine::mpi::feeders {
//==============================================================================
class ScalarFieldInFlow {
 public:
  using bbox_t = std::array<double, 8>;
  bbox_t bbox;

  ScalarFieldInFlow() : bbox{0, 0, 0, 0, 0, 0, 0, 0} {}
  explicit ScalarFieldInFlow(const bbox_t& bbox) : bbox(bbox) {}
  virtual ~ScalarFieldInFlow(){};

  virtual double          s(double x, double y, double z, double t) = 0;
  virtual vec3 g(double x, double y, double z, double t) = 0;
  virtual vec3 v(double x, double y, double z, double t) = 0;
};

class EggBox : public ScalarFieldInFlow {
  // A  % Amplitude
  // Bx % Wellenlänge
  // By % Wellenlänge
  // Cx % Phasenverschiebung
  // Cy % Phasenverschiebung
 public:
  double a, bx, by, cx, cy;

  EggBox(double a, double bx, double by, double cx, double cy);

  double          s(double x, double y, double z, double t);
  vec3 g(double x, double y, double z, double t);
  vec3 v(double x, double y, double z, double t);
};

class TileBox : public ScalarFieldInFlow {
 public:
  double k;  // speed of curling behavior
  double l;  // speed of translation in x direction

  TileBox(double k, double l);
  double          s(double x, double y, double z, double t);
  vec3 g(double x, double y, double z, double t);
  vec3 v(double x, double y, double z, double t);
};

class Plane : public ScalarFieldInFlow {
 public:
  Plane();
  double          s(double x, double y, double z, double t);
  vec3 g(double x, double y, double z, double t);
  vec3 v(double x, double y, double z, double t);
};

/// Field that returns linearly interpolated values between time steps of
/// another Scalar Field in Flow
class InterpolatedField : public ScalarFieldInFlow {
 public:
  InterpolatedField(std::unique_ptr<ScalarFieldInFlow>&& base,
                    std::set<double>                     steps);
  double          s(double x, double y, double z, double t);
  vec3 g(double x, double y, double z, double t);
  vec3 v(double x, double y, double z, double t);

 private:
  std::unique_ptr<ScalarFieldInFlow> _base;
  std::set<double>                   _steps;
};
//==============================================================================
}  // namespace tatooine::mpi::feeders
//==============================================================================
#endif
