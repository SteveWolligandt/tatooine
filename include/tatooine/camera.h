#ifndef CG_CAMERA_H
#define CG_CAMERA_H
//==============================================================================
#include <tatooine/ray.h>
#include <tatooine/clonable.h>
#include <tatooine/concepts.h>
#include <array>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief Interface for camera implementations.
///
/// Implementations must override the ray method that casts rays through the
/// camera's image plane.
template <real_number Real>
class camera : public clonable<camera<Real>> {
  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  const std::array<size_t, 2> m_resolution;
 public:
  using this_t            = camera<Real>;
  using parent_clonable_t = clonable<camera<Real>>;
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  camera(size_t res_x, size_t res_y) : m_resolution{res_x, res_y} {}
  virtual ~camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in x-direction.
  size_t plane_width() const { return m_resolution[0]; }
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in y-direction.
  size_t plane_height() const { return m_resolution[1]; }
  //----------------------------------------------------------------------------
  // interface methods
  //----------------------------------------------------------------------------
  /// \brief Gets a ray through plane at pixel with coordinate [x,y].
  ///
  /// [0,0] is bottom left.
  /// ray goes through center of pixel.
  /// This method must be overridden in camera implementations.
  virtual tatooine::ray<Real, 3> ray(Real x, Real y) const = 0;
};
//==============================================================================
}  // namespace cg
//==============================================================================
#endif
