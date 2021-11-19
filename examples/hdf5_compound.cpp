#include <tatooine/hdf5.h>
#include <tatooine/tensor.h>
#include <tatooine/geometry/ellipse.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/reflection.h>
#include <tatooine/static_multidim_array.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
class S {
  std::array<int, 5> m_a{1, 2, 3, 4, 5};
  float  m_b;
  double m_c;
 public:
  S() = default;
  S(std::array<int, 5> const& a, float const b, double const c)
      : m_a{a}, m_b{b}, m_c{c} {}

  auto a() const -> auto const& { return m_a; }
  auto a(size_t const i) const { return m_a[i]; }
  auto b() const { return m_b; }
  auto c() const { return m_c; }

  auto set_a(std::array<int, 5> const& a) { m_a = a; }
  auto set_a(size_t const i, int const a) { m_a[i] = a; }
  auto set_b(float const b) { m_b = b; }
  auto set_c(double const c) { m_c = c; }
};
namespace tatooine::reflection {
TATOOINE_MAKE_ADT_REFLECTABLE(S,
    TATOOINE_REFLECTION_INSERT_METHOD(a, a()),
    TATOOINE_REFLECTION_INSERT_METHOD(b, b()),
    TATOOINE_REFLECTION_INSERT_METHOD(c, c()));
}
//==============================================================================
auto IO_S() {
  auto path = filesystem::path{"SDScompound.h5"};
  {
    auto file    = hdf5::file{path};
    auto dataset = file.create_dataset<S>("data", 1);

    auto s = S{{1, 2, 2, 2, 2}, 2.0f, 15.0};
    dataset.write(&s);
  }
  {
    auto file = hdf5::file{path};
    auto data = file.dataset<S>("data").read();

    auto const& s = data(0);
    std::cout << "[" << s.a(0);
    for (size_t i = 1; i < 5; ++i) {
      std::cout << ", " << s.a(i);
    }
    std::cout << "]\n";
    std::cout << s.b() << '\n';
    std::cout << s.c() << '\n';
  }
}
//==============================================================================
auto IO_static_multidim_array() {
  auto path   = filesystem::path{"static_multidim_array.h5"};
  using arr_t = static_multidim_array<double, x_fastest, tag::stack, 3, 3, 3>;
  auto s0     = arr_t::randu();
  {
    auto file    = hdf5::file{path};
    auto dataset = file.create_dataset<arr_t>("data", 1);

    dataset.write(&s0);
  }
  {
    auto file = hdf5::file{path};
    auto data = file.dataset<arr_t>("data").read();

    auto const& s1        = data(0);
    auto const  iteration = [&](auto const... is) {
      if (s0(is...) != s1(is...)) {
        std::cout << "static multidim array wrong!\n";
        return false;
      }
      return true;
    };
  }
}
//==============================================================================
auto IO_tensor() {
  auto path   = filesystem::path{"tensor.h5"};
  using tensor_t = tensor333;
  auto t0     = tensor_t::randu();
  {
    auto file    = hdf5::file{path};
    auto dataset = file.create_dataset<tensor_t>("data", 1);

    dataset.write(&t0);
  }
  {
    auto file = hdf5::file{path};
    auto data = file.dataset<tensor_t>("data").read();

    auto const& t1        = data(0);
    auto const  iteration = [&](auto const... is) {
      if (t0(is...) != t1(is...)) {
        std::cout << "tensor wrong!\n";
        return false;
      }
      return true;
    };
    for_loop(iteration, 3, 3, 3);
  }
}
//==============================================================================
auto IO_ellipse() {
  auto path   = filesystem::path{"ellipse.h5"};
  using ell_t = geometry::ellipse<real_t>;
  auto ell0     = ell_t{};
  {
    auto file    = hdf5::file{path};
    auto dataset = file.create_dataset<ell_t>("data", 1);
    dataset.write(&ell0);
  }
  {
    auto file = hdf5::file{path};
    auto data = file.dataset<ell_t>("data").read();

    auto const& ell1        = data(0);
    auto const  iteration_center = [&](auto const... is) {
      if (ell0.center()(is...) != ell1.center()(is...)) {
        std::cout << "ellipse center wrong!\n";
        return false;
      }
      return true;
    };
    auto const  iteration_S = [&](auto const... is) {
      if (ell0.S()(is...) != ell1.S()(is...)) {
        std::cout << "ellipse S wrong!\n";
        return false;
      }
      return true;
    };
    for_loop(iteration_S, 2, 2);
    for_loop(iteration_center, 2);
  }
}
//==============================================================================
auto IO_autonomous_particles() {
  auto path   = filesystem::path{"autonomous_particle.h5"};
  using particle_t = autonomous_particle2;
  auto particle0        = particle_t{vec{0.5, 0.5}, 0.0, 0.01};
  {
    auto file    = hdf5::file{path};
    auto dataset = file.create_dataset<particle_t>("data", 1);
    dataset.write(&particle0);
  }
  {
    auto file = hdf5::file{path};
    auto data = file.dataset<particle_t>("data").read();

    //auto const& particle1        = data(0);
    //auto const  iteration_center = [&](auto const... is) {
    //  if (particle0.center()(is...) != particle1.center()(is...)) {
    //    std::cout << "ellipse center wrong!\n";
    //    return false;
    //  }
    //  return true;
    //};
    //auto const  iteration_S = [&](auto const... is) {
    //  if (particle0.S()(is...) != particle1.S()(is...)) {
    //    std::cout << "ellipse S wrong!\n";
    //    return false;
    //  }
    //  return true;
    //};
    //for_loop(iteration_S, 2, 2);
    //for_loop(iteration_center, 2);
  }
}
//==============================================================================
auto main() -> int {
  auto dimsf = std::array{hsize_t(0)};
  auto maxdims = std::array{ H5S_UNLIMITED};
  auto dataset_space =
      H5Screate_simple(2, dimsf.data(), maxdims.data());
  H5Sclose(dataset_space);
  //IO_S();
  //IO_static_multidim_array();
  //IO_tensor();
  //IO_ellipse();
  //IO_autonomous_particles();
}
