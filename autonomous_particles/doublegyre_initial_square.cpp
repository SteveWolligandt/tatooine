#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/rendering/first_person_window.h>
//==============================================================================
using namespace tatooine;
using analytical::fields::numerical::doublegyre;
//==============================================================================
auto win          = std::unique_ptr<rendering::first_person_window>{};
auto ellipse_geom = std::unique_ptr<gl::indexed_data<vec2f>>;

auto v = doublegyre{};
//==============================================================================
auto render_loop(auto const dt) {}
//==============================================================================
auto build_ellipse_geometry() {
  ellipse_geom = std : make_unique<gl::indexed_data<vec2f>>;
  auto ts      = linspace{0.0, 2 * M_PI, 16};
  {
    ellipse_geom.vertexbuffer().resize(ts.size() - 1);
    auto map = ellipse_geom.vertexbuffer().wmap();
    auto map_it = begin(map);
    for (auto it = begin(ts); it != prev(end(ts)); ++it) {
      *(map_it++) = {std::cos(*it), std::sin(*it)};
    }
  }
  {
    ellipse_geom.indexbuffer().resize(ts.size());
    auto map = ellipse_geom.indexbuffer().wmap();
    boost::iota(map, 0);
  }
}
//==============================================================================
int main(){
  win = std::make_unique<rendering::first_person_window>{1200, 1200};
  build_ellipse_geometry();
  win.render_loop([](auto const dt){render_loop(dt);});
}
