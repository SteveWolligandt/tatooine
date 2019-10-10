#include <tatooine/algorithm.h>
#include <tatooine/mesh.h>
#include <tatooine/quadtree.h>
#include <algorithm>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/numeric.hpp>
#include <catch2/catch.hpp>
#include <chrono>
#include <iostream>

//==============================================================================
namespace tatooine::test {
//==============================================================================
struct quadtree_test : quadtree<double, mesh<double, 2>> {
  using mesh_t   = ::tatooine::mesh<double, 2>;
  using parent_t = quadtree<double, mesh_t>;
  using vertex   = mesh_t::vertex;
  using edge     = mesh_t::edge;
  using face     = mesh_t::face;
  mesh_t m;
  quadtree_test() : parent_t{m} {}
  quadtree_test(const boundingbox<double, 2>& bb) : parent_t{m, bb} {}
};

struct quadtree_test0 : quadtree_test {
  quadtree_test0() {
    m.insert_vertex(-1, -1);
    m.insert_vertex(1, -1);
    m.insert_vertex(-1, 1);
    m.insert_vertex(1, 1);
    m.insert_edge(0, 1);
    m.insert_edge(0, 2);
    m.insert_edge(1, 3);
    m.insert_edge(2, 3);
    calc_boundaries();
    insert_all_vertices();
  }
};
TEST_CASE_METHOD(quadtree_test0, "quadtree", "[quadtree]") {
  // indices per child node
  std::vector<std::vector<vertex>> vertex_requirements{
      {vertex{0}}, {vertex{1}}, {vertex{2}}, {vertex{3}}};
  std::vector<std::vector<edge>> edge_requirements{{edge{0}, edge{1}},
                                                   {edge{0}, edge{2}},
                                                   {edge{1}, edge{3}},
                                                   {edge{2}, edge{3}}};

  std::vector<vertex> all_vertices(m.num_vertices(), vertex{0});
  std::vector<edge>   all_edges(m.num_edges(), edge{0});

  boost::range::iota(all_vertices, 0ul);
  boost::range::iota(all_edges, 0ul);

  std::array<std::vector<vertex>, 4> inv_vertex_requirements;
  std::array<std::vector<edge>, 4>   inv_edge_requirements;

  for (size_t i = 0; i < 4; ++i) {
    boost::set_difference(all_vertices, vertex_requirements[i],
                          std::back_inserter(inv_vertex_requirements[i]));
    boost::set_difference(all_edges, edge_requirements[i],
                          std::back_inserter(inv_edge_requirements[i]));
  }

  for (unsigned int i = 0; i < 4; ++i) {
    auto vertices = at(i).vertices();
    auto edges    = at(i).edges();

    for (const auto& r : vertex_requirements[i]) REQUIRE(contains(r, vertices));
    for (const auto& r : edge_requirements[i]) REQUIRE(contains(r, edges));

    for (const auto& r : inv_vertex_requirements[i])
      REQUIRE(!contains(r, vertices));
    for (const auto& r : inv_edge_requirements[i]) REQUIRE(!contains(r, edges));
  }
}

//==============================================================================
struct quadtree_test1 : quadtree_test {
  quadtree_test1() {
    for (auto y : linspace(-1.0, 1.0, 100)) {
      for (auto x : linspace(-1.0, 1.0, 100)) { m.insert_vertex(x, y); }
    }
    calc_boundaries();
    insert_all_vertices();
  }
};
TEST_CASE_METHOD(quadtree_test1, "quadtree_predicate",
                 "[quadtree][predicate]") {
  using timer = std::chrono::high_resolution_clock;
  using ms    = std::chrono::duration<double, std::milli>;
  using namespace boost;
  using namespace boost::range;
  using namespace boost::adaptors;

  // specify spatial boundaries for search area
  double min_x = 0.2, max_x = 0.3;
  double min_y = 0.6, max_y = 0.8;

  // create constraints for filtering
  auto x_constraint = make_x_constraint(min_x, max_x);
  auto y_constraint = make_y_constraint(min_y, max_y);

  // filter points and measure time
  std::set<vertex> quadtree_filtered;
  std::vector<ms>  quadtree_times;
  for (size_t i = 0; i < 1000; ++i) {
    quadtree_filtered.clear();
    auto before       = timer::now();
    quadtree_filtered = vertices(x_constraint, y_constraint);
    auto after        = timer::now();
    quadtree_times.push_back(after - before);
  }

  // do the same with bruteforce
  std::vector<ms>     brute_force_times;
  std::vector<vertex> brute_force_filtered;
  for (size_t i = 0; i < 1000; ++i) {
    brute_force_filtered.clear();
    brute_force_filtered.reserve(1);
    auto before = timer::now();
    for (size_t i = 0; i < m.num_vertices(); ++i)
      if (min_x <= m[vertex{i}](0) && m[vertex{i}](0) <= max_x &&
          min_y <= m[vertex{i}](1) && m[vertex{i}](1) <= max_y)
        brute_force_filtered.push_back(vertex{i});
    auto after = timer::now();
    brute_force_times.push_back(after - before);
  }

  // require that quadtree search is faster
  {
    auto to_count             = [](const auto& t) { return t.count(); };
    auto quadtree_time_counts = quadtree_times | transformed(to_count);
    auto bf_time_counts       = brute_force_times | transformed(to_count);
    auto quadtree_timing =
        accumulate(quadtree_time_counts, 0.0) / quadtree_times.size();
    auto bf_timing = accumulate(bf_time_counts, 0.0) / brute_force_times.size();
    INFO("quadtree filtering slower than bruteforce filtering");
    REQUIRE(quadtree_timing < bf_timing);
    std::cout << "quadtree timing  : " << quadtree_timing << "ms\n";
    std::cout << "bruteforce timing: " << bf_timing << "ms\n";
  }

  // check if points are equal
  REQUIRE(brute_force_filtered.size() == quadtree_filtered.size());
  for (const auto& i : brute_force_filtered)
    REQUIRE(quadtree_filtered.find(i) != quadtree_filtered.end());

  // check if points are actually in defined range
  for (const auto& i : quadtree_filtered) {
    const auto& x = m[i];
    INFO(x);
    REQUIRE(x(0) >= min_x);
    REQUIRE(x(1) >= min_y);
    REQUIRE(x(0) <= max_x);
    REQUIRE(x(1) <= max_y);
  }
}

//==============================================================================
struct quadtree_test2 : quadtree_test {
  vertex v0, v1, v2, v3, v4;
  face   f;

  quadtree_test2() : quadtree_test{boundingbox{vec{0.0, 0.0}, vec{4.0, 4.0}}} {
    v0 = m.insert_vertex(0.2, 3.3);
    v1 = m.insert_vertex(1.5, 0.1);
    v2 = m.insert_vertex(2.5, 3.5);
    v3 = m.insert_vertex(2.5, 0.5);
    v4 = m.insert_vertex(2.5, 1.5);
    f  = m.insert_face(v0, v1, v2);
    insert_all_vertices();
  }
};
TEST_CASE_METHOD(quadtree_test2, "quadtree_face", "[quadtree][face]") {
  REQUIRE(contains(v0, at(LEFT_TOP).local_vertices()));
  REQUIRE(contains(v1, at(LEFT_BOTTOM).local_vertices()));
  REQUIRE(contains(v2, at(RIGHT_TOP).local_vertices()));

  REQUIRE(at(RIGHT_BOTTOM).is_splitted());
  REQUIRE(contains(f, at(LEFT_TOP).local_faces()));
  REQUIRE(contains(f, at(RIGHT_TOP).local_faces()));
  REQUIRE(contains(f, at(LEFT_BOTTOM).local_faces()));
  REQUIRE(at(RIGHT_BOTTOM).local_faces().empty());
  REQUIRE_FALSE(contains(f, at(RIGHT_BOTTOM).local_faces()));

  REQUIRE(contains(f, at(RIGHT_BOTTOM).at(LEFT_TOP).local_faces()));
  REQUIRE_FALSE(contains(f, at(RIGHT_BOTTOM).at(RIGHT_TOP).local_faces()));
  REQUIRE_FALSE(contains(f, at(RIGHT_BOTTOM).at(LEFT_BOTTOM).local_faces()));
  REQUIRE_FALSE(contains(f, at(RIGHT_BOTTOM).at(RIGHT_BOTTOM).local_faces()));

  REQUIRE(at(LEFT_TOP).local_edges().size() == 2);
  REQUIRE(at(RIGHT_TOP).local_edges().size() == 2);
  REQUIRE(at(LEFT_BOTTOM).local_edges().size() == 2);
  REQUIRE(at(RIGHT_BOTTOM).at(LEFT_TOP).local_edges().size() == 1);
  REQUIRE(at(RIGHT_BOTTOM).at(RIGHT_TOP).local_edges().empty());
  REQUIRE(at(RIGHT_BOTTOM).at(LEFT_BOTTOM).local_edges().empty());
  REQUIRE(at(RIGHT_BOTTOM).at(RIGHT_BOTTOM).local_edges().empty());
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
