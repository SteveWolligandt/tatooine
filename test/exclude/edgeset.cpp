#include <tatooine/edgeset.h>
#include <boost/range/algorithm.hpp>
#include <iostream>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

class edgeset_test : public edgeset<double, 3> {
 public:
  vertex v0, v1, v2, v3;
  edge   e0, e1, e2;
  edgeset_test() {
    v0 = insert_vertex(1, 2, 3);
    v1 = insert_vertex(2, 4, 6);
    v2 = insert_vertex(3, 7, 9);
    v3 = insert_vertex(4, 11, 13);

    e0 = insert_edge(v0, v1);
    e1 = insert_edge(v1, v2);
    e2 = insert_edge(v2, v3);
  }
};

//==============================================================================
TEST_CASE_METHOD(edgeset_test, "[edgeset] insertion", "[edgeset]") {
  using namespace boost;

  REQUIRE(num_vertices() == 4);
  REQUIRE(num_edges() == 3);

  REQUIRE(num_edges(v0) == 1);
  REQUIRE(num_edges(v1) == 2);
  REQUIRE(num_edges(v2) == 2);
  REQUIRE(num_edges(v3) == 1);

  REQUIRE(find(edges(v0), e0) != edges(v0).end());
  REQUIRE(find(edges(v0), e1) == edges(v0).end());
  REQUIRE(find(edges(v0), e2) == edges(v0).end());

  REQUIRE(find(edges(v1), e0) != edges(v1).end());
  REQUIRE(find(edges(v1), e1) != edges(v1).end());
  REQUIRE(find(edges(v1), e2) == edges(v1).end());

  REQUIRE(find(edges(v2), e0) == edges(v2).end());
  REQUIRE(find(edges(v2), e1) != edges(v2).end());
  REQUIRE(find(edges(v2), e2) != edges(v2).end());

  REQUIRE(find(edges(v3), e0) == edges(v3).end());
  REQUIRE(find(edges(v3), e1) == edges(v3).end());
  REQUIRE(find(edges(v3), e2) != edges(v3).end());
}

//==============================================================================
TEST_CASE_METHOD(edgeset_test, "[edgeset] removal", "[edgeset]") {
  using namespace boost;
  remove(e1);

  SECTION("first removal before tidy up") {
    REQUIRE(num_vertices() == 4);
    REQUIRE(num_edges() == 2);

    REQUIRE(num_edges(v0) == 1);
    REQUIRE(num_edges(v1) == 1);
    REQUIRE(num_edges(v2) == 1);
    REQUIRE(num_edges(v3) == 1);

    REQUIRE(find(edges(v0), e0) != edges(v0).end());
    REQUIRE(find(edges(v0), e1) == edges(v0).end());

    REQUIRE(find(edges(v1), e0) != edges(v1).end());
    REQUIRE(find(edges(v1), e1) == edges(v1).end());

    REQUIRE(find(edges(v2), e0) == edges(v2).end());
    REQUIRE(find(edges(v2), e2) != edges(v2).end());

    REQUIRE(find(edges(v3), e0) == edges(v3).end());
    REQUIRE(find(edges(v3), e2) != edges(v3).end());
  }

  tidy_up();
  // invalidates e1 and all successors
  // -> e1 becomes e2
  // -> e2 is out of bounds
  SECTION("first removal after tidy up") {
    REQUIRE(num_vertices() == 4);
    REQUIRE(num_edges() == 2);

    REQUIRE(num_edges(v0) == 1);
    REQUIRE(num_edges(v1) == 1);
    REQUIRE(num_edges(v2) == 1);
    REQUIRE(num_edges(v3) == 1);

    REQUIRE(find(edges(v0), e0) != edges(v0).end());
    REQUIRE(find(edges(v0), e1) == edges(v0).end());

    REQUIRE(find(edges(v1), e0) != edges(v1).end());
    REQUIRE(find(edges(v1), e1) == edges(v1).end());

    REQUIRE(find(edges(v2), e0) == edges(v2).end());
    REQUIRE(find(edges(v2), e1) != edges(v2).end());

    REQUIRE(find(edges(v3), e0) == edges(v3).end());
    REQUIRE(find(edges(v3), e1) != edges(v3).end());
  }

  remove(e0);
  SECTION("second removal before tidy") {
    REQUIRE(num_vertices() == 2);
    REQUIRE(num_edges() == 1);

    REQUIRE(num_edges(v2) == 1);
    REQUIRE(num_edges(v3) == 1);

    REQUIRE(find(edges(v2), e1) != edges(v2).end());
    REQUIRE(find(edges(v3), e1) != edges(v3).end());
  }

  tidy_up();
  // invalidates e0 and all successors
  // -> e0 becomes e1
  // -> e1 is out of bounds
  // deletes v0 and v1
  // -> v0 becomes v2
  // -> v1 becomes v3
  SECTION("second removal after tidy") {
    REQUIRE(num_vertices() == 2);
    REQUIRE(num_edges() == 1);
    REQUIRE(num_edges(v0) == 1);
    REQUIRE(num_edges(v1) == 1);
    REQUIRE(find(edges(v0), e0) != edges(v0).end());
    REQUIRE(find(edges(v1), e0) != edges(v1).end());
  }
}

//==============================================================================
TEST_CASE_METHOD(edgeset_test, "[edgeset] vertex removal", "[edgeset]") {
  using namespace boost;
  remove(v1);

  // SECTION("removal before tidy") {
  REQUIRE(num_vertices() == 2);
  REQUIRE(num_edges() == 1);

  REQUIRE(num_edges(v0) == 0);
  REQUIRE(num_edges(v1) == 0);
  REQUIRE(num_edges(v2) == 1);
  REQUIRE(num_edges(v3) == 1);

  REQUIRE(find(edges(v0), e0) == edges(v0).end());
  REQUIRE(find(edges(v0), e1) == edges(v0).end());
  REQUIRE(find(edges(v0), e2) == edges(v0).end());

  REQUIRE(find(edges(v1), e0) == edges(v1).end());
  REQUIRE(find(edges(v1), e1) == edges(v1).end());
  REQUIRE(find(edges(v1), e2) == edges(v1).end());

  REQUIRE(find(edges(v2), e0) == edges(v2).end());
  REQUIRE(find(edges(v2), e1) == edges(v2).end());
  REQUIRE(find(edges(v2), e2) != edges(v2).end());

  REQUIRE(find(edges(v3), e0) == edges(v3).end());
  REQUIRE(find(edges(v3), e1) == edges(v3).end());
  REQUIRE(find(edges(v3), e2) != edges(v3).end());

  REQUIRE(at(e2)[0] == v2);
  REQUIRE(at(e2)[1] == v3);
  // }

  tidy_up();
  // invalidates e1 and all successors
  // -> e1 becomes e2
  // -> e2 is out of bounds
  // SECTION("first removal after tidy up") {
  REQUIRE(num_vertices() == 2);
  REQUIRE(num_edges() == 1);

  REQUIRE(num_edges(v0) == 1);
  REQUIRE(num_edges(v1) == 1);

  REQUIRE(find(edges(v0), e0) != edges(v0).end());
  REQUIRE(find(edges(v1), e0) != edges(v1).end());

  REQUIRE(at(e0)[0] == v0);
  REQUIRE(at(e0)[1] == v1);
  // }

  remove(e0, false);
  // SECTION("edge removal before tidy up") {
  REQUIRE(num_vertices() == 2);
  REQUIRE(num_edges() == 0);

  REQUIRE(num_edges(v0) == 0);
  REQUIRE(num_edges(v1) == 0);

  REQUIRE(find(edges(v0), e0) == edges(v0).end());
  REQUIRE(find(edges(v1), e0) == edges(v1).end());
  // }

  tidy_up();
  // SECTION("edge removal after tidy up") {
  REQUIRE(num_vertices() == 2);
  REQUIRE(num_edges() == 0);

  REQUIRE(num_edges(v0) == 0);
  REQUIRE(num_edges(v1) == 0);

  REQUIRE(find(edges(v0), e0) == edges(v0).end());
  REQUIRE(find(edges(v1), e0) == edges(v1).end());
  // }
}

//==============================================================================
TEST_CASE_METHOD(edgeset_test, "[edgeset] vertex removal2", "[edgeset]") {
  using namespace boost;
  remove(v1);
  remove(e2, false);
  tidy_up();
  REQUIRE(num_vertices() == 2);
  REQUIRE(num_edges() == 0);

  REQUIRE(num_edges(v0) == 0);
  REQUIRE(num_edges(v1) == 0);

  REQUIRE(find(edges(v0), e0) == edges(v0).end());
  REQUIRE(find(edges(v1), e0) == edges(v1).end());
}

//==============================================================================
TEST_CASE_METHOD(edgeset_test, "[edgeset] Property", "[edgeset]") {
  using namespace boost;
  auto& prop = add_edge_property<double>("e:prop");
  REQUIRE(prop.size() == 3);
}

//------------------------------------------------------------------------------
TEST_CASE("[edgeset] copy", "[edgeset]") {
  edgeset<double, 2> es;

  std::vector v{es.insert_vertex(1, 2), es.insert_vertex(2, 3),
                es.insert_vertex(3, 4)};
  std::vector e{es.insert_edge(v[0], v[1]), es.insert_edge(v[1], v[2])};

  auto& vp = es.add_vertex_property<int>("vertex_property");
  auto& ep = es.add_edge_property<int>("edge_property");
  vp[v[0]] = 1;
  vp[v[1]] = 2;
  vp[v[2]] = 4;

  ep[e[0]] = 500;
  ep[e[1]] = 1000;

  edgeset<double, 2> copy{es};

  REQUIRE(es.num_vertices() == copy.num_vertices());
  REQUIRE(es.num_edges() == copy.num_edges());
  REQUIRE(copy.num_edges(v[0]) == 1);
  REQUIRE(copy.num_edges(v[1]) == 2);
  REQUIRE(copy.num_edges(v[2]) == 1);
  const auto& vp_copy = copy.vertex_property<int>("vertex_property");
  const auto& ep_copy = copy.edge_property<int>("edge_property");
  for (auto v : es.vertices()) {
    REQUIRE(approx_equal(es[v], copy[v], 0));
    REQUIRE(vp[v] == vp_copy[v]);
  }
  for (auto e : es.edges()) {
    REQUIRE(es[e] == copy[e]);
    REQUIRE(ep[e] == ep_copy[e]);
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
