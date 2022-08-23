#include <tatooine/mesh.h>
#include <boost/range/algorithm.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace boost::range;

//==============================================================================
namespace tatooine::test {
//==============================================================================
/// creates a mesh with 4 vertices, 2 faces and 5 edges.
///
///       [v2]
///         x
///        /|\
///       / | \
/// [v0] x  |  x [v3]
///       \ | /
///        \|/
///         x
///        [v1]
//------------------------------------------------------------------------------
struct mesh_test : public mesh<double, 3> {
  //----------------------------------------------------------------------------
  using parent_t = mesh<double, 3>;
  using pos_type  = typename parent_t::pos_type;
  using parent_t::edge;
  using parent_t::face;
  using parent_t::vertex;

  //----------------------------------------------------------------------------
  static const pos_type p0;
  static const pos_type p1;
  static const pos_type p2;
  static const pos_type p3;

  //----------------------------------------------------------------------------
  vertex v0, v1, v2, v3;
  edge   e0, e1, e2, e3, e4;
  face   f0, f1;

  //----------------------------------------------------------------------------
  //! constructor
  mesh_test() {
    // add vertices
    v0 = insert_vertex(p0);
    v1 = insert_vertex(p1);
    v2 = insert_vertex(p2);
    v3 = insert_vertex(p3);

    // add faces and automatically create edges
    f0 = insert_face(v0, v1, v2);
    f1 = insert_face(v1, v3, v2);

    // extract edges
    auto edge_iter = edges().begin();
    e0             = *edge_iter;
    ++edge_iter;
    e1 = *edge_iter;
    ++edge_iter;
    e2 = *edge_iter;
    ++edge_iter;
    e3 = *edge_iter;
    ++edge_iter;
    e4 = *edge_iter;
  }
  //----------------------------------------------------------------------------
};  // MeshTestFixture
//------------------------------------------------------------------------------

const mesh_test::pos_type mesh_test::p0{1, 2, 3};
const mesh_test::pos_type mesh_test::p1{2, 3, 4};
const mesh_test::pos_type mesh_test::p2{3, 4, 5};
const mesh_test::pos_type mesh_test::p3{4, 5, 6};

//==============================================================================
// TEST_CASE_METHOD(mesh_test, "[mesh] number of edges", "[mesh]") {
//   // at default duplicate edges must be filtered out
//   REQUIRE(num_edges() == 5);
//
//   // number of edges and capacity must be zero
//   clear_edges();
//   REQUIRE(m_edges.size() == 0);
//   REQUIRE(m_edges.capacity() == 0);
//
//   // you can allow duplicate edges by disabling the duplicate_check
//   // when inserting edges of faces
//   clear_edges();
//   insert_edges_of_faces(false);
//   REQUIRE(num_edges() == 6);
// }

//==============================================================================
TEST_CASE_METHOD(mesh_test, "[mesh] edges of face", "[mesh]") {
  std::map<face, std::vector<edge>> reqirements;
  reqirements[f0] = {e0, e1, e2};
  reqirements[f1] = {e1, e3, e4};

  REQUIRE(num_faces() == 2);

  SECTION("check if all found edges are in requirements") {
    for (auto f : faces()) {
      auto found_edges = edges(f);
      REQUIRE(found_edges.size() == 3);
      for (auto edge_of_face : found_edges) {
        INFO("found edge in requirement: e" << edge_of_face.i);
        REQUIRE(find(reqirements[f], edge_of_face) != end(reqirements[f]));
      }
    }
  }

  SECTION("check if all requirements are in found edges") {
    for (auto f : faces()) {
      auto found_edges = edges(f);
      REQUIRE(found_edges.size() == 3);
      for (auto requirement : reqirements[f]) {
        INFO("requirement in found edges: e" << requirement.i);
        REQUIRE(find(found_edges, requirement) != end(found_edges));
      }
    }
  }
}

//==============================================================================
TEST_CASE_METHOD((edgeset<double, 3>), "[edgeset] delete edge", "[edgeset]") {
  auto v0 = insert_vertex(1, 2, 3);
  auto v1 = insert_vertex(2, 3, 4);
  auto v2 = insert_vertex(3, 4, 5);
  auto v3 = insert_vertex(4, 5, 6);

  [[maybe_unused]] auto e0 = insert_edge(v0, v1);
  [[maybe_unused]] auto e1 = insert_edge(v1, v2);
  [[maybe_unused]] auto e2 = insert_edge(v2, v0);
  [[maybe_unused]] auto e3 = insert_edge(v1, v3);
  [[maybe_unused]] auto e4 = insert_edge(v3, v2);

  REQUIRE(num_edges() == 5);
  REQUIRE(vertices().size() == 4);

  remove(e0);
  REQUIRE(num_edges() == 4);
  REQUIRE(vertices().size() == 4);

  remove(e2);
  REQUIRE(num_edges() == 3);
  REQUIRE(vertices().size() == 3);

  tidy_up();
  REQUIRE(m_points.size() == 3);
  REQUIRE(m_edges.size() == 3);
}

//==============================================================================
TEST_CASE_METHOD(mesh_test, "[mesh] remove face", "[mesh]") {
  using vertex = mesh<double, 3>::vertex;
  using edge   = mesh<double, 3>::edge;
  using face   = mesh<double, 3>::face;
  remove(f0);

  REQUIRE(vertices().size() == 3);
  REQUIRE(num_edges() == 3);
  REQUIRE(num_faces() == 1);

  tidy_up();

  REQUIRE(vertices().size() == 3);
  REQUIRE(num_edges() == 3);
  REQUIRE(num_faces() == 1);

  SECTION("vertex Positions") {
    REQUIRE(at(vertex{0})(0) == 2);
    REQUIRE(at(vertex{0})(1) == 3);
    REQUIRE(at(vertex{0})(2) == 4);

    REQUIRE(at(vertex{1})(0) == 3);
    REQUIRE(at(vertex{1})(1) == 4);
    REQUIRE(at(vertex{1})(2) == 5);

    REQUIRE(at(vertex{2})(0) == 4);
    REQUIRE(at(vertex{2})(1) == 5);
    REQUIRE(at(vertex{2})(2) == 6);
  }

  SECTION("Edges") {
    REQUIRE(at(edge{0})[0] == vertex{0});
    REQUIRE(at(edge{0})[1] == vertex{1});

    REQUIRE(at(edge{1})[0] == vertex{0});
    REQUIRE(at(edge{1})[1] == vertex{2});

    REQUIRE(at(edge{2})[0] == vertex{2});
    REQUIRE(at(edge{2})[1] == vertex{1});
  }

  SECTION("Faces") {
    REQUIRE(at(face{0})[0] == vertex{0});
    REQUIRE(at(face{0})[1] == vertex{2});
    REQUIRE(at(face{0})[2] == vertex{1});
  }
}

//==============================================================================
TEST_CASE("[mesh] copy", "[edgeset]") {
  mesh<double, 2> m;

  std::vector v{m.insert_vertex(1, 2), m.insert_vertex(2, 3),
                m.insert_vertex(3, 4)};
  std::vector f{m.insert_face(v[0], v[1], v[2])};

  auto& vp               = m.add_vertex_property<int>("vertex_property");
  auto& ep               = m.add_edge_property<int>("edge_property");
  auto& fp               = m.add_face_property<int>("face_property");
  vp[v[0]]               = 1;
  vp[v[1]]               = 2;
  vp[v[2]]               = 4;
  ep[*m.edges().begin()] = 500;
  fp[f.front()]          = 200;

  mesh<double, 2> copy{m};

  REQUIRE(m.vertices().size() == copy.num_vertices());
  REQUIRE(m.num_edges() == copy.num_edges());
  REQUIRE(m.num_faces() == copy.num_faces());
  REQUIRE(copy.num_edges(v[0]) == 2);
  REQUIRE(copy.num_edges(v[1]) == 2);
  REQUIRE(copy.num_edges(v[2]) == 2);
  REQUIRE(copy.num_faces(v[0]) == 1);
  REQUIRE(copy.num_faces(v[1]) == 1);
  REQUIRE(copy.num_faces(v[2]) == 1);

  auto& vp_copy = copy.vertex_property<int>("vertex_property");
  auto& ep_copy = copy.edge_property<int>("edge_property");
  auto& fp_copy = copy.face_property<int>("face_property");

  for (auto v : m.vertices()) {
    REQUIRE(approx_equal(m[v], copy[v], 0));
    REQUIRE(vp[v] == vp_copy[v]);
  }
  for (auto e : m.edges()) {
    REQUIRE(m[e] == copy[e]);
    REQUIRE(ep[e] == ep_copy[e]);
  }
  for (auto f : m.faces()) {
    REQUIRE(m[f] == copy[f]);
    REQUIRE(fp[f] == fp_copy[f]);
    REQUIRE(m[f][0] == copy[f][0]);
    REQUIRE(m[f][1] == copy[f][1]);
    REQUIRE(m[f][2] == copy[f][2]);
  }

  fp_copy[f.front()] = 100;
  REQUIRE(fp[f.front()] != fp_copy[f.front()]);
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
