#ifndef __STEADIFIED_VECTORFIELD_H__
#define __STEADIFIED_VECTORFIELD_H__

#include <Tatooine/mesh.h>
#include <Tatooine/quadtree.h>
#include <Tatooine/vectorfield.h>

template <typename real_t>
struct steadified_vectorfield
    : tatooine::vectorfield<2, real_t, steadified_vectorfield<real_t>> {
  //============================================================================
  using this_t   = steadified_vectorfield<real_t>;
  using parent_t = tatooine::vectorfield<2, real_t, this_t>;
  using pos_t    = typename parent_t::pos_t;
  using vec_t    = typename parent_t::vec_t;

  using mesh_t     = tatooine::mesh<2, real_t>;
  using quadtree_t = tatooine::quadtree<2, real_t, mesh_t>;
  using face_t     = typename mesh_t::face;

  using mesh_list     = std::vector<std::vector<mesh_t>>;
  using quadtree_list = std::vector<std::vector<quadtree_t>>;
  using typename parent_t::out_of_domain;

  static constexpr size_t max_quadtree_depth = 10;

  //============================================================================
  mesh_list     meshes;
  quadtree_list quadtrees;

  //============================================================================
  steadified_vectorfield(const mesh_list& _meshes) : meshes(_meshes) {
    create_quadtrees();
  }

  //----------------------------------------------------------------------------
  steadified_vectorfield(mesh_list&& _meshes) : meshes(std::move(_meshes)) {
    create_quadtrees();
  }

  //============================================================================
  void create_quadtrees() {
    auto to_quadtree = [](const auto& ribbon_mesh) -> quadtree_t {
      return {ribbon_mesh, max_quadtree_depth};
    };
    for (const auto& surf : meshes)
      boost::transform(surf, std::back_inserter(quadtrees.emplace_back()),
                       to_quadtree);

    for (auto& surf : quadtrees)
      for (auto& q : surf) {
        std::cout << q.mesh().num_faces() << '\n';
        q.insert_all_vertices(false, true);
        std::cout << "done!\n";
      }
  }

  //----------------------------------------------------------------------------
  std::optional<std::tuple<const mesh_t*, face_t, tatooine::Vec<real_t, 3>>>
  in_domain(const pos_t& x, real_t) const {
    for (const auto& surf : quadtrees)
      for (const auto& ribbon_quadtree : surf) {
        const auto& mesh = ribbon_quadtree.mesh();
        const auto& uv =
            mesh.template vertex_property<tatooine::Vec<real_t, 2>>("uv");
        const auto& v =
            mesh.template vertex_property<tatooine::Vec<real_t, 2>>("v");
        auto matches = ribbon_quadtree.faces(x);
        if (!matches.empty()) {
          real_t                   min_tau = std::numeric_limits<real_t>::max();
          face_t                   nearest_face;
          auto                     nearest_tau_match = begin(matches);
          tatooine::Vec<real_t, 3> nearest_bary;
          for (auto it = begin(matches); it != end(matches); ++it) {
            const auto& [f, bary] = *it;
            real_t tau            =            //
                bary(0) * uv[mesh[f][0]](1) +  //
                bary(1) * uv[mesh[f][1]](1) +  //
                bary(2) * uv[mesh[f][2]](1);
            if (tau < min_tau) {
              min_tau           = tau;
              nearest_face      = f;
              nearest_tau_match = it;
              nearest_bary      = bary;
            }
          }

          return std::tuple{&mesh, nearest_face, nearest_bary};
        }
      }
    return {};
  }

  //----------------------------------------------------------------------------

  constexpr vec_t evaluate(const pos_t& x, real_t t = 0) const {
    if (auto res = in_domain(x, t); res) {
      const auto& [m, face, bary] = *res;
      const auto& mesh            = *m;
      const auto& v =
          mesh.template vertex_property<tatooine::Vec<real_t, 2>>("v");

      return bary(0) * v[mesh[face][0]] +  //
             bary(1) * v[mesh[face][1]] +  //
             bary(2) * v[mesh[face][2]];
    }
    // throw out_of_domain{};
    return {0, 0};
  }
};

#endif
