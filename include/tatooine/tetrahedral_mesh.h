#ifndef TATOOINE_TETRAHEDRAL_MESH_H
#define TATOOINE_TETRAHEDRAL_MESH_H
//==============================================================================
#include <tatooine/pointset.h>
#include <tatooine/property.h>
#include <tatooine/vtk_legacy.h>

#include <boost/range/algorithm/copy.hpp>
#include <vector>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class tetrahedral_mesh : public pointset<Real, N> {
 public:
  using this_t   = tetrahedral_mesh<Real, N>;
  using parent_t = pointset<Real, N>;
  using parent_t::at;
  using typename parent_t::vertex_index;
  using parent_t::operator[];
  using parent_t::is_valid;
  template <typename T>
  using vertex_property_t = typename parent_t::template vertex_property_t<T>;
  //----------------------------------------------------------------------------
  struct tetrahedron_index : handle {
    using handle::handle;
    constexpr bool operator==(tetrahedron_index other) const {
      return this->i == other.i;
    }
    constexpr bool operator!=(tetrahedron_index other) const {
      return this->i != other.i;
    }
    constexpr bool operator<(tetrahedron_index other) const {
      return this->i < other.i;
    }
    static constexpr auto invalid() {
      return tetrahedron_index{handle::invalid_idx};
    }
  };
  //----------------------------------------------------------------------------
  struct tetrahedron_iterator
      : boost::iterator_facade<tetrahedron_iterator, tetrahedron_index,
                               boost::bidirectional_traversal_tag,
                               tetrahedron_index> {
    tetrahedron_iterator(tetrahedron_index i, tetrahedral_mesh const* mesh)
        : m_index{i}, m_mesh{mesh} {}
    tetrahedron_iterator(tetrahedron_iterator const& other)
        : m_index{other.m_index}, m_mesh{other.m_mesh} {}

   private:
    tetrahedron_index       m_index;
    tetrahedral_mesh const* m_mesh;

    friend class boost::iterator_core_access;

    auto increment() {
      do
        ++m_index;
      while (!m_mesh->is_valid(m_index));
    }
    auto decrement() {
      do
        --m_index;
      while (!m_mesh->is_valid(m_index));
    }

    auto equal(tetrahedron_iterator const& other) const {
      return m_index == other.m_index;
    }
    auto dereference() const { return m_index; }
  };
  //----------------------------------------------------------------------------
  struct tetrahedron_container {
    using iterator       = tetrahedron_iterator;
    using const_iterator = tetrahedron_iterator;

    tetrahedral_mesh const* m_mesh;

    auto begin() const {
      tetrahedron_iterator vi{tetrahedron_index{0}, m_mesh};
      if (!m_mesh->is_valid(*vi)) { ++vi; }
      return vi;
    }

    auto end() const {
      return tetrahedron_iterator{
          tetrahedron_index{m_mesh->m_tetrahedrons.size()}, m_mesh};
    }
  };
  //----------------------------------------------------------------------------
  template <typename T>
  using tetrahedron_property_t = vector_property_impl<tetrahedron_index, T>;
  using tetrahedron_property_container_t =
      std::map<std::string,
               std::unique_ptr<vector_property<tetrahedron_index>>>;
  //============================================================================
 private:
  std::vector<std::array<vertex_index, 4>> m_tetrahedrons;
  std::vector<tetrahedron_index>           m_invalid_tetrahedrons;
  tetrahedron_property_container_t         m_tetrahedron_properties;

 public:
  //============================================================================
  constexpr tetrahedral_mesh() = default;
  //============================================================================
 public:
  tetrahedral_mesh(tetrahedral_mesh const& other)
      : parent_t{other}, m_tetrahedrons{other.m_tetrahedrons} {
    for (auto const& [key, fprop] : other.m_tetrahedron_properties) {
      m_tetrahedron_properties.insert(std::pair{key, fprop->clone()});
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  tetrahedral_mesh(tetrahedral_mesh&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(tetrahedral_mesh const& other) -> tetrahedral_mesh& {
    parent_t::operator=(other);
    m_tetrahedron_properties.clear();
    m_tetrahedrons    = other.m_tetrahedrons;
    for (auto const& [key, fprop] : other.m_tetrahedron_properties) {
      m_tetrahedron_properties.insert(std::pair{key, fprop->clone()});
    }
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(tetrahedral_mesh&& other) noexcept
    -> tetrahedral_mesh& = default;
  //----------------------------------------------------------------------------
  tetrahedral_mesh(std::string const& file) { read(file); }
  //============================================================================
  auto operator[](tetrahedron_index t) const -> auto const& {
    return m_tetrahedrons[t.i];
  }
  auto operator[](tetrahedron_index t) -> auto& { return m_tetrahedrons[t.i]; }
  //----------------------------------------------------------------------------
  auto at(tetrahedron_index t) const -> auto const& {
    return m_tetrahedrons[t.i];
  }
  auto at(tetrahedron_index t) -> auto& { return m_tetrahedrons[t.i]; }
  //----------------------------------------------------------------------------
  auto insert_tetrahedron(vertex_index v0, vertex_index v1, vertex_index v2, vertex_index v3) {
    m_tetrahedrons.push_back(std::array{v0, v1, v2, v3});
    for (auto& [key, prop] : m_tetrahedron_properties) { prop->push_back(); }
    return tetrahedron_index{m_tetrahedrons.size() - 1};
  }
  //----------------------------------------------------------------------------
  auto insert_tetrahedron(size_t v0, size_t v1, size_t v2, size_t v3) {
    return insert_tetrahedron(vertex_index{v0}, vertex_index{v1}, vertex_index{v2}, vertex_index{v3});
  }
  //----------------------------------------------------------------------------
  auto insert_tetrahedron(std::array<vertex_index, 4> const& t) {
    m_tetrahedrons.push_back(t);
    for (auto& [key, prop] : m_tetrahedron_properties) { prop->push_back(); }
    return tetrahedron_index{m_tetrahedrons.size() - 1};
  }
  //----------------------------------------------------------------------------
  auto clear() {
    parent_t::clear();
    m_tetrahedrons.clear();
  }
  //----------------------------------------------------------------------------
  auto tetrahedrons() const { return tetrahedron_container{this}; }
  //----------------------------------------------------------------------------
  auto num_tetrahedrons() const { return m_tetrahedrons.size(); }
  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 2 || _N == 3, bool> = true>
  auto write_vtk(std::string const& path,
                 std::string const& title = "tatooine tetrahedral mesh") const
      -> bool {
    using boost::copy;
    using boost::adaptors::transformed;
    vtk::legacy_file_writer writer(path, vtk::POLYDATA);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();
      if constexpr (N == 2) {
        auto three_dims = [](vec<Real, 2> const& v2) {
          return vec<Real, 3>{v2(0), v2(1), 0};
        };
        std::vector<vec<Real, 3>> v3s(this->m_vertices.size());
        auto                      three_dimensional = transformed(three_dims);
        copy(this->m_vertices | three_dimensional, begin(v3s));
        writer.write_points(v3s);

      } else if constexpr (N == 3) {
        writer.write_points(this->m_vertices);
      }

      std::vector<std::vector<size_t>> polygons;
      polygons.reserve(num_tetrahedrons());
      for (auto const& tetrahedron_index : m_tetrahedrons) {
        polygons.push_back(
            std::vector{tetrahedron_index[0].i, tetrahedron_index[1].i,
                        tetrahedron_index[2].i, tetrahedron_index[3].i});
      }
      writer.write_polygons(polygons);

      // write vertex_index data
      writer.write_point_data(this->num_vertices());
      for (auto const& [name, prop] : this->m_vertex_properties) {
        if (prop->type() == typeid(vec<Real, 4>)) {
        } else if (prop->type() == typeid(vec<Real, 3>)) {
        } else if (prop->type() == typeid(vec<Real, 2>)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<vec<Real, 2>> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        } else if (prop->type() == typeid(Real)) {
          auto const& casted_prop =
              *dynamic_cast<vertex_property_t<Real> const*>(prop.get());
          writer.write_scalars(name, casted_prop.container());
        }
      }

      writer.close();
      return true;
    } else {
      return false;
    }
  }
  //----------------------------------------------------------------------------
  auto read(std::string const& path) {
    auto ext = path.substr(path.find_last_of(".") + 1);
    if constexpr (N == 2 || N == 3) {
      if (ext == "vtk") { read_vtk(path); }
    }
  }
  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 2 || _N == 3, bool> = true>
  auto read_vtk(std::string const& path) {
    struct listener_t : vtk::legacy_file_listener {
      tetrahedral_mesh& mesh;

      listener_t(tetrahedral_mesh& _mesh) : mesh(_mesh) {}

      void on_dataset_type(vtk::DatasetType t) override {
        if (t != vtk::POLYDATA) {
          throw std::runtime_error{
              "[tetrahedral_mesh] need polydata when reading vtk legacy"};
        }
      }

      void on_points(std::vector<std::array<float, 3>> const& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1], p[2]); }
      }
      void on_points(std::vector<std::array<double, 3>> const& ps) override {
        for (auto& p : ps) { mesh.insert_vertex(p[0], p[1], p[2]); }
      }
      void on_polygons(std::vector<int> const& ps) override {
        for (size_t i = 0; i < ps.size();) {
          auto const& size = ps[i++];
          if (size == 4) {
            mesh.insert_tetrahedron(size_t(ps[i]), size_t(ps[i + 1]),
                                    size_t(ps[i + 2]), size_t(ps[i + 2]));
          }
          i += size;
        }
      }
      void on_scalars(std::string const& data_name,
                      std::string const& /*lookup_table_name*/,
                      size_t num_comps, std::vector<double> const& scalars,
                      vtk::ReaderData data) override {
        if (data == vtk::POINT_DATA) {
          if (num_comps == 1) {
            auto& prop = mesh.template add_vertex_property<double>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) { prop[i] = scalars[i]; }
          } else if (num_comps == 2) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 2>>(data_name);

            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 3) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 3>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          } else if (num_comps == 4) {
            auto& prop =
                mesh.template add_vertex_property<vec<double, 4>>(data_name);
            for (size_t i = 0; i < prop.size(); ++i) {
              for (size_t j = 0; j < num_comps; ++j) {
                prop[i][j] = scalars[i * num_comps + j];
              }
            }
          }
        }
      }
    } listener{*this};
    vtk::legacy_file f{path};
    f.add_listener(listener);
    f.read();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto tetrahedron_property(std::string const& name) -> auto& {
    auto prop = m_tetrahedron_properties.at(name).get();
    auto casted_prop =  dynamic_cast<tetrahedron_property_t<T>*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto tetrahedron_property(std::string const& name) const -> auto const& {
    auto prop = m_tetrahedron_properties.at(name).get();
    auto casted_prop =  dynamic_cast<tetrahedron_property_t<T> const*>(prop);
    assert(typeid(T) == casted_prop->type());
    return *casted_prop;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto add_tetrahedron_property(std::string const& name,
                                 T const&           value = T{}) -> auto& {
    auto [it, suc] = m_tetrahedron_properties.insert(
        std::pair{name, std::make_unique<tetrahedron_property_t<T>>(value)});
    auto fprop = dynamic_cast<tetrahedron_property_t<T>*>(it->second.get());
    fprop->resize(m_tetrahedrons.size());
    return *fprop;
  }
  //----------------------------------------------------------------------------
  constexpr bool is_valid(tetrahedron_index t) const {
    return boost::find(m_invalid_tetrahedrons, t) ==
           end(m_invalid_tetrahedrons);
  }
};
//==============================================================================
tetrahedral_mesh()->tetrahedral_mesh<double, 3>;
tetrahedral_mesh(std::string const&)->tetrahedral_mesh<double, 3>;
//==============================================================================
// namespace detail {
// template <typename MeshCont>
// auto write_mesh_container_to_vtk(MeshContc onst& meshes, std::string const&
// path,
//                                 std::string const& title) {
//  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
//  if (writer.is_open()) {
//    size_t num_pts = 0;
//    size_t cur_first = 0;
//    for (auto const& m : meshes) { num_pts += m.num_vertices(); }
//    std::vector<std::array<typename MeshCont::value_type::real_t, 3>> points;
//    std::vector<std::vector<size_t>> tetrahedrons; points.reserve(num_pts);
//    tetrahedrons.reserve(meshes.size());
//
//    for (auto const& m : meshes) {
//      // add points
//      for (auto const& v : m.vertices()) {
//        points.push_back(std::array{m[v](0), m[v](1), m[v](2)});
//      }
//
//      // add tetrahedrons
//      for (auto t : m.tetrahedrons()) {
//        tetrahedrons.emplace_back();
//        tetrahedrons.back().push_back(cur_first + m[t][0].i);
//        tetrahedrons.back().push_back(cur_first + m[t][1].i);
//        tetrahedrons.back().push_back(cur_first + m[t][2].i);
//      }
//      cur_first += m.num_vertices();
//    }
//
//    // write
//    writer.set_title(title);
//    writer.write_header();
//    writer.write_points(points);
//    writer.write_polygons(tetrahedrons);
//    //writer.write_point_data(num_pts);
//    writer.close();
//  }
//}
//}  // namespace detail
////==============================================================================
// template <typename Real>
// auto write_vtk(std::vector<tetrahedral_mesh<Real, 3>> const& meshes,
// std::string const& path,
//               std::string const& title = "tatooine meshes") {
//  detail::write_mesh_container_to_vtk(meshes, path, title);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

