#ifndef TATOOINE_TETRAEDER_MESH_H
#define TATOOINE_TETRAEDER_MESH_H

#include "mesh.h"
#include "vtk_legacy.h"
#include "grid.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
class tetraedermesh : public mesh<Real, N> {
 public:
  using this_t     = tetraedermesh<Real, N>;
  using parent_t   = mesh<Real, N>;
  using pointset_t = pointset<Real, N>;
  using edgeset_t  = edgeset<Real, N>;
  using mesh_t     = mesh<Real, N>;

  using typename parent_t::handle;
  using typename parent_t::vertex;
  using typename parent_t::edge;
  using typename parent_t::face;
  using typename parent_t::point_t;

  using parent_t::at;
  using parent_t::operator[];
  using parent_t::remove;
  using parent_t::is_valid;
  using parent_t::vertices;
  using parent_t::edges;
  using parent_t::faces;
  using parent_t::num_vertices;
  using parent_t::num_edges;
  using parent_t::num_faces;
  using parent_t::insert_vertex;
  using parent_t::insert_face;

  //============================================================================
  struct tetraeder : handle {
    auto& operator=(tetraeder other) {
      this->i = other.i;
      return *this;
    }
    bool operator==(const tetraeder& other) const { return this->i == other.i; }
    bool operator!=(const tetraeder& other) const { return this->i != other.i; }
    bool operator<(const tetraeder& other) const { return this->i < other.i; }
    static constexpr auto invalid() { return tetraeder{handle::invalid_idx}; }
  };

  //============================================================================
  struct tet_iterator
      : public boost::iterator_facade<tet_iterator, tetraeder,
                                      boost::bidirectional_traversal_tag,
                                      tetraeder> {
    tet_iterator(tetraeder _t, const tetraedermesh* _tm)
        : t{_t}, tm{_tm} {}
    tet_iterator(const tet_iterator& other)
        : t{other.t}, tm{other.tm} {}

   private:
    tetraeder            t;
    const tetraedermesh* tm;
    friend class boost::iterator_core_access;

    void increment() {
      do
        ++t.i;
      while (!tm->is_valid(t));
    }
    void decrement() {
      do
        --t.i;
      while (!tm->is_valid(t));
    }

    bool equal(const tet_iterator& other) const {
      return t.i == other.t.i;
    }
    auto dereference() const { return t; }
  };

  //============================================================================
  struct tetraeder_container {
    using iterator       = tet_iterator;
    using const_iterator = tet_iterator;
    const tetraedermesh* tetmesh;

    auto begin() const {
      tet_iterator fi{tetraeder{0}, tetmesh};
      if (!tetmesh->is_valid(*fi)) ++fi;
      return fi;
    }
    auto end() const {
      return tet_iterator{tetraeder{tetmesh->m_tetraeders.size()},
                               tetmesh};
    }
  };

  //============================================================================
  template <typename T>
  using vertex_prop = typename parent_t::template vertex_prop<T>;

  //----------------------------------------------------------------------------
  template <typename T>
  using edge_prop = typename parent_t::template edge_prop<T>;

  //----------------------------------------------------------------------------
  template <typename T>
  using face_prop = typename parent_t::template face_prop<T>;

  //----------------------------------------------------------------------------
  template <typename T>
  struct tetprop : public property_type<T> {
    using property_type<T>::property_type;
    auto&       at(tetraeder t) { return property_type<T>::at(t.i); }
    const auto& at(tetraeder t) const { return property_type<T>::at(t.i); }
    auto& operator[](tetraeder t) { return property_type<T>::operator[](t.i); }
    const auto& operator[](tetraeder t) const {
      return property_type<T>::operator[](t.i);
    }
    std::unique_ptr<property> clone() const override {
      return std::unique_ptr<tetprop<T>>(
          new tetprop<T>{*this});
    }
  };

  //============================================================================
 protected:
  std::vector<std::array<vertex, 4>>               m_tetraeders;
  std::vector<tetraeder>                           m_invalid_tetraeders;
  std::map<std::string, std::unique_ptr<property>> m_tetraeder_properties;

  vertex_prop<std::vector<tetraeder>>* m_tetraeders_of_vertices = nullptr;
  edge_prop<std::vector<tetraeder>>*   m_tetraeders_of_edges    = nullptr;
  face_prop<std::vector<tetraeder>>*   m_tetraeders_of_faces    = nullptr;

  tetprop<std::vector<edge>>* m_edges_of_tetraeders = nullptr;
  tetprop<std::vector<face>>* m_faces_of_tetraeders = nullptr;

 public:
  //============================================================================
  constexpr tetraedermesh() { add_link_properties(); }

  //----------------------------------------------------------------------------
  constexpr tetraedermesh(std::initializer_list<point_t>&& vertices)
      : parent_t(std::move(vertices)) {
    add_link_properties();
  }

  //============================================================================
 public:
  tetraedermesh(const tetraedermesh& other)
      : parent_t(other),
        m_tetraeders(other.m_tetraeders),
        m_invalid_tetraeders(other.m_invalid_tetraders) {
    m_tetraeder_properties.clear();
    for (const auto& [name, prop] : other.m_tetraeder_properties)
      m_tetraeder_properties[name] = prop->clone();
    find_link_properties();
  }

  //----------------------------------------------------------------------------
  tetraedermesh(tetraedermesh&& other)
      : parent_t(std::move(other)),
        m_tetraeders(std::move(other.m_tetraeders)),
        m_invalid_tetraeders(std::move(other.m_invalid_tetraeders)),
        m_tetraeder_properties(std::move(other.m_tetraeder_properties)) {
    find_link_properties();
  }

  //----------------------------------------------------------------------------
  auto& operator=(const tetraedermesh& other) {
    parent_t::operator=(other);
    m_tetraeders         = other.m_tetraeders;
    m_invalid_tetraeders = other.m_invalid_tetraeders;
    for (const auto& [name, prop] : other.m_tetraeder_properties)
      m_tetraeder_properties[name] = prop->clone();
    find_link_properties();
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(tetraedermesh&& other) {
    parent_t::operator=(std::move(other));
    m_tetraeders           = std::move(other.m_tetraeders);
    m_invalid_tetraeders   = std::move(other.m_invalid_tetraeders);
    m_tetraeder_properties = std::move(other.m_tetraeder_properties);
    find_link_properties();
    return *this;
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 3>...,
            typename... OtherReals>
  tetraedermesh(const linspace<OtherReals>&... ls)
      : tetraedermesh{grid{ls...}} {
    static_assert(sizeof...(OtherReals) == 3);
  }

  //----------------------------------------------------------------------------
  template <size_t _N = N, std::enable_if_t<_N == 3>..., typename OtherReal>
  tetraedermesh(const grid<OtherReal, 3>& g) {
    add_link_properties();
    const vec   res{g.dimension(0).resolution,
                    g.dimension(1).resolution,
                    g.dimension(2).resolution};
    const auto  bb    = g.boundingbox();
    const auto& min   = bb.min;
    const auto& max   = bb.max;
    const auto  range = max - min;

    std::vector<std::vector<std::vector<vertex>>> vs;

    for (size_t x = 0; x < res[0]; ++x) {
      vs.emplace_back();
      for (size_t y = 0; y < res[1]; ++y) {
        vs.back().emplace_back();
        for (size_t z = 0; z < res[2]; ++z) {
          vs.back().back().emplace_back();
          vec p =
              vec{Real(x) / (res[0] - 1),
                  Real(y) / (res[1] - 1),
                  Real(z) / (res[2] - 1)} * range + min;
          vs[x][y][z] = this->insert_vertex(p);
        }
      }
    }

    constexpr auto sign = [](auto i) { return i % 2 == 0 ? 1 : -1; };
    for (size_t z = 0; z < res[2] - 1; ++z) {
      for (size_t y = 0; y < res[1] - 1; ++y) {
        for (size_t x = 0; x < res[0] - 1; ++x) {
          const bool is_odd = sign(x) * sign(y) * sign(z) == -1;
          if (is_odd) {
            insert_tetraeder(vs[x][y][z], vs[x + 1][y][z], vs[x + 1][y][z + 1],
                             vs[x + 1][y + 1][z]);
            insert_tetraeder(vs[x][y][z], vs[x][y + 1][z], vs[x + 1][y + 1][z],
                             vs[x][y + 1][z + 1]);
            insert_tetraeder(vs[x][y][z], vs[x][y][z + 1], vs[x][y + 1][z + 1],
                             vs[x + 1][y][z + 1]);
            insert_tetraeder(vs[x + 1][y + 1][z], vs[x + 1][y][z + 1],
                             vs[x][y + 1][z + 1], vs[x + 1][y + 1][z + 1]);
            insert_tetraeder(vs[x][y][z], vs[x + 1][y][z + 1],
                             vs[x + 1][y + 1][z], vs[x][y + 1][z + 1]);
          } else {
            insert_tetraeder(vs[x][y][z], vs[x + 1][y][z], vs[x][y + 1][z],
                             vs[x][y][z + 1]);
            insert_tetraeder(vs[x + 1][y][z], vs[x + 1][y + 1][z],
                             vs[x][y + 1][z], vs[x + 1][y + 1][z + 1]);
            insert_tetraeder(vs[x + 1][y][z], vs[x + 1][y][z + 1],
                             vs[x][y][z + 1], vs[x + 1][y + 1][z + 1]);
            insert_tetraeder(vs[x][y + 1][z], vs[x][y + 1][z + 1],
                             vs[x][y][z + 1], vs[x + 1][y + 1][z + 1]);
            insert_tetraeder(vs[x + 1][y][z], vs[x][y + 1][z], vs[x][y][z + 1],
                             vs[x + 1][y + 1][z + 1]);
          }
        }
      }
    }
  }
  // template <size_t _N = N, std::enable_if_t<_N == 3>...>
  // tetraedermesh(const tetgenio& io) : parent_t{io} {
  //   add_link_properties();
  //   for (int i = 0; i < io.numberoftetrahedra; ++i) {
  //     insert_tetraeder(io.tetrahedronlist[i * 4], io.tetrahedronlist[i * 4 + 1],
  //                      io.tetrahedronlist[i * 4 + 2],
  //                      io.tetrahedronlist[i * 4 + 3]);
  //   }
  // }

  //============================================================================
 private:
  void add_link_properties() {
    m_tetraeders_of_vertices =
        dynamic_cast<vertex_prop<std::vector<tetraeder>>*>(
            &this->template add_vertex_property<std::vector<tetraeder>>(
                "v:tetraeders"));
    m_tetraeders_of_edges =
      dynamic_cast<edge_prop<std::vector<tetraeder>>*>(
            &this->template add_edge_property<std::vector<tetraeder>>(
                "e:tetraeders"));
    m_tetraeders_of_faces =
      dynamic_cast<face_prop<std::vector<tetraeder>>*>(
            &this->template add_face_property<std::vector<tetraeder>>(
                "f:tetraeders"));

    m_edges_of_tetraeders = dynamic_cast<tetprop<std::vector<edge>>*>(
        &this->template add_tetraeder_property<std::vector<edge>>("t:edges"));
    m_faces_of_tetraeders = dynamic_cast<tetprop<std::vector<face>>*>(
        &this->template add_tetraeder_property<std::vector<face>>("t:faces"));
  }

  //----------------------------------------------------------------------------
  void find_link_properties() {
    m_tetraeders_of_vertices =
        dynamic_cast<vertex_prop<std::vector<tetraeder>>*>(
            &this->template vertex_property<std::vector<tetraeder>>(
                "v:tetraeders"));
    m_tetraeders_of_edges =
        dynamic_cast<edge_prop<std::vector<tetraeder>>*>(
            &this->template edge_property<std::vector<tetraeder>>(
                "e:tetraeders"));
    m_tetraeders_of_faces =
        dynamic_cast<face_prop<std::vector<tetraeder>>*>(
            &this->template face_property<std::vector<tetraeder>>(
                "f:tetraeders"));

    m_edges_of_tetraeders = dynamic_cast<tetprop<std::vector<edge>>*>(
        &tetraeder_property<std::vector<edge>>("t:edges"));
    m_faces_of_tetraeders = dynamic_cast<tetprop<std::vector<face>>*>(
        &tetraeder_property<std::vector<face>>("t:faces"));
  }

 public:
  //============================================================================
  constexpr auto&       at(tetraeder t) { return m_tetraeders[t.i]; }
  constexpr const auto& at(tetraeder t) const { return m_tetraeders[t.i]; }

  //----------------------------------------------------------------------------
  constexpr auto&       operator[](tetraeder t) { return at(t); }
  constexpr const auto& operator[](tetraeder t) const { return at(t); }

  //----------------------------------------------------------------------------
  constexpr auto insert_tetraeder(size_t v0, size_t v1,
                                  size_t v2, size_t v3) {
    return insert_tetraeder(vertex{v0}, vertex{v1}, vertex{v2}, vertex{v3});
  }

  //----------------------------------------------------------------------------
  constexpr auto insert_tetraeder(const vertex& v0, const vertex& v1,
                                  const vertex& v2, const vertex& v3) {
    tetraeder t{m_tetraeders.size()};
    m_tetraeders.push_back({v0, v1, v2, v3});
    for (auto& [key, prop] : m_tetraeder_properties) prop->push_back();

    std::array fs{insert_face(v0, v1, v2), insert_face(v0, v1, v3),
                  insert_face(v0, v2, v3), insert_face(v1, v2, v3)};

    // link vertices
    for (auto v : vertices(t)) tetraeders(v).push_back(t);

    // link faces
    for (auto f : fs) {
      faces(t).push_back(f);
      tetraeders(f).push_back(t);
    }

    // link edges
    std::set<edge> es;
    for (auto f : fs)
      for (auto e : edges(f)) es.insert(e);
    for (auto e : es) {
      edges(t).push_back(e);
      tetraeders(e).push_back(t);
    }

    return t;
  }

  //----------------------------------------------------------------------------
  constexpr void remove(tetraeder t) {
    using namespace boost;
    if (is_valid(t) &&
        find(m_invalid_tetraeders, t) == m_invalid_tetraeders.end())
      m_invalid_tetraeders.push_back(t);
  }

  //----------------------------------------------------------------------------
  void remove(vertex v) {
    using namespace boost;
    parent_t::remove(v);
    for (auto t : tetraeders(v))
      if (find(m_invalid_tetraeders, t) == end(m_invalid_tetraeders))
        remove(t);
  }

  //----------------------------------------------------------------------------
  void remove(edge e, bool remove_orphaned_vertices = true) {
    using namespace boost;
    parent_t::remove(e, remove_orphaned_vertices);
    for (auto t : tetraeders(e))
      if (find(m_invalid_tetraeders, t) == end(m_invalid_tetraeders)) remove(t);
  }

  //----------------------------------------------------------------------------
  void remove(face f, bool remove_orphaned_vertices = true,
              bool remove_orphaned_edges = true) {
    using namespace boost;
    parent_t::remove(f, remove_orphaned_vertices, remove_orphaned_edges);
    for (auto t : tetraeders(f))
      if (find(m_invalid_tetraeders, t) == end(m_invalid_tetraeders)) remove(t);
  }

  //----------------------------------------------------------------------------
  // constexpr void remove(tetraeder t, bool remove_orphaned_vertices = true,
  //                       bool remove_orphaned_edges = true,
  //                       bool remove_orphaned_faces = true) {
  //   using namespace boost;
  //   if (is_valid(t)) {
  //     if (find(m_invalid_tetraeders, t) == end(m_invalid_tetraeders))
  //       m_invalid_tetraeders.push_back(t);
  //
  //     if (remove_orphaned_vertices)
  //       for (auto v : vertices(t))
  //         if (num_tetraeders(v) <= 1) remove(v);
  //
  //     if (remove_orphaned_edges)
  //       for (auto e : edges(t))
  //         if (num_tetraeders(e) <= 1) remove(e, remove_orphaned_vertices);
  //
  //     if (remove_orphaned_faces)
  //       for (auto f : faces(t))
  //         if (num_tetraeders(f) <= 1)
  //           remove(e, remove_orphaned_vertices, remove_orphaned_edges);
  //
  //     // remove tetraeder link from vertices
  //     for (auto v : vertices(t)) tetraeders(v).erase(find(tetraeders(v), t));
  //     // remove tetraeder link from edges
  //     for (auto e : edges(t)) tetraeders(e).erase(find(tetraeders(e), t));
  //     // remove tetraeder link from faces
  //     for (auto f : faces(t)) tetraeders(f).erase(find(tetraeders(f), t));
  //
  //     vertices(t).clear();
  //     edges(t).clear();
  //     faces(t).clear();
  //   }
  // }

  //----------------------------------------------------------------------------
  //! tidies up invalid vertices, edges and faces
  void tidy_up() {
    // erase actual faces
    for (const auto t : m_invalid_tetraeders) {
      // reindex deleted faces indices;
      for (auto& t_to_reindex : m_invalid_tetraeders)
        if (t_to_reindex.i > t.i) --t_to_reindex.i;

      m_tetraeders.erase(m_tetraeders.begin() + t.i);
      for (const auto& [key, prop] : m_tetraeder_properties) {
        prop->erase(t.i);
      }
    }
    m_invalid_tetraeders.clear();

    // tidy up mesh
    parent_t::tidy_up();
  }

  //----------------------------------------------------------------------------
  constexpr bool is_valid(tetraeder t) const {
    return boost::find(m_invalid_tetraeders, t) == m_invalid_tetraeders.end();
  }

  //----------------------------------------------------------------------------
  constexpr void clear_tetraeders() {
    m_tetraeders.clear();
    m_tetraeders.shrink_to_fit();
    m_invalid_tetraeders.clear();
    m_invalid_tetraeders.shrink_to_fit();
  }

  //----------------------------------------------------------------------------
  void clear() {
    parent_t::clear();
    clear_tetraeders();
  }

  //----------------------------------------------------------------------------
  constexpr auto tetraeders() const { return tetraeder_container{this}; }
  auto&       tetraeders(vertex v) { return m_tetraeders_of_vertices->at(v); }
  const auto& tetraeders(vertex v) const { return m_tetraeders_of_vertices->at(v); }
  auto&       tetraeders(edge e) { return m_tetraeders_of_edges->at(e); }
  const auto& tetraeders(edge e) const { return m_tetraeders_of_edges->at(e); }
  auto&       tetraeders(face f) { return m_tetraeders_of_faces->at(f); }
  const auto& tetraeders(face f) const { return m_tetraeders_of_faces->at(f); }

  auto&       vertices(tetraeder t) { return m_tetraeders[t.i]; }
  const auto& vertices(tetraeder t) const { return m_tetraeders[t.i]; }
  auto&       edges(tetraeder t) { return m_edges_of_tetraeders->at(t); }
  const auto& edges(tetraeder t) const { return m_edges_of_tetraeders->at(t); }
  auto&       faces(tetraeder t) { return m_faces_of_tetraeders->at(t); }
  const auto& faces(tetraeder t) const { return m_faces_of_tetraeders->at(t); }

  auto num_tetraeders() const { return m_tetraeders.size(); }
  auto num_tetraeders(vertex v) const { return tetraeders(v).size(); }
  auto num_tetraeders(edge e) const { return tetraeders(e).size(); }
  auto num_tetraeders(face f) const { return tetraeders(f).size(); }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& tetraeder_property(const std::string& name) {
    return *dynamic_cast<tetprop<T>*>(
        m_tetraeder_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  const auto& tetraeder_property(const std::string& name) const {
    return *dynamic_cast<tetprop<T>*>(
        m_tetraeder_properties.at(name).get());
  }

  //----------------------------------------------------------------------------
  template <typename T>
  auto& add_tetraeder_property(const std::string& name, const T& value = T{}) {
    auto [it, suc] = m_tetraeder_properties.insert(
        std::pair{name, std::make_unique<tetprop<T>>(value)});
    auto prop = dynamic_cast<tetprop<T>*>(it->second.get());
    prop->resize(num_tetraeders());
    return *prop;
  }

  //----------------------------------------------------------------------------
  void write_vtk_as_mesh(const std::string& filepath,
                 const std::string& title = "tatooine tetmesh") {
    parent_t::write_vtk(filepath, title);
  }

  //----------------------------------------------------------------------------
  void write_vtk(const std::string& filepath,
                 const std::string& title = "tatooine tetmesh") {
    vtk::legacy_file_writer writer(filepath, vtk::UNSTRUCTURED_GRID);
    if (writer.is_open()) {
      writer.set_title(title);
      writer.write_header();

      // write points
      std::vector<std::array<Real, 3>> points;
      points.reserve(this->m_points.size());
      for (const auto& p : this->m_points)
        if constexpr (N == 3)
          points.push_back({p(0), p(1), p(2)});
        else
          points.push_back({p(0), p(1), 0});
      writer.write_points(points);

      // write tets
      std::vector<std::vector<size_t>> tets;
      std::vector<vtk::CellType> cell_types;
      tets.reserve(num_tetraeders());
      cell_types.reserve(num_tetraeders());
      for (auto t : tetraeders()) {
        tets.emplace_back();
        for (auto v : at(t)) tets.back().push_back(v.i);
        cell_types.push_back(vtk::TETRA);
      }
      writer.write_cells(tets);
      writer.write_cell_types(cell_types);

      // write point data
      if (this->m_vertex_properties.size() > 3)
        writer.write_point_data(this->m_points.size());
      for (const auto& [name, prop] : this->m_vertex_properties) {
        if (name != "v:edges" && name != "v:tetraeders" && name != "v:faces") {
          std::vector<std::vector<Real>> data;
          data.reserve(this->m_points.size());

          if (prop->type() == typeid(vec<Real, 4>)) {
            for (const auto& v4 :
                 *dynamic_cast<const vertex_prop<vec<Real, 4>>*>(
                     prop.get()))
              data.push_back({v4(0), v4(1), v4(2), v4(3)});

          } else if (prop->type() == typeid(vec<Real, 3>)) {
            for (const auto& v3 :
                 *dynamic_cast<const vertex_prop<vec<Real, 3>>*>(
                     prop.get()))
              data.push_back({v3(0), v3(1), v3(2)});

          } else if (prop->type() == typeid(vec<Real, 2>)) {
            const auto& casted_prop =
                *dynamic_cast<const vertex_prop<vec<Real, 2>>*>(
                    prop.get());
            for (const auto& v2 : casted_prop) data.push_back({v2(0), v2(1)});

          } else if (prop->type() == typeid(Real)) {
            for (const auto& scalar :
                 *dynamic_cast<const vertex_prop<Real>*>(prop.get()))
              data.push_back({scalar});
          }
          if (!data.empty()) writer.write_scalars(name, data);
        }
      }

      // write cell data
      if (m_tetraeder_properties.size() > 2)
        writer.write_cell_data(num_tetraeders());
      for (const auto& [name, prop] : m_tetraeder_properties) {
        if (name != "t:edges" && name != "t:faces") {
          if (prop->type() == typeid(vec<Real, 4>)) {
            std::vector<std::vector<Real>> data;
            data.reserve(m_tetraeders.size());
            for (const auto& v4 :
                 *dynamic_cast<const tetprop<vec<Real, 4>>*>(
                     prop.get()))
              data.push_back({v4(0), v4(1), v4(2), v4(3)});
            writer.write_scalars(name, data);

          } else if (prop->type() == typeid(vec<Real, 3>)) {
            std::vector<std::vector<Real>> data;
            data.reserve(m_tetraeders.size());
            for (const auto& v3 :
                 *dynamic_cast<const tetprop<vec<Real, 3>>*>(
                     prop.get()))
              data.push_back({v3(0), v3(1), v3(2)});
            writer.write_scalars(name, data);

          } else if (prop->type() == typeid(vec<Real, 2>)) {
            std::vector<std::vector<Real>> data;
            data.reserve(m_tetraeders.size());
            for (const auto& v2 :
                 *dynamic_cast<const tetprop<vec<Real, 2>>*>(
                     prop.get()))
              data.push_back({v2(0), v2(1)});
            writer.write_scalars(name, data);

          } else if (prop->type() == typeid(double)) {
            std::vector<std::vector<double>> data;
            data.reserve(m_tetraeders.size());
            for (const auto& scalar :
                 *dynamic_cast<const tetprop<double>*>(prop.get()))
              data.push_back({scalar});
            writer.write_scalars(name, data);

          } else if (prop->type() == typeid(float)) {
            std::vector<std::vector<float>> data;
            data.reserve(m_tetraeders.size());
            for (const auto& scalar :
                 *dynamic_cast<const tetprop<float>*>(prop.get()))
              data.push_back({scalar});
            writer.write_scalars(name, data);
          }
        }
      }
      writer.close();
    }
  }

  //------------------------------------------------------------------------------
  // void delaunay(bool quiet = true) {
  //   tetgenio in, out;
  //   pointset_t::to_tetgen_io(in);
  //   tetgenbehavior b;
  //   if (quiet) { b.quiet = 1; }
  //   tetrahedralize(&b, &in, &out);
  //   convert_indices(out);
  //   insert_tetraeders(out);
  // }

  //------------------------------------------------------------------------------
  // void constrained_delaunay(bool quiet = true) {
  //   tetgenio in, out;
  //   mesh_t::to_tetgen_io(in);
  //   tetgenbehavior b;
  //   b.nobisect = 1;
  //   b.plc      = 1;
  //   if (quiet) { b.quiet = 1; }
  //   tetrahedralize(&b, &in, &out);
  //
  //   convert_indices(out);
  //   insert_tetraeders(out);
  // }

  //----------------------------------------------------------------------------
  // void constrained_delaunay(const std::vector<face>& fs, bool quiet = true) {
  //   tetgenio in, out;
  //   mesh_t::to_tetgen_io(in, fs);
  //   tetgenbehavior b;
  //   b.nobisect = 1;
  //   if (quiet) { b.quiet = 1; }
  //   b.plc = 1;
  //   tetrahedralize(&b, &in, &out);
  //   convert_indices(out);
  //   insert_tetraeders(out);
  // }

  //----------------------------------------------------------------------------
  //! inserts tetraeders from a tetgenio object
  // void insert_tetraeders(const tetgenio& io) {
  //   for (int t = 0; t < io.numberoftetrahedra; ++t) {
  //     auto tet = &io.tetrahedronlist[t * 4];
  //     insert_tetraeder(
  //         io.pointattributelist[tet[0]], io.pointattributelist[tet[1]],
  //         io.pointattributelist[tet[2]], io.pointattributelist[tet[3]]);
  //   }
  // }

  //-----------------------------------------------------------------------------
  //! inserts new vertices and sets vertex indices to tetgenio
  // void convert_indices(tetgenio& io) {
  //   for (int i = 0; i < io.numberofpoints; ++i) {
  //     // point from input
  //     if (io.pointmarkerlist[i] < 0) {
  //       io.pointmarkerlist[i] = static_cast<int>(io.pointattributelist[i]);
  //     }
  //
  //     // new point
  //     else {
  //       io.pointmarkerlist[i] =
  //           insert_vertex(io.pointlist[i * 3],
  //                         io.pointlist[i * 3 + 1],
  //                         io.pointlist[i * 3 + 2]).i;
  //     }
  //   }
  // }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
