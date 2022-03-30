#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/rendering/interactive.h>
#include <tatooine/rendering/interactive/shaders.h>
#include <tatooine/steady_field.h>

#include <boost/program_options.hpp>
//==============================================================================
using namespace tatooine;
auto constexpr advection_direction = forward;
//==============================================================================
using autonomous_particle_flowmap_type =
    AutonomousParticleFlowmapDiscretization<
        2, AutonomousParticle<2>::split_behaviors::five_splits>;
//==============================================================================
struct vis {
  using pass_through =
      rendering::interactive::shaders::colored_pass_through_2d_without_matrices;
  using vec2_gl = Vec2<GLfloat>;
  using vec3_gl = Vec3<GLfloat>;
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "layout (location = 1) in vec3 color;\n"
        "out vec3 frag_color;\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "void main() {\n"
        "  frag_color = color;\n"
        "  gl_Position = projection_matrix *\n"
        "                view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "in vec3 frag_color;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  out_color = vec4(frag_color, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_view_matrix(Mat4<GLfloat>::eye());
    }
    //------------------------------------------------------------------------------
   public:
    //------------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data());
    }
    //------------------------------------------------------------------------------
    auto set_view_matrix(Mat4<GLfloat> const& MV) -> void {
      set_uniform_mat4("view_matrix", MV.data());
    }
  };
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  uniform_rectilinear_grid<double, 2> grid;
  double                              agranovsky_error                = 0;
  double            autonomous_particles_barycentric_coordinate_error = 0;
  double            autonomous_particles_nearest_neighbor_error       = 0;
  double            autonomous_particles_inverse_distance_error       = 0;
  vec2d             cursor_pos;
  vec2d             cursor_pos_projected;
  vec2d             current_point{1, 0.5};
  std::vector<bool> hovered;
  int               point_size = 20;
  rendering::orthographic_camera<GLfloat>                cam;
  std::vector<autonomous_particle2::sampler_type> const& samplers;
  unstructured_triangular_grid2                          local_positions;
  gl::indexeddata<vec2_gl, vec3_gl>                      local_positions_gpu;
  gl::indexeddata<vec2_gl, vec3_gl> local_triangulation_gpu;
  bool                              mouse_down = false;
  std::vector<std::size_t>          hovered_indices;
  float                             minimap_range = 0.25f;
  enum class mode_t { number, radius };
  mode_t                                mode           = mode_t::number;
  std::size_t                           nearest_number = 3;
  double                                nearest_radius = 0.1;
  std::vector<pointset2::vertex_handle> nearest_neighbor_indices;

  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  vis(uniform_rectilinear_grid<double, 2> const& g, auto const& samplers)
      : grid{g},
        hovered(size(samplers), false),
        cam{vec3_gl{0, 0, 0},
            vec3_gl{0, 0, -1},
            -minimap_range,
            minimap_range,
            -minimap_range,
            minimap_range,
            -1,
            1,
            Vec4<std::size_t>{10, 10, 500, 500}},
        samplers{samplers} {
    local_positions.vertices().resize(size(samplers));
    local_positions_gpu.vertexbuffer().resize(size(samplers));
    local_positions_gpu.indexbuffer().resize(size(samplers));
    for (std::size_t i = 0; i < size(samplers); ++i) {
      local_positions_gpu.indexbuffer()[i] = i;
    }
    update_points();
  }
  //----------------------------------------------------------------------------
  auto properties() {
    ImGui::Text("Vis");
    if (ImGui::DragFloat("scale", &minimap_range, 0.01f, 0.0f, FLT_MAX,
                         "%.06f")) {
      cam = {vec3_gl{0, 0, 0},
             vec3_gl{0, 0, -1},
             -minimap_range,
             minimap_range,
             -minimap_range,
             minimap_range,
             -1,
             1,
             Vec4<std::size_t>{10, 10, 500, 500}};
    }
    ImGui::Text("current point: [%e, %e]", current_point.x(),
                current_point.y());
    ImGui::Text("Agran error: %e", agranovsky_error);

    ImGui::Text("barycentric coordinate error: %e ",
                autonomous_particles_barycentric_coordinate_error);
    ImGui::Text(
        "barycentric coordinate diff:  %e ",
        agranovsky_error - autonomous_particles_barycentric_coordinate_error);

    ImGui::Text("nearest neighbor error: %e ",
                autonomous_particles_nearest_neighbor_error);
    ImGui::Text("nearest neighbor diff:  %e ",
                agranovsky_error - autonomous_particles_nearest_neighbor_error);

    ImGui::Text("inverse distance error: %e ",
                autonomous_particles_inverse_distance_error);
    ImGui::Text("inverse distance diff:  %e ",
                agranovsky_error - autonomous_particles_inverse_distance_error);

    if (ImGui::BeginCombo("##combo",
                          mode == mode_t::number ? "number" : "radius")) {
      if (ImGui::Selectable("Number", mode == mode_t::number)) {
        mode = mode_t::number;
      }
      if (mode == mode_t::number) {
        ImGui::SetItemDefaultFocus();
      }
      if (ImGui::Selectable("Radius", mode == mode_t::radius)) {
        mode = mode_t::radius;
      }
      if (mode == mode_t::radius) {
        ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    switch (mode) {
      case mode_t::number:
        if (ImGui::DragSizeT("number of nearest neighbors", &nearest_number, 1,
                             1, 1000)) {
          update_nearest_neighbors();
        }
        break;
      case mode_t::radius:
        if (ImGui::DragDouble("nearest neighbor radius", &nearest_radius, 0.01,
                              0.0, FLT_MAX)) {
          update_nearest_neighbors();
        }
        break;
    }
  }
  //----------------------------------------------------------------------------
  auto render(auto const& renderable, rendering::camera auto const& cam) {
    shader::get().set_projection_matrix(cam.projection_matrix());
    shader::get().set_view_matrix(cam.view_matrix());
    shader::get().bind();
    gl::point_size(static_cast<GLfloat>(point_size));
    {
      auto p = gl::indexeddata<vec2_gl, vec3_gl>{};
      p.vertexbuffer().resize(1);
      p.indexbuffer().resize(1);
      {
        auto data = p.vertexbuffer().map();
        data[0]   = {vec2_gl{current_point}, vec3_gl{0, 0, 0}};
      }
      {
        auto data = p.indexbuffer().map();
        data[0]   = 0;
      }
      p.draw_points();
    }

    auto& ellipse_shader = rendering::interactive::renderer<
        geometry::ellipse<double>>::shader::get();
    auto& ellipse_geometry = rendering::interactive::renderer<
        geometry::ellipse<double>>::geometry::get();
    ellipse_shader.set_projection_matrix(cam.projection_matrix());
    ellipse_shader.bind();
    ellipse_shader.set_color(1, 0, 0);
    gl::line_width(3);
    for (std::size_t i = 0; i < size(hovered); ++i) {
      if (hovered[i]) {
        ellipse_shader.set_model_view_matrix(
            cam.view_matrix() *
            rendering::interactive::renderer<geometry::ellipse<double>>::
                construct_model_matrix(samplers[i].S(advection_direction),
                                       samplers[i].x0(advection_direction)));
        ellipse_geometry.draw_line_loop();
      }
    }
    ellipse_shader.set_color(0, 1, 0);
    for (auto v : nearest_neighbor_indices) {
      ellipse_shader.set_model_view_matrix(
          cam.view_matrix() *
          rendering::interactive::renderer<geometry::ellipse<double>>::
              construct_model_matrix(
                  samplers[v.index()].S(advection_direction),
                  samplers[v.index()].x0(advection_direction)));
      ellipse_geometry.draw_line_loop();
    }
  }
  //----------------------------------------------------------------------------
  auto late_render() {
    cam.set_gl_viewport();
    {
      auto outline = gl::indexeddata<vec2_gl>{};
      outline.vertexbuffer().resize(4);
      {
        auto data = outline.vertexbuffer().map();
        data[0]   = vec2_gl{-0.999, -0.999};
        data[1]   = vec2_gl{0.999, -0.999};
        data[2]   = vec2_gl{0.999, 0.999};
        data[3]   = vec2_gl{-0.999, 0.999};
      }
      outline.indexbuffer().resize(6);
      {
        auto data = outline.indexbuffer().map();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 3;
        data[3]   = 1;
        data[4]   = 2;
        data[5]   = 3;
      }
      pass_through::get().bind();
      pass_through::get().set_color(1, 1, 1);
      gl::disable_depth_test();
      outline.draw_triangles();

      outline.indexbuffer().resize(4);
      {
        auto data = outline.indexbuffer().map();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 2;
        data[3]   = 3;
      }
      gl::line_width(3);
      pass_through::get().set_color(0, 0, 0);
      outline.draw_line_loop();
      gl::enable_depth_test();
    }
    {
      auto axes = gl::indexeddata<vec2_gl>{};
      axes.vertexbuffer().resize(4);
      axes.indexbuffer().resize(4);
      {
        auto data = axes.vertexbuffer().map();
        data[0]   = vec2_gl{0, -1};
        data[1]   = vec2_gl{0, 1};
        data[2]   = vec2_gl{-1, 0};
        data[3]   = vec2_gl{1, 0};
      }
      {
        auto data = axes.indexbuffer().map();
        data[0]   = 0;
        data[1]   = 1;
        data[2]   = 2;
        data[3]   = 3;
      }
      gl::line_width(1);
      axes.draw_lines();
    }

    shader::get().bind();
    shader::get().set_projection_matrix(cam.projection_matrix());
    shader::get().set_view_matrix(cam.view_matrix());
    gl::point_size(5);
    local_positions_gpu.draw_points();
    local_triangulation_gpu.draw_lines();
  }
  //----------------------------------------------------------------------------
  auto update_points() {
    local_positions.invalidate_kd_tree();
    {
      auto map = local_positions_gpu.vertexbuffer().wmap();
      for_loop(
          [&](auto const i) {
            // auto const vertex_pos =
            //     -samplers[i].local_pos(current_point, advection_direction);
            auto const vertex_pos        = samplers[i].x0(advection_direction);
            local_positions.vertex_at(i) = vertex_pos;
            get<0>(map[i])               = vec2_gl{vertex_pos};
          },
          execution_policy::parallel, samplers.size());
    }
    {
      local_positions.build_delaunay_mesh();
      local_triangulation_gpu.vertexbuffer() =
          local_positions_gpu.vertexbuffer();
      if (local_positions.triangles().size() > 0) {
        local_triangulation_gpu.indexbuffer().resize(
            size(local_positions.triangles()) * 6);
        auto map = local_triangulation_gpu.indexbuffer().wmap();
        for_loop(
            [&](auto const i) {
              auto const [v0, v1, v2] = local_positions
                  [unstructured_triangular_grid2::triangle_handle{i}];
              map[i * 6]     = v0.index();
              map[i * 6 + 1] = v1.index();
              map[i * 6 + 2] = v1.index();
              map[i * 6 + 3] = v2.index();
              map[i * 6 + 4] = v2.index();
              map[i * 6 + 5] = v0.index();
            },
            execution_policy::parallel, size(local_positions.triangles()));
      }
    }
    agranovsky_error = grid.scalar_vertex_property("error_agranovksy")
                           .linear_sampler()(current_point);
    autonomous_particles_barycentric_coordinate_error =
        grid.scalar_vertex_property("error_barycentric_coordinate")
            .linear_sampler()(current_point);
    autonomous_particles_nearest_neighbor_error =
        grid.scalar_vertex_property("error_nearest_neighbor")
            .linear_sampler()(current_point);
    autonomous_particles_inverse_distance_error =
        grid.scalar_vertex_property("error_inverse_distance")
            .linear_sampler()(current_point);
  }
  //----------------------------------------------------------------------------
  auto update_nearest_neighbors() -> void {
    switch (mode) {
      case mode_t::number:
        nearest_neighbor_indices =
            local_positions
                .nearest_neighbors(cursor_pos_projected, nearest_number)
                .first;
        break;
      case mode_t::radius:
        nearest_neighbor_indices =
            local_positions
                .nearest_neighbors_radius(cursor_pos_projected, nearest_radius)
                .first;
        break;
    }
    {
      auto map = local_positions_gpu.vertexbuffer().rwmap();
      for (auto& v : map) {
        get<1>(v) = {0, 0, 0};
      }
      for (auto const& v : nearest_neighbor_indices) {
        get<1>(map[v.index()]) = {0, 1, 0};
      }
    }
  }
  //----------------------------------------------------------------------------
  auto update_hovered() {
    for (auto const v : local_positions.vertices()) {
      auto const proj    = this->cam.project(vec2f{local_positions[v]}).xy();
      hovered[v.index()] = euclidean_distance(proj, cursor_pos) < 10 ||
                           samplers[v.index()].is_inside(cursor_pos_projected,
                                                         advection_direction);
      if (hovered[v.index()]) {
        local_positions_gpu.vertexbuffer().write_only_element_at(v.index()) = {
            vec2_gl{local_positions[v]}, vec3_gl{1, 0, 0}};
        //} else {
        //  local_positions_gpu.vertexbuffer().write_only_element_at(v.index())
        //  = {
        //      vec2_gl{local_positions[v]}, vec3_gl{0, 0, 0}};
      }
    }
  }
  //----------------------------------------------------------------------------
  auto on_cursor_moved(double const cursor_x, double const cursor_y,
                       rendering::camera auto const& cam) {
    cursor_pos           = {cursor_x, cursor_y};
    cursor_pos_projected = vec2{cam.unproject(vec2f{cursor_pos}).xy()};
    if (mouse_down) {
      current_point = vec2{cam.unproject(vec2f{cursor_pos}).xy()};
      update_points();
    }
    update_nearest_neighbors();
    update_hovered();
  }
  //----------------------------------------------------------------------------
  auto on_button_pressed(gl::button b, rendering::camera auto const& cam) {
    if (b == gl::button::left) {
      mouse_down    = true;
      current_point = vec2{cam.unproject(vec2f{cursor_pos}).xy()};
      update_points();
      update_nearest_neighbors();
    }
  }
  //----------------------------------------------------------------------------
  auto on_button_released(gl::button b) {
    if (b == gl::button::left) {
      mouse_down = false;
    }
  }
};
struct velocity_janos : vectorfield<velocity_janos, real_number, 2> {
  constexpr auto evaluate(vec2 const& p, real_number const t) const
      -> vectorfield<velocity_janos, real_number, 2>::tensor_type {
    return vec2{-p.x() * (2 * p.x() * p.x() - 2) / 2, -p.y()};
  }
};
struct flowmap_janos {
  static constexpr auto num_dimensions() -> std::size_t { return 2; }
  auto                  operator()(auto const& p, real_number const t,
                  real_number const tau) const {
    return vec2{p(0) / sqrt(-exp(-2 * tau) * p(0) * p(0) + p(0) * p(0) + gcem::exp(-2 * tau)),
                p(1) * gcem::exp(-tau)};
  };

  template <std::size_t N>
  auto operator()(Mat<2, N> const& p, real_number const t, real_number const tau) const {
    auto out = Mat<2, N>{};
    for (std::size_t i = 0; i < N; ++i) {
      out.col(i) = operator()(p.col(i), t, tau);
    }
    return out;
  }
};
//------------------------------------------------------------------------------
auto doit(auto& resample_grid, auto const& initial_particles,
          auto& uuid_generator, auto const t_0, auto const t_end,
          auto const inverse_distance_num_samples) {
  auto const tau = t_end - t_0;
  auto       phi = flowmap_janos{};

  using clock = std::chrono::high_resolution_clock;
  auto before = std::chrono::time_point<clock>{};
  auto after  = std::chrono::time_point<clock>{};
  std::cout << "advecting " << size(initial_particles)
            << " autonomous particles...\n";
  before                            = clock::now();
  auto flowmap_autonomous_particles = autonomous_particle_flowmap_type{
      phi, t_end, 0.01, initial_particles, uuid_generator};
  after = clock::now();
  std::cout << "advecting " << size(initial_particles)
            << " autonomous particles... done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  std::cout << "building mesh with centers...\n";
  before = clock::now();
  auto resample_grid_with_particle_centers =
      unstructured_simplicial_grid<real_number, 2, 2>{};
  resample_grid.vertices().iterate_indices([&](auto const... is) {
    resample_grid_with_particle_centers.insert_vertex(
        resample_grid.vertex_at(is...));
  });
  for (auto const& p : flowmap_autonomous_particles.samplers()) {
    resample_grid_with_particle_centers.insert_vertex(
        p.x0(advection_direction));
  }
  resample_grid_with_particle_centers.build_delaunay_mesh();
  after = clock::now();
  std::cout << "building mesh with centers... done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  //////
  // flowmap discretizations
  //////
  auto& flowmap_numerical_prop =
      resample_grid_with_particle_centers.vec2_vertex_property("numerical");
  auto& flowmap_regular_prop =
      resample_grid_with_particle_centers.vec2_vertex_property("regular");
  // auto&      flowmap_autonomous_particles_barycentric_coordinate_prop =
  //     resample_grid_with_particle_centers.vec2_vertex_property("barycentric_coordinate");
  // auto& flowmap_autonomous_particles_nearest_neighbor_prop =
  //     resample_grid_with_particle_centers.vec2_vertex_property("nearest_neighbor");
  // auto& flowmap_autonomous_particles_inverse_distance_prop =
  //    resample_grid_with_particle_centers.vec2_vertex_property(
  //        "inverse_distance");
  // auto& flowmap_autonomous_particles_inverse_distance_local_prop =
  //     resample_grid_with_particle_centers.vec2_vertex_property("inverse_distance_local");
  // auto& flowmap_autonomous_particles_inverse_distance_without_gradient_prop =
  //    resample_grid_with_particle_centers.vec2_vertex_property(
  //        "inverse_distance_without_gradient");
  auto& flowmap_autonomous_particles_radial_basis_functions_prop =
      resample_grid_with_particle_centers.vec2_vertex_property(
          "radialbasis_functions");
  auto& flowmap_agranovsky_prop =
      resample_grid_with_particle_centers.vec2_vertex_property("agranovksy");

  //////
  // flowmap errors
  //////
  auto& flowmap_error_regular_prop =
      resample_grid_with_particle_centers.scalar_vertex_property(
          "error_regular");
  // auto& flowmap_error_autonomous_particles_barycentric_coordinate_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_barycentric_coordinate");

  // auto& flowmap_error_autonomous_particles_nearest_neighbor_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_nearest_neighbor");

  // auto& flowmap_error_autonomous_particles_inverse_distance_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property(
  //         "error_inverse_distance");

  // auto& flowmap_error_autonomous_particles_inverse_distance_local_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_inverse_distance_local");

  // auto&
  //     flowmap_error_autonomous_particles_inverse_distance_without_gradient_prop
  //     =
  //         resample_grid_with_particle_centers.scalar_vertex_property(
  //             "error_inverse_distance_without_gradients");

  auto& flowmap_error_autonomous_particles_radial_basis_functions_prop =
      resample_grid_with_particle_centers.scalar_vertex_property(
          "error_radial_basis_functions");

  auto& flowmap_error_agranovksy_prop =
      resample_grid_with_particle_centers.scalar_vertex_property(
          "error_agranovksy");

  //////
  // flowmap comparisons
  //////

  // auto& flowmap_error_diff_regular_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_diff_regular");

  // auto& flowmap_error_diff_barycentric_coordinate_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_diff_barycentric_coordinate");

  // auto& flowmap_error_diff_nearest_neighbor_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_diff_nearest_neighbor");

  // auto& flowmap_error_diff_inverse_distance_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property(
  //         "error_diff_inverse_distance");

  // auto& flowmap_error_diff_inverse_distance_local_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property("error_diff_inverse_distance_local");

  // auto& flowmap_error_diff_inverse_distance_without_gradient_prop =
  //     resample_grid_with_particle_centers.scalar_vertex_property(
  //         "error_diff_inverse_distance_without_gradient");

  auto& flowmap_error_diff_radial_basis_functions_prop =
      resample_grid_with_particle_centers.scalar_vertex_property(
          "error_diff_radial_basis_functions");

  std::cout << "measuring autonomous particles flowmap...\n";
  before = clock::now();

#pragma omp parallel for
  for (std::size_t i = 0;
       i < resample_grid_with_particle_centers.vertices().size(); ++i) {
    auto const v =
        unstructured_simplicial_grid<real_number, 2, 2>::vertex_handle{i};
    auto const q = resample_grid_with_particle_centers[v];
    // flowmap_autonomous_particles_barycentric_coordinate_prop[v] =
    //     flowmap_autonomous_particles.sample_barycentric_coordinate(
    //         q, advection_direction, execution_policy::sequential);
    // flowmap_autonomous_particles_nearest_neighbor_prop[v] =
    //     flowmap_autonomous_particles.sample_nearest_neighbor(
    //         q, advection_direction, execution_policy::sequential);

    // flowmap_autonomous_particles_inverse_distance_prop[v] =
    //     flowmap_autonomous_particles.sample_inverse_distance(
    //         inverse_distance_num_samples, q, advection_direction,
    //         execution_policy::sequential);

    // flowmap_autonomous_particles_inverse_distance_local_prop[v] =
    //     flowmap_autonomous_particles.sample_inverse_distance_local(
    //         inverse_distance_num_samples, q, advection_direction,
    //         execution_policy::sequential);

    // flowmap_autonomous_particles_inverse_distance_without_gradient_prop[v] =
    //     flowmap_autonomous_particles.sample_inverse_distance_without_gradient(
    //         inverse_distance_num_samples, q, advection_direction,
    //         execution_policy::sequential);

    flowmap_autonomous_particles_radial_basis_functions_prop[v] =
        flowmap_autonomous_particles.sample_radial_basis_functions(
            q, 0.03, advection_direction, execution_policy::sequential);
  }
  after = clock::now();
  std::cout << "measuring autonomous particles flowmap done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  // auto const num_particles_after_advection = 10000;
  auto const num_particles_after_advection =
      flowmap_autonomous_particles.num_particles();
  std::cout << "num_particles_after_advection: "
            << num_particles_after_advection << '\n';

  std::cout << "measuring numerical flowmap...\n";
  before = clock::now();
#pragma omp parallel for
  for (std::size_t i = 0;
       i < resample_grid_with_particle_centers.vertices().size(); ++i) {
    auto const v =
        unstructured_simplicial_grid<real_number, 2, 2>::vertex_handle{i};
    auto const q = resample_grid_with_particle_centers[v];
    if (advection_direction == backward) {
      flowmap_numerical_prop[v] = phi(q, t_end, -tau);
    } else {
      flowmap_numerical_prop[v] = phi(q, t_0, tau);
    }
  }
  after = clock::now();
  std::cout << "measuring numerical flowmap done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  auto const regularized_height = static_cast<std::size_t>(
      std::ceil(std::sqrt((num_particles_after_advection / 2))));
  auto const regularized_width = regularized_height * 2;
  std::cout << "regularized_width: " << regularized_width << '\n';
  std::cout << "regularized_height: " << regularized_height << '\n';
  std::cout << "number of particles in regular sampler: "
            << regularized_width * regularized_height << '\n';
  std::cout << "advecting regular flowmap...\n";
  before               = clock::now();
  auto flowmap_regular = regular_flowmap_discretization2{phi,
                                                         t_0,
                                                         tau,
                                                         resample_grid.front(),
                                                         resample_grid.back(),
                                                         regularized_width,
                                                         regularized_height};
  after                = clock::now();
  std::cout << "advecting regular flowmap... done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  std::cout << "measuring regular flowmap...\n";
  before = clock::now();
#pragma omp parallel for
  for (std::size_t i = 0;
       i < resample_grid_with_particle_centers.vertices().size(); ++i) {
    auto const v =
        unstructured_simplicial_grid<real_number, 2, 2>::vertex_handle{i};
    auto const q = resample_grid_with_particle_centers[v];
    try {
      flowmap_regular_prop[v] = flowmap_regular.sample(q, advection_direction);
    } catch (...) {
      flowmap_regular_prop[v] = vec2{0.0 / 0.0, 0.0 / 0.0};
    }
  }
  after = clock::now();
  std::cout << "measuring regular flowmap done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  auto const agranovsky_delta_t = 0.5;
  auto const num_agranovksy_steps =
      static_cast<std::size_t>(std::ceil((t_end - t_0) / agranovsky_delta_t));
  auto const regularized_height_agranovksky = static_cast<std::size_t>(
      std::ceil(regularized_height /
                std::sqrt(static_cast<double>(num_agranovksy_steps))));
  auto const regularized_width_agranovksky = static_cast<std::size_t>(
      std::ceil(regularized_width * regularized_height /
                (std::sqrt(static_cast<double>(num_agranovksy_steps)) *
                 regularized_height)));
  std::cout << "num_agranovksy_steps: " << num_agranovksy_steps << '\n';
  std::cout << "regularized_width_agranovksky: "
            << regularized_width_agranovksky << '\n';
  std::cout << "regularized_height_agranovksky: "
            << regularized_height_agranovksky << '\n';
  std::cout << "number of particles in agranovsky sampler: "
            << regularized_width_agranovksky * regularized_height_agranovksky *
                   num_agranovksy_steps
            << '\n';

  std::cout << "advecting agranovsky flowmap...\n";
  before = clock::now();
  auto flowmap_agranovsky =
      agranovsky_flowmap_discretization2{phi,
                                         t_0,
                                         tau,
                                         agranovsky_delta_t,
                                         resample_grid.front(),
                                         resample_grid.back(),
                                         regularized_width_agranovksky,
                                         regularized_height_agranovksky};
  after = clock::now();
  std::cout << "advecting agranovsky flowmap... done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";
  std::cout << "measuring agranovsky flowmap...\n";
  before = clock::now();
#pragma omp parallel for
  for (std::size_t i = 0;
       i < resample_grid_with_particle_centers.vertices().size(); ++i) {
    auto const v =
        unstructured_simplicial_grid<real_number, 2, 2>::vertex_handle{i};
    auto const q = resample_grid_with_particle_centers[v];
    try {
      flowmap_agranovsky_prop[v] =
          flowmap_agranovsky.sample(q, advection_direction);
    } catch (...) {
      flowmap_agranovsky_prop[v] = vec2{0.0 / 0.0, 0.0 / 0.0};
    }
  }
  after = clock::now();
  std::cout << "measuring agranovsky flowmap done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  std::cout << "measuring errors...\n";
  before = clock::now();
#pragma omp parallel for
  for (std::size_t i = 0;
       i < resample_grid_with_particle_centers.vertices().size(); ++i) {
    auto const v =
        unstructured_simplicial_grid<real_number, 2, 2>::vertex_handle{i};
    auto const q   = resample_grid_with_particle_centers[v];

    // flowmap_error_autonomous_particles_barycentric_coordinate_prop[v]
    // =
    //     euclidean_distance(
    //         flowmap_numerical_prop[v],
    //         flowmap_autonomous_particles_barycentric_coordinate_prop[v]);
    //
    // flowmap_error_autonomous_particles_nearest_neighbor_prop[v] =
    //     euclidean_distance(
    //         flowmap_numerical_prop[v],
    //         flowmap_autonomous_particles_nearest_neighbor_prop[v]);

    // flowmap_error_autonomous_particles_inverse_distance_prop[v] =
    //     euclidean_distance(
    //         flowmap_numerical_prop[v],
    //         flowmap_autonomous_particles_inverse_distance_prop[v]);

    // flowmap_error_autonomous_particles_inverse_distance_local_prop[v] =
    //     euclidean_distance(
    //         flowmap_numerical_prop[v],
    //         flowmap_autonomous_particles_inverse_distance_local_prop[v]);

    // flowmap_error_autonomous_particles_inverse_distance_without_gradient_prop
    //     [v] = euclidean_distance(
    //         flowmap_numerical_prop[v],
    //         flowmap_autonomous_particles_inverse_distance_without_gradient_prop
    //             [v]);

    flowmap_error_autonomous_particles_radial_basis_functions_prop[v] =
        euclidean_distance(
            flowmap_numerical_prop[v],
            flowmap_autonomous_particles_radial_basis_functions_prop[v]);

    flowmap_error_regular_prop[v] =
        euclidean_distance(flowmap_numerical_prop[v], flowmap_regular_prop[v]);

    flowmap_error_agranovksy_prop[v] = euclidean_distance(
        flowmap_numerical_prop[v], flowmap_agranovsky_prop[v]);

    // flowmap_error_diff_barycentric_coordinate_prop[v] =
    //     flowmap_error_agranovksy_prop[v] -
    //     flowmap_error_autonomous_particles_barycentric_coordinate_prop[v];

    // flowmap_error_diff_nearest_neighbor_prop[v] =
    //     flowmap_error_agranovksy_prop[v] -
    //     flowmap_error_autonomous_particles_nearest_neighbor_prop[v];

    // flowmap_error_diff_inverse_distance_prop[v] =
    //     flowmap_error_agranovksy_prop[v] -
    //     flowmap_error_autonomous_particles_inverse_distance_prop[v];

    // flowmap_error_diff_inverse_distance_local_prop[v] =
    //     flowmap_error_agranovksy_prop[v] -
    //     flowmap_error_autonomous_particles_inverse_distance_local_prop[v];

    // flowmap_error_diff_inverse_distance_without_gradient_prop[v] =
    //     flowmap_error_agranovksy_prop[v] -
    //     flowmap_error_autonomous_particles_inverse_distance_without_gradient_prop
    //         [v];

    flowmap_error_diff_radial_basis_functions_prop[v] =
        flowmap_error_agranovksy_prop[v] -
        flowmap_error_autonomous_particles_radial_basis_functions_prop[v];
  }
  after = clock::now();
  std::cout << "measuring errors done! ("
            << std::chrono::duration_cast<std::chrono::milliseconds>(after -
                                                                     before)
                       .count() /
                   1000.0
            << "s)\n";

  rendering::interactive::pre_setup();
  auto m = vis{resample_grid, flowmap_autonomous_particles.samplers()};
  auto advected_particles = std::vector<geometry::ellipse<real_number>>{};
  std::ranges::copy(flowmap_autonomous_particles.samplers() |
                        std::views::transform([](auto const& s) {
                          return s.ellipse(advection_direction);
                        }),
                    std::back_inserter(advected_particles));

  auto regular_flowmap_grid =
      rectilinear_grid{linspace{-2.0, 2.0, regularized_width},
                       linspace{-1.0, 1.0, regularized_height}};
  auto regular_flowmap_grid_agranovsky =
      rectilinear_grid{linspace{-2.0, 2.0, regularized_width_agranovksky},
                       linspace{-1.0, 1.0, regularized_height_agranovksky}};

  rendering::interactive::render(
      rectilinear_grid{linspace{-2.0, 2.0, 3}, linspace{-1.0, 1.0, 2}},
      // regular_flowmap_grid,regular_flowmap_grid_agranovsky,
      initial_particles, advected_particles, m,
      resample_grid_with_particle_centers);
};
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto                  uuid_generator               = std::atomic_uint64_t{};
  [[maybe_unused]] auto t_0                          = real_number(0);
  [[maybe_unused]] auto t_end                        = real_number(4);
  [[maybe_unused]] auto num_frames                   = std::size_t(10);
  [[maybe_unused]] auto inverse_distance_num_samples = std::size_t(5);

  namespace po                                       = boost::program_options;
  auto desc = po::options_description{"Allowed options"};
  desc.add_options()("help", "produce help message")(
      "inverse_distance_num_samples", po::value<real_number>(),
      "inverse distance number of samples")("t_0", po::value<real_number>(),
                                            "start time of integration")(
      "t_end", po::value<real_number>(), "end time of integration");

  auto variables_map = po::variables_map{};
  po::store(po::parse_command_line(argc, argv, desc), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help") > 0) {
    std::cout << desc;
    return 0;
  }

  if (variables_map.count("inverse_distance_num_samples") > 0) {
    num_frames =
        variables_map["inverse_distance_num_samples"].as<std::size_t>();
  }
  if (variables_map.count("t_0") > 0) {
    t_0 = variables_map["t_0"].as<real_number>();
  }
  if (variables_map.count("t_end") > 0) {
    t_end = variables_map["t_end"].as<real_number>();
  } else {
    throw std::runtime_error{"Flag --t_end not specified."};
  }
  //============================================================================
  auto resample_grid =
      rectilinear_grid{linspace{-2.0, 2.0, 201}, linspace{-1.0, 1.0, 101}};
  auto const initial_particles_dg = autonomous_particle2::particles_from_grid(
      t_0, rectilinear_grid{linspace{-2.0, 2.0, 61}, linspace{-1.0, 1.0, 31}},
      uuid_generator);
  doit(resample_grid, initial_particles_dg, uuid_generator, t_0, t_end,
       inverse_distance_num_samples);
}