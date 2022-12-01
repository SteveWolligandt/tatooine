#ifndef TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalFlowmapDiscretization>
struct staggered_flowmap_discretization {
  using internal_flowmap_discretization_type = InternalFlowmapDiscretization;
  using real_type = typename internal_flowmap_discretization_type::real_type;
  static auto constexpr num_dimensions() -> std::size_t {
    return internal_flowmap_discretization_type::num_dimensions();
  }
  using vec_type = vec<real_type, num_dimensions()>;
  using pos_type = vec_type;
  //============================================================================
  mutable std::vector<std::unique_ptr<internal_flowmap_discretization_type>>
                                m_steps              = {};
  std::vector<filesystem::path> m_filepaths_to_steps = {};
  bool                          m_write_to_disk      = false;
  mutable std::mutex            m_deletion_mutex;
  //============================================================================
  staggered_flowmap_discretization(
      staggered_flowmap_discretization const &other)
      : m_filepaths_to_steps{std::move(other.m_filepaths_to_steps)},
        m_write_to_disk{other.m_write_to_disk} {
    for (auto const &step : m_steps) {
      m_steps.emplace_back(new internal_flowmap_discretization_type{step});
    }
  }
  //----------------------------------------------------------------------------
  staggered_flowmap_discretization(
      staggered_flowmap_discretization &&other) noexcept
      : m_steps{std::move(other.m_steps)},
        m_filepaths_to_steps{std::move(other.m_filepaths_to_steps)},
        m_write_to_disk{other.m_write_to_disk} {}
  //----------------------------------------------------------------------------
  auto operator=(staggered_flowmap_discretization const &other)
      -> staggered_flowmap_discretization & {
    for (auto const &step : m_steps) {
      m_steps.emplace_back(new internal_flowmap_discretization_type{step});
    }
    m_filepaths_to_steps = other.m_filepaths_to_steps;
    m_write_to_disk      = other.m_write_to_disk;
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(staggered_flowmap_discretization &&other)
      -> staggered_flowmap_discretization & {
    m_steps              = std::move(other.m_steps);
    m_filepaths_to_steps = std::move(other.m_filepaths_to_steps);
    m_write_to_disk      = std::move(other.m_write_to_disk);
    return *this;
  }
  //============================================================================
  template <typename Flowmap, typename... InternalFlowmapArgs>
  staggered_flowmap_discretization(Flowmap &&flowmap, arithmetic auto const t0,
                                   arithmetic auto const tau,
                                   arithmetic auto const delta_t,
                                   InternalFlowmapArgs &&...args) {
    auto       cur_t0 = real_type(t0);
    auto const t_end = static_cast<real_type>(t0) + static_cast<real_type>(tau);
    m_steps.reserve(static_cast<std::size_t>((t_end - t0) / delta_t) + 2);
    static auto const eps    = real_type(1e-10);
    auto              cnt    = std::size_t{};
    auto const        prefix = random::alpha_numeric_string(5) + "_";
    while (cur_t0 + eps < t0 + tau) {
      std::cout << "begin of while\n";
      auto cur_tau = static_cast<real_type>(delta_t);
      if (cur_t0 + cur_tau > t_end) {
        cur_tau = static_cast<real_type>(t0) + static_cast<real_type>(tau) -
                  static_cast<real_type>(cur_t0);
      }
      std::cout << "cur_tau: " << cur_tau << '\n';
      std::cout << "generating path\n";
      auto const &path =
          m_filepaths_to_steps.emplace_back(prefix + std::to_string(cnt++));
      std::cout << "advecting " << path << '\n';
      m_steps.emplace_back(new internal_flowmap_discretization_type{
          std::forward<Flowmap>(flowmap), cur_t0, cur_tau,
          std::forward<InternalFlowmapArgs>(args)...});
      cur_t0 += cur_tau;

      if (m_write_to_disk) {
        std::cout << "writing " << path << '\n';
        m_steps.back()->write(path);
        std::cout << "resetting " << path << '\n';
        m_steps.back().reset();
      }
      std::cout << "done with " << path << '\n';
    }
  }
  //============================================================================
  auto write_to_disk(bool const w = true) { m_write_to_disk = w; }
  //----------------------------------------------------------------------------
  auto num_steps() const { return m_steps.size(); }
  //============================================================================
  auto step(std::size_t const i) const -> auto const & {
    if (m_write_to_disk && m_steps[i] == nullptr) {
      auto lock = std::lock_guard{m_deletion_mutex};
      if (m_steps[i] == nullptr) {
        for (auto &step : m_steps) {
          step.reset();
        }
      }
      m_steps[i]->read(m_filepaths_to_steps[i]);
    }
    return *m_steps[i];
  }
  //----------------------------------------------------------------------------
  auto step(std::size_t const i) -> auto & {
    if (m_write_to_disk && m_steps[i] == nullptr) {
      auto lock = std::lock_guard{m_deletion_mutex};
      if (m_steps[i] == nullptr) {
        for (auto &step : m_steps) {
          if (step != nullptr) {
            step.reset();
          }
        }
      }
      m_steps[i] = std::make_unique<internal_flowmap_discretization_type>(
          m_filepaths_to_steps[i]);
    }
    return *m_steps[i];
  }
  //============================================================================
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t_end - t0)
  auto sample(pos_type x, forward_tag const tag) const {
    for (std::size_t i = 0; i < num_steps(); ++i) {
      x = step(i).sample(x, tag);
    }
    return x;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in backward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t_end, t0 - t_end)
  auto sample(pos_type x, backward_tag const tag) const {
    for (std::int64_t i = static_cast<std::int64_t>(num_steps() - 1); i >= 0;
         --i) {
      x = step(i).sample(x, tag);
    }
    return x;
  }
};
//==============================================================================
} // namespace tatooine
//==============================================================================
#endif
