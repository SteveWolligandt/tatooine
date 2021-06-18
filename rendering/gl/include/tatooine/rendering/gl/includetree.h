#ifndef YAVIN_INCLUDE_TREE_H
#define YAVIN_INCLUDE_TREE_H
//==============================================================================
#include <filesystem>
#include <iostream>
#include <list>
#include <string>
//==============================================================================
namespace yavin {
//==============================================================================
struct include_tree {
 private:
  int                     m_line_number;
  size_t                  m_num_lines;
  std::filesystem::path   m_path;
  std::list<include_tree> m_nested_include_trees;
  include_tree const*     m_parent;

 public:
  auto line_number() const { return m_line_number; }
  auto line_number() -> auto& { return m_line_number; }
  auto num_lines() const { return m_num_lines; }
  auto num_lines() ->auto& { return m_num_lines; }
  auto path() const -> auto const& { return m_path; }
  auto path() -> auto& { return m_path; }
  auto nested_include_trees() const -> auto const& {
    return m_nested_include_trees;
  }
  auto nested_include_trees() -> auto& { return m_nested_include_trees; }
  auto parent() const -> auto const& { return *m_parent; }
  auto has_parent() const -> bool { return m_parent != nullptr; }

 public:
  include_tree() : m_line_number{-1}, m_num_lines{0} {}
  //----------------------------------------------------------------------------
  include_tree(size_t line_number, size_t num_lines,
               std::filesystem::path const&   path,
               std::list<include_tree> const& nested_include_trees,
               include_tree const&            parent)
      : m_line_number{static_cast<int>(line_number)},
        m_num_lines{num_lines},
        m_path{path},
        m_nested_include_trees{nested_include_trees},
        m_parent{&parent} {}
  //----------------------------------------------------------------------------
  include_tree(size_t line_number, size_t num_lines,
               std::filesystem::path const&   path,
               std::list<include_tree> const& nested_include_trees)
      : m_line_number{static_cast<int>(line_number)},
        m_num_lines{num_lines},
        m_path{path},
        m_nested_include_trees{nested_include_trees} {}
  //----------------------------------------------------------------------------
  include_tree(size_t line_number, size_t num_lines,
               std::filesystem::path const& path)
      : m_line_number{static_cast<int>(line_number)},
        m_num_lines{num_lines},
        m_path{path} {}
  //----------------------------------------------------------------------------
  include_tree(size_t line_number, size_t num_lines,
               std::filesystem::path const& path, include_tree const& parent)
      : m_line_number{static_cast<int>(line_number)},
        m_num_lines{num_lines},
        m_path{path},
        m_parent{&parent} {}
  //----------------------------------------------------------------------------
  /// returns file and line number
  auto parse_line(size_t n) const -> std::pair<include_tree const&, size_t> {
    size_t cur_offset = 0;
    for (auto const& nesting : m_nested_include_trees) {
      if (n >= nesting.m_line_number + cur_offset &&
          n < nesting.m_line_number + cur_offset +
                  nesting.num_lines_with_includes())
        return nesting.parse_line(n - cur_offset - nesting.m_line_number);
      else
        cur_offset += nesting.num_lines_with_includes() - 1;
    }
    return {*this, n - cur_offset};
  }
  //----------------------------------------------------------------------------
  auto num_lines_with_includes() const -> size_t {
    size_t n = m_num_lines;
    n -= m_nested_include_trees.size();
    for (auto const& nesting : m_nested_include_trees)
      n += nesting.num_lines_with_includes();
    return n;
  }
  //----------------------------------------------------------------------------
  auto print(size_t indent = 0) const -> void {
    for (size_t i = 0; i < indent; ++i) {
      std::cerr << ' ';
    }
    std::cerr << m_line_number << "/" << std::to_string(m_num_lines) << ": "
              << m_path << '\n';
    for (auto const& nesting : m_nested_include_trees) {
      nesting.print(indent + 2);
    }
  }
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
