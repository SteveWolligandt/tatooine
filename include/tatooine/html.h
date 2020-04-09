#ifndef TATOOINE_HTML_H
#define TATOOINE_HTML_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include <fstream>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <string>
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine::html {
//╒══════════════════════════════════════════════════════════════════════════╕
struct content {
  auto to_string() const -> std::string {
    std::stringstream stream;
    to_stream(stream);
    return stream.str();
  }
  virtual auto to_stream(std::ostream&) const -> std::ostream& = 0;
};  // content
//╞══════════════════════════════════════════════════════════════════════════╡
struct text : content {
  std::string txt;
  bool centered = true;

  text(std::string txt_) : txt{std::move(txt_)} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<div class=textfield>";
    if (centered) {stream << "<center>";}
    stream << "<p>";
    stream << txt;
    stream << "</p>";
    if (centered) {stream << "</center>";}
    stream << "</div>";
    return stream;
  }
};  // text
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const text& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
template <typename Content>
auto to_string(Content&& content) -> std::string {
  return content.to_string();
}
auto to_string(const std::string& content) -> const std::string& {
  return content;
}
auto to_string(const char* content) -> std::string {
  return content;
}
template <typename Content>
auto to_content(Content&& content) -> decltype(auto) {
  return std::forward<Content>(content);
}
auto to_content(const std::string& content) {
  return text{content};
}
auto to_content(const char* content) {
  return text{content};
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct vertical_container : content {
  std::string contents;

  template <typename... Contents>
  vertical_container(Contents&&... contents_)
      : contents{((to_content(contents_).to_string() + ' ') + ...)} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    return stream << contents;
  }
};  // vertical_container
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const vertical_container& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct horizontal_container : content {
  std::string contents;

  template <typename... Contents>
  horizontal_container(Contents&&... contents_)
      : contents{"<table table-layout=\"fixed\"><tr>\n" +
                 (("<td>" + to_content(contents_).to_string() + "</td>") + ...) +
                 "</tr></table>\n"} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    return stream << contents;
  }
};  // horizontal_container
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const horizontal_container& cnt)
    -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct image : content {
  std::string path;

  image(std::string path_) : path{std::move(path_)} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<img width=100% src=\"" << path << "\">\n";
    return stream;
  }
};  // image
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const image& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct video : content {
  std::string path;

  video(std::string path_) : path{std::move(path_)} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<video width=\"100%\" controls><source src=\"" << path
           << "\" type=\"video/mp4\"></video>\n";
    return stream;
  }
};  // video
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const video& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct table : content {
  using table_data_t = std::vector<std::vector<std::string>>;

  table_data_t table_data;
  bool top_header;
  bool equal_width;

  table(const table_data_t& table_data_, bool top_header_ = true,
        bool equal_width_ = true)
      : table_data{table_data_},
        top_header{top_header_},
        equal_width{equal_width_} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<table ";
    if (equal_width) { stream << "table-layout=\"fixed\" "; }
    stream << "class=\"table\">\n";
    if (top_header) {
      stream << "<tr>";
      for (const auto& th : table_data.front()) {
        stream << "<th>" << th << "</th>";
      }
      stream << "</tr>\n";
    }
    for (size_t i = (top_header ? 1 : 0); i < table_data.size(); ++i) {
      stream << "<tr>";
      for (const auto& td : table_data[i]) {
        stream << "<td>" << td << "</td>";
      }
      stream << "</tr>\n";
    }
    stream << "</table>\n";
    return stream;
  }
};  // table
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const table& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
struct slider : content {
  std::vector<std::string> contents;

  slider(std::vector<std::string> contents_) : contents{std::move(contents_)} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<div id=\"carouselExampleIndicators\" class=\"carousel\">\n"
           << "  <ol class=\"carousel-indicators\">\n"
           << "    <li data-target=\"#carouselExampleIndicators\" "
              "data-slide-to=\"0\" class=\"active\"></li>\n";

    for (size_t i = 1; i < contents.size(); ++i) {
      stream << "<li data-target=\"#carouselExampleIndicators\" "
                "data-slide-to=\""
             << i << "\"></li>\n";
    }

    stream << "</ol>\n"
           << "<div class=\"carousel-inner\">\n"
           << "<div class=\"carousel-item active\">\n"
           << contents[0] << "\n</div>\n";
    for (size_t i = 1; i < contents.size(); ++i) {
      stream << "<div class=\"carousel-item\">\n"
             << contents[i] << "\n</div>\n";
    }

    stream << "</div>\n"
           << "<a class=\"carousel-control-prev\" "
              "href=\"#carouselExampleIndicators\" role=\"button\" "
              "data-slide=\"prev\">\n"
           << "<span class=\"carousel-control-prev-icon\" "
              "aria-hidden=\"true\"></span>\n"
           << "<span class=\"sr-only\">Previous</span>\n"
           << "</a>\n"
           << "<a class=\"carousel-control-next\" "
              "href=\"#carouselExampleIndicators\" role=\"button\" "
              "data-slide=\"next\">\n"
           << "<span class=\"carousel-control-next-icon\" "
              "aria-hidden=\"true\"></span>\n"
           << "<span class=\"sr-only\">Next</span>\n"
           << "</a>\n"
           << "</div>\n";

    return stream;
  }

};  // slider
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const slider& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct chart : content {
  std::vector<double>      data;
  std::string              name;
  std::string              color;
  std::vector<std::string> labels;

  chart(const std::vector<double>& data_, const std::string& name_,
        const std::string&              color_  = "#FF0000",
        const std::vector<std::string>& labels_ = std::vector<std::string>{})
      : data{data_}, name{name_}, color{color_}, labels{labels_} {}

  auto to_stream(std::ostream& stream) const -> std::ostream& override {
    stream << "<canvas id=\"chart" << name << "\" width=100%></canvas>\n"
           << "<script>\n"
           << "new Chart(document.getElementById(\"chart" << name << "\"), {\n"
           << "type: 'line',\n"
           << "data: {\n";
    if (!labels.empty()) {
      stream << "  labels: [\"" << labels.front() << "\"";
      for (size_t i = 1; i < labels.size(); ++i) {
        stream << ", \"" << labels[i] << "\"";
      }
      stream << "],\n";
    }
    stream << "  datasets: [{ \n"
           << "    data: [" << data.front();
    for (size_t i = 1; i < data.size(); ++i) { stream << ", " << data[i]; }
    stream << "],\n"
           << "    label: \"" << name << "\",\n"
           << "    borderColor: \"" << color << "\",\n"
           << "    fill: false}]},\n"
           << "    options: {\n"
           << "    title: {\n"
           << "      display: true,\n"
           << "      text: 'coverage'\n"
           << "    }\n"
           << "}\n"
           << "});</script>\n";
    return stream;
  }
};  // chart
//├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
auto operator<<(std::ostream& stream, const chart& cnt) -> std::ostream& {
  return cnt.to_stream(stream);
}
//╞══════════════════════════════════════════════════════════════════════════╡
struct doc {
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ members                                                              │
  //├──────────────────────────────────────────────────────────────────────┤
  private:
    std::string m_path;
    std::string m_title;
    bool m_wrap_width;
    bool m_use_bootstrap;
    bool m_use_chart_js;
    std::vector<std::string> m_stylesheets;
    std::vector<std::string> m_contents;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ ctors                                                                │
  //├──────────────────────────────────────────────────────────────────────┤
  public:
   explicit doc(std::string path) : m_path{std::move(path)} { wrap_width(); }
   doc(const doc&)     = default;
   doc(doc&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
   auto operator=(const doc&) -> doc& = default;
   auto operator=(doc&&) noexcept -> doc& = default;

   //┌─────────────────────────────────────────────────────────────────────┐
   //│ methods                                                             │
   //├─────────────────────────────────────────────────────────────────────┤
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void set_title(std::string title) { m_title = std::move(title); }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void wrap_width(bool wrap = true) { m_wrap_width = wrap; }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void use_bootstrap(bool use = true) { m_use_bootstrap = use; }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void use_chart_js(bool use = true) { m_use_chart_js = use; }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void write() const {
     std::ofstream file{m_path};
     if (file.is_open()) {
       write(file);
     }
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write(std::ostream& stream) const -> std::ostream& {
     stream << "<!DOCTYPE html>\n"
            << "<html>\n";
     write_head(stream);
     write_body(stream);
     stream << "</html>";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_head(std::ostream& stream) const ->std::ostream&{
     stream << "<head>\n";
     if (!m_title.empty()) { write_title(stream); }
     if (m_use_bootstrap) { write_bootstrap_head(stream); }
     if (m_wrap_width) { write_wrap_css(stream); }
     stream << "<link "
            << "href=\"https://fonts.googleapis.com/"
            << "css?family=Butterfly+Kids|Roboto\" rel=\"stylesheet\">";
     for (const auto& css : m_stylesheets) { stream << css << '\n'; }
     stream << "</head>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_body(std::ostream& stream) const ->std::ostream&{
     stream << "<body>\n";
     if (m_wrap_width) {
       stream << "<div id=\"outerwrap\"><div id=\"innerwrap\">\n";
     }
     if (m_use_bootstrap) { write_bootstrap_body(stream); }
     if (m_use_chart_js) { write_chart_js_body(stream); }
     stream << '\n';
     stream << m_contents.front() << '\n';
     for (size_t i = 1; i < m_contents.size(); ++i) {
       stream << "<hr>\n\n";
       stream << m_contents[i] << '\n';
     }
     if (m_wrap_width) { stream << "</div></div>\n"; }
     stream << "</body>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_wrap_css(std::ostream& stream) const -> std::ostream&{
     stream << "<style type=\"text/css\" title=\"text/css\">\n"
            << "#outerwrap {\n"
            << "  margin:auto;\n"
            << "  width:800px;\n"
            << "  border:1px solid #CCCCCC;\n"
            << "  padding 10px;\n"
            << "}\n"
            << "#innerwrap {\n"
            << "  margin:10px;\n"
            << "}\n"
            << "hr {\n"
            << "  border-top: 1px dashed #CCCCCC;\n"
            << "}\n"
            << ".textfield {\n"
            << "  font-family: \'Roboto\';\n"
            << "  font-size: 1cm;\n"
            << "}\n"
            << "</style>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_title(std::ostream& stream) const -> std::ostream& {
     stream << "<title>" << m_title << "</title>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_bootstrap_head(std::ostream& stream) const ->std::ostream& {
     stream << "<meta charset=\"utf-8\">\n";
     stream << "<meta name=\"viewport\" content=\"width=device-width, "
               "initial-scale=1, shrink-to-fit=no\">\n";
     stream << "<link rel=\"stylesheet\" "
               "href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/"
               "css/bootstrap.min.css\" "
               "integrity=\"sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/"
               "iJTQUOhcWr7x9JvoRxT2MZw1T\" crossorigin=\"anonymous\">\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_bootstrap_body(std::ostream& stream) const -> std::ostream& {
     stream
         << "<script src=\"https://code.jquery.com/jquery-3.3.1.slim.min.js\" "
            "integrity=\"sha384-q8i/"
            "X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo\" "
            "crossorigin=\"anonymous\"></script>\n"
         << "<script "
            "src=\"https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/"
            "popper.min.js\" "
            "integrity=\"sha384-"
            "UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1\""
            " "
            "crossorigin=\"anonymous\"></script>\n"
         << "<script "
            "src=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/"
            "bootstrap.min.js\" "
            "integrity=\"sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/"
            "nJGzIxFDsf4x0xIM+B07jRM\" crossorigin=\"anonymous\"></script>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   auto write_chart_js_body(std::ostream& stream) const -> std::ostream& {
     stream
         << "<script "
            "src=\"https://cdn.jsdelivr.net/npm/chart.js@2.8.0\"></script>\n";
     return stream;
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add_content(const std::string& content) {
     m_contents.push_back(content);
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add_stylesheet(const std::string& css) { m_stylesheets.push_back(css); }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add(const std::string& cnt) {
     text t{cnt};
     add_content(tatooine::html::to_string(t));
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add(const content& cnt) {
     add_content(tatooine::html::to_string(cnt));
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add(const table& cnt) {
     use_bootstrap();
     add_content(tatooine::html::to_string(cnt));
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add(const chart& cnt) {
     use_chart_js();
     add_content(tatooine::html::to_string(cnt));
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename... Contents,
             std::enable_if_t<(sizeof...(Contents) > 1), bool> = true>
   void add(Contents&&... contents) {
     use_bootstrap();
     use_chart_js();
     add(horizontal_container{std::forward<Contents>(contents)...});
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename... Contents,
             std::enable_if_t<(sizeof...(Contents) > 1), bool> = true>
   void add_horizontal(Contents&&... contents) {
     use_bootstrap();
     use_chart_js();
     add(horizontal_container{std::forward<Contents>(contents)...});
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename... Contents,
             std::enable_if_t<(sizeof...(Contents) > 1), bool> = true>
   void add_vertical(Contents&&... contents) {
     use_bootstrap();
     use_chart_js();
     add(vertical_container{std::forward<Contents>(contents)...});
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename... Contents,
             std::enable_if_t<(sizeof...(Contents) > 1), bool> = true>
   void add_slider(Contents&&... contents) {
     use_bootstrap();
     use_chart_js();
     add(slider{std::vector{tatooine::html::to_string(contents)...}});
   }
}; // doc
//╘══════════════════════════════════════════════════════════════════════════╛
}  // namespace tatooine::html
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
