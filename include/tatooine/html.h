#ifndef TATOOINE_HTML_H
#define TATOOINE_HTML_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {

//╒══════════════════════════════════════════════════════════════════════════╕
struct html {
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
   explicit html(std::string path) : m_path{std::move(path)} { wrap_width(); }
   html(const html&)     = default;
   html(html&&) noexcept = default;
  //┌──────────────────────────────────────────────────────────────────────┐
  //│ assignment operators                                                 │
  //├──────────────────────────────────────────────────────────────────────┤
   auto operator=(const html&) -> html& = default;
   auto operator=(html&&) noexcept -> html& = default;

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
   void use_chart_js(bool use = true) { m_use_bootstrap = use; }
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
   void add_table(const std::vector<std::vector<std::string>>& table_data,
                  bool top_header = true, bool equal_width = true) {
     use_bootstrap();
     std::stringstream stream;
     table(stream, table_data, top_header, equal_width);
     add_content(stream.str());
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static void table(
       std::ostream&                                stream,
       const std::vector<std::vector<std::string>>& table_data,
       bool top_header = true, bool equal_width = true) {
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
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static std::string table(
       const std::vector<std::vector<std::string>>& table_data,
       bool top_header = true, bool equal_width = true) {
     std::stringstream stream;
     table(stream, table_data, top_header, equal_width);
     return stream.str();
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename... Strings>
   void add_side_by_side(Strings&&... strings) {
     add_table(std::vector<std::vector<std::string>>{{strings...}});
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename Real, typename Label = std::string>
   void add_chart(const std::vector<Real>&  data,
       const std::string& name,
                  const std::string&        color  = "#FF0000",
                  const std::vector<Label>& labels = std::vector<Label>{}) {
     use_chart_js();
     std::stringstream stream;
     chart(stream, data, name, color, labels);
     add_content(stream.str());
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename Real, typename Label = std::string>
   static void chart(std::ostream& stream, const std::vector<Real>& data,
                     const std::string&        name,
                     const std::string&        color  = "#FF0000",
                     const std::vector<Label>& labels = std::vector<Label>{}) {
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
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename Real, typename Label = std::string>
   static std::string chart(const std::vector<Real>& data,
                     const std::string&        name,
                     const std::string&        color  = "#FF0000",
                     const std::vector<Label>& labels = std::vector<Label>{}) {
     std::stringstream stream;
     chart(data, name, color, labels);
     return stream.str();
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add_image(const std::string& path) {
     std::stringstream stream;
     image(stream, path);
     add_content(stream.str());
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static void image(std::ostream& stream, const std::string& path) {
     stream << "<img width=100% src=\"" << path << "\">\n";
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static auto image(const std::string& path) {
     std::stringstream stream;
     image(stream, path);
     return stream.str();
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add_video(const std::string& path) {
     std::stringstream stream;
     video(stream, path);
     add_content(stream.str());
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static void video(std::ostream& stream, const std::string& path) {
     stream << "<video width=\"100%\" controls><source src=\"" << path
            << "\" type=\"video/mp4\"></video>\n";
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   static auto video(const std::string& path) {
     std::stringstream stream;
     video(stream, path);
     return stream.str();
   }
}; // html
//╘══════════════════════════════════════════════════════════════════════════╛
}
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
