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
   void add_stylesheet(const std::string& css) { m_stylesheets.push_back(css); }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   void add_table(const std::vector<std::vector<std::string>>& table_data,
                  bool top_header = true, bool equal_width = true) {
     use_bootstrap();
     std::stringstream table;
     table << "<table ";
     if (equal_width) { table << "table-layout=\"fixed\" "; }
     table << "class=\"table\">\n";
     if (top_header) {
       table << "<tr>";
       for (const auto& th : table_data.front()) {
         table << "<th>" << th << "</th>";
       }
       table << "</tr>\n";
     }
     for (size_t i = (top_header ? 1 : 0); i < table_data.size(); ++i) {
       table << "<tr>";
       for (const auto& td : table_data[i]) {
         table << "<td>" << td << "</td>";
       }
       table << "</tr>\n";
     }
     table << "</table>\n";
     m_contents.push_back(table.str());
   }
   //├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   template <typename Real, typename Label = std::string>
   void add_chart(const std::vector<Real>&  data,
       const std::string& name,
                  const std::string&        color  = "#FF0000",
                  const std::vector<Label>& labels = std::vector<Label>{}) {
     use_chart_js();
     std::stringstream chart;
     chart << "<canvas id=\"chart" << m_contents.size()
           << "\" width=100%></canvas>\n"
           << "<script>\n"
           << "new Chart(document.getElementById(\"chart" << m_contents.size()
           << "\"), {\n"
           << "type: 'line',\n"
           << "data: {\n";
     if (!labels.empty()) {
       chart << "  labels: [\"" << labels.front() << "\"";
       for (size_t i = 1; i < labels.size(); ++i) {
         chart << ", \"" << labels[i] << "\"";
       }
       chart << "],\n";
     }
     chart << "  datasets: [{ \n"
           << "    data: [" << data.front();
     for (size_t i = 1; i < data.size(); ++i) { chart << ", " << data[i]; }
     chart << "],\n"
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
     m_contents.push_back(chart.str());
   }
}; // html
//╘══════════════════════════════════════════════════════════════════════════╛
}
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
