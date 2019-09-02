#ifndef __TATOOINE_MEMORY_USAGE_H__
#define __TATOOINE_MEMORY_USAGE_H__

#include <unistd.h>
#include <optional>
#include <fstream>
#include <string>

//==============================================================================
namespace tatooine {
//==============================================================================

/// Total amount of RAM in kB
inline size_t total_memory() {
  std::string   token;
  std::ifstream file{"/proc/meminfo"};
  while (file >> token) {
    if (token == "MemTotal:") {
      unsigned long mem;
      if (file >> mem) {
        return mem;
      } else {
        throw std::runtime_error{"could not get total RAM"};
      }
    }
    // ignore rest of the line
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  throw std::runtime_error{"could not get total RAM"};
}

//! Attempts to read the system-dependent data for a process' virtual memory
//! size and resident set size, and return the results in Byte.
inline auto memory_usage() {
  // 'file' stat seems to give the most reliable results
  std::ifstream stat_stream{"/proc/self/stat", std::ios_base::in};

  // dummy vars for leading entries in stat that we don't care about
  std::string pid, comm, state, ppid, pgrp, session, tty_nr, tpgid, flags,
      minflt, cminflt, majflt, cmajflt, utime, stime, cutime, cstime, priority,
      nice, O, itrealvalue, starttime;

  // the two fields we want
  unsigned long vsize;
  long          rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
      starttime >> vsize >> rss;
  // don't care about the rest

  stat_stream.close();

  // in case x86-64 is configured to use 2MB pages
  auto page_size_b = sysconf(_SC_PAGE_SIZE);
  return std::pair{vsize, rss * page_size_b};
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
