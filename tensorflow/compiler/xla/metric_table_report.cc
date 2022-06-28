#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/metric_table_report.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

void MetricTableReport::AddEntry(Entry entry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::AddEntry");

  entries_.push_back(std::move(entry));
}

void MetricTableReport::SetMetricName(std::string metric_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("metric_name: \"" + metric_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::SetMetricName");

  metric_name_ = std::move(metric_name);
}

void MetricTableReport::SetEntryName(std::string entry_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("entry_name: \"" + entry_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_2(mht_2_v, 215, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::SetEntryName");

  entry_name_ = std::move(entry_name);
}

void MetricTableReport::SetShowAllEntries() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_3(mht_3_v, 222, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::SetShowAllEntries");

  max_entries_to_show_ = std::numeric_limits<int64_t>::max();
  max_entries_per_category_to_show_ = std::numeric_limits<int64_t>::max();
  max_metric_proportion_to_show_ = 1.1;  // more than 100%
}

void MetricTableReport::SetShowCategoryTable() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_4(mht_4_v, 231, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::SetShowCategoryTable");
 show_category_table_ = true; }

void MetricTableReport::SetShowEntryTable() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_5(mht_5_v, 236, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::SetShowEntryTable");
 show_entry_table_ = true; }

std::string MetricTableReport::MakeReport(double expected_metric_sum) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_6(mht_6_v, 241, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::MakeReport");

  expected_metric_sum_ = expected_metric_sum;
  report_.clear();

  // Sort the entries.
  const auto metric_greater = [](const Entry& a, const Entry& b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_7(mht_7_v, 249, "", "./tensorflow/compiler/xla/metric_table_report.cc", "lambda");

    return a.metric > b.metric;
  };
  absl::c_sort(entries_, metric_greater);

  // Create the report
  AppendLine();
  AppendHeader();

  if (show_category_table_) {
    AppendLine();
    AppendCategoryTable();
  }
  if (show_entry_table_) {
    AppendLine();
    AppendEntryTable();
  }
  AppendLine();

  return std::move(report_);
}

void MetricTableReport::WriteReportToInfoLog(double expected_metric_sum) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_8(mht_8_v, 274, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::WriteReportToInfoLog");

  // Write something to the log normally to get the date-time and file prefix.
  LOG(INFO) << "Writing report to log.";

  int64_t pos = 0;
  const std::string report = MakeReport(expected_metric_sum);
  const int report_size = report.size();
  while (pos < report_size) {
    int64_t end_of_line = report.find('\n', pos);
    const int64_t _npos = std::string::npos;
    if (end_of_line == _npos) {
      end_of_line = report.size();
    }
    absl::string_view line(report.data() + pos, end_of_line - pos);

    // TODO(b/34779244): Figure out how to do this without the verbose log-line
    // prefix. The usual way didn't compile on open source.
    LOG(INFO) << line;

    pos = end_of_line + 1;
  }
}

std::vector<MetricTableReport::Category> MetricTableReport::MakeCategories(
    const std::vector<Entry>* entries) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_9(mht_9_v, 301, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::MakeCategories");

  // Create the categories using a category_text -> category map.
  absl::flat_hash_map<std::string, Category> category_map;
  for (const Entry& entry : *entries) {
    Category& category = category_map[entry.category_text];
    category.metric_sum += entry.metric;
    category.entries.push_back(&entry);
  }

  // Move the categories to a vector.
  std::vector<Category> categories;
  categories.reserve(category_map.size());
  for (auto& key_value_pair : category_map) {
    categories.push_back(std::move(key_value_pair.second));
    categories.back().category_text = key_value_pair.first;
  }

  // Sort the categories.
  auto metric_sum_greater = [](const Category& a, const Category& b) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_10(mht_10_v, 322, "", "./tensorflow/compiler/xla/metric_table_report.cc", "lambda");

    return a.metric_sum > b.metric_sum;
  };
  absl::c_sort(categories, metric_sum_greater);

  return categories;
}

void MetricTableReport::AppendHeader() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_11(mht_11_v, 333, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::AppendHeader");

  AppendLine("********** ", metric_name_, " report **********");
  AppendLine("There are ", MetricString(expected_metric_sum_), " ",
             metric_name_, " in total.");
  AppendLine("There are ", MetricString(UnaccountedMetric()), " ", metric_name_,
             " (", MetricPercent(UnaccountedMetric()),
             ") not accounted for by the data.");
  AppendLine("There are ", entries_.size(), " ", entry_name_, ".");
}

void MetricTableReport::AppendCategoryTable() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_12(mht_12_v, 346, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::AppendCategoryTable");

  const std::vector<Category> categories = MakeCategories(&entries_);

  AppendLine("********** categories table for ", metric_name_, " **********");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64_t categories_shown = 0;
  for (const auto& category : categories) {
    if (categories_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++categories_shown;
    metric_sum += category.metric_sum;

    // Show the category.
    std::string text = category.category_text;
    if (text.empty()) {
      text = "[no category]";
    }
    absl::StrAppend(&text, " (", category.entries.size(), " ", entry_name_,
                    ")");
    AppendTableRow(text, category.metric_sum, metric_sum);

    // Show the top entries in the category.
    const char* const kIndentPrefix = "                              * ";
    int64_t entries_to_show = std::min<int64_t>(
        max_entries_per_category_to_show_, category.entries.size());
    const int64_t category_entries_size = category.entries.size();
    if (category_entries_size == entries_to_show + 1) {
      // May as well show the last entry on the line that would otherwise say
      // that there is a single entry not shown.
      ++entries_to_show;
    }
    for (int64_t i = 0; i < entries_to_show; ++i) {
      AppendLine(kIndentPrefix, MetricPercent(category.entries[i]->metric), " ",
                 category.entries[i]->short_text);
    }
    const int64_t remaining_entries = category.entries.size() - entries_to_show;
    if (remaining_entries > 0) {
      AppendLine(kIndentPrefix, "... (", remaining_entries, " more ",
                 entry_name_, ")");
    }
  }
  const int64_t remaining_categories = categories.size() - categories_shown;
  if (remaining_categories > 0) {
    AppendTableRow(
        absl::StrCat("... (", remaining_categories, " more categories)"),
        expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendEntryTable() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_13(mht_13_v, 402, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::AppendEntryTable");

  AppendLine("********** ", entry_name_, " table for ", metric_name_,
             " **********");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64_t entries_shown = 0;
  for (const auto& entry : entries_) {
    if (entries_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++entries_shown;
    metric_sum += entry.metric;

    std::string text = entry.text;
    if (text.empty()) {
      text = "[no entry text]";
    }
    AppendTableRow(text, entry.metric, metric_sum);
  }
  const int64_t remaining_entries = entries_.size() - entries_shown;
  if (remaining_entries > 0) {
    AppendTableRow(
        absl::StrCat("... (", remaining_entries, " more ", entry_name_, ")"),
        expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendTableRow(const std::string& text,
                                       const double metric,
                                       const double running_metric_sum) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_14(mht_14_v, 437, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::AppendTableRow");

  // This is the widest metric number possible, assuming non-negative metrics,
  // so align to that width.
  const int64_t max_metric_string_size =
      MetricString(expected_metric_sum_).size();
  std::string metric_string = MetricString(metric);

  // Don't try to make a gigantic string and crash if expected_metric_sum_ is
  // wrong somehow.
  int64_t padding_len = 1;
  const int64_t metric_string_size = metric_string.size();
  if (max_metric_string_size >= metric_string_size) {
    padding_len += max_metric_string_size - metric_string.size();
  }
  std::string padding(padding_len, ' ');
  AppendLine(padding, metric_string, " (", MetricPercent(metric), " Σ",
             MetricPercent(running_metric_sum), ")   ", text);
}

double MetricTableReport::UnaccountedMetric() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_15(mht_15_v, 459, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::UnaccountedMetric");

  double metric_sum = 0.0;
  for (const auto& entry : entries_) {
    metric_sum += entry.metric;
  }
  return expected_metric_sum_ - metric_sum;
}

std::string MetricTableReport::MetricString(double metric) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_16(mht_16_v, 470, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::MetricString");

  // Round to integer and stringify.
  std::string s1 = absl::StrCat(std::llround(metric));

  // Code below commafies the string, e.g. "1234" becomes "1,234".
  absl::string_view sp1(s1);
  std::string output;
  // Copy leading non-digit characters unconditionally.
  // This picks up the leading sign.
  while (!sp1.empty() && !absl::ascii_isdigit(sp1[0])) {
    output.push_back(sp1[0]);
    sp1.remove_prefix(1);
  }
  // Copy rest of input characters.
  for (int64_t i = 0, end = sp1.size(); i < end; ++i) {
    if (i > 0 && (sp1.size() - i) % 3 == 0) {
      output.push_back(',');
    }
    output.push_back(sp1[i]);
  }
  return output;
}

std::string MetricTableReport::MetricPercent(double metric) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSmetric_table_reportDTcc mht_17(mht_17_v, 496, "", "./tensorflow/compiler/xla/metric_table_report.cc", "MetricTableReport::MetricPercent");

  return absl::StrFormat("%5.2f%%", metric / expected_metric_sum_ * 100.0);
}

}  // namespace xla
