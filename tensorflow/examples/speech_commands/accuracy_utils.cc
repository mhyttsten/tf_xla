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
class MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc() {
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

#include "tensorflow/examples/speech_commands/accuracy_utils.h"

#include <fstream>
#include <iomanip>
#include <unordered_set>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

Status ReadGroundTruthFile(const string& file_name,
                           std::vector<std::pair<string, int64_t>>* result) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + file_name + "\"");
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/examples/speech_commands/accuracy_utils.cc", "ReadGroundTruthFile");

  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Ground truth file '", file_name,
                                        "' not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    std::vector<string> pieces = tensorflow::str_util::Split(line, ',');
    if (pieces.size() != 2) {
      continue;
    }
    float timestamp;
    if (!tensorflow::strings::safe_strtof(pieces[1].c_str(), &timestamp)) {
      return tensorflow::errors::InvalidArgument(
          "Wrong number format at line: ", line);
    }
    string label = pieces[0];
    auto timestamp_int64 = static_cast<int64_t>(timestamp);
    result->push_back({label, timestamp_int64});
  }
  std::sort(result->begin(), result->end(),
            [](const std::pair<string, int64>& left,
               const std::pair<string, int64>& right) {
              return left.second < right.second;
            });
  return Status::OK();
}

void CalculateAccuracyStats(
    const std::vector<std::pair<string, int64_t>>& ground_truth_list,
    const std::vector<std::pair<string, int64_t>>& found_words,
    int64_t up_to_time_ms, int64_t time_tolerance_ms,
    StreamingAccuracyStats* stats) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc mht_1(mht_1_v, 235, "", "./tensorflow/examples/speech_commands/accuracy_utils.cc", "CalculateAccuracyStats");

  int64_t latest_possible_time;
  if (up_to_time_ms == -1) {
    latest_possible_time = std::numeric_limits<int64_t>::max();
  } else {
    latest_possible_time = up_to_time_ms + time_tolerance_ms;
  }
  stats->how_many_ground_truth_words = 0;
  for (const std::pair<string, int64_t>& ground_truth : ground_truth_list) {
    const int64_t ground_truth_time = ground_truth.second;
    if (ground_truth_time > latest_possible_time) {
      break;
    }
    ++stats->how_many_ground_truth_words;
  }

  stats->how_many_false_positives = 0;
  stats->how_many_correct_words = 0;
  stats->how_many_wrong_words = 0;
  std::unordered_set<int64_t> has_ground_truth_been_matched;
  for (const std::pair<string, int64_t>& found_word : found_words) {
    const string& found_label = found_word.first;
    const int64_t found_time = found_word.second;
    const int64_t earliest_time = found_time - time_tolerance_ms;
    const int64_t latest_time = found_time + time_tolerance_ms;
    bool has_match_been_found = false;
    for (const std::pair<string, int64_t>& ground_truth : ground_truth_list) {
      const int64_t ground_truth_time = ground_truth.second;
      if ((ground_truth_time > latest_time) ||
          (ground_truth_time > latest_possible_time)) {
        break;
      }
      if (ground_truth_time < earliest_time) {
        continue;
      }
      const string& ground_truth_label = ground_truth.first;
      if ((ground_truth_label == found_label) &&
          (has_ground_truth_been_matched.count(ground_truth_time) == 0)) {
        ++stats->how_many_correct_words;
      } else {
        ++stats->how_many_wrong_words;
      }
      has_ground_truth_been_matched.insert(ground_truth_time);
      has_match_been_found = true;
      break;
    }
    if (!has_match_been_found) {
      ++stats->how_many_false_positives;
    }
  }
  stats->how_many_ground_truth_matched = has_ground_truth_been_matched.size();
}

void PrintAccuracyStats(const StreamingAccuracyStats& stats) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSexamplesPSspeech_commandsPSaccuracy_utilsDTcc mht_2(mht_2_v, 291, "", "./tensorflow/examples/speech_commands/accuracy_utils.cc", "PrintAccuracyStats");

  if (stats.how_many_ground_truth_words == 0) {
    LOG(INFO) << "No ground truth yet, " << stats.how_many_false_positives
              << " false positives";
  } else {
    float any_match_percentage =
        (stats.how_many_ground_truth_matched * 100.0f) /
        stats.how_many_ground_truth_words;
    float correct_match_percentage = (stats.how_many_correct_words * 100.0f) /
                                     stats.how_many_ground_truth_words;
    float wrong_match_percentage = (stats.how_many_wrong_words * 100.0f) /
                                   stats.how_many_ground_truth_words;
    float false_positive_percentage =
        (stats.how_many_false_positives * 100.0f) /
        stats.how_many_ground_truth_words;

    LOG(INFO) << std::setprecision(1) << std::fixed << any_match_percentage
              << "% matched, " << correct_match_percentage << "% correctly, "
              << wrong_match_percentage << "% wrongly, "
              << false_positive_percentage << "% false positives ";
  }
}

}  // namespace tensorflow
