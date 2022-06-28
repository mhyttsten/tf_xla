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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/compilation_stats.h"

#include <iostream>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class NoopStats : public CompilationStats {
 public:
  NoopStats() = default;

  void StartPass(absl::string_view pass_name) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "StartPass");
}

  void EndPass(absl::string_view pass_name) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "EndPass");
}

  void CompilationReport() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_2(mht_2_v, 215, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "CompilationReport");
}

  int GetPassesSize() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_3(mht_3_v, 220, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "GetPassesSize");
 return 0; }
};

class Stats : public CompilationStats {
 public:
  Stats() = default;

  void StartPass(absl::string_view pass_name) override;

  void EndPass(absl::string_view pass_name) override;

  void CompilationReport() override;

  int GetPassesSize() override;

 private:
  struct PassInfo {
    PassInfo(absl::string_view name, double duration)
        : name(name), duration_ms(duration) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "PassInfo");
}

    std::string name;
    int num_runs = 1;
    double duration_ms;
  };

  // Info about the passes that have been run so far.
  std::vector<PassInfo> passes_;
  // Used to avoid nested calls to StartPass.
  bool pass_running_ = false;
  std::string current_pass_;
  // The start time of the currently running pass.
  uint64_t start_micros_;
};

/* static */
std::unique_ptr<CompilationStats> CompilationStats::MakeNoopStats() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "CompilationStats::MakeNoopStats");

  return absl::make_unique<NoopStats>();
}

/* static */
std::unique_ptr<CompilationStats> CompilationStats::MakeStats() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_6(mht_6_v, 270, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "CompilationStats::MakeStats");

  return absl::make_unique<Stats>();
}

void Stats::StartPass(absl::string_view pass_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_7(mht_7_v, 278, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "Stats::StartPass");

  CHECK(!pass_running_) << "Can't start " << pass_name << " while running "
                        << current_pass_;
  pass_running_ = true;
  current_pass_ = std::string(pass_name);
  start_micros_ = tensorflow::Env::Default()->NowMicros();
}

void Stats::EndPass(absl::string_view pass_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_8(mht_8_v, 290, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "Stats::EndPass");

  CHECK(pass_running_);
  CHECK_EQ(current_pass_, std::string(pass_name));
  pass_running_ = false;
  uint64_t end_micros = tensorflow::Env::Default()->NowMicros();
  double duration_ms = (end_micros - start_micros_) / 1000.0;
  passes_.push_back(PassInfo(current_pass_, duration_ms));
}

void Stats::CompilationReport() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_9(mht_9_v, 302, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "Stats::CompilationReport");

  CHECK(!pass_running_) << "EndPass never called for " << current_pass_;
  absl::flat_hash_map<std::string, PassInfo> summary;
  double total_duration = 0;

  for (auto& pass_run : passes_) {
    auto pass_name = pass_run.name;
    total_duration += pass_run.duration_ms;
    auto it = summary.find(pass_name);
    if (it == summary.end()) {
      summary.insert(std::make_pair(pass_name, pass_run));
    } else {
      ++summary.at(pass_name).num_runs;
      summary.at(pass_name).duration_ms += pass_run.duration_ms;
    }
  }

  std::vector<PassInfo> sorted_summary;
  sorted_summary.reserve(summary.size());
  for (auto& it : summary) {
    sorted_summary.push_back(it.second);
  }
  absl::c_sort(sorted_summary, [](const PassInfo& a, const PassInfo& b) {
    // Sort passes that take the longest first, break ties using pass names.
    return std::make_pair(b.duration_ms, a.name) <
           std::make_pair(a.duration_ms, b.name);
  });
  LOG(INFO) << "Total runtime (ms) of HLO passes: " << total_duration;
  LOG(INFO) << "Pass name, num runs, time (ms)";
  for (auto& pass_info : sorted_summary) {
    LOG(INFO) << pass_info.name << ", " << pass_info.num_runs << ", "
              << pass_info.duration_ms;
  }
}

int Stats::GetPassesSize() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilation_statsDTcc mht_10(mht_10_v, 340, "", "./tensorflow/compiler/xla/service/compilation_stats.cc", "Stats::GetPassesSize");
 return passes_.size(); }

}  // namespace xla
