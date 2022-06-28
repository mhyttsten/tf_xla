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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh() {
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


#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// HumanReadableProfileBuilder helps you create a textual profile of a
// computation, suitable for consumption by humans.
class HumanReadableProfileBuilder {
 public:
  explicit HumanReadableProfileBuilder(absl::string_view computation_name,
                                       bool is_entry_computation,
                                       int64_t total_cycles,
                                       double clock_rate_ghz)
      : computation_name_(computation_name),
        is_entry_computation_(is_entry_computation),
        total_cycles_(total_cycles),
        clock_rate_ghz_(clock_rate_ghz) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("computation_name: \"" + std::string(computation_name.data(), computation_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh mht_0(mht_0_v, 209, "", "./tensorflow/compiler/xla/service/human_readable_profile_builder.h", "HumanReadableProfileBuilder");

    CHECK_GE(clock_rate_ghz, 1e-9);
  }

  int64_t total_cycles() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/human_readable_profile_builder.h", "total_cycles");
 return total_cycles_; }

  // Adds an operation to the profile.  If you don't know the number of
  // floating-point ops or bytes touched by the op, or if you don't know how
  // fast it would run optimally, pass -1 for that param.
  void AddOp(absl::string_view op_name, absl::string_view short_name,
             absl::string_view category, int64_t cycles, int64_t flop_count,
             int64_t transcendental_count, int64_t bytes_accessed,
             float optimal_seconds) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op_name: \"" + std::string(op_name.data(), op_name.size()) + "\"");
   mht_2_v.push_back("short_name: \"" + std::string(short_name.data(), short_name.size()) + "\"");
   mht_2_v.push_back("category: \"" + std::string(category.data(), category.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh mht_2(mht_2_v, 230, "", "./tensorflow/compiler/xla/service/human_readable_profile_builder.h", "AddOp");

    op_infos_.push_back({std::string(op_name), std::string(short_name),
                         std::string(category), cycles, flop_count,
                         transcendental_count, bytes_accessed,
                         optimal_seconds});
  }

  // Gets the human-readable profile.
  std::string ToString() const;

 private:
  struct OpInfo {
    std::string name;
    std::string short_name;
    std::string category;
    int64_t cycles;
    int64_t flop_count;  // -1 if unknown
    int64_t transcendental_count;
    int64_t bytes_accessed;  // -1 if unknown
    float optimal_seconds;   // -1 if unknown
  };

  double CyclesToSeconds(int64_t cycles) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh mht_3(mht_3_v, 255, "", "./tensorflow/compiler/xla/service/human_readable_profile_builder.h", "CyclesToSeconds");

    return cycles / clock_rate_ghz_ / 1e9;
  }
  double CyclesToMicroseconds(int64_t cycles) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShuman_readable_profile_builderDTh mht_4(mht_4_v, 261, "", "./tensorflow/compiler/xla/service/human_readable_profile_builder.h", "CyclesToMicroseconds");

    return cycles / clock_rate_ghz_ / 1000.0;
  }

  std::string computation_name_;
  bool is_entry_computation_;
  int64_t total_cycles_;
  double clock_rate_ghz_;
  std::vector<OpInfo> op_infos_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HUMAN_READABLE_PROFILE_BUILDER_H_
