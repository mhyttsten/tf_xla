/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
#define TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh() {
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


#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/match.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/stack_frame.h"

namespace tensorflow {

// Returns "true" on filenames which should be skipped.
using StackTraceFilter = std::function<bool(const char*)>;

using SourceLoc = std::pair<std::string, int>;

// Using absl::Hash breaks NVCC under Windows :P
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    std::size_t h1 = std::hash<T1>()(pair.first);
    std::size_t h2 = std::hash<T2>()(pair.second);
    return h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
  }
};

// Maps filename/line_no combination into a stack frame.
using SourceMap = std::unordered_map<SourceLoc, StackFrame, PairHash>;

using ToStackFramesFunctor = std::vector<StackFrame>(int, const SourceMap&,
                                                     const StackTraceFilter&,
                                                     bool, int);

// Returns whether the given frame is internal to TF.
inline bool IsInternalFrameForFilename(absl::string_view file_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_name: \"" + std::string(file_name.data(), file_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh mht_0(mht_0_v, 223, "", "./tensorflow/core/util/managed_stack_trace.h", "IsInternalFrameForFilename");

  // Use a simple heuristic for now.
  // TODO(cheshire): Build a more sophisticated mechanism, rely on @tf.export.
  return (absl::StrContains(file_name, "tensorflow/python") ||
          absl::StrContains(file_name, "tensorflow\\python")) &&
         !absl::StrContains(file_name, "keras") &&
         !absl::StrContains(file_name, "test.py");
}

// Language agnostic stack trace class. It only saves an id, and language
// clients are responsible for managing the actual stack trace objects.
class ManagedStackTrace {
 public:
  ManagedStackTrace(int id, ToStackFramesFunctor* to_stack_frames)
      : id_(id), to_stack_frames_(to_stack_frames) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/util/managed_stack_trace.h", "ManagedStackTrace");
}

  // Returns stack trace as a vector of `StackFrame`s.
  std::vector<StackFrame> ToStackFrames(const SourceMap& source_map,
                                        const StackTraceFilter& filtered,
                                        bool reverse_traversal = false,
                                        int limit = -1) const {
    return to_stack_frames_(id_, source_map, filtered, reverse_traversal,
                            limit);
  }

 private:
  int id_;
  ToStackFramesFunctor* to_stack_frames_;
};

// Generates a message with a definition location based on a provided stack
// trace, or an empty one if the stack trace is empty.
inline std::string DefinitionLocationMsg(
    const absl::optional<ManagedStackTrace>& stack_trace) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSmanaged_stack_traceDTh mht_2(mht_2_v, 262, "", "./tensorflow/core/util/managed_stack_trace.h", "DefinitionLocationMsg");

  if (stack_trace.has_value()) {
    std::vector<StackFrame> stack_frames =
        stack_trace->ToStackFrames({}, IsInternalFrameForFilename,
                                   /*reverse_traversal=*/true,
                                   /*limit=*/1);
    if (!stack_frames.empty()) {
      const StackFrame& last_frame = stack_frames[0];
      return absl::StrCat(" (defined @ ", last_frame.file_name, ":",
                          last_frame.line_number, ")");
    }
  }
  return "";
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_ABSTRACT_STACK_TRACE_H_
