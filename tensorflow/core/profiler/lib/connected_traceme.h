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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh() {
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
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace profiler {

/*
 * TraceMeProducer and TraceMeConsumer are used to correlate TraceMe events on
 * different threads. TraceMeProducer generates the context information to be
 * passed to TraceMeConsumer, which consists of the context id and optionally
 * the context type. They may be provided by the user. Then, the events of the
 * same context information can be correlated during the analysis.
 *
 * Example Usages:
 * (1) Using the user-provided context type and id. The user is responsible for
 *     providing the same context type and id to TraceMeProducer and
 *     TraceMeConsumer.
 * [Producer Thread]
 * // user_context_id is provided by the user.
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); },
 *     ContextType::kTfExecutor, user_context_id);
 * [Consumer Thread]
 * // user_context_id is provided by the user.
 * TraceMeConsumer consumer(
 *     [&] { return "op_execute"; }, ContextType::kTfExecutor, user_context_id);
 *
 * (2) Using the user-provided context type and generic id. The user is
 *     responsible for passing the TraceMeProducer's context id to
 *     TraceMeConsumer as well as providing the same context type to
 *     TraceMeProducer and TraceMeConsumer.
 * [Producer Thread]
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); },
 *     ContextType::kTfExecutor);
 * context_id = producer.GetContextId();
 * // Pass context_id to the consumer thread.
 * [Consumer Thread]
 * // context_id is passed from the producer thread.
 * TraceMeConsumer consumer(
 *     [&] { return "op_execute"; }, ContextType::kTfExecutor, context_id);
 *
 * (3) Using the generic context information. The user is responsible for
 *     passing the TraceMeProducer's context id to TraceMeConsumer.
 * [Producer Thread]
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); });
 * context_id = producer.GetContextId();
 * // Pass context_id to the consumer thread.
 * [Consumer Thread]
 * // context_id is passed from the producer thread.
 * TraceMeConsumer consumer([&] { return "op_execute"; }, context_id);
 */
class TraceMeProducer {
 public:
  template <typename NameT>
  explicit TraceMeProducer(NameT&& name,
                           ContextType context_type = ContextType::kGeneric,
                           absl::optional<uint64> context_id = absl::nullopt,
                           int level = 2)
      : context_id_(context_id.has_value() ? context_id.value()
                                           : TraceMe::NewActivityId()),
        trace_me_(std::forward<NameT>(name), level) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh mht_0(mht_0_v, 255, "", "./tensorflow/core/profiler/lib/connected_traceme.h", "TraceMeProducer");

    trace_me_.AppendMetadata([&] {
      return TraceMeEncode({{"_pt", context_type}, {"_p", context_id_}});
    });
  }

  uint64 GetContextId() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh mht_1(mht_1_v, 264, "", "./tensorflow/core/profiler/lib/connected_traceme.h", "GetContextId");
 return context_id_; }

 private:
  uint64 context_id_;
  TraceMe trace_me_;
};

class TraceMeConsumer {
 public:
  template <typename NameT>
  TraceMeConsumer(NameT&& name, ContextType context_type, uint64 context_id,
                  int level = 2)
      : trace_me_(std::forward<NameT>(name), level) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh mht_2(mht_2_v, 279, "", "./tensorflow/core/profiler/lib/connected_traceme.h", "TraceMeConsumer");

    trace_me_.AppendMetadata([&] {
      return TraceMeEncode({{"_ct", context_type}, {"_c", context_id}});
    });
  }

  template <typename NameT>
  TraceMeConsumer(NameT&& name, uint64 context_id, int level = 2)
      : TraceMeConsumer(std::forward<NameT>(name), ContextType::kGeneric,
                        context_id, level) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSconnected_tracemeDTh mht_3(mht_3_v, 291, "", "./tensorflow/core/profiler/lib/connected_traceme.h", "TraceMeConsumer");
}

 private:
  TraceMe trace_me_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_
