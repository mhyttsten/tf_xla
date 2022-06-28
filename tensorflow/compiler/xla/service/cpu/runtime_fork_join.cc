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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fork_joinDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fork_joinDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fork_joinDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

using ComputeFunctionType = void (*)(void*, const void*, const void**, void**,
                                     void*, int64_t*, uint64_t*);

// Dispatches 'num_partitions - 1' calls to 'function_ptr' in parallel.
// Calls 'function_ptr' for first partition inline.
// Uses blocking counter to synchronize threads after parallel calls complete.
//
// The 'partitions' array has a total number of elements equal to
// 'num_partitions * num_partitioned_dims * 2' (the '2' is necessary to specify
// dimension start and limit indices).
//
// The 'partitions' array layout stores array elements in memory with dimension
// start limit as the most-minor dimension, followed by dimension, then
// partition.
//
// EX: Layout of 'partitions' array with 'num_partitions = 2', and
//     'num_partitioned_dims = 3'
//
//   [partition0_dim0_start]
//   [partition0_dim0_limit]
//   [partition0_dim1_start]
//   [partition0_dim1_limit]
//   [partition0_dim2_start]
//   [partition0_dim2_limit]
//   [partition1_dim0_start]
//   [partition1_dim0_limit]
//   [partition1_dim1_start]
//   [partition1_dim1_limit]
//   [partition1_dim2_start]
//   [partition1_dim2_limit]
//
ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ParallelForkJoin(
    void* result_ptr, const void* run_options_ptr, const void** params,
    void** buffer_table, void* status, uint64_t* prof_counters,
    int32_t num_partitions, int64_t* partitions, int32_t num_partitioned_dims,
    void* function_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fork_joinDTcc mht_0(mht_0_v, 233, "", "./tensorflow/compiler/xla/service/cpu/runtime_fork_join.cc", "__xla_cpu_runtime_ParallelForkJoin");

  VLOG(2) << "ParallelForkJoin ENTRY"
          << " num_partitions: " << num_partitions
          << " num_partitioned_dims: " << num_partitioned_dims;
  CHECK_EQ(params, nullptr);
  CHECK_GT(num_partitions, 1);
  CHECK_GT(num_partitioned_dims, 0);
  CHECK_NE(function_ptr, nullptr);
  CHECK_NE(partitions, nullptr);
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  CHECK_NE(run_options, nullptr);
  CHECK_NE(run_options->intra_op_thread_pool(), nullptr);

  ComputeFunctionType function =
      reinterpret_cast<ComputeFunctionType>(function_ptr);
  // Compute partition stride in 'partitions' array.
  const int64_t stride = 2 * num_partitioned_dims;

  std::vector<XlaCustomCallStatus> statuses(num_partitions);

  // Dispatch 'num_partitions - 1' compute functions to run in parallel.
  tensorflow::BlockingCounter bc(num_partitions - 1);
  for (int32_t i = 1; i < num_partitions; ++i) {
    const int64_t offset = i * stride;
    run_options->intra_op_thread_pool()->enqueueNoNotification(
        [i, function, result_ptr, run_options_ptr, buffer_table, prof_counters,
         partitions, offset, &bc, &statuses]() {
          function(result_ptr, run_options_ptr, nullptr, buffer_table,
                   &statuses[i], &partitions[offset], prof_counters);
          bc.DecrementCount();
          VLOG(3) << "ParallelForkJoin partition " << i << " done.";
        });
  }

  // Call first compute function inline.
  function(result_ptr, run_options_ptr, params, buffer_table, &statuses[0],
           &partitions[0], prof_counters);
  VLOG(3) << "ParallelForkJoin partition 0 done.";
  bc.Wait();

  // Collect all error messages (if any).
  std::vector<std::pair<int32_t, absl::string_view>> error_messages;
  for (int32_t i = 0; i < num_partitions; ++i) {
    absl::optional<absl::string_view> msg =
        xla::CustomCallStatusGetMessage(&statuses[i]);
    if (msg) {
      error_messages.emplace_back(i, *msg);
    }
  }

  if (!error_messages.empty()) {
    // Join all error messages into a single string to serve as the message for
    // the returned status.
    std::string error_message = absl::StrJoin(
        error_messages, "\n",
        [](std::string* out, std::pair<int32_t, absl::string_view> p) {
          int32_t idx = p.first;
          absl::string_view msg = p.second;
          absl::StrAppend(out,
                          absl::StrFormat("Partition %d error: %s", idx, msg));
        });
    XlaCustomCallStatusSetFailure(
        reinterpret_cast<XlaCustomCallStatus*>(status), error_message.data(),
        error_message.length());
  }
  VLOG(2) << "ParallelForkJoin EXIT";
}
