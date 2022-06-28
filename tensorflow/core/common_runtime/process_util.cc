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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/process_util.h"

#if defined(ENABLE_MKL) && defined(ENABLE_ONEDNN_OPENMP)
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP
#endif  // defined(ENABLE_MKL) && defined(ENABLE_ONEDNN_OPENMP)
#include <string.h>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

namespace {

// Use environment setting if specified (init once)
int32 GetEnvNumInterOpThreads() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/common_runtime/process_util.cc", "GetEnvNumInterOpThreads");

  static int32_t env_num_threads = NumInterOpThreadsFromEnvironment();
  return env_num_threads;
}

int32 DefaultNumInterOpThreads() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/process_util.cc", "DefaultNumInterOpThreads");

#ifndef __ANDROID__
  int32_t env_num_threads = GetEnvNumInterOpThreads();
  if (env_num_threads > 0) {
    return env_num_threads;
  }

  // Default to the maximum parallelism for the current process.
  return port::MaxParallelism();
#else
  // Historically, -D__ANDROID__ resulted in the inter-op threadpool not being
  // used (regardless of what was chosen here); instead, all work was done on
  // the thread(s) calling Session::Run. That's no longer the case, but we'd
  // like to avoid suddenly higher concurrency and peak resource usage (for the
  // same device shape, graph, and options) versus prior versions - as best we
  // can:
  //
  //   - Single Session::Run (none concurrent), and default options:
  //     Behavior is mostly the same as before.
  //
  //   - Concurrent Session::Runs, and default options:
  //     Reduced concurrency versus before.
  //
  //   - Thread-pool size set explicitly (>1):
  //     Increased concurrency versus before.
  //
  // (We assume the first case is the most common)
  return 1;
#endif
}

static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/common_runtime/process_util.cc", "InitComputePool");

  int32_t inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    inter_op_parallelism_threads = DefaultNumInterOpThreads();
  }
  return new thread::ThreadPool(
      Env::Default(), ThreadOptions(), "Compute", inter_op_parallelism_threads,
      !options.config.experimental().disable_thread_spinning(),
      /*allocator=*/nullptr);
}

}  // namespace

thread::ThreadPool* ComputePool(const SessionOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/common_runtime/process_util.cc", "ComputePool");

  static thread::ThreadPool* compute_pool = InitComputePool(options);
  return compute_pool;
}

int32 NumInterOpThreadsFromEnvironment() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/common_runtime/process_util.cc", "NumInterOpThreadsFromEnvironment");

  int32_t num;
  const char* val = std::getenv("TF_NUM_INTEROP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}

int32 NumIntraOpThreadsFromEnvironment() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/common_runtime/process_util.cc", "NumIntraOpThreadsFromEnvironment");

  int32_t num;
  const char* val = std::getenv("TF_NUM_INTRAOP_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}
#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 OMPThreadsFromEnvironment() {
  // 1) std::getenv is thread-safe (as long as no other function modifies the
  // host env) from C++11 onward. 2) Most of TF code (except tests and
  // experimental code) doesn't call setenv and unsetenv
  int32 num;
  const char* val = std::getenv("OMP_NUM_THREADS");
  return (val && strings::safe_strto32(val, &num)) ? num : 0;
}

int32 DefaultNumIntraOpThreads() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/common_runtime/process_util.cc", "DefaultNumIntraOpThreads");

  // Use environment setting if specified (init once)
  static int env_num_threads = NumIntraOpThreadsFromEnvironment();
  if (env_num_threads > 0) {
    return env_num_threads;
  }

  // Default to the maximum parallelism for the current process.
  return port::MaxParallelism();
}
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
int32 NumInterOpThreadsFromSessionOptions(const SessionOptions& options) {
  const int32_t inter_op = options.config.inter_op_parallelism_threads();
  if (inter_op > 0) return inter_op;
  const int32_t env_inter_op = GetEnvNumInterOpThreads();
  if (env_inter_op > 0) return env_inter_op;

#if defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  if (IsMKLEnabled()) {
    // MKL library executes ops in parallel using OMP threads.
    // Setting inter_op conservatively to avoid thread oversubscription that
    // could lead to severe perf degradations and OMP resource exhaustion.
    // Inter ops are set such that mkl_inter_op * mkl_intra_op <= NumCores.
    const int32 intra_op = options.config.intra_op_parallelism_threads();
    const int32 omp_max_threads = OMPThreadsFromEnvironment();
    const int32 mkl_intra_op =
        (omp_max_threads > 0)
            ? omp_max_threads
            : (intra_op > 0) ? intra_op : DefaultNumIntraOpThreads();
    DCHECK_GE(mkl_intra_op, 1);
    const int32 mkl_inter_op = std::max(
        (DefaultNumInterOpThreads() + mkl_intra_op - 1) / mkl_intra_op, 2);
    VLOG(0)
        << "Creating new thread pool with default inter op setting: "
        << mkl_inter_op
        << ". Tune using inter_op_parallelism_threads for best performance.";
    return mkl_inter_op;
  }
#endif  // defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_MKL)
  return DefaultNumInterOpThreads();
}

thread::ThreadPool* NewThreadPoolFromSessionOptions(
    const SessionOptions& options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_7(mht_7_v, 347, "", "./tensorflow/core/common_runtime/process_util.cc", "NewThreadPoolFromSessionOptions");

  const int32_t num_threads = NumInterOpThreadsFromSessionOptions(options);
  VLOG(1) << "Session inter op parallelism threads: " << num_threads;
  return new thread::ThreadPool(
      options.env, ThreadOptions(), "Compute", num_threads,
      !options.config.experimental().disable_thread_spinning(),
      /*allocator=*/nullptr);
}

void SchedClosure(std::function<void()> closure) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_8(mht_8_v, 359, "", "./tensorflow/core/common_runtime/process_util.cc", "SchedClosure");

  if (!tracing::EventCollector::IsEnabled()) {
    return Env::Default()->SchedClosure(std::move(closure));
  }
  uint64 id = tracing::GetUniqueArg();
  tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);

  Env::Default()->SchedClosure([id, closure = std::move(closure)]() {
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure, id);
    closure();
  });
}

void SchedNonBlockingClosureAfter(int64_t micros,
                                  std::function<void()> closure) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSprocess_utilDTcc mht_9(mht_9_v, 376, "", "./tensorflow/core/common_runtime/process_util.cc", "SchedNonBlockingClosureAfter");

  Env::Default()->SchedClosureAfter(micros, std::move(closure));
}

}  // namespace tensorflow
