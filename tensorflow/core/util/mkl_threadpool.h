
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

#ifndef TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
#define TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh() {
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

#ifdef INTEL_MKL

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnnl_threadpool.hpp"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/threadpool.h"
#define EIGEN_USE_THREADS

namespace tensorflow {

#ifndef ENABLE_ONEDNN_OPENMP
using dnnl::threadpool_interop::threadpool_iface;

// Divide 'n' units of work equally among 'teams' threads. If 'n' is not
// divisible by 'teams' and has a remainder 'r', the first 'r' teams have one
// unit of work more than the rest. Returns the range of work that belongs to
// the team 'tid'.
// Parameters
//   n        Total number of jobs.
//   team     Number of workers.
//   tid      Current thread_id.
//   n_start  start of range operated by the thread.
//   n_end    end of the range operated by the thread.

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T* n_start, T* n_end) {
  if (team <= 1 || n == 0) {
    *n_start = 0;
    *n_end = n;
    return;
  }
  T min_per_team = n / team;
  T remainder = n - min_per_team * team;  // i.e., n % teams.
  *n_start = tid * min_per_team + std::min(tid, remainder);
  *n_end = *n_start + min_per_team + (tid < remainder);
}

struct MklDnnThreadPool : public threadpool_iface {
  MklDnnThreadPool() = default;

  MklDnnThreadPool(OpKernelContext* ctx, int num_threads = -1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_0(mht_0_v, 235, "", "./tensorflow/core/util/mkl_threadpool.h", "MklDnnThreadPool");

    eigen_interface_ = ctx->device()
                           ->tensorflow_cpu_worker_threads()
                           ->workers->AsEigenThreadPool();
    num_threads_ =
        (num_threads == -1) ? eigen_interface_->NumThreads() : num_threads;
  }
  virtual int get_num_threads() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_1(mht_1_v, 245, "", "./tensorflow/core/util/mkl_threadpool.h", "get_num_threads");
 return num_threads_; }
  virtual bool get_in_parallel() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_2(mht_2_v, 249, "", "./tensorflow/core/util/mkl_threadpool.h", "get_in_parallel");

    return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
  }
  virtual uint64_t get_flags() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_3(mht_3_v, 255, "", "./tensorflow/core/util/mkl_threadpool.h", "get_flags");
 return ASYNCHRONOUS; }
  virtual void parallel_for(int n,
                            const std::function<void(int, int)>& fn) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_4(mht_4_v, 260, "", "./tensorflow/core/util/mkl_threadpool.h", "parallel_for");

    // Should never happen (handled by DNNL)
    if (n == 0) return;

    // Should never happen (handled by DNNL)
    if (n == 1) {
      fn(0, 1);
      return;
    }

    int nthr = get_num_threads();
    int njobs = std::min(n, nthr);
    bool balance = (nthr < n);
    for (int i = 0; i < njobs; i++) {
      eigen_interface_->ScheduleWithHint(
          [balance, i, n, njobs, fn]() {
            if (balance) {
              int start, end;
              balance211(n, njobs, i, &start, &end);
              for (int j = start; j < end; j++) fn(j, n);
            } else {
              fn(i, n);
            }
          },
          i, i + 1);
    }
  }
  ~MklDnnThreadPool() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_5(mht_5_v, 290, "", "./tensorflow/core/util/mkl_threadpool.h", "~MklDnnThreadPool");
}

 private:
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
  int num_threads_ = 1;  // Execute in caller thread.
};

#else

// This struct was just added to enable successful OMP-based build.
struct MklDnnThreadPool {
  MklDnnThreadPool() = default;
  MklDnnThreadPool(OpKernelContext* ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_6(mht_6_v, 305, "", "./tensorflow/core/util/mkl_threadpool.h", "MklDnnThreadPool");
}
  MklDnnThreadPool(OpKernelContext* ctx, int num_threads) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSmkl_threadpoolDTh mht_7(mht_7_v, 309, "", "./tensorflow/core/util/mkl_threadpool.h", "MklDnnThreadPool");
}
};

#endif  // !ENABLE_ONEDNN_OPENMP

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
