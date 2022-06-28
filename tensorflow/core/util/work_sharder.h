/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_WORK_SHARDER_H_
#define TENSORFLOW_CORE_UTIL_WORK_SHARDER_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSwork_sharderDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSwork_sharderDTh() {
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

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// DEPRECATED: Prefer threadpool->ParallelFor with SchedulingStrategy, which
// allows you to specify the strategy for choosing shard sizes, including using
// a fixed shard size. Use this function only if you want to manually cap
// parallelism.
//
// Shards the "total" unit of work assuming each unit of work having
// roughly "cost_per_unit". Each unit of work is indexed 0, 1, ...,
// total - 1. Each shard contains 1 or more units of work and the
// total cost of each shard is roughly the same. The calling thread and the
// "workers" are used to compute each shard (calling work(start,
// limit). A common configuration is that "workers" is a thread pool
// with at least "max_parallelism" threads.
//
// "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
// if not CPU-bound) to complete a unit of work. Overestimating creates too
// many shards and CPU time will be dominated by per-shard overhead, such as
// Context creation. Underestimating may not fully make use of the specified
// parallelism.
//
// "work" should be a callable taking (int64, int64) arguments.
// work(start, limit) computes the work units from [start,
// limit), i.e., [start, limit) is a shard.
//
// Too much parallelism can also cause excessive thread switches,
// therefore, Shard() often limits the maximum parallelism. Each
// caller can provide the 1st argument max_parallelism. A thread can
// call SetMaxParallelism() so that all Shard() calls later limits the
// thread parallelism.
//
// REQUIRES: max_parallelism >= 0
// REQUIRES: workers != nullptr
// REQUIRES: total >= 0
// REQUIRES: cost_per_unit >= 0
void Shard(int max_parallelism, thread::ThreadPool* workers, int64_t total,
           int64_t cost_per_unit, std::function<void(int64_t, int64_t)> work);

// Each thread has an associated option to express the desired maximum
// parallelism. Its default is a very large quantity.
//
// Within TF runtime, per-thread max parallelism affects Shard() and
// intra-op parallelism. E.g., if SetPerThreadMaxParallelism(1) is
// arranged to be called by a tf_compute thread, Shard() calls and
// eigen device assignment happens in that thread afterwards becomes
// single-threaded.
void SetPerThreadMaxParallelism(int max_parallelism);
int GetPerThreadMaxParallelism();

// Helper to set and unset per-thread max parallelism.
class ScopedPerThreadMaxParallelism {
 public:
  ScopedPerThreadMaxParallelism(int max_parallelism)
      : previous_(GetPerThreadMaxParallelism()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharderDTh mht_0(mht_0_v, 246, "", "./tensorflow/core/util/work_sharder.h", "ScopedPerThreadMaxParallelism");

    SetPerThreadMaxParallelism(max_parallelism);
  }

  ~ScopedPerThreadMaxParallelism() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharderDTh mht_1(mht_1_v, 253, "", "./tensorflow/core/util/work_sharder.h", "~ScopedPerThreadMaxParallelism");
 SetPerThreadMaxParallelism(previous_); }

 private:
  int previous_ = -1;
};

// Implementation details for Shard().
class Sharder {
 public:
  typedef std::function<void()> Closure;
  typedef std::function<void(Closure)> Runner;
  typedef std::function<void(int64_t, int64_t)> Work;

  // Refers to Shard()'s comment for the meaning of total,
  // cost_per_unit, work, max_parallelism. runner is an interface to
  // schedule a closure. Shard() uses thread::ThreadPool instead.
  static void Do(int64_t total, int64_t cost_per_unit, const Work& work,
                 const Runner& runner, int max_parallelism);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_WORK_SHARDER_H_
