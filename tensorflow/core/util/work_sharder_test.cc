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
class MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc() {
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

#include "tensorflow/core/util/work_sharder.h"

#include <atomic>
#include <vector>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

void RunSharding(int64_t num_workers, int64_t total, int64_t cost_per_unit,
                 int64_t per_thread_max_parallelism,
                 thread::ThreadPool* threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/util/work_sharder_test.cc", "RunSharding");

  mutex mu;
  int64_t num_shards = 0;
  int64_t num_done_work = 0;
  std::vector<bool> work(total, false);
  Shard(num_workers, threads, total, cost_per_unit,
        [=, &mu, &num_shards, &num_done_work, &work](int64_t start,
                                                     int64_t limit) {
          VLOG(1) << "Shard [" << start << "," << limit << ")";
          EXPECT_GE(start, 0);
          EXPECT_LE(limit, total);
          mutex_lock l(mu);
          ++num_shards;
          for (; start < limit; ++start) {
            EXPECT_FALSE(work[start]);  // No duplicate
            ++num_done_work;
            work[start] = true;
          }
        });
  LOG(INFO) << num_workers << " " << total << " " << cost_per_unit << " "
            << num_shards;
  EXPECT_EQ(num_done_work, total);
  if (std::min(num_workers, per_thread_max_parallelism) <
      threads->NumThreads()) {
    // If the intention is to limit the parallelism explicitly, we'd
    // better honor it. Ideally, even if per_thread_max_parallelism >
    // num_workers, we should expect that Shard() implementation do
    // not over-shard. Unfortunately, ThreadPoolDevice::parallelFor
    // tends to over-shard.
    EXPECT_LE(num_shards, 1 + per_thread_max_parallelism);
  }
}

TEST(Shard, Basic) {
  thread::ThreadPool threads(Env::Default(), "test", 16);
  for (auto workers : {0, 1, 2, 3, 5, 7, 10, 11, 15, 100, 1000}) {
    for (auto total : {0, 1, 7, 10, 64, 100, 256, 1000, 9999}) {
      for (auto cost_per_unit : {0, 1, 11, 102, 1003, 10005, 1000007}) {
        for (auto maxp : {1, 2, 4, 8, 100}) {
          ScopedPerThreadMaxParallelism s(maxp);
          RunSharding(workers, total, cost_per_unit, maxp, &threads);
        }
      }
    }
  }
}

TEST(Shard, OverflowTest) {
  thread::ThreadPool threads(Env::Default(), "test", 3);
  for (auto workers : {1, 2, 3}) {
    const int64_t total_elements = 1LL << 32;
    const int64_t cost_per_unit = 10;
    std::atomic<int64_t> num_elements(0);
    Shard(workers, &threads, total_elements, cost_per_unit,
          [&num_elements](int64_t start, int64_t limit) {
            num_elements += limit - start;
          });
    EXPECT_EQ(num_elements.load(), total_elements);
  }
}

void BM_Sharding(::testing::benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc mht_1(mht_1_v, 265, "", "./tensorflow/core/util/work_sharder_test.cc", "BM_Sharding");

  const int arg = state.range(0);

  thread::ThreadPool threads(Env::Default(), "test", 16);
  const int64_t total = 1LL << 30;
  auto lambda = [](int64_t start, int64_t limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSwork_sharder_testDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/util/work_sharder_test.cc", "lambda");
};
  auto work = std::cref(lambda);
  for (auto s : state) {
    Shard(arg - 1, &threads, total, 1, work);
  }
}
BENCHMARK(BM_Sharding)->Range(1, 128);

}  // namespace
}  // namespace tensorflow
