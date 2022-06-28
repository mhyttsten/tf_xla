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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/recent_request_ids.h"

#include <algorithm>

#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

Status TrackUnique(int64_t request_id, RecentRequestIds* recent_request_ids) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/distributed_runtime/recent_request_ids_test.cc", "TrackUnique");

  RecvTensorRequest request;
  request.set_request_id(request_id);
  return recent_request_ids->TrackUnique(request_id, "recent_request_ids_test",
                                         request);
}

// request_id 0 is always valid.
TEST(RecentRequestIds, Zero) {
  RecentRequestIds recent_request_ids(1);
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
  EXPECT_TRUE(TrackUnique(0, &recent_request_ids).ok());
}

TEST(RecentRequestIds, Unordered) {
  // Capacity for 6 numbers.
  RecentRequestIds recent_request_ids(6);

  // Some unordered numbers to insert into request_id_set.
  std::vector<int64_t> numbers = {53754,  23351,  164101, 7476,
                                  162432, 130761, 164102};

  // Insert numbers[0..6) and check that all previously inserted numbers remain
  // in the set.
  for (int i = 0; i < 6; ++i) {
    TF_EXPECT_OK(TrackUnique(numbers[i], &recent_request_ids));

    for (int j = 0; j <= i; ++j) {
      EXPECT_FALSE(TrackUnique(numbers[j], &recent_request_ids).ok())
          << "i=" << i << " j=" << j;
    }
  }

  // Insert numbers[6]. Inserting this 7th number should evict the first number
  // from the set. The set should only contain numbers[1..7).
  TF_EXPECT_OK(TrackUnique(numbers[6], &recent_request_ids));
  for (int i = 1; i < 7; ++i) {
    EXPECT_FALSE(TrackUnique(numbers[i], &recent_request_ids).ok())
        << "i=" << i;
  }

  // Insert numbers[0] again. This should succeed because we just evicted it
  // from the set.
  TF_EXPECT_OK(TrackUnique(numbers[0], &recent_request_ids));
}

// Check that the oldest request_id is evicted.
void TestOrdered(int num_request_ids) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/distributed_runtime/recent_request_ids_test.cc", "TestOrdered");

  RecentRequestIds recent_request_ids(num_request_ids);

  // Insert [1..101). The current number and the (num_request_ids - 1) preceding
  // numbers should still be in the set.
  for (int i = 1; i < 101; ++i) {
    TF_EXPECT_OK(TrackUnique(i, &recent_request_ids));

    for (int j = std::max(1, i - num_request_ids + 1); j <= i; ++j) {
      EXPECT_FALSE(TrackUnique(j, &recent_request_ids).ok())
          << "i=" << i << " j=" << j;
    }
  }
}

// Test eviction with various numbers of buckets.
TEST(RecentRequestIds, Ordered2) { TestOrdered(2); }
TEST(RecentRequestIds, Ordered3) { TestOrdered(3); }
TEST(RecentRequestIds, Ordered4) { TestOrdered(4); }
TEST(RecentRequestIds, Ordered5) { TestOrdered(5); }

static void BM_TrackUnique(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_ids_testDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/distributed_runtime/recent_request_ids_test.cc", "BM_TrackUnique");

  RecentRequestIds recent_request_ids(100000);
  RecvTensorRequest request;
  for (auto s : state) {
    TF_CHECK_OK(recent_request_ids.TrackUnique(GetUniqueRequestId(),
                                               "BM_TrackUnique", request));
  }
}

BENCHMARK(BM_TrackUnique);

}  // namespace tensorflow
