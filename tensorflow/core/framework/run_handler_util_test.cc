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
class MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_util_testDTcc() {
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

#include "tensorflow/core/framework/run_handler_util.h"

#include <vector>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
namespace tensorflow {
namespace {

void VerifySchedulingRanges(int num_active_requests, int num_threads,
                            int min_threads_per_request,
                            bool print_stats = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrun_handler_util_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/framework/run_handler_util_test.cc", "VerifySchedulingRanges");

  if (print_stats) {
    LOG(INFO) << "Test case# num_active_requests: " << num_active_requests
              << " num_threads: " << num_threads
              << " min_threads: " << min_threads_per_request;
  }
  std::vector<std::uint_fast32_t> start(num_active_requests);
  std::vector<std::uint_fast32_t> end(num_active_requests);

  ComputeInterOpSchedulingRanges(num_active_requests, num_threads,
                                 min_threads_per_request, &start, &end);
  string range_str = "";
  for (int i = 0; i < num_active_requests; ++i) {
    if (i > 0) range_str += " ";
    range_str += strings::StrCat("[", start[i], ", ", end[i], ")");

    ASSERT_GE(start[i], 0) << range_str;
    ASSERT_LE(end[i], num_threads) << range_str;
    if (i > 0) {
      // Due to linearly decreasing demand, #threads(i - 1) >= #threads(i)
      ASSERT_GE(end[i - 1] - start[i - 1], end[i] - start[i]) << range_str;
      // No missing threads.
      ASSERT_GE(end[i - 1], start[i]) << range_str;
    }
    // Each interval is at least of size 'min_threads_per_request'.
    ASSERT_GE((end[i] - start[i]), min_threads_per_request) << range_str;
    // Verify that assigned (quantized) threads is not overly estimated
    // from real demand, when the demand is high (>=
    // min_threads_per_request).
    float entry_weight = num_active_requests - i;
    float total_weight = 0.5f * num_active_requests * (num_active_requests + 1);
    float thread_demand = (entry_weight * num_threads) / total_weight;
    if (thread_demand > min_threads_per_request) {
      // We expect some over-estimation of threads due to quantization,
      // but we hope it's not more than 1 extra thread.
      ASSERT_NEAR(end[i] - start[i], thread_demand, 1.0)
          << "Ranges: " << range_str << " thread_demand: " << thread_demand
          << " i: " << i;
    }
  }
  ASSERT_EQ(end[num_active_requests - 1], num_threads);
  ASSERT_EQ(start[0], 0);
  if (print_stats) {
    LOG(INFO) << "Assigned ranges: " << range_str;
  }
}

TEST(RunHandlerUtilTest, TestComputeInterOpSchedulingRanges) {
  const int kMinThreadsPerRequestBound = 12;
  const int kMaxActiveRequests = 128;
  const int kMaxThreads = 128;

  for (int min_threads_per_request = 1;
       min_threads_per_request <= kMinThreadsPerRequestBound;
       ++min_threads_per_request) {
    for (int num_active_requests = 1; num_active_requests <= kMaxActiveRequests;
         ++num_active_requests) {
      for (int num_threads = min_threads_per_request;
           num_threads <= kMaxThreads; ++num_threads) {
        VerifySchedulingRanges(num_active_requests, num_threads,
                               min_threads_per_request);
      }
    }
  }
}

TEST(RunHandlerUtilTest, TestComputeInterOpStealingRanges) {
  int num_inter_op_threads = 9;
  std::vector<std::uint_fast32_t> start_vec(num_inter_op_threads);
  std::vector<std::uint_fast32_t> end_vec(num_inter_op_threads);

  // When there is 9 threads, there should be two thread groups.
  // The first group has threads [0, 6) with stealing range [0, 6)
  // The second group has threads [6, 9) with stealing range [3, 9)

  ComputeInterOpStealingRanges(num_inter_op_threads, 6, &start_vec, &end_vec);
  int stealing_ranges[2][2] = {{0, 6}, {3, 9}};

  for (int i = 0; i < num_inter_op_threads; ++i) {
    int expected_start = stealing_ranges[i / 6][0];
    int expected_end = stealing_ranges[i / 6][1];
    string message =
        strings::StrCat("Stealing range of thread ", i, " should be [",
                        expected_start, ", ", expected_end, "]");
    ASSERT_EQ(start_vec[i], expected_start) << message;
    ASSERT_EQ(end_vec[i], expected_end) << message;
  }
}

TEST(RunHandlerUtilTest, TestExponentialRequestDistribution) {
  int num_active_requests = 3;
  int num_threads = 10;
  std::vector<int> actual_distribution =
      ChooseRequestsWithExponentialDistribution(num_active_requests,
                                                num_threads);

  std::vector<int> expected_distribution{0, 0, 0, 0, 0, 1, 1, 1, 2, 2};
  ASSERT_EQ(actual_distribution, expected_distribution);
}

TEST(RunHandlerUtilTest, TestParamFromEnvWithDefault) {
  std::vector<double> result = ParamFromEnvWithDefault(
      "RUN_HANDLER_TEST_ENV", std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 0);
  EXPECT_EQ(result[2], 0);

  std::vector<int> result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                                     std::vector<int>{0, 0, 0});
  EXPECT_EQ(result2.size(), 3);
  EXPECT_EQ(result2[0], 0);
  EXPECT_EQ(result2[1], 0);
  EXPECT_EQ(result2[2], 0);

  bool result3 =
      ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, false);

  // Set environment variable.
  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV", "1,2,3", true), 0);
  result = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                   std::vector<double>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
  result2 = ParamFromEnvWithDefault("RUN_HANDLER_TEST_ENV",
                                    std::vector<int>{0, 0, 0});
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result2[0], 1);
  EXPECT_EQ(result2[1], 2);
  EXPECT_EQ(result2[2], 3);

  EXPECT_EQ(setenv("RUN_HANDLER_TEST_ENV_BOOL", "true", true), 0);
  result3 = ParamFromEnvBoolWithDefault("RUN_HANDLER_TEST_ENV_BOOL", false);
  EXPECT_EQ(result3, true);
}

}  // namespace
}  // namespace tensorflow
