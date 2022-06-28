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
class MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc() {
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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* PTruncatedNormal(int num_batches, int samples_per_batch) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/kernels/parameterized_truncated_normal_op_test.cc", "PTruncatedNormal");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  // Use mean 0 and stdev 1
  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);

  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setRandom();
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(5.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormal2SD(int num_batches, int samples_per_batch) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/kernels/parameterized_truncated_normal_op_test.cc", "PTruncatedNormal2SD");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(-2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(2.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormalOneTail(int num_batches, int samples_per_batch) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSparameterized_truncated_normal_op_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/kernels/parameterized_truncated_normal_op_test.cc", "PTruncatedNormalOneTail");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(std::numeric_limits<float>::infinity());

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

#define BM_PTruncatedNormalDev(DEVICE, B, S)                                   \
  static void BM_PTruncatedNormal_##DEVICE##_##B##_##S(                        \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal(B, S),                           \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_2SD(DEVICE, B, S)                               \
  static void BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S(                    \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal2SD(B, S),                        \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_OneTail(DEVICE, B, S)                           \
  static void BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S(                \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormalOneTail(B, S),                    \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S);

BM_PTruncatedNormalDev(cpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(cpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(cpu, 10000, 100);
BM_PTruncatedNormalDev(gpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(gpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(gpu, 10000, 100);

}  // namespace tensorflow
