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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_swish_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_swish_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_swish_op_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "absl/strings/match.h"
#include "dnnl.hpp"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// This is a special case, because EIGEN kernels does not have Swish Kerenls.
// Compare the performance of default tensorflow kernels (Eigen) with
// MKL kernels on CPU.
//
// Then you could use below command to test mkl and eigen performance:
// $ bazel run --action_env=TF_ENABLE_ONEDNN_OPTS=1 -c opt \
//  //tensorflow/core/kernels/mkl:mkl_swish_op_test -- --benchmark_filter=all
//

namespace tensorflow {

// --------------------------------------------------------------------------//
//  Test Swish Kernels accuracy and performance                              //
// --------------------------------------------------------------------------//
template <typename T>
static Graph* SwishGraph(const string& kind, const TensorShape& shape) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("kind: \"" + kind + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_swish_op_testDTcc mht_0(mht_0_v, 226, "", "./tensorflow/core/kernels/mkl/mkl_swish_op_test.cc", "SwishGraph");

  auto* graph = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::v();
  Tensor input_t(dtype, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  const bool isDefault = (kind == "Default");

  Node* sigmoid;
  Node* mul;
  Node* swish;
  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_sigmoid"), "Sigmoid")
                    .Input(input)
                    .Attr("T", dtype)
                    .Finalize(graph, &sigmoid));

    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_mul"), "Mul")
                    .Input(input)
                    .Input(sigmoid)
                    .Attr("T", dtype)
                    .Finalize(graph, &mul));
    return graph;
  }
  // Mkl Swish op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("Mkl_swish"), "_MklSwish")
                  .Input(input)
                  .Attr("T", dtype)
                  .Finalize(graph, &swish));
  return graph;
}

#define BM_SWISH(kind, A, B, C, D, type, T)                                \
  static void BM_SWISH_##kind##_##type##_##A##_##B##_##C##_##D##_##T(      \
      ::testing::benchmark::State& state) {                                \
    int64 num_computed_elements = (A) * (B) * (C) * (D);                   \
    int64 flops_per_iter = num_computed_elements;                          \
                                                                           \
    test::Benchmark(#type, SwishGraph<T>(#kind, {A, B, C, D})).Run(state); \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);          \
  }                                                                        \
  BENCHMARK(BM_SWISH_##kind##_##type##_##A##_##B##_##C##_##D##_##T)

#define BENCHMARK_SWISH(A, B, C, D, type, T) \
  BM_SWISH(Default, A, B, C, D, type, T);    \
  BM_SWISH(Mkl, A, B, C, D, type, T);

#define BENCHMARK_DTYPE(T)                    \
  BENCHMARK_SWISH(1, 16, 16, 3, cpu, T);      \
  BENCHMARK_SWISH(16, 32, 32, 1, cpu, T);     \
  BENCHMARK_SWISH(16, 64, 64, 128, cpu, T);   \
  BENCHMARK_SWISH(32, 64, 64, 128, cpu, T);   \
  BENCHMARK_SWISH(32, 256, 256, 128, cpu, T); \
  BENCHMARK_SWISH(32, 512, 512, 128, cpu, T);

BENCHMARK_DTYPE(float)
BENCHMARK_DTYPE(bfloat16)

}  // namespace tensorflow

#endif  // INTEL_MKL
