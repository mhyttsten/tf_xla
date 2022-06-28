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
class MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_op_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename InputShape>
static Graph* BroadcastTo(int dim0, int dim1, InputShape input_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSbroadcast_to_op_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/kernels/broadcast_to_op_test.cc", "BroadcastTo");

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_FLOAT, input_shape(dim0, dim1));
  input.flat<float>() = input.flat<float>().setRandom();

  Tensor shape(DT_INT32, TensorShape({2}));
  shape.flat<int32>()(0) = dim0;
  shape.flat<int32>()(1) = dim1;

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, shape))
                  .Attr("T", DT_FLOAT)
                  .Attr("Tidx", DT_INT32)
                  .Finalize(g, &node));
  return g;
}

#define BM_BroadcastTo_InnerDim(DIM0, DIM1, type)                             \
  static void BM_BroadcastTo_Inner##_##type##_##DIM0##_##DIM1(                \
      ::testing::benchmark::State& state) {                                   \
    test::Benchmark(#type,                                                    \
                    BroadcastTo(DIM0, DIM1,                                   \
                                [](int dim0, int dim1) {                      \
                                  return TensorShape({dim0, 1});              \
                                }),                                           \
                    /*old_benchmark_api=*/false)                              \
        .Run(state);                                                          \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * DIM0 * \
                            DIM1);                                            \
  }                                                                           \
  BENCHMARK(BM_BroadcastTo_Inner##_##type##_##DIM0##_##DIM1)->UseRealTime();

#define BM_BroadcastTo_OuterDim(DIM0, DIM1, type)                             \
  static void BM_BroadcastTo_Outer##_##type##_##DIM0##_##DIM1(                \
      ::testing::benchmark::State& state) {                                   \
    test::Benchmark(#type,                                                    \
                    BroadcastTo(DIM0, DIM1,                                   \
                                [](int dim0, int dim1) {                      \
                                  return TensorShape({1, dim1});              \
                                }),                                           \
                    /*old_benchmark_api=*/false)                              \
        .Run(state);                                                          \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * DIM0 * \
                            DIM1);                                            \
  }                                                                           \
  BENCHMARK(BM_BroadcastTo_Outer##_##type##_##DIM0##_##DIM1)->UseRealTime();

BM_BroadcastTo_InnerDim(64, 64, cpu);
BM_BroadcastTo_InnerDim(128, 128, cpu);
BM_BroadcastTo_InnerDim(256, 256, cpu);
BM_BroadcastTo_InnerDim(512, 512, cpu);
BM_BroadcastTo_InnerDim(1024, 1024, cpu);
BM_BroadcastTo_InnerDim(500, 20000, cpu);

BM_BroadcastTo_OuterDim(64, 64, cpu);
BM_BroadcastTo_OuterDim(128, 128, cpu);
BM_BroadcastTo_OuterDim(256, 256, cpu);
BM_BroadcastTo_OuterDim(512, 512, cpu);
BM_BroadcastTo_OuterDim(1024, 1024, cpu);
BM_BroadcastTo_OuterDim(500, 20000, cpu);

}  // end namespace tensorflow
