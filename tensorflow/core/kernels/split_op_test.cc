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
class MHTracer_DTPStensorflowPScorePSkernelsPSsplit_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsplit_op_testDTcc() {
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

#include <initializer_list>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* MakeGraph(int split_dim, int num_split,
                        std::initializer_list<int64_t> chunk_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsplit_op_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/split_op_test.cc", "MakeGraph");

  Graph* g = new Graph(OpRegistry::Global());
  TensorShape in_shape(chunk_size);
  in_shape.set_dim(split_dim, in_shape.dim_size(split_dim) * num_split);
  Tensor in(DataTypeToEnum<float>::value, in_shape);
  in.flat<float>().setRandom();
  Tensor split_dim_tensor = test::AsScalar<int32>(split_dim);
  Node* split;
  TF_CHECK_OK(NodeBuilder(g->NewName("split"), "Split")
                  .Input(test::graph::Constant(g, split_dim_tensor))
                  .Input(test::graph::Constant(g, in))
                  .Attr("num_split", num_split)
                  .Finalize(g, &split));
  return g;
}

#define BM_SPLIT_1D(num_split, chunk_size)                                  \
  static void BM_Split_1d_##num_split##_##chunk_size(                       \
      ::testing::benchmark::State& state) {                                 \
    auto label =                                                            \
        strings::Printf("1-D %d chunks of %d each", num_split, chunk_size); \
    state.SetLabel(label);                                                  \
    auto g = MakeGraph(/* split_dim = */ 0, num_split, {chunk_size});       \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);      \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *      \
                            num_split * chunk_size);                        \
  }                                                                         \
  BENCHMARK(BM_Split_1d_##num_split##_##chunk_size)->UseRealTime();

#define BM_SPLIT_2D(split_dim, num_split, chunk_size0, chunk_size1)          \
  static void                                                                \
      BM_Split_2d_##split_dim##_##num_split##_##chunk_size0##_##chunk_size1( \
          ::testing::benchmark::State& state) {                              \
    auto label =                                                             \
        strings::Printf("2-D %d chunks in dim %d of (%d * %d) each",         \
                        num_split, split_dim, chunk_size0, chunk_size1);     \
    state.SetLabel(label);                                                   \
    auto g = MakeGraph(split_dim, num_split, {chunk_size0, chunk_size1});    \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);       \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *       \
                            num_split * chunk_size0 * chunk_size1);          \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_Split_2d_##split_dim##_##num_split##_##chunk_size0##_##chunk_size1) \
      ->UseRealTime();

BM_SPLIT_1D(5, 1);
BM_SPLIT_1D(262144, 1);
BM_SPLIT_1D(1, 100000);
BM_SPLIT_1D(5, 100000);
BM_SPLIT_1D(10, 4194304);
BM_SPLIT_1D(2, 4194304);
BM_SPLIT_1D(100, 1024);
BM_SPLIT_1D(32768, 1024);

BM_SPLIT_2D(0, 1024, 1, 10);
BM_SPLIT_2D(0, 1024, 10, 10);
BM_SPLIT_2D(0, 512, 1024, 256);
BM_SPLIT_2D(0, 20, 100000, 5);
BM_SPLIT_2D(0, 2, 3, 524288);
BM_SPLIT_2D(0, 100, 4096, 512);

BM_SPLIT_2D(1, 1024, 1, 10);
BM_SPLIT_2D(1, 1024, 10, 10);
BM_SPLIT_2D(1, 512, 1024, 256);
BM_SPLIT_2D(1, 20, 100000, 5);
BM_SPLIT_2D(1, 2, 3, 524288);
BM_SPLIT_2D(1, 100, 4096, 512);

}  // namespace tensorflow
