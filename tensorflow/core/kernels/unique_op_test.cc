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
class MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc() {
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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

const int kMaxStrLen = 40;

TensorProto GetRandomInt32TensorProto(int dim, int max_int) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/unique_op_test.cc", "GetRandomInt32TensorProto");

  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    tensor_proto.add_int_val(int_val);
  }
  return tensor_proto;
}

TensorProto GetRandomInt32TensorProtoWithRepeat(int dim, int repeat,
                                                int max_int) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/kernels/unique_op_test.cc", "GetRandomInt32TensorProtoWithRepeat");

  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int int_val = std::rand() % max_int;
    for (int j = 0; j < repeat; ++j) {
      tensor_proto.add_int_val(int_val);
    }
  }
  return tensor_proto;
}

void BM_Unique_INT32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/unique_op_test.cc", "BM_Unique_INT32");

  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomInt32TensorProto(dim, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(int32));
}

void BM_Unique_INT32_Repeat(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/unique_op_test.cc", "BM_Unique_INT32_Repeat");

  const int dim = state.range(0);
  const int max_int = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_INT32, TensorShape({dim * 200}));
  CHECK(
      input.FromProto(GetRandomInt32TensorProtoWithRepeat(dim, 200, max_int)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);

  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim * 200 *
                          sizeof(int32));
}

TensorProto GetRandomStringsTensorProto(int dim, int max_str_len) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_4(mht_4_v, 295, "", "./tensorflow/core/kernels/unique_op_test.cc", "GetRandomStringsTensorProto");

  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(dim);
  tensor_proto.mutable_tensor_shape()->set_unknown_rank(false);
  for (int i = 0; i < dim; ++i) {
    const int len = std::rand() % max_str_len + 1;
    string rand_str;
    rand_str.resize(len);
    for (int j = 0; j < len; ++j) {
      rand_str[j] = static_cast<char>(j % 256);
    }
    tensor_proto.add_string_val(rand_str);
  }
  return tensor_proto;
}

void BM_Unique_STRING(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSunique_op_testDTcc mht_5(mht_5_v, 315, "", "./tensorflow/core/kernels/unique_op_test.cc", "BM_Unique_STRING");

  const int dim = state.range(0);

  Graph* g = new Graph(OpRegistry::Global());

  Tensor input(DT_STRING, TensorShape({dim}));
  CHECK(input.FromProto(GetRandomStringsTensorProto(dim, kMaxStrLen)));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Unique")
                  .Input(test::graph::Constant(g, input))
                  .Attr("T", DT_STRING)
                  .Finalize(g, &node));
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api*/ false)
      .Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * dim *
                          sizeof(tstring));
}

BENCHMARK(BM_Unique_INT32)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_INT32_Repeat)
    ->UseRealTime()
    ->ArgPair(32, 1024 * 1024)
    ->ArgPair(256, 1024 * 1024)
    ->ArgPair(1024, 1024 * 1024)
    ->ArgPair(4 * 1024, 1024 * 1024)
    ->ArgPair(16 * 1024, 1024 * 1024)
    ->ArgPair(64 * 1024, 1024 * 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(4 * 1024 * 1024, 1024 * 1024)
    ->ArgPair(32, 32 * 1024 * 1024)
    ->ArgPair(256, 32 * 1024 * 1024)
    ->ArgPair(1024, 32 * 1024 * 1024)
    ->ArgPair(4 * 1024, 32 * 1024 * 1024)
    ->ArgPair(16 * 1024, 32 * 1024 * 1024)
    ->ArgPair(64 * 1024, 32 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 32 * 1024 * 1024)
    ->ArgPair(32, 64 * 1024 * 1024)
    ->ArgPair(256, 64 * 1024 * 1024)
    ->ArgPair(1024, 64 * 1024 * 1024)
    ->ArgPair(4 * 1024, 64 * 1024 * 1024)
    ->ArgPair(16 * 1024, 64 * 1024 * 1024)
    ->ArgPair(64 * 1024, 64 * 1024 * 1024)
    ->ArgPair(1024 * 1024, 64 * 1024 * 1024);

BENCHMARK(BM_Unique_STRING)
    ->UseRealTime()
    ->Arg(32)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(4 * 1024)
    ->Arg(16 * 1024)
    ->Arg(64 * 1024)
    ->Arg(256 * 1024);

}  // namespace
}  // namespace tensorflow
