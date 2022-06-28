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
class MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc() {
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
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace {

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension"
template <typename T>
static void SliceHelper(::testing::benchmark::State& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/strided_slice_op_test.cc", "SliceHelper");

  const int size = state.range(0);
  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor end(DT_INT32, TensorShape({2}));
  end.flat<int32>()(0) = 10 + kDim;
  end.flat<int32>()(1) = 10 + size;

  Tensor strides(DT_INT32, TensorShape({2}));
  strides.flat<int32>()(0) = 1;
  strides.flat<int32>()(1) = 1;

  Tensor input(dt, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "StridedSlice")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, begin))
                  .Input(test::graph::Constant(g, end))
                  .Input(test::graph::Constant(g, strides))
                  .Attr("T", dt)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim *
                          size * sizeof(T));
}

void BM_SliceFloat(::testing::benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/kernels/strided_slice_op_test.cc", "BM_SliceFloat");

  SliceHelper<float>(state);
}

BENCHMARK(BM_SliceFloat)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_SliceComplex64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/kernels/strided_slice_op_test.cc", "BM_SliceComplex64");

  SliceHelper<std::complex<float>>(state);
}

BENCHMARK(BM_SliceComplex64)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_SliceBFloat16(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/strided_slice_op_test.cc", "BM_SliceBFloat16");

  SliceHelper<bfloat16>(state);
}

BENCHMARK(BM_SliceBFloat16)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_ValidateStridedSliceOp(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_op_testDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/kernels/strided_slice_op_test.cc", "BM_ValidateStridedSliceOp");

  int kDim = 100;
  int kMaxSize = 15000;
  int size = 100;
  Tensor begin = test::AsTensor<int32>({10, 10});
  Tensor end = test::AsTensor<int32>({10 + kDim, 10 + size});
  Tensor strides = test::AsTensor<int32>({1, 1});
  TensorShape input_shape({2 * kDim, kMaxSize});

  for (auto s : state) {
    TensorShape processing_shape, final_shape;
    bool is_identity = true, slice_dim0 = true, is_simple_slice = true;
    gtl::InlinedVector<int64_t, 4> begin_out, end_out, strides_out;
    const int32_t begin_mask = 0;
    const int32_t end_mask = 0;
    const int32_t ellipsis_mask = 0;
    const int32_t new_axis_mask = 0;
    const int32_t shrink_axis_mask = 0;

    TF_CHECK_OK(ValidateStridedSliceOp(
        &begin, &end, strides, input_shape, begin_mask, end_mask, ellipsis_mask,
        new_axis_mask, shrink_axis_mask, &processing_shape, &final_shape,
        &is_identity, &is_simple_slice, &slice_dim0, &begin_out, &end_out,
        &strides_out));
  }
}

BENCHMARK(BM_ValidateStridedSliceOp);

}  // namespace
}  // namespace tensorflow
