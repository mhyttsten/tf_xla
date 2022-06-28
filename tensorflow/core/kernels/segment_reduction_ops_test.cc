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
class MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc() {
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
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

static void BM_UnsortedSegmentReduction(::testing::benchmark::State& state,
                                        const string& reduction, int num_rows,
                                        int num_cols, int segment_size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("reduction: \"" + reduction + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_UnsortedSegmentReduction");

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  gtl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input(DT_FLOAT, shape1);
  // input.flat<float>().setRandom();
  reduction_inputs.push_back({nullptr, &input});

  TensorShape shape2({num_rows});
  Tensor indices(DT_INT32, shape2);
  test::FillFn<int>(&indices,
                    [&segment_size](int i) -> int { return i % segment_size; });
  reduction_inputs.push_back({nullptr, &indices});

  Tensor num_segments(DT_INT32, TensorShape({}));
  num_segments.scalar<int>()() = segment_size;
  reduction_inputs.push_back({nullptr, &num_segments});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_INT32))
                  .Finalize(&reduction_node_def));
  Status status;
  std::unique_ptr<OpKernel> reduction_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     reduction_node_def, TF_GRAPH_DEF_VERSION, &status));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &reduction_inputs;
  params.op_kernel = reduction_op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(&params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  for (auto s : state) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64_t bytes_per_iter =
      static_cast<int64_t>(num_rows * num_cols * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

#define BM_UnsortedReduce(O, R, C, S)                                        \
  static void BM_##O##_##R##_##C##_##S(::testing::benchmark::State& state) { \
    BM_UnsortedSegmentReduction(state, #O, R, C, S);                         \
  }                                                                          \
  BENCHMARK(BM_##O##_##R##_##C##_##S);

#define BM_UnsortedReduce_Arg(R, C, S) \
  BM_UnsortedReduce(UnsortedSegmentSum, R, C, S);

BM_UnsortedReduce_Arg(4096, 1024, 1);
BM_UnsortedReduce_Arg(4096, 1024, 128);

template <typename Index>
static void BM_SegmentReduction(::testing::benchmark::State& state,
                                const string& reduction, Index num_rows,
                                Index num_cols, Index segment_size) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("reduction: \"" + reduction + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_1(mht_1_v, 286, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SegmentReduction");

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  gtl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input1(DT_FLOAT, shape1);
  reduction_inputs.push_back({nullptr, &input1});

  TensorShape shape2({num_rows});
  Tensor input2(DataTypeToEnum<Index>::v(), shape2);
  test::FillFn<Index>(&input2, [&num_rows, &segment_size](Index i) -> Index {
    return std::min(i / segment_size, num_rows - 1);
  });
  reduction_inputs.push_back({nullptr, &input2});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DataTypeToEnum<Index>::v()))
                  .Finalize(&reduction_node_def));
  Status status;
  std::unique_ptr<OpKernel> reduction_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     reduction_node_def, TF_GRAPH_DEF_VERSION, &status));
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &reduction_inputs;
  params.op_kernel = reduction_op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(&params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  for (auto s : state) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64_t bytes_per_iter =
      static_cast<int64_t>(num_rows * num_cols * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

#define BM_Reduce(O, R, C, S)                          \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int32( \
      ::testing::benchmark::State & state) {           \
    BM_SegmentReduction<int32>(state, #O, R, C, S);    \
  }                                                    \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int64( \
      ::testing::benchmark::State & state) {           \
    BM_SegmentReduction<int64_t>(state, #O, R, C, S);  \
  }                                                    \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int32);  \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int64);

#define BM_Reduce_Arg(R, C, S)    \
  BM_Reduce(SegmentSum, R, C, S); \
  BM_Reduce(SegmentMean, R, C, S);

BM_Reduce_Arg(64, 32, 1);
BM_Reduce_Arg(4096, 128, 1);

BM_Reduce_Arg(16, 8, 2);
BM_Reduce_Arg(64, 32, 2);
BM_Reduce_Arg(4096, 32, 2);
BM_Reduce_Arg(4096, 128, 2);

template <DataType T>
static void SparseSegmentMeanGradHelper(::testing::benchmark::State& state,
                                        float uniqueness, int size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_2(mht_2_v, 363, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "SparseSegmentMeanGradHelper");

  typedef typename EnumToDataType<T>::Type DT;
  Graph* g = new Graph(OpRegistry::Global());
  CHECK_LE(uniqueness, 1.0);
  CHECK_GT(uniqueness, 0.0);

  const int kNumIndices = size;
  Tensor indices(DT_INT32, TensorShape({kNumIndices}));
  auto indices_flat = indices.flat<int32>();
  Tensor segments(DT_INT32, TensorShape({kNumIndices}));
  auto segments_flat = segments.flat<int32>();

  int kUniqueIndices = uniqueness * kNumIndices;
  Tensor output_dim0(DT_INT32, TensorShape({}));
  output_dim0.scalar<int32>()() = kUniqueIndices;

  for (int i = 0; i < kNumIndices; ++i) {
    indices_flat(i) = (i * 31) % kUniqueIndices;
    segments_flat(i) = i * .8;
  }

  const int kDim1 = segments_flat(kNumIndices - 1) + 1;
  const int kDim2 = 128;
  Tensor input(T, TensorShape({kDim1, kDim2}));
  input.flat<DT>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseSegmentMeanGrad")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, indices))
                  .Input(test::graph::Constant(g, segments))
                  .Input(test::graph::Constant(g, output_dim0))
                  .Attr("T", T)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (kDim1 * kDim2) * sizeof(float));
}

static void BM_SparseSegmentMeanGrad_Low_FP32(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_3(mht_3_v, 407, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_Low_FP32");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_FLOAT>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_FP32(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_4(mht_4_v, 417, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_High_FP32");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_FLOAT>(state, 0.01, size);
}

static void BM_SparseSegmentMeanGrad_Low_BF16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_5(mht_5_v, 427, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_Low_BF16");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_BFLOAT16>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_BF16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_6(mht_6_v, 437, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_High_BF16");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_BFLOAT16>(state, 0.01, size);
}

static void BM_SparseSegmentMeanGrad_Low_FP16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_7(mht_7_v, 447, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_Low_FP16");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_HALF>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_FP16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsegment_reduction_ops_testDTcc mht_8(mht_8_v, 457, "", "./tensorflow/core/kernels/segment_reduction_ops_test.cc", "BM_SparseSegmentMeanGrad_High_FP16");

  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_HALF>(state, 0.01, size);
}

BENCHMARK(BM_SparseSegmentMeanGrad_Low_FP32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_FP32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_Low_BF16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_BF16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_Low_FP16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_FP16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);

}  // namespace tensorflow
