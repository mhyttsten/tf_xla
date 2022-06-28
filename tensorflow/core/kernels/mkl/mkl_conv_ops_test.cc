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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// TODO(intel-tf): Add numerical tests that will compare results of default
// (aka Eigen) convolutions with MKL convolutions.

// -------------------------------------------------------------------------- //
// Performance Benchmarks.                                                    //
// -------------------------------------------------------------------------- //

// Compare performance of default Tensorflow convolution kernels (Eigen) with
// MKL kernels on CPU.

// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}

namespace tensorflow {

struct Conv2DDimensions {
  Conv2DDimensions(int n, int h, int w, int c, int fc, int fh, int fw)
      : input_batches(n),
        input_height(h),
        input_width(w),
        input_depth(c),
        filter_count(fc),
        filter_height(fh),
        filter_width(fw) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "Conv2DDimensions");
}

  int input_batches;
  int input_height;
  int input_width;
  int input_depth;
  int filter_count;
  int filter_height;
  int filter_width;
};

static Tensor GetRandomTensor(const TensorShape& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetRandomTensor");

  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setRandom();
  return tensor;
}

// Get a random Tensor for the Conv2D input.
static Tensor GetRandomInputTensor(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetRandomInputTensor");

  return GetRandomTensor({dims.input_batches, dims.input_height,
                          dims.input_width, dims.input_depth});
}

// Get a random Tensor for the Conv2D filter.
static Tensor GetRandomFilterTensor(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetRandomFilterTensor");

  return GetRandomTensor({dims.filter_height, dims.filter_width,
                          dims.input_depth, dims.filter_count});
}

// Get a random Tensor for the Conv2D output (assuming SAME padding).
static Tensor GetRandomOutputTensor(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetRandomOutputTensor");

  return GetRandomTensor({dims.input_batches, dims.input_height,
                          dims.input_width, dims.filter_count});
}

// Get a Tensor encoding Conv2D input shape.
static Tensor GetInputSizesTensor(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetInputSizesTensor");

  return test::AsTensor<int32>({dims.input_batches, dims.input_height,
                                dims.input_width, dims.input_depth});
}

// Get a Tensor encoding Conv2D filter shape.
static Tensor GetFilterSizesTensor(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "GetFilterSizesTensor");

  return test::AsTensor<int32>({dims.filter_height, dims.filter_width,
                                dims.input_depth, dims.filter_count});
}

static Graph* DefaultConv2D(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_7(mht_7_v, 292, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "DefaultConv2D");

  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d"), "Conv2D")
                  .Input(input)
                  .Input(filter)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Finalize(graph, &conv2d));

  return graph;
}

static Graph* MklConv2D(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "MklConv2D");

  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("mkl_conv_2d"), "_MklConv2D")
                  .Input(input)
                  .Input(filter)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d));

  return graph;
}

static Graph* DefaultConv2DBwdInput(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_9(mht_9_v, 346, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "DefaultConv2DBwdInput");

  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_sizes_t = GetInputSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input_sizes =
      test::graph::Constant(graph, input_sizes_t, "input_sizes");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* conv2d_bwd_input;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_2d_bwd_input"), "Conv2DBackpropInput")
          .Input(input_sizes)
          .Input(filter)
          .Input(out_backprop)
          .Attr("T", DT_FLOAT)
          .Attr("strides", {1, 1, 1, 1})
          .Attr("padding", "SAME")
          .Finalize(graph, &conv2d_bwd_input));

  return graph;
}

static Graph* MklConv2DBwdInput(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_10(mht_10_v, 376, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "MklConv2DBwdInput");

  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_sizes_t = GetInputSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input_sizes =
      test::graph::Constant(graph, input_sizes_t, "input_sizes");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d_bwd_input;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d_bwd_input"),
                          "_MklConv2DBackpropInput")
                  .Input(input_sizes)
                  .Input(filter)
                  .Input(out_backprop)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d_bwd_input));

  return graph;
}

static Graph* DefaultConv2DBwdFilter(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_11(mht_11_v, 413, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "DefaultConv2DBwdFilter");

  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_sizes_t = GetFilterSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter_sizes =
      test::graph::Constant(graph, filter_sizes_t, "filter_sizes");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* conv2d_bwd_filter;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_2d_bwd_filter"), "Conv2DBackpropFilter")
          .Input(input)
          .Input(filter_sizes)
          .Input(out_backprop)
          .Attr("T", DT_FLOAT)
          .Attr("strides", {1, 1, 1, 1})
          .Attr("padding", "SAME")
          .Finalize(graph, &conv2d_bwd_filter));

  return graph;
}

static Graph* MklConv2DBwdFilter(const Conv2DDimensions& dims) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_conv_ops_testDTcc mht_12(mht_12_v, 444, "", "./tensorflow/core/kernels/mkl/mkl_conv_ops_test.cc", "MklConv2DBwdFilter");

  Graph* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_sizes_t = GetFilterSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter_sizes =
      test::graph::Constant(graph, filter_sizes_t, "filter_sizes");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d_bwd_filter;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d_bwd_filter"),
                          "_MklConv2DBackpropFilter")
                  .Input(input)
                  .Input(filter_sizes)
                  .Input(out_backprop)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d_bwd_filter));

  return graph;
}

// Macro arguments names: --------------------------------------------------- //
//    N: batch size
//    H: height
//    W: width
//    C: channels
//   FC: filter count
//   FH: filter height
//   FW: filter width

#define BM_CONCAT(a, b) a##b

#define BM_NAME(p, type, N, H, W, C, FC, FH, FW) \
  BM_CONCAT(BM_##p##_##type##_in_##N##_##H##_##W##_##C, _f_##FC##_##FH##_##FW)

// Flops computation in these benchmarks are the same as in
// eigen_benchmark_cpu_test.cc.

#define BM_Conv2DT(kind, N, H, W, C, FC, FH, FW, type, LABEL)           \
  static void BM_NAME(Conv2D_##kind, type, N, H, W, C, FC, FH,          \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (N) * (H) * (W) * (FC);               \
    int64 flops_per_iter = num_computed_elements * ((C) * (FH) * (FW)); \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2D)(dims),               \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2D_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2D(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

#define BM_Conv2DBwdInputT(kind, N, H, W, C, FC, FH, FW, type, LABEL)   \
  static void BM_NAME(Conv2DBwdInput_##kind, type, N, H, W, C, FC, FH,  \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (N) * (H) * (W) * (C);                \
    int64 flops_per_iter = num_computed_elements * ((C) * (FH) * (FW)); \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2DBwdInput)(dims),       \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2DBwdInput_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2DBwdInput(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DBwdInputT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DBwdInputT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

#define BM_Conv2DBwdFilterT(kind, N, H, W, C, FC, FH, FW, type, LABEL)  \
  static void BM_NAME(Conv2DBwdFilter_##kind, type, N, H, W, C, FC, FH, \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (FH) * (FW) * (C) * (FC);             \
    int64 flops_per_iter = num_computed_elements * ((N) * (H) * (W));   \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2DBwdFilter)(dims),      \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2DBwdFilter_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2DBwdFilter(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DBwdFilterT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DBwdFilterT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

// ImageNet Convolutions ---------------------------------------------------- //

BM_Conv2D(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2D(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2D(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2D(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2D(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2D(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2D(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

BM_Conv2DBwdInput(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2DBwdInput(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2DBwdInput(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2DBwdInput(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2DBwdInput(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2DBwdInput(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2DBwdInput(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

BM_Conv2DBwdFilter(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2DBwdFilter(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2DBwdFilter(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2DBwdFilter(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2DBwdFilter(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2DBwdFilter(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2DBwdFilter(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

}  // namespace tensorflow
