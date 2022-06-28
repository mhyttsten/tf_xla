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
class MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc() {
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

#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

////////////////////////////////////////////////////////////////////////////////
// Performance benchmarks for the Conv2DOp and FusedConv2Op.                  //
////////////////////////////////////////////////////////////////////////////////

struct Conv2DGraph {
  Graph* graph;
  Node* conv2d;
};

struct Conv2DWithBiasGraph {
  Graph* graph;
  Node* conv2d;
  Node* bias;
};

struct Conv2DWithBiasAndActivationGraph {
  Graph* graph;
  Node* conv2d;
  Node* bias;
  Node* activation;
};

struct Conv2DWithBatchNormGraph {
  Graph* graph;
  Node* conv2d;
  Node* batch_norm;
};

struct Conv2DWithBatchNormAndActivationGraph {
  Graph* graph;
  Node* conv2d;
  Node* batch_norm;
  Node* activation;
};

template <typename T>
static Tensor MakeRandomTensor(const TensorShape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "MakeRandomTensor");

  Tensor tensor(DataTypeToEnum<T>::value, TensorShape(shape));
  tensor.flat<T>() = tensor.flat<T>().setRandom();
  return tensor;
}

// Creates a simple Tensorflow graph with single Conv2D node.
template <typename T>
static Conv2DGraph Conv2D(int batch, int height, int width, int in_depth,
                          int filter_w, int filter_h, int out_depth,
                          TensorFormat data_format = FORMAT_NHWC) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_1(mht_1_v, 250, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "Conv2D");

  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = data_format == FORMAT_NHWC
                        ? MakeRandomTensor<T>({batch, height, width, in_depth})
                        : MakeRandomTensor<T>({batch, in_depth, height, width});

  // Filter is always in HWIO.
  Tensor filter_t =
      MakeRandomTensor<T>({filter_w, filter_h, in_depth, out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "Conv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("data_format", ToString(data_format))
                  .Finalize(graph, &conv2d));

  return {graph, conv2d};
}

// Creates a Tensorflow graph with a Conv2D node followed by BiasAdd.
template <typename T>
static Conv2DWithBiasGraph Conv2DWithBias(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, TensorFormat data_format = FORMAT_NHWC) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_2(mht_2_v, 284, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "Conv2DWithBias");

  Conv2DGraph conv_graph = Conv2D<T>(batch, height, width, in_depth, filter_w,
                                     filter_h, out_depth, data_format);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;

  Tensor bias_t = MakeRandomTensor<T>({out_depth});
  Node* bias = test::graph::Constant(graph, bias_t, "bias");

  Node* out;
  TF_CHECK_OK(NodeBuilder(graph->NewName("bias"), "BiasAdd")
                  .Input(conv2d)
                  .Input(bias)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Attr("data_format", ToString(data_format))
                  .Finalize(graph, &out));

  return {graph, conv2d, out};
}

// Creates a Tensorflow graph with a Conv2D node followed by BiasAdd and
// activation (Relu, Relu6, etc...).
template <typename T>
static Conv2DWithBiasAndActivationGraph Conv2DWithBiasAndActivation(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const string& activation_type,
    TensorFormat data_format = FORMAT_NHWC) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("activation_type: \"" + activation_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_3(mht_3_v, 315, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "Conv2DWithBiasAndActivation");

  Conv2DWithBiasGraph conv_graph =
      Conv2DWithBias<T>(batch, height, width, in_depth, filter_w, filter_h,
                        out_depth, data_format);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;
  Node* bias = conv_graph.bias;

  Node* activation;
  TF_CHECK_OK(NodeBuilder(graph->NewName("activation"), activation_type)
                  .Input(bias)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Finalize(graph, &activation));

  return {graph, conv2d, bias, activation};
}

// Creates a Tensorflow graph with a Conv2D node followed by FusedBatchNorm.
template <typename T>
static Conv2DWithBatchNormGraph Conv2DWithBatchNorm(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, TensorFormat data_format = FORMAT_NHWC) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_4(mht_4_v, 340, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "Conv2DWithBatchNorm");

  Conv2DGraph conv_graph = Conv2D<T>(batch, height, width, in_depth, filter_w,
                                     filter_h, out_depth, data_format);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;

  Tensor scale_t = MakeRandomTensor<T>({out_depth});
  Tensor offset_t = MakeRandomTensor<T>({out_depth});
  Tensor mean_t = MakeRandomTensor<T>({out_depth});
  Tensor variance_t = MakeRandomTensor<T>({out_depth});

  Node* scale = test::graph::Constant(graph, scale_t, "scale");
  Node* offset = test::graph::Constant(graph, offset_t, "offset");
  Node* mean = test::graph::Constant(graph, mean_t, "mean");
  Node* variance = test::graph::Constant(graph, variance_t, "variance");

  Node* out;
  TF_CHECK_OK(NodeBuilder(graph->NewName("batch_norm"), "FusedBatchNorm")
                  .Input(conv2d)
                  .Input(scale)
                  .Input(offset)
                  .Input(mean)
                  .Input(variance)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Attr("is_training", false)
                  .Attr("data_format", ToString(data_format))
                  .Finalize(graph, &out));

  return {graph, conv2d, out};
}

// Creates a Tensorflow graph with a Conv2D node followed by FusedBatchNorm and
// activation (Relu, Relu6, etc...).
template <typename T>
static Conv2DWithBatchNormAndActivationGraph Conv2DWithBatchNormAndActivation(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const string& activation_type,
    TensorFormat data_format = FORMAT_NHWC) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("activation_type: \"" + activation_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSconv_ops_benchmark_testDTcc mht_5(mht_5_v, 382, "", "./tensorflow/core/kernels/conv_ops_benchmark_test.cc", "Conv2DWithBatchNormAndActivation");

  Conv2DWithBatchNormGraph conv_graph =
      Conv2DWithBatchNorm<T>(batch, height, width, in_depth, filter_w, filter_h,
                             out_depth, data_format);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;
  Node* batch_norm = conv_graph.batch_norm;

  Node* activation;
  TF_CHECK_OK(NodeBuilder(graph->NewName("activation"), activation_type)
                  .Input(batch_norm)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Finalize(graph, &activation));

  return {graph, conv2d, batch_norm, activation};
}

// Creates a tensorflow graph with a single FusedConv2D (with BiasAdd) node and
// fuses into it additional computations (e.g. Relu).
template <typename T>
static Graph* FusedConv2DWithBias(int batch, int height, int width,
                                  int in_depth, int filter_w, int filter_h,
                                  int out_depth,
                                  const std::vector<string>& fused_ops = {},
                                  TensorFormat data_format = FORMAT_NHWC) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = data_format == FORMAT_NHWC
                        ? MakeRandomTensor<T>({batch, height, width, in_depth})
                        : MakeRandomTensor<T>({batch, in_depth, height, width});

  // Filter is always in HWIO.
  Tensor filter_t =
      MakeRandomTensor<T>({filter_w, filter_h, in_depth, out_depth});
  Tensor bias_t = MakeRandomTensor<T>({out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* bias = test::graph::Constant(graph, bias_t, "bias");

  std::vector<NodeBuilder::NodeOut> args = {bias};

  Node* conv;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "_FusedConv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("num_args", 1)
                  .Input(args)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("fused_ops", fused_ops)
                  .Finalize(graph, &conv));

  return graph;
}

// Creates a tensorflow graph with a single FusedConv2D (with FusedBatchNorm)
// node and fuses into it additional computations (e.g. Relu).
template <typename T>
static Graph* FusedConv2DWithBatchNorm(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const std::vector<string>& fused_ops = {},
    TensorFormat data_format = FORMAT_NHWC) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = data_format == FORMAT_NHWC
                        ? MakeRandomTensor<T>({batch, height, width, in_depth})
                        : MakeRandomTensor<T>({batch, in_depth, height, width});

  // Filter is always in HWIO.
  Tensor filter_t =
      MakeRandomTensor<T>({filter_w, filter_h, in_depth, out_depth});
  Tensor scale_t = MakeRandomTensor<T>({out_depth});
  Tensor offset_t = MakeRandomTensor<T>({out_depth});
  Tensor mean_t = MakeRandomTensor<T>({out_depth});
  Tensor variance_t = MakeRandomTensor<T>({out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* scale = test::graph::Constant(graph, scale_t, "scale");
  Node* offset = test::graph::Constant(graph, offset_t, "offset");
  Node* mean = test::graph::Constant(graph, mean_t, "mean");
  Node* variance = test::graph::Constant(graph, variance_t, "variance");

  std::vector<NodeBuilder::NodeOut> args = {scale, offset, mean, variance};

  Node* conv;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "_FusedConv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("num_args", 4)
                  .Input(args)
                  .Attr("T", DataTypeToEnum<T>::value)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("fused_ops", fused_ops)
                  .Finalize(graph, &conv));

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

// -------------------------------------------------------------------------- //
// The following benchmarks are always using 'float' data type with NHWC layout.
// -------------------------------------------------------------------------- //

#define BM_SET_INFO(N, H, W, C, type, LABEL, NAME)                         \
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * (N) * \
                          (H) * (W) * (C));                                \
  state.SetLabel(LABEL);

#define BM_NAME(name, type, N, H, W, C, FW, FH, FC) \
  name##_##type##_##N##_##H##_##W##_##C##_##FW##_##FH##_##FC

#define BM_Conv2D(N, H, W, C, FW, FH, FC, type, LABEL)                  \
  static void BM_NAME(BM_Conv2D, type, N, H, W, C, FW, FH,              \
                      FC)(::testing::benchmark::State & state) {        \
    test::Benchmark(#type, Conv2D<float>(N, H, W, C, FW, FH, FC).graph, \
                    /*old_benchmark_api=*/false)                        \
        .Run(state);                                                    \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                       \
  }                                                                     \
  BENCHMARK(BM_NAME(BM_Conv2D, type, N, H, W, C, FW, FH, FC))           \
      ->Arg(/*unused arg*/ 1);

#define BM_Conv2DWithBias(N, H, W, C, FW, FH, FC, type, LABEL)           \
  static void BM_NAME(BM_Conv2DWithBias, type, N, H, W, C, FW, FH,       \
                      FC)(::testing::benchmark::State & state) {         \
    test::Benchmark(#type,                                               \
                    Conv2DWithBias<float>(N, H, W, C, FW, FH, FC).graph, \
                    /*old_benchmark_api=*/false)                         \
        .Run(state);                                                     \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                        \
  }                                                                      \
  BENCHMARK(BM_NAME(BM_Conv2DWithBias, type, N, H, W, C, FW, FH, FC))    \
      ->Arg(/*unused arg*/ 1);

#define BM_Conv2DWithBiasAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)        \
  static void BM_NAME(BM_Conv2DWithBiasAndRelu, type, N, H, W, C, FW, FH,    \
                      FC)(::testing::benchmark::State & state) {             \
    test::Benchmark(                                                         \
        #type,                                                               \
        Conv2DWithBiasAndActivation<float>(N, H, W, C, FW, FH, FC, "Relu")   \
            .graph,                                                          \
        /*old_benchmark_api=*/false)                                         \
        .Run(state);                                                         \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                            \
  }                                                                          \
  BENCHMARK(BM_NAME(BM_Conv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, FC)) \
      ->Arg(/*unused arg*/ 1);

#define BM_FusedConv2DWithBias(N, H, W, C, FW, FH, FC, type, LABEL)        \
  static void BM_NAME(BM_FusedConv2DWithBias, type, N, H, W, C, FW, FH,    \
                      FC)(::testing::benchmark::State & state) {           \
    test::Benchmark(                                                       \
        #type,                                                             \
        FusedConv2DWithBias<float>(N, H, W, C, FW, FH, FC, {"BiasAdd"}),   \
        /*old_benchmark_api=*/false)                                       \
        .Run(state);                                                       \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                          \
  }                                                                        \
  BENCHMARK(BM_NAME(BM_FusedConv2DWithBias, type, N, H, W, C, FW, FH, FC)) \
      ->Arg(/*unused arg*/ 1);

#define BM_FusedConv2DWithBiasAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_FusedConv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, \
                      FC)(::testing::benchmark::State & state) {               \
    test::Benchmark(#type,                                                     \
                    FusedConv2DWithBias<float>(N, H, W, C, FW, FH, FC,         \
                                               {"BiasAdd", "Relu"}),           \
                    /*old_benchmark_api=*/false)                               \
        .Run(state);                                                           \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                              \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(BM_FusedConv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, FC))    \
      ->Arg(/*unused arg*/ 1);

#define BM_Conv2DWithBatchNorm(N, H, W, C, FW, FH, FC, type, LABEL)           \
  static void BM_NAME(BM_Conv2DWithBatchNorm, type, N, H, W, C, FW, FH,       \
                      FC)(::testing::benchmark::State & state) {              \
    test::Benchmark(#type,                                                    \
                    Conv2DWithBatchNorm<float>(N, H, W, C, FW, FH, FC).graph, \
                    /*old_benchmark_api=*/false)                              \
        .Run(state);                                                          \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                             \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_Conv2DWithBatchNorm, type, N, H, W, C, FW, FH, FC))    \
      ->Arg(/*unused arg*/ 1);

#define BM_Conv2DWithBatchNormAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_Conv2DWithBatchNormAndRelu, type, N, H, W, C, FW, FH, \
                      FC)(::testing::benchmark::State & state) {               \
    test::Benchmark(#type,                                                     \
                    Conv2DWithBatchNormAndActivation<float>(N, H, W, C, FW,    \
                                                            FH, FC, "Relu")    \
                        .graph,                                                \
                    /*old_benchmark_api=*/false)                               \
        .Run(state);                                                           \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                              \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(BM_Conv2DWithBatchNormAndRelu, type, N, H, W, C, FW, FH, FC))    \
      ->Arg(/*unused arg*/ 1);

#define BM_FusedConv2DWithBatchNorm(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_FusedConv2DWithBatchNorm, type, N, H, W, C, FW, FH, \
                      FC)(::testing::benchmark::State & state) {             \
    test::Benchmark(#type,                                                   \
                    FusedConv2DWithBatchNorm<float>(N, H, W, C, FW, FH, FC,  \
                                                    {"FusedBatchNorm"}),     \
                    /*old_benchmark_api=*/false)                             \
        .Run(state);                                                         \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                            \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_NAME(BM_FusedConv2DWithBatchNorm, type, N, H, W, C, FW, FH, FC))    \
      ->Arg(/*unused arg*/ 1);

#define BM_FusedConv2DWithBatchNormAndRelu(N, H, W, C, FW, FH, FC, type,      \
                                           LABEL)                             \
  static void BM_NAME(BM_FusedConv2DWithBatchNormAndRelu, type, N, H, W, C,   \
                      FW, FH, FC)(::testing::benchmark::State & state) {      \
    test::Benchmark(#type,                                                    \
                    FusedConv2DWithBatchNorm<float>(                          \
                        N, H, W, C, FW, FH, FC, {"FusedBatchNorm", "Relu"}),  \
                    /*old_benchmark_api=*/false)                              \
        .Run(state);                                                          \
    BM_SET_INFO(N, H, W, C, type, LABEL, Conv2D);                             \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_FusedConv2DWithBatchNormAndRelu, type, N, H, W, C, FW, \
                    FH, FC))                                                  \
      ->Arg(/*unused arg*/ 1);

// -------------------------------------------------------------------------- //
// Pixel CNN convolutions.
// -------------------------------------------------------------------------- //

// 1x1 Convolution: MatMulFunctor

BM_Conv2D(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2D(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2D(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

// 1) BiasAdd {+ Relu}

BM_Conv2DWithBias(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBias(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBias(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBias(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBias(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBias(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

// 2) FusedBatchNorm {+ Relu}

BM_Conv2DWithBatchNorm(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNorm(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNorm(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBatchNormAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNormAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNormAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNorm(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNorm(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBatchNorm(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNormAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNormAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu,
                                   "1x1 /b 16");
BM_FusedConv2DWithBatchNormAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu,
                                   "1x1 /b 32");

// -------------------------------------------------------------------------- //
// 3x3 Convolution: SpatialConvolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2D(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2D(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

// 1) BiasAdd {+ Relu}

BM_Conv2DWithBias(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBias(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBias(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBias(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBias(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_FusedConv2DWithBias(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

// 2) FusedBatchNorm {+ Relu}

BM_Conv2DWithBatchNorm(8, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNorm(16, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNorm(32, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBatchNormAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBatchNormAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBatchNormAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBatchNorm(8, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNorm(16, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBatchNorm(32, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNormAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBatchNormAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu,
                                   "3x3 /b 16");
BM_FusedConv2DWithBatchNormAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu,
                                   "3x3 /b 32");

#if GOOGLE_CUDA
// -------------------------------------------------------------------------- //
// 1x1 Convolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_Conv2D(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_Conv2D(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

// -------------------------------------------------------------------------- //
// 3x3 Convolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_Conv2D(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_Conv2D(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");
#endif

// Macro arguments names: --------------------------------------------------- //
//      T: data type
// FORMAT: data format (NHWC or NCHW)
//      N: batch size
//      H: height
//      W: width
//      C: channels
//     FC: filter count
//     FH: filter height
//     FW: filter width

// -------------------------------------------------------------------------- //
// The following benchmarks are used to compare different data format
// performance for different data types. They make sense only when CUDA enabled,
// because on CPU we only support data in NHWC.
// -------------------------------------------------------------------------- //

#define BM_LONG_NAME(name, type, T, FORMAT, N, H, W, C, FW, FH, FC) \
  name##_##T##_##FORMAT##_##type##_##N##_##H##_##W##_##C##_##FW##_##FH##_##FC

#define BM_Conv2DFmt(T, FORMAT, N, H, W, C, FW, FH, FC, type)                 \
  static void BM_LONG_NAME(BM_Conv2D, type, T, FORMAT, N, H, W, C, FW, FH,    \
                           FC)(::testing::benchmark::State & state) {         \
    test::Benchmark(#type,                                                    \
                    Conv2D<T>(N, H, W, C, FW, FH, FC, FORMAT_##FORMAT).graph, \
                    /*old_benchmark_api=*/false)                              \
        .Run(state);                                                          \
    BM_SET_INFO(N, H, W, C, type, "", Conv2D);                                \
  }                                                                           \
  BENCHMARK(BM_LONG_NAME(BM_Conv2D, type, T, FORMAT, N, H, W, C, FW, FH, FC)) \
      ->Arg(/*unused arg*/ 1);

#if GOOGLE_CUDA
using fp32 = float;
using fp16 = Eigen::half;

// ResNet50-ish convolutions.
#define BENCHMARK_RESNET50(DATA_FORMAT, BATCH, T)                    \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 56, 56, 64, 1, 1, 64, gpu);    \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 56, 56, 64, 1, 1, 256, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 56, 56, 256, 1, 1, 64, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 56, 56, 64, 3, 3, 64, gpu);    \
                                                                     \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 28, 28, 128, 1, 1, 128, gpu);  \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 28, 28, 128, 1, 1, 512, gpu);  \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 28, 28, 512, 1, 1, 128, gpu);  \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 28, 28, 512, 3, 3, 128, gpu);  \
                                                                     \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 14, 14, 256, 1, 1, 256, gpu);  \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 14, 14, 256, 1, 1, 1024, gpu); \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 14, 14, 1024, 1, 1, 256, gpu); \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 14, 14, 256, 3, 3, 256, gpu)

// NASnet-ish convolutions (Tensorflow models: slim/nets/nasnet).
#define BENCHMARK_NASNET(DATA_FORMAT, BATCH, T)                       \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 165, 165, 96, 1, 1, 96, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 83, 83, 84, 1, 1, 84, gpu);     \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 42, 42, 336, 1, 1, 336, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 42, 42, 168, 1, 1, 168, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 21, 21, 1008, 1, 1, 1008, gpu); \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 21, 21, 336, 1, 1, 336, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 11, 11, 672, 1, 1, 672, gpu);   \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 21, 21, 2016, 1, 1, 2016, gpu); \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 11, 11, 2688, 1, 1, 2688, gpu); \
  BM_Conv2DFmt(T, DATA_FORMAT, BATCH, 11, 11, 4032, 1, 1, 4032, gpu)

#define BENCHMARK_DTYPE(DATA_FORMAT, BATCH, T) \
  BENCHMARK_RESNET50(DATA_FORMAT, BATCH, T);   \
  BENCHMARK_NASNET(DATA_FORMAT, BATCH, T)

BENCHMARK_DTYPE(NHWC, 16, fp32);
BENCHMARK_DTYPE(NCHW, 16, fp32);

BENCHMARK_DTYPE(NHWC, 16, fp16);
BENCHMARK_DTYPE(NCHW, 16, fp16);

BENCHMARK_DTYPE(NHWC, 32, fp32);
BENCHMARK_DTYPE(NCHW, 32, fp32);

BENCHMARK_DTYPE(NHWC, 32, fp16);
BENCHMARK_DTYPE(NCHW, 32, fp16);

BENCHMARK_DTYPE(NHWC, 64, fp32);
BENCHMARK_DTYPE(NCHW, 64, fp32);

BENCHMARK_DTYPE(NHWC, 64, fp16);
BENCHMARK_DTYPE(NCHW, 64, fp16);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
