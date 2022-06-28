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
class MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc() {
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

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

template <typename T>
class FusedMatMulOpTest : public OpsTestBase {
 protected:
  using BiasAddGraphRunner =
      std::function<void(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fetch: \"" + fetch + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/matmul_op_test.cc", "RunAndFetch");

    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunMatMulWithBias(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, bool transpose_a,
                         bool transpose_b, Tensor* output,
                         bool allow_gpu_device = false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_1(mht_1_v, 276, "", "./tensorflow/core/kernels/matmul_op_test.cc", "RunMatMulWithBias");

    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    RunAndFetch(root, "with_bias", output, allow_gpu_device);
  }

  void RunMatMulWithBiasAndActivation(
      const Tensor& lhs_data, const Tensor& rhs_data, const Tensor& bias_data,
      bool transpose_a, bool transpose_b, const string& activation_type,
      Tensor* output, bool allow_gpu_device = false) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("activation_type: \"" + activation_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/kernels/matmul_op_test.cc", "RunMatMulWithBiasAndActivation");

    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    if (activation_type == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Relu6") {
      ops::Relu6(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Elu") {
      ops::Elu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "LeakyRelu") {
      ops::internal::LeakyRelu(root.WithOpName("with_activation"), with_bias);
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedMatMulOp(const Tensor& lhs_data, const Tensor& rhs_data,
                        const std::vector<Tensor>& args_data,
                        const std::vector<string>& fused_ops, bool transpose_a,
                        bool transpose_b, Tensor* output,
                        bool allow_gpu_device = false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_3(mht_3_v, 334, "", "./tensorflow/core/kernels/matmul_op_test.cc", "RunFusedMatMulOp");

    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args_data.size());

    Output lhs =
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data));
    Output rhs =
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data));

    std::vector<NodeDefBuilder::NodeOut> args;
    for (int i = 0; i < num_args; ++i) {
      Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                              Input::Initializer(args_data[i]));
      args.emplace_back(arg.name(), 0, dtype);
    }

    NodeDef fused_matmul;
    TF_EXPECT_OK(NodeDefBuilder("fused_matmul", "_FusedMatMul")
                     .Input({lhs.name(), 0, dtype})
                     .Input({rhs.name(), 0, dtype})
                     .Input(args)
                     .Attr("num_args", num_args)
                     .Attr("T", dtype)
                     .Attr("fused_ops", fused_ops)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Finalize(&fused_matmul));

    RunAndFetch(root, fused_matmul.name(), output, allow_gpu_device,
                &fused_matmul);
  }

  void VerifyBiasAddTensorsNear(int m, int k, int n,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_4(mht_4_v, 373, "", "./tensorflow/core/kernels/matmul_op_test.cc", "VerifyBiasAddTensorsNear");

    DataType dtype = DataTypeToEnum<T>::v();

    Tensor lhs(dtype, {m, k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor rhs(dtype, {k, n});
    rhs.flat<T>() = rhs.flat<T>().setRandom();
    rhs.flat<T>() -= rhs.flat<T>().constant(static_cast<T>(0.5f));

    // Bias added to the inner dimension.
    const int bias_size = n;
    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().setRandom();
    bias.flat<T>() += bias.flat<T>().constant(static_cast<T>(0.5f));

    Tensor matmul;
    Tensor fused_matmul;

    run_default(lhs, rhs, bias, &matmul);
    run_fused(lhs, rhs, bias, &fused_matmul);

    ASSERT_EQ(matmul.dtype(), fused_matmul.dtype());
    ASSERT_EQ(matmul.shape(), fused_matmul.shape());

    test::ExpectClose(matmul, fused_matmul, /*atol=*/1e-5);
  }

  // Verifies that computing MatMul+BiasAdd in a graph is identical to
  // FusedMatMul.
  void VerifyMatMulWithBias(int m, int k, int n, bool transpose_a,
                            bool transpose_b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_5(mht_5_v, 408, "", "./tensorflow/core/kernels/matmul_op_test.cc", "VerifyMatMulWithBias");

    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_6(mht_6_v, 414, "", "./tensorflow/core/kernels/matmul_op_test.cc", "lambda");

          RunMatMulWithBias(input_data, filter_data, bias_data, transpose_a,
                            transpose_b, out);
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_7(mht_7_v, 424, "", "./tensorflow/core/kernels/matmul_op_test.cc", "lambda");

          RunFusedMatMulOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           transpose_a, transpose_b, out);
        };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
  }

  // Verifies that computing MatMul+BiasAdd+{Activation} in a graph is identical
  // to FusedMatMul.
  void VerifyConv2DWithBiasAndActivation(int m, int k, int n, bool transpose_a,
                                         bool transpose_b,
                                         const string& activation) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("activation: \"" + activation + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_8(mht_8_v, 440, "", "./tensorflow/core/kernels/matmul_op_test.cc", "VerifyConv2DWithBiasAndActivation");

    const BiasAddGraphRunner run_default = [&](const Tensor& input_data,
                                               const Tensor& filter_data,
                                               const Tensor& bias_data,
                                               Tensor* out) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_9(mht_9_v, 447, "", "./tensorflow/core/kernels/matmul_op_test.cc", "lambda");

      RunMatMulWithBiasAndActivation(input_data, filter_data, bias_data,
                                     transpose_a, transpose_b, activation, out);
    };

    const BiasAddGraphRunner run_fused = [&](const Tensor& input_data,
                                             const Tensor& filter_data,
                                             const Tensor& bias_data,
                                             Tensor* out) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_10(mht_10_v, 458, "", "./tensorflow/core/kernels/matmul_op_test.cc", "lambda");

      RunFusedMatMulOp(input_data, filter_data, {bias_data},
                       {"BiasAdd", activation}, transpose_a, transpose_b, out);
    };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
  }
};

// MatMul with BatchNorm can be tested only with `T=float`, because default
// `FusedBatchNorm` kernel supports only floats for scale, mean and variance.

template <typename T>
class FusedMatMulWithBiasOpTest : public FusedMatMulOpTest<T> {};

TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest);

// -------------------------------------------------------------------------- //
// MatMul + BiasAdd + {Activation}                                            //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256) {
  this->VerifyMatMulWithBias(256, 256, 256, false, false);
  this->VerifyMatMulWithBias(256, 256, 256, true, false);
  this->VerifyMatMulWithBias(256, 256, 256, false, true);
  this->VerifyMatMulWithBias(256, 256, 256, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256) {
  this->VerifyMatMulWithBias(1, 256, 256, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1) {
  this->VerifyMatMulWithBias(256, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1) {
  this->VerifyMatMulWithBias(1, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, true,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, true,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 256, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 1, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu", "LeakyRelu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 1, false, false,
                                            activation);
  }
}

REGISTER_TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest,        //
                            MatMul256x256x256,                //
                            MatMul1x256x256,                  //
                            MatMul256x256x1,                  //
                            MatMul1x256x1,                    //
                            MatMul256x256x256WithActivation,  //
                            MatMul1x256x256WithActivation,    //
                            MatMul256x256x1WithActivation,    //
                            MatMul1x256x1WithActivation);

// TODO(ezhulenev): Add support for more data types.
using FusedBiasAddDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedMatMulWithBiasOpTest,
                               FusedBiasAddDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_11(mht_11_v, 556, "", "./tensorflow/core/kernels/matmul_op_test.cc", "Matmul");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(state);   \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2);               \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE)   \
      ->MeasureProcessCPUTime();

#ifdef GOOGLE_CUDA

#define BM_Matmul(M, K, N, TA, TB)                                       \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, gpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \
  /* Uncomment to enable benchmarks for double/complex128: */            \
  // BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu); \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

#else

#define BM_Matmul(M, K, N, TA, TB)                     \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu);

#endif  // GOOGLE_CUDA

// LINT.IfChange

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);

// LINT.ThenChange(//tensorflow/core/kernels/mkl/mkl_matmul_op_benchmark.cc)

// Benchmarks for batched matmul with broadcasting.
Node* BroadcastTo(Graph* g, Node* input, Node* shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_12(mht_12_v, 679, "", "./tensorflow/core/kernels/matmul_op_test.cc", "BroadcastTo");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(input)
                  .Input(shape)
                  .Finalize(g, &ret));
  return ret;
}

Node* BatchMatmulV2(Graph* g, Node* in0, Node* in1, bool adj_x, bool adj_y) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_13(mht_13_v, 691, "", "./tensorflow/core/kernels/matmul_op_test.cc", "BatchMatmulV2");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BatchMatMulV2")
                  .Input(in0)
                  .Input(in1)
                  .Attr("adj_x", adj_x)
                  .Attr("adj_y", adj_y)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* BatchMatmul(int b, int m, int k, int n, bool adjoint_a,
                          bool adjoint_b, DataType type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_14(mht_14_v, 707, "", "./tensorflow/core/kernels/matmul_op_test.cc", "BatchMatmul");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, adjoint_a ? TensorShape({b, k, m}) : TensorShape({b, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, adjoint_b ? TensorShape({b, n, k}) : TensorShape({b, k, n}));
  in1.flat<T>().setRandom();
  test::graph::BatchMatmul(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, in1), adjoint_a, adjoint_b);
  return g;
}

template <typename T>
static Graph* BatchMatmulWithBroadcast(int b0, int b1, int m, int k, int n,
                                       bool manual_broadcast, DataType type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_testDTcc mht_15(mht_15_v, 723, "", "./tensorflow/core/kernels/matmul_op_test.cc", "BatchMatmulWithBroadcast");

  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, TensorShape({b0, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, TensorShape({b1, k, n}));
  in1.flat<T>().setRandom();

  Tensor broadcasted_in0_shape(DT_INT64, TensorShape({3}));
  Tensor broadcasted_in1_shape(DT_INT64, TensorShape({3}));

  Node* in0_node = nullptr;
  Node* in1_node = nullptr;
  if (manual_broadcast) {
    for (int i = 0; i < 3; ++i) {
      auto vec0 = broadcasted_in0_shape.vec<int64_t>();
      auto vec1 = broadcasted_in1_shape.vec<int64_t>();
      vec0(i) = (i == 0 ? std::max(b0, b1) : in0.shape().dim_size(i));
      vec1(i) = (i == 0 ? std::max(b0, b1) : in1.shape().dim_size(i));
    }
    in0_node = BroadcastTo(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, broadcasted_in0_shape));
    in1_node = BroadcastTo(g, test::graph::Constant(g, in1),
                           test::graph::Constant(g, broadcasted_in1_shape));
  } else {
    in0_node = test::graph::Constant(g, in0);
    in1_node = test::graph::Constant(g, in1);
  }

  BatchMatmulV2(g, in0_node, in1_node, false, false);
  return g;
}

// NOLINTBEGIN
// Function names are already longer than 80 chars.
#define BM_BatchMatmulDev(B, M, K, N, TA, TB, T, TFTYPE, DEVICE)                  \
  static void                                                                     \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
          ::testing::benchmark::State& state) {                                   \
    test::Benchmark(#DEVICE, BatchMatmul<T>(B, M, K, N, TA, TB, TFTYPE),          \
                    /*old_benchmark_api*/ false)                                  \
        .Run(state);                                                              \
    state.SetItemsProcessed(state.iterations() * B * M * K * N * 2);              \
  }                                                                               \
  BENCHMARK(                                                                      \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE) \
      ->MeasureProcessCPUTime();
// NOLINTEND

#define BM_BatchMatmul(B, M, K, N, TA, TB) \
  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, cpu);
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// cpu);
//  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, gpu);
/* Uncomment to enable benchmarks for double & complex types: */
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// gpu);
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu); \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu);
// \
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu); \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

// Macro arguments names: --------------------------------------------------- //
//   B1: batch size of LHS
//   B2: batch size of RHS
//    M: outer dimension of LHS
//    K: inner dimensions of LHS and RHS
//    N: outer dimension of RHS
//   MB: boolean indicating whether to use manual broadcasting
//    T: C++ type of scalars (e.g. float, std::complex)
//   TT: TensorFlow type of scalars (e.g. DT_FLOAT, DT_COMPLEX128
//    D: Device (e.g. cpu, gpu)
#define BM_BatchMatmulBCastDev(B1, B2, M, K, N, MB, T, TT, D)                  \
  static void                                                                  \
      BM_BatchMatmulBCast##_##B1##_##B2##_##M##_##K##_##N##_##MB##_##TT##_##D( \
          ::testing::benchmark::State& state) {                                \
    test::Benchmark(#D, BatchMatmulWithBroadcast<T>(B1, B2, M, K, N, MB, TT),  \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(state.iterations() * std::max(B1, B2) * M * K *    \
                            N * 2);                                            \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_BatchMatmulBCast##_##B1##_##B2##_##M##_##K##_##N##_##MB##_##TT##_##D) \
      ->MeasureProcessCPUTime();

#define BM_BatchMatmulBCast(B1, B2, M, K, N, MB) \
  BM_BatchMatmulBCastDev(B1, B2, M, K, N, MB, float, DT_FLOAT, cpu);

// Typical fully connected layers
BM_BatchMatmulBCast(1, 128, 1, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 1, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 1, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 1, 1024, 1024, false);
BM_BatchMatmulBCast(1, 128, 128, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 128, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 128, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 128, 1024, 1024, false);

// Square matmul.
BM_BatchMatmulBCast(1, 128, 512, 512, 512, true);
BM_BatchMatmulBCast(1, 128, 512, 512, 512, false);
BM_BatchMatmulBCast(128, 1, 512, 512, 512, true);
BM_BatchMatmulBCast(128, 1, 512, 512, 512, false);
BM_BatchMatmulBCast(1, 128, 1024, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 1024, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 1024, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 1024, 1024, 1024, false);

// Matrix-vector multiplies.
BM_BatchMatmulBCast(1, 128, 10000, 200, 1, true);
BM_BatchMatmulBCast(1, 128, 10000, 200, 1, false);
BM_BatchMatmulBCast(128, 1, 10000, 200, 1, true);
BM_BatchMatmulBCast(128, 1, 10000, 200, 1, false);

// Vector-matrix multiplies.
BM_BatchMatmulBCast(1, 128, 1, 200, 10000, true);
BM_BatchMatmulBCast(1, 128, 1, 200, 10000, false);
BM_BatchMatmulBCast(128, 1, 1, 200, 10000, true);
BM_BatchMatmulBCast(128, 1, 1, 200, 10000, false);

// Typical fully connected layers
BM_BatchMatmul(1, 1, 1024, 1024, false, false);
BM_BatchMatmul(1, 8, 1024, 1024, false, false);
BM_BatchMatmul(1, 16, 1024, 1024, false, false);
BM_BatchMatmul(1, 128, 1024, 1024, false, false);
BM_BatchMatmul(2, 1, 1024, 1024, false, false);
BM_BatchMatmul(2, 8, 1024, 1024, false, false);
BM_BatchMatmul(2, 16, 1024, 1024, false, false);
BM_BatchMatmul(2, 128, 1024, 1024, false, false);
BM_BatchMatmul(8, 1, 1024, 1024, false, false);
BM_BatchMatmul(8, 8, 1024, 1024, false, false);
BM_BatchMatmul(8, 16, 1024, 1024, false, false);
BM_BatchMatmul(8, 128, 1024, 1024, false, false);
BM_BatchMatmul(32, 1, 1024, 1024, false, false);
BM_BatchMatmul(32, 8, 1024, 1024, false, false);
BM_BatchMatmul(32, 16, 1024, 1024, false, false);
BM_BatchMatmul(32, 128, 1024, 1024, false, false);

// Square matmul.
BM_BatchMatmul(1, 32, 32, 32, false, false);
BM_BatchMatmul(1, 128, 128, 128, false, false);
BM_BatchMatmul(1, 256, 256, 256, false, false);
BM_BatchMatmul(1, 1024, 1024, 1024, false, false);
BM_BatchMatmul(1, 2048, 2048, 2048, false, false);
BM_BatchMatmul(2, 32, 32, 32, false, false);
BM_BatchMatmul(2, 128, 128, 128, false, false);
BM_BatchMatmul(2, 256, 256, 256, false, false);
BM_BatchMatmul(2, 1024, 1024, 1024, false, false);
BM_BatchMatmul(2, 2048, 2048, 2048, false, false);
BM_BatchMatmul(4, 32, 32, 32, false, false);
BM_BatchMatmul(4, 128, 128, 128, false, false);
BM_BatchMatmul(4, 256, 256, 256, false, false);
BM_BatchMatmul(4, 1024, 1024, 1024, false, false);
BM_BatchMatmul(4, 2048, 2048, 2048, false, false);
BM_BatchMatmul(8, 32, 32, 32, false, false);
BM_BatchMatmul(8, 128, 128, 128, false, false);
BM_BatchMatmul(8, 256, 256, 256, false, false);
BM_BatchMatmul(8, 1024, 1024, 1024, false, false);
BM_BatchMatmul(8, 2048, 2048, 2048, false, false);
BM_BatchMatmul(32, 32, 32, 32, false, false);
BM_BatchMatmul(32, 128, 128, 128, false, false);
BM_BatchMatmul(32, 256, 256, 256, false, false);
BM_BatchMatmul(32, 1024, 1024, 1024, false, false);
BM_BatchMatmul(32, 2048, 2048, 2048, false, false);

// Matrix-vector multiplies.
BM_BatchMatmul(1, 10000, 200, 1, false, false);
BM_BatchMatmul(8, 10000, 200, 1, false, false);
BM_BatchMatmul(32, 10000, 200, 1, false, false);
BM_BatchMatmul(1, 10000, 200, 1, true, false);
BM_BatchMatmul(8, 10000, 200, 1, true, false);
BM_BatchMatmul(32, 10000, 200, 1, true, false);
BM_BatchMatmul(1, 10000, 200, 1, false, true);
BM_BatchMatmul(8, 10000, 200, 1, false, true);
BM_BatchMatmul(32, 10000, 200, 1, false, true);
BM_BatchMatmul(1, 10000, 200, 1, true, true);
BM_BatchMatmul(8, 10000, 200, 1, true, true);
BM_BatchMatmul(32, 10000, 200, 1, true, true);

// Vector-matrix multiplies.
BM_BatchMatmul(1, 1, 200, 10000, false, false);
BM_BatchMatmul(8, 1, 200, 10000, false, false);
BM_BatchMatmul(32, 1, 200, 10000, false, false);
BM_BatchMatmul(1, 1, 200, 10000, true, false);
BM_BatchMatmul(8, 1, 200, 10000, true, false);
BM_BatchMatmul(32, 1, 200, 10000, true, false);
BM_BatchMatmul(1, 1, 200, 10000, false, true);
BM_BatchMatmul(8, 1, 200, 10000, false, true);
BM_BatchMatmul(32, 1, 200, 10000, false, true);
BM_BatchMatmul(1, 1, 200, 10000, true, true);
BM_BatchMatmul(8, 1, 200, 10000, true, true);
BM_BatchMatmul(32, 1, 200, 10000, true, true);

}  // namespace
}  // namespace tensorflow
