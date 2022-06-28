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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

using ::tensorflow::Scope;
using ::tensorflow::ops::Conv2D;
using ::tensorflow::ops::Identity;
using ::tensorflow::ops::RandomUniform;

constexpr int kBatchSize = 32;
constexpr int kWidth = 10;
constexpr int kHeight = 10;
constexpr int kDepthIn = 8;
constexpr int kKernel = 3;
constexpr int kDepthOut = 16;

// When there is a GPU, we test generic_layout_optimization for the conversion
// from NHWC to NCHW format. When there is only CPU, we test the conversion
// from NCHW to NHWC format. The following macros help setting tensor shapes,
// source and destination format strings, and transpose permutation vectors
// appropriately for NHWC -> NCHW conversion (when GPU) and NCHW -> NHWC
// conversion (when only CPU).

#if (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
#define DIMS(n, h, w, c) \
  { n, h, w, c }
#define SRC_DATA_FORMAT "NHWC"
#define DST_DATA_FORMAT "NCHW"
#define DEVICE "GPU"
#define REWRITER_CONFIG \
  RewriterConfig::DEFAULT, RewriterConfig::NO_CONVERSION_ON_CPU
#define PERMUTATION_SRC_TO_DST \
  { 0, 3, 1, 2 }
#define PERMUTATION_DST_TO_SRC \
  { 0, 2, 3, 1 }
#else
#define DIMS(n, h, w, c) \
  { n, c, h, w }
#define SRC_DATA_FORMAT "NCHW"
#define DST_DATA_FORMAT "NHWC"
#define DEVICE "CPU"
#define REWRITER_CONFIG RewriterConfig::DEFAULT, RewriterConfig::NCHW_TO_NHWC
#define PERMUTATION_SRC_TO_DST \
  { 0, 2, 3, 1 }
#define PERMUTATION_DST_TO_SRC \
  { 0, 3, 1, 2 }
#endif  // (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)

template <typename T = float>
Output SimpleConv2D(tensorflow::Scope* s, int input_size, int filter_size,
                    const string& padding, const string& device) {
  int batch_size = 8;
  int input_height = input_size;
  int input_width = input_size;
  int input_depth = 3;
  int filter_count = 2;
  int stride = 1;
  TensorShape input_shape(
      DIMS(batch_size, input_height, input_width, input_depth));
  Tensor input_data(DataTypeToEnum<T>::value, input_shape);
  test::FillIota<T>(&input_data, static_cast<T>(1));
  Output input =
      ops::Const(s->WithOpName("Input"), Input::Initializer(input_data));

  TensorShape filter_shape(
      {filter_size, filter_size, input_depth, filter_count});
  Tensor filter_data(DataTypeToEnum<T>::value, filter_shape);
  test::FillIota<T>(&filter_data, static_cast<T>(1));
  Output filter =
      ops::Const(s->WithOpName("Filter"), Input::Initializer(filter_data));

  Output conv = ops::Conv2D(s->WithOpName("Conv2D").WithDevice(device), input,
                            filter, DIMS(1, stride, stride, 1), padding,
                            ops::Conv2D::Attrs().DataFormat(SRC_DATA_FORMAT));
  return conv;
}

Output SimpleConv2DBackpropInput(tensorflow::Scope* s, int input_size,
                                 int filter_size, const string& padding,
                                 bool dilated, const int input_sizes_length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_0(mht_0_v, 285, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "SimpleConv2DBackpropInput");

  int batch_size = 128;
  int input_height = input_size;
  int input_width = input_size;
  int input_depth = 3;
  int filter_count = 2;
  int stride = 1;
  TensorShape input_sizes_shape({input_sizes_length});
  Tensor input_data(DT_INT32, input_sizes_shape);
  if (input_sizes_length == 4) {
    test::FillValues<int>(
        &input_data, DIMS(batch_size, input_height, input_width, input_depth));
  } else {
    test::FillValues<int>(&input_data, {input_height, input_width});
  }
  Output input_sizes =
      ops::Const(s->WithOpName("InputSizes"), Input::Initializer(input_data));

  TensorShape filter_shape(
      {filter_size, filter_size, input_depth, filter_count});
  Output filter =
      ops::Variable(s->WithOpName("Filter"), filter_shape, DT_FLOAT);

  int output_height = input_height;
  int output_width = input_width;
  TensorShape output_shape(
      DIMS(batch_size, output_height, output_width, filter_count));
  Tensor output_data(DT_FLOAT, output_shape);
  test::FillIota<float>(&output_data, 1.0f);
  Output output =
      ops::Const(s->WithOpName("Output"), Input::Initializer(output_data));

  Output conv_backprop_input;
  Output input_sizes_i =
      ops::Identity(s->WithOpName("InputSizesIdentity"), input_sizes);
  ops::Conv2DBackpropInput::Attrs attrs;
  attrs = attrs.DataFormat(SRC_DATA_FORMAT);
  if (dilated) {
    attrs = attrs.Dilations(DIMS(1, 2, 2, 1));
  }
  conv_backprop_input = ops::Conv2DBackpropInput(
      s->WithOpName("Conv2DBackpropInput"), input_sizes_i, filter, output,
      DIMS(1, stride, stride, 1), padding, attrs);

  return conv_backprop_input;
}

class GenericLayoutOptimizerTest : public GrapplerTest {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_1(mht_1_v, 337, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "SetUp");

    bool gpu_available = GetNumAvailableGPUs() > 0;

    if (gpu_available) {
      virtual_cluster_ =
          absl::make_unique<SingleMachine>(/*timeout_s=*/10, 1, 1);
    } else {
      DeviceProperties cpu_device;
      cpu_device.set_type("CPU");
      cpu_device.set_frequency(1000);
      cpu_device.set_num_cores(4);
      cpu_device.set_bandwidth(32);
      cpu_device.set_l1_cache_size(32 * 1024);
      cpu_device.set_l2_cache_size(256 * 1024);
      cpu_device.set_l3_cache_size(4 * 1024 * 1024);
      cpu_device.set_memory_size(1024 * 1024);
#if (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
      DeviceProperties gpu_device;
      gpu_device.set_type("GPU");
      gpu_device.mutable_environment()->insert({"architecture", "6"});
      virtual_cluster_ =
          absl::WrapUnique(new VirtualCluster({{"/CPU:0", cpu_device},
                                               { "/GPU:1",
                                                 gpu_device }}));
#else
      virtual_cluster_ =
          absl::WrapUnique(new VirtualCluster({{"/CPU:0", cpu_device}}));
#endif  // (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
    }
    TF_ASSERT_OK(virtual_cluster_->Provision());
  }

  void TearDown() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_2(mht_2_v, 372, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "TearDown");
 TF_ASSERT_OK(virtual_cluster_->Shutdown()); }

  std::unique_ptr<Cluster> virtual_cluster_;
};

void VerifyRegularFaninMatch(const utils::NodeView* node, int port,
                             absl::string_view fanin_name, int fanin_port) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fanin_name: \"" + std::string(fanin_name.data(), fanin_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_3(mht_3_v, 382, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "VerifyRegularFaninMatch");

  ASSERT_GE(node->NumRegularFanins(), port);
  const auto& fanin = node->GetRegularFanin(port);
  EXPECT_EQ(fanin.node_view()->GetName(), fanin_name);
  EXPECT_EQ(fanin.index(), fanin_port);
}

void VerifyRegularFanoutMatch(const utils::NodeView* node, int port,
                              absl::string_view fanout_name, int fanout_port) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fanout_name: \"" + std::string(fanout_name.data(), fanout_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_4(mht_4_v, 394, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "VerifyRegularFanoutMatch");

  bool found = false;
  for (const auto& regular_fanout : node->GetRegularFanout(port)) {
    if (regular_fanout.node_view()->GetName() == fanout_name &&
        regular_fanout.index() == fanout_port) {
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

void VerifyDataFormatAttributeMatch(const utils::NodeView* node,
                                    absl::string_view attr_value) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("attr_value: \"" + std::string(attr_value.data(), attr_value.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_testDTcc mht_5(mht_5_v, 410, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_test.cc", "VerifyDataFormatAttributeMatch");

  const auto* attr = node->GetAttr("data_format");
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->s(), attr_value);
}

TEST_F(GenericLayoutOptimizerTest, OptimizeSimpleConv2DGraph) {
  // A simple graph contains 1 Conv2D node, 2 input and 1 output nodes.
  // Data format is NHWC on GPU, while NCHW on CPU.
  Scope scope = Scope::NewRootScope();

  auto conv2d = SimpleConv2D(&scope, 4, 2, "VALID", "");
  auto identity = Identity(scope.WithOpName("Output"), conv2d);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  // The expected optimized graph contains 2 extra sets of Transpose nodes and
  // has the Conv2D's data_format set to "NCHW" on GPU, while "NHWC" on CPU.
  auto* input_transpose_node = graph_view.GetNode(
      absl::StrCat("Conv2D-0-Transpose", SRC_DATA_FORMAT, "To", DST_DATA_FORMAT,
                   "-LayoutOptimizer"));

  ASSERT_NE(input_transpose_node, nullptr);
  ASSERT_EQ(input_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_transpose_node, 0, "Input", 0);

  auto* conv2d_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv2d_node, nullptr);
  ASSERT_EQ(conv2d_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(conv2d_node, 0, input_transpose_node->GetName(), 0);
  VerifyRegularFaninMatch(conv2d_node, 1, "Filter", 0);
  VerifyDataFormatAttributeMatch(conv2d_node, DST_DATA_FORMAT);

  auto* output_transpose_node = graph_view.GetNode(
      absl::StrCat("Conv2D-0-0-Transpose", DST_DATA_FORMAT, "To",
                   SRC_DATA_FORMAT, "-LayoutOptimizer"));
  ASSERT_NE(output_transpose_node, nullptr);
  ASSERT_EQ(output_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_transpose_node, 0, conv2d_node->GetName(), 0);

  auto* output_node = graph_view.GetNode("Output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_transpose_node->GetName(), 0);
}

TEST_F(GenericLayoutOptimizerTest, PreserveFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "");
  auto i = ops::Identity(s.WithOpName("i"), conv);
  GrapplerItem item;
  item.fetch.push_back("Conv2D");
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, SRC_DATA_FORMAT);
}

TEST_F(GenericLayoutOptimizerTest, EmptyDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, DST_DATA_FORMAT);
}

TEST_F(GenericLayoutOptimizerTest, GPUDevice) {
#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  GTEST_SKIP() << "Neither CUDA nor ROCm is enabled";
#endif  // !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv =
      SimpleConv2D(&s, 4, 2, "VALID", "/job:w/replica:0/task:0/device:GPU:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, "NCHW");
}

TEST_F(GenericLayoutOptimizerTest, CPUDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D(&s, 4, 2, "VALID", "/CPU:0");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
#if (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
  VerifyDataFormatAttributeMatch(conv_node, "NHWC");
#else
  VerifyDataFormatAttributeMatch(conv_node, DST_DATA_FORMAT);
#endif  // (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
}

TEST_F(GenericLayoutOptimizerTest, NoOptimizeIntegerConvolution) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto conv = SimpleConv2D<int32>(&s, 4, 2, "VALID", "");
  Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
  GrapplerItem item;
  TF_ASSERT_OK(s.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv_node, nullptr);
  VerifyDataFormatAttributeMatch(conv_node, SRC_DATA_FORMAT);
}

TEST_F(GenericLayoutOptimizerTest, Connectivity) {
  Scope scope = Scope::NewRootScope();
  auto conv = SimpleConv2D(&scope, 4, 2, "VALID",
                           absl::StrCat("/device:", DEVICE, ":0"));
  auto i1 = ops::Identity(scope.WithOpName("i1"), conv);
  auto i2 = ops::Identity(scope.WithOpName("i2"), i1);
  auto i3 = ops::Identity(scope.WithOpName("i3"), i2);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  // Make the graph not in topological order to test the handling of multi-hop
  // connectivity (here we say two nodes are connected if all nodes in the
  // middle are layout agnostic). If the graph is already in topological order,
  // the problem is easier, where layout optimizer only needs to check
  // single-hop connectivity.
  Status status;
  utils::GraphView graph_view_original(&item.graph, &status);
  const int i1_index = graph_view_original.GetNode("i1")->node_index();
  const int i2_index = graph_view_original.GetNode("i2")->node_index();
  item.graph.mutable_node()->SwapElements(i1_index, i2_index);

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* node_i2_output = graph_view.GetNode("i2");
  ASSERT_NE(node_i2_output, nullptr);
  // Layout optimizer should process i2, as it detects i2 is connected with the
  // Conv2D node two hops away. Similarly i1 is processed as well, as i1 is
  // directly connected to the Conv2D node.
  ASSERT_EQ(node_i2_output->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(node_i2_output, 0, "i1", 0);
}

TEST_F(GenericLayoutOptimizerTest, Conv2DBackpropInputNonConstInputSizes) {
  for (const int input_sizes_length : {2, 4}) {
    Scope s = Scope::NewRootScope();
    auto conv = SimpleConv2DBackpropInput(&s, 7, 2, "SAME", /*dilated=*/false,
                                          input_sizes_length);
    Output fetch = ops::Identity(s.WithOpName("Fetch"), {conv});
    GrapplerItem item;
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

    Status status;
    utils::GraphView graph_view(&output, &status);
    TF_ASSERT_OK(status);
    auto* conv2d_backprop_node = graph_view.GetNode("Conv2DBackpropInput");
    ASSERT_NE(conv2d_backprop_node, nullptr);
    ASSERT_EQ(conv2d_backprop_node->NumRegularFanins(), 3);
    VerifyRegularFaninMatch(
        conv2d_backprop_node, 0,
        absl::StrCat("Conv2DBackpropInput-0-DataFormatVecPermute",
                     SRC_DATA_FORMAT, "To", DST_DATA_FORMAT,
                     "-LayoutOptimizer"),
        0);
    auto* input_sizes_node = graph_view.GetNode(absl::StrCat(
        "Conv2DBackpropInput-0-DataFormatVecPermute", SRC_DATA_FORMAT, "To",
        DST_DATA_FORMAT, "-LayoutOptimizer"));
    ASSERT_NE(input_sizes_node, nullptr);
    EXPECT_EQ(input_sizes_node->GetOp(), "DataFormatVecPermute");
    ASSERT_EQ(input_sizes_node->NumRegularFanins(), 1);
    VerifyRegularFaninMatch(input_sizes_node, 0, "InputSizesIdentity", 0);
  }
}

TEST_F(GenericLayoutOptimizerTest, Conv2DDataFormatVecPermuteCollapse) {
  Scope scope =
      Scope::NewRootScope().WithDevice(absl::StrCat("/device:", DEVICE, ":0"));
  auto conv = SimpleConv2D(&scope, 4, 2, "VALID",
                           absl::StrCat("/device:", DEVICE, ":0"));
  auto shape = ops::Shape(scope.WithOpName("shape"), conv);
  auto value = ops::Const(scope.WithOpName("value"), 0, {});
  auto fill = ops::Fill(scope.WithOpName("fill"), shape, value);
  auto i = ops::Identity(scope.WithOpName("i"), fill);
  GrapplerItem item;
  TF_ASSERT_OK(scope.ToGraphDef(&item.graph));

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  // Graph before optimization:
  // input -> conv2d -> shape -> fill -> output
  //
  // Graph after expansion:
  // input -> T -> conv2d -> T' -> T -> shape -> D' -> D -> fill -> T' -> output
  //
  // Graph after collapsion:
  // input -> T -> conv2d -> shape -> fill -> T' -> output
  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);
  auto* conv2d_node = graph_view.GetNode("Conv2D");
  ASSERT_NE(conv2d_node, nullptr);
  ASSERT_EQ(conv2d_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(
      conv2d_node, 0,
      absl::StrCat("Conv2D-0-Transpose", SRC_DATA_FORMAT, "To", DST_DATA_FORMAT,
                   "-LayoutOptimizer"),
      0);

  auto* shape_node = graph_view.GetNode("shape");
  ASSERT_NE(shape_node, nullptr);
  ASSERT_EQ(shape_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(shape_node, 0, conv2d_node->GetName(), 0);

  auto* fill_node = graph_view.GetNode("fill");
  ASSERT_NE(fill_node, nullptr);
  ASSERT_EQ(fill_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(fill_node, 0, shape_node->GetName(), 0);
  VerifyRegularFanoutMatch(
      fill_node, 0,
      absl::StrCat("fill-0-0-Transpose", DST_DATA_FORMAT, "To", SRC_DATA_FORMAT,
                   "-LayoutOptimizer"),
      0);

  auto* graph_output = graph_view.GetNode("i");
  ASSERT_NE(graph_output, nullptr);
  ASSERT_EQ(graph_output->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(
      graph_output, 0,
      absl::StrCat("fill-0-0-Transpose", DST_DATA_FORMAT, "To", SRC_DATA_FORMAT,
                   "-LayoutOptimizer"),
      0);
}

TEST_F(GenericLayoutOptimizerTest, DoNotPruneNonAddedCancellableTransposes) {
  GrapplerItem item;
  {
    Scope scope = Scope::NewRootScope().WithDevice(
        absl::StrCat("/device:", DEVICE, ":0"));
    auto input = ops::RandomUniform(scope.WithOpName("input"),
                                    DIMS(kBatchSize, kHeight, kWidth, kDepthIn),
                                    DT_FLOAT);
    // Permutation for source to destination data format.
    // GPU: NHWC -> NCHW: {0, 3, 1, 2}
    // CPU: NCHW -> NHWC: {0, 2, 3, 1}
    auto input_in_transpose =
        ops::Transpose(scope.WithOpName("input_in_transpose"), input,
                       ops::Const(scope, PERMUTATION_SRC_TO_DST, {4}));
    // Permutation for destination to source data format.
    // GPU: NCHW -> NHWC: {0, 2, 3, 1}
    // CPU: NHWC -> NCHW: {0, 3, 1, 2}
    auto input_out_transpose = ops::Transpose(
        scope.WithOpName("input_out_transpose"), input_in_transpose,
        ops::Const(scope, PERMUTATION_DST_TO_SRC, {4}));
    Tensor bias_data(DT_FLOAT, TensorShape({kDepthIn}));
    test::FillIota<float>(&bias_data, 1.0f);
    auto bias_add = ops::BiasAdd(
        scope.WithOpName("bias_add"), input_out_transpose, bias_data,
        ops::BiasAdd::Attrs().DataFormat(SRC_DATA_FORMAT));
    auto output_in_transpose =
        ops::Transpose(scope.WithOpName("output_in_transpose"), bias_add,
                       ops::Const(scope, PERMUTATION_SRC_TO_DST, {4}));
    auto output_out_transpose = ops::Transpose(
        scope.WithOpName("output_out_transpose"), output_in_transpose,
        ops::Const(scope, PERMUTATION_DST_TO_SRC, {4}));
    auto output =
        ops::Identity(scope.WithOpName("output"), output_out_transpose);
    TF_ASSERT_OK(scope.ToGraphDef(&item.graph));
  }

  GenericLayoutOptimizer optimizer(REWRITER_CONFIG);
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);

  auto* input_node = graph_view.GetNode("input");
  ASSERT_NE(input_node, nullptr);

  auto* input_in_transpose_node = graph_view.GetNode("input_in_transpose");
  ASSERT_NE(input_in_transpose_node, nullptr);
  ASSERT_EQ(input_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_in_transpose_node, 0, input_node->GetName(), 0);

  auto* input_out_transpose_node = graph_view.GetNode("input_out_transpose");
  ASSERT_NE(input_out_transpose_node, nullptr);
  ASSERT_EQ(input_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(input_out_transpose_node, 0,
                          input_in_transpose_node->GetName(), 0);

  auto* bias_add_in_transpose_node = graph_view.GetNode(
      absl::StrCat("bias_add-0-Transpose", SRC_DATA_FORMAT, "To",
                   DST_DATA_FORMAT, "-LayoutOptimizer"));
  ASSERT_NE(bias_add_in_transpose_node, nullptr);
  ASSERT_EQ(bias_add_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_in_transpose_node, 0,
                          input_out_transpose_node->GetName(), 0);

  auto* bias_add_node = graph_view.GetNode("bias_add");
  ASSERT_NE(bias_add_node, nullptr);
  ASSERT_EQ(bias_add_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_node, 0,
                          bias_add_in_transpose_node->GetName(), 0);

  auto* bias_add_out_transpose_node = graph_view.GetNode(
      absl::StrCat("bias_add-0-0-Transpose", DST_DATA_FORMAT, "To",
                   SRC_DATA_FORMAT, "-LayoutOptimizer"));
  ASSERT_NE(bias_add_out_transpose_node, nullptr);
  ASSERT_EQ(bias_add_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(bias_add_out_transpose_node, 0,
                          bias_add_node->GetName(), 0);

  auto* output_in_transpose_node = graph_view.GetNode("output_in_transpose");
  ASSERT_NE(output_in_transpose_node, nullptr);
  ASSERT_EQ(output_in_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_in_transpose_node, 0,
                          bias_add_out_transpose_node->GetName(), 0);

  auto* output_out_transpose_node = graph_view.GetNode("output_out_transpose");
  ASSERT_NE(output_out_transpose_node, nullptr);
  ASSERT_EQ(output_out_transpose_node->NumRegularFanins(), 2);
  VerifyRegularFaninMatch(output_out_transpose_node, 0,
                          output_in_transpose_node->GetName(), 0);

  auto* output_node = graph_view.GetNode("output");
  ASSERT_NE(output_node, nullptr);
  ASSERT_EQ(output_node->NumRegularFanins(), 1);
  VerifyRegularFaninMatch(output_node, 0, output_out_transpose_node->GetName(),
                          0);
}

TEST_F(GenericLayoutOptimizerTest, CancelTransposeAroundPad) {
  using test::function::NDef;

  GenericLayoutOptimizer optimizer(
      RewriterConfig::AGGRESSIVE,
      RewriterConfig::NCHW_TO_NHWC /* CPU settings*/);

  const Tensor kPermuteNhwcToNchw = test::AsTensor<int32>({0, 3, 1, 2});
  const Tensor kPermuteNchwToNhwc = test::AsTensor<int32>({0, 2, 3, 1});
  const Tensor kPad = test::AsTensor<int32>({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});

  GrapplerItem item;
  item.graph = test::function::GDef({
      NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}),

      NDef("paddings", "Const", {}, {{"dtype", DT_INT32}, {"value", kPad}}),
      NDef("perm_nhwc_to_nchw", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNhwcToNchw}}),
      NDef("perm_nchw_to_nhwc", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNchwToNhwc}}),

      NDef("transpose_0", "Transpose", {"x", "perm_nhwc_to_nchw"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
      NDef("pad", "Pad", {"transpose_0", "paddings"},
           {{"T", DT_FLOAT}, {"Tpaddings", DT_INT32}}),
      NDef("transpose_1", "Transpose", {"pad", "perm_nchw_to_nhwc"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
      NDef("transpose_2", "Transpose", {"pad", "perm_nchw_to_nhwc"},
           {{"T", DT_FLOAT}, {"Tperm", DT_INT32}}),
  });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  const Tensor kPermutedPaddings =
      test::AsTensor<int32>({1, 2, 5, 6, 7, 8, 3, 4}, {4, 2});

  GraphDef expected = test::function::GDef({
      NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}),

      NDef("paddings", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermutedPaddings}}),
      NDef("perm_nhwc_to_nchw", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNhwcToNchw}}),
      NDef("perm_nchw_to_nhwc", "Const", {},
           {{"dtype", DT_INT32}, {"value", kPermuteNchwToNhwc}}),

      // Transpose nodes replaced by Identity nodes.
      NDef("transpose_0", "Identity", {"x"}, {{"T", DT_FLOAT}}),
      NDef("pad", "Pad", {"transpose_0", "paddings"},
           {{"T", DT_FLOAT}, {"Tpaddings", DT_INT32}}),
      NDef("transpose_1", "Identity", {"pad"}, {{"T", DT_FLOAT}}),
      NDef("transpose_2", "Identity", {"pad"}, {{"T", DT_FLOAT}}),
  });

  CompareGraphs(expected, output);

  Tensor x = GenerateRandomTensor<DT_FLOAT>({2, 6, 6, 8});
  item.fetch = {"transpose_1", "transpose_2"};
  item.feed.emplace_back("x", x);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors.size(), 2);
  ASSERT_EQ(tensors_expected.size(), 2);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

TEST_F(GenericLayoutOptimizerTest, PreserveInputShapes) {
  using test::function::NDef;

  GenericLayoutOptimizer optimizer(RewriterConfig::AGGRESSIVE);

  AttrValue output_shapes;
  auto* shape = output_shapes.mutable_list()->add_shape();
  shape->add_dim()->set_size(-1);

  GrapplerItem item;
  item.graph = test::function::GDef({NDef(
      "x", "_Arg", {},
      {{"T", DT_FLOAT}, {"index", 0}, {"_output_shapes", output_shapes}})});
  item.feed.emplace_back("x", Tensor(DT_FLOAT));

  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  Status status;
  utils::GraphView graph_view(&output, &status);
  TF_ASSERT_OK(status);

  auto* arg = graph_view.GetNode("x");
  ASSERT_NE(arg, nullptr);
  EXPECT_TRUE(arg->HasAttr("_output_shapes"));
  EXPECT_EQ(arg->GetAttr("_output_shapes")->DebugString(),
            output_shapes.DebugString());
}

// TODO(yanzha): Add more complex Graph for test.

}  // namespace grappler
}  // namespace tensorflow
