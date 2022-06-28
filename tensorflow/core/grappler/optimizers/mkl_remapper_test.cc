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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc() {
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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/remapper.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {
namespace grappler {

class MklRemapperTest : public GrapplerTest {
 public:
  const string kAddNOp = "AddN";
  const string kAddOp = "Add";
  const string kAddV2Op = "AddV2";

 protected:
  void FuseConv2DWithBiasAndAddNOrAdd(const string& data_format,
                                      const string& activation, string add_op,
                                      bool add_with_bcast) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("data_format: \"" + data_format + "\"");
   mht_0_v.push_back("activation: \"" + activation + "\"");
   mht_0_v.push_back("add_op: \"" + add_op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "FuseConv2DWithBiasAndAddNOrAdd");

    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = (data_format == "NHWC")
                           ? ops::Placeholder::Shape({8, 32, 32, 3})
                           : ops::Placeholder::Shape({8, 3, 32, 32});
    auto input_shape_addn = ops::Placeholder::Shape({});
    if (data_format == "NHWC") {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({128});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 32, 32, 128});
    } else {
      if (add_with_bcast)
        input_shape_addn = ops::Placeholder::Shape({32});
      else
        input_shape_addn = ops::Placeholder::Shape({8, 128, 32, 32});
    }
    auto filter_shape = ops::Placeholder::Shape({1, 1, 3, 128});
    auto bias_shape = ops::Placeholder::Shape({128});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto input_addn =
        Placeholder(s.WithOpName("input_addn"), DT_FLOAT, input_shape_addn);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    std::vector<int> strides = {1, 1, 1, 1};
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), input, filter, strides, "SAME",
                    ops::Conv2D::Attrs().DataFormat(data_format));
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias,
                                 ops::BiasAdd::Attrs().DataFormat(data_format));

    auto addfetch = [&](::tensorflow::Input addop) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_1(mht_1_v, 251, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "lambda");

      auto activate = s.WithOpName("activation");
      auto fetch = s.WithOpName("fetch");
      if (activation == "Relu") {
        ops::Identity(fetch, ops::Relu(activate, addop));
      } else if (activation == "Relu6") {
        ops::Identity(fetch, ops::Relu6(activate, addop));
      } else if (activation == "Elu") {
        ops::Identity(fetch, ops::Elu(activate, addop));
      } else if (activation == "LeakyRelu") {
        ops::Identity(fetch, ops::internal::LeakyRelu(activate, addop));
      } else {
        DCHECK(activation == "None");
        ops::Identity(fetch, addop);
      }
    };

    if (add_op == kAddNOp) {
      auto addn = ops::AddN(s.WithOpName(add_op),
                            std::initializer_list<Input>{input_addn, bias_add});
      addfetch(addn);
    } else if (add_op == kAddV2Op) {
      auto add = ops::AddV2(s.WithOpName(add_op), input_addn, bias_add);
      addfetch(add);
    } else {
      auto add = ops::Add(s.WithOpName(add_op), input_addn, bias_add);
      addfetch(add);
    }
    auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape.shape_.dim_sizes()));
    auto input_addn_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape_addn.shape_.dim_sizes()));
    auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(filter_shape.shape_.dim_sizes()));
    auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(bias_shape.shape_.dim_sizes()));

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"filter", filter_tensor},
                 {"bias", bias_tensor},
                 {"input_addn", input_addn_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    // Set Rewriter config to AGGRESSIVE so that we can use Placeholder shape
    // to test that Add with both inputs having same shape get fused with
    // Conv2D. Setting this config to AGGRESSIVE is not required for the feature
    // though.
    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    bool check_fusion = !add_with_bcast;
    int found = 0;
    for (const NodeDef& node : output.node()) {
      auto fetch_node_name = activation != "None" ? "activation" : add_op;
      if (node.name() == fetch_node_name) {
        if (check_fusion) {
          EXPECT_EQ("_FusedConv2D", node.op());
          EXPECT_EQ("input", node.input(0));
          EXPECT_EQ("filter", node.input(1));

          EXPECT_EQ(2, node.attr().at("num_args").i());
          EXPECT_EQ("bias", node.input(2));
          EXPECT_EQ("input_addn", node.input(3));

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          if (activation != "None") {
            EXPECT_EQ(3, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
            EXPECT_EQ(activation, fused_ops[2]);
          } else {
            EXPECT_EQ(2, fused_ops.size());
            EXPECT_EQ("BiasAdd", fused_ops[0]);
            EXPECT_EQ("Add", fused_ops[1]);
          }
        } else {
          if (activation != "None") {
            EXPECT_EQ(node.op(), activation);
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), add_op);
          } else {
            EXPECT_EQ(node.op(), add_op);
            ASSERT_EQ(node.input_size(), 2);
          }
        }
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    // Using relative tolerance since oneDNN could produce different results
    // when float32 numbers need to be rounded during accumulation.
    test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-6);
  }
};

#define CREATE_CONV2DFUSION_TEST(data_format, addop, activation, bcast)                          \
  TEST_F(                                                                                        \
      MklRemapperTest,                                                                           \
      FuseConv2DWithBiasAnd##addop##_##data_format##_activation##activation##_addbcast##bcast) { \
    FuseConv2DWithBiasAndAddNOrAdd(#data_format, #activation, #addop, bcast);                    \
  }

#define CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(data_format, addop, bcast) \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Relu, bcast);               \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Relu6, bcast);              \
  CREATE_CONV2DFUSION_TEST(data_format, addop, Elu, bcast);                \
  CREATE_CONV2DFUSION_TEST(data_format, addop, LeakyRelu, bcast);          \
  CREATE_CONV2DFUSION_TEST(data_format, addop, None, bcast);

#define CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(addop)            \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false);

CREATE_CONV2DFUSION_ADD_NOBCAST_TEST(AddN);

#define CREATE_CONV2DFUSION_ADD_BCAST_TEST(addop)              \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, false); \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NHWC, addop, true);  \
  CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST(NCHW, addop, true);

CREATE_CONV2DFUSION_ADD_BCAST_TEST(Add);
CREATE_CONV2DFUSION_ADD_BCAST_TEST(AddV2);

#undef CREATE_CONV2DFUSION_ADD_NOBCAST_TEST
#undef CREATE_CONV2DFUSION_ADD_BCAST_TEST
#undef CREATE_CONV2DFUSION_ADD_ACTIVATION_TEST
#undef CREATE_CONV2DFUSION_TEST

#define REGISTER_TEST(NAME, T, INPUT)                                         \
  TEST_F(MklRemapperTest, NAME##_##T) {                                       \
    using ::tensorflow::ops::Placeholder;                                     \
                                                                              \
    for (const string& activation : {"Relu", "Relu6", "Elu", "None"}) {       \
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();                \
                                                                              \
      auto input_shape = Placeholder::Shape({8, 32, 32, 3});                  \
      auto filter_shape = Placeholder::Shape({1, 1, 3, 1});                   \
      auto bias_shape = Placeholder::Shape({3});                              \
                                                                              \
      auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape); \
      auto filter =                                                           \
          Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);        \
      auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);    \
                                                                              \
      std::vector<int> strides = {1, 1, 1, 1};                                \
      auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),  \
                                             input, filter, strides, "SAME"); \
      auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), conv, bias);     \
                                                                              \
      ops::Identity fetch = [&]() -> ops::Identity {                          \
        auto activate = s.WithOpName("activation");                           \
        auto fetch = s.WithOpName("fetch");                                   \
                                                                              \
        if (activation == "Relu") {                                           \
          return ops::Identity(fetch, ops::Relu(activate, bias_add));         \
        } else if (activation == "Relu6") {                                   \
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));        \
        } else if (activation == "Elu") {                                     \
          return ops::Identity(fetch, ops::Elu(activate, bias_add));          \
        }                                                                     \
                                                                              \
        DCHECK(activation == "None");                                         \
        return ops::Identity(fetch, bias_add);                                \
      }();                                                                    \
                                                                              \
      auto input_t = GenerateRandomTensor<DT_FLOAT>({8, 32, 32, 3});          \
      auto filter_t = GenerateRandomTensor<DT_FLOAT>({1, 1, 3, 1});           \
      auto bias_t = GenerateRandomTensor<DT_FLOAT>({3});                      \
                                                                              \
      GrapplerItem item;                                                      \
      item.fetch = {"fetch"};                                                 \
      item.feed = {                                                           \
          {"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};        \
      TF_CHECK_OK(s.ToGraphDef(&item.graph));                                 \
                                                                              \
      for (int i = 0; i < item.graph.node_size(); ++i) {                      \
        item.graph.mutable_node(i)->set_device("/device:CPU:0");              \
      }                                                                       \
                                                                              \
      Remapper optimizer(RewriterConfig::ON);                                 \
      GraphDef output;                                                        \
      TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));                \
                                                                              \
      int found = 0;                                                          \
      for (const NodeDef& node : output.node()) {                             \
        if (node.name() != "bias_add" && node.name() != "activation")         \
          continue;                                                           \
                                                                              \
        EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");                  \
        ASSERT_EQ(node.input_size(), 3);                                      \
        EXPECT_EQ(node.input(0), "input");                                    \
        EXPECT_EQ(node.input(1), "filter");                                   \
                                                                              \
        EXPECT_EQ(node.attr().at("num_args").i(), 1);                         \
        EXPECT_EQ(node.input(2), "bias");                                     \
                                                                              \
        const auto fused_ops = node.attr().at("fused_ops").list().s();        \
        if (node.name() == "bias_add") {                                      \
          ASSERT_EQ(fused_ops.size(), 1);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          found++;                                                            \
        }                                                                     \
        if (node.name() == "activation") {                                    \
          ASSERT_EQ(fused_ops.size(), 2);                                     \
          EXPECT_EQ(fused_ops[0], "BiasAdd");                                 \
          EXPECT_EQ(fused_ops[1], activation);                                \
          found++;                                                            \
        }                                                                     \
      }                                                                       \
      EXPECT_EQ(found, 1);                                                    \
                                                                              \
      auto tensors_expected =                                                 \
          EvaluateNodes(item.graph, item.fetch, item.feed);                   \
      ASSERT_EQ(tensors_expected.size(), 1);                                  \
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);            \
      ASSERT_EQ(tensors.size(), 1);                                           \
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);   \
    }                                                                         \
  }
REGISTER_TEST_ALL_TYPES(FuseDepthwiseConv2DWithBiasAndActivation);
#undef REGISTER_TEST

TEST_F(MklRemapperTest, FuseBatchNormWithRelu) {
  using ::tensorflow::ops::Placeholder;

  for (bool is_training : {true, false}) {
    for (bool has_side_input : {true, false}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      const int num_channels = 24;

      TensorShape channel_shape({num_channels});
      TensorShape empty_shape({0});

      auto input =
          Placeholder(s.WithOpName("input"), DT_FLOAT,
                      ops::Placeholder::Shape({2, 8, 8, num_channels}));
      auto input_cast = ops::Cast(s.WithOpName("input_cast"), input, DT_FLOAT);
      auto scale = Placeholder(s.WithOpName("scale"), DT_FLOAT);
      auto offset = Placeholder(s.WithOpName("offset"), DT_FLOAT);
      auto mean = Placeholder(s.WithOpName("mean"), DT_FLOAT);
      auto var = Placeholder(s.WithOpName("var"), DT_FLOAT);

      float epsilon = 0.1f;
      auto fbn =
          ops::FusedBatchNormV3(s.WithOpName("fused_batch_norm"), input_cast,
                                scale, offset, mean, var,
                                ops::FusedBatchNormV3::IsTraining(is_training)
                                    .Epsilon(epsilon)
                                    .DataFormat("NHWC"));

      if (has_side_input) {
        auto side_input =
            Placeholder(s.WithOpName("side_input"), DT_FLOAT,
                        ops::Placeholder::Shape({2, 8, 8, num_channels}));
        auto side_input_cast =
            ops::Cast(s.WithOpName("side_input_cast"), side_input, DT_FLOAT);
        auto add = ops::Add(s.WithOpName("add"), fbn.y, side_input_cast);
        auto relu = ops::Relu(s.WithOpName("relu"), add);
      } else {
        auto relu = ops::Relu(s.WithOpName("relu"), fbn.y);
      }

      auto input_t = GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});
      auto scale_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto offset_t = GenerateRandomTensor<DT_FLOAT>(channel_shape);
      auto mean_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                               : channel_shape);
      auto var_t = GenerateRandomTensor<DT_FLOAT>(is_training ? empty_shape
                                                              : channel_shape);
      auto side_input_t =
          GenerateRandomTensor<DT_FLOAT>({2, 8, 8, num_channels});

      GrapplerItem item;
      item.fetch = {"relu"};
      if (has_side_input)
        item.feed = {{"input", input_t},   {"scale", scale_t},
                     {"offset", offset_t}, {"mean", mean_t},
                     {"var", var_t},       {"side_input", side_input_t}};
      else
        item.feed = {{"input", input_t},
                     {"scale", scale_t},
                     {"offset", offset_t},
                     {"mean", mean_t},
                     {"var", var_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::AGGRESSIVE);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      if (has_side_input) {
        for (const NodeDef& node : output.node()) {
          if (node.name() == "add") {
            EXPECT_EQ(node.op(), "Add");
            ASSERT_EQ(node.input_size(), 2);
            EXPECT_EQ(node.input(0), "fused_batch_norm");
            EXPECT_EQ(node.input(1), "side_input_cast");
            found++;
          }
          if (node.name() == "relu") {
            EXPECT_EQ(node.op(), "Relu");
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), "add");
            found++;
          }
          if (node.name() == "fused_batch_norm") {
            EXPECT_EQ(node.op(), "FusedBatchNormV3");
            ASSERT_EQ(node.input_size(), 5);
            EXPECT_EQ(node.input(0), "input_cast");
            EXPECT_EQ(node.input(1), "scale");
            EXPECT_EQ(node.input(2), "offset");
            EXPECT_EQ(node.input(3), "mean");
            EXPECT_EQ(node.input(4), "var");
            found++;
          }
        }
        EXPECT_EQ(found, 3);
      } else {
        for (const NodeDef& node : output.node()) {
          if (node.name() == "relu") {
            EXPECT_EQ(node.op(), "Identity");
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input(0), "fused_batch_norm");
            found++;
          }
          if (node.name() == "fused_batch_norm") {
            EXPECT_EQ(node.op(), "_FusedBatchNormEx");
            ASSERT_EQ(node.input_size(), 5);
            EXPECT_EQ(node.input(0), "input_cast");
            EXPECT_EQ(node.input(1), "scale");
            EXPECT_EQ(node.input(2), "offset");
            EXPECT_EQ(node.input(3), "mean");
            EXPECT_EQ(node.input(4), "var");

            auto attr = node.attr();
            EXPECT_EQ(attr["num_side_inputs"].i(), 0);
            EXPECT_EQ(attr["activation_mode"].s(), "Relu");
            found++;
          }
        }
        EXPECT_EQ(found, 2);
      }

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
    }
  }
}

TEST_F(MklRemapperTest, FuseMatMulWithBiasAddAndAdd) {
  using ::tensorflow::ops::Placeholder;

  for (const string& add_op : {"BiasAdd", "AddV2", "Add"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape = ops::Placeholder::Shape({4, 32});
    auto input_shape_add = ops::Placeholder::Shape({4, 8});
    auto filter_shape = ops::Placeholder::Shape({32, 8});
    auto bias_shape = ops::Placeholder::Shape({8});

    auto input = Placeholder(s.WithOpName("input"), DT_FLOAT, input_shape);
    auto input_add =
        Placeholder(s.WithOpName("input_add"), DT_FLOAT, input_shape_add);
    auto filter = Placeholder(s.WithOpName("filter"), DT_FLOAT, filter_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DT_FLOAT, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), input, filter);
    Output bias_add;
    if (add_op == "BiasAdd")
      bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);
    else if (add_op == "AddV2")
      bias_add = ops::AddV2(s.WithOpName("bias_add"), matmul, bias);
    else if (add_op == "Add")
      bias_add = ops::Add(s.WithOpName("bias_add"), bias, matmul);

    auto fetch = s.WithOpName("fetch");
    auto add = ops::Add(s.WithOpName("add"), bias_add, input_add);

    ops::Identity(fetch, add);

    auto input_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape.shape_.dim_sizes()));
    auto input_add_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(input_shape_add.shape_.dim_sizes()));
    auto filter_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(filter_shape.shape_.dim_sizes()));
    auto bias_tensor = GenerateRandomTensor<DT_FLOAT>(
        TensorShape(bias_shape.shape_.dim_sizes()));

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_tensor},
                 {"filter", filter_tensor},
                 {"bias", bias_tensor},
                 {"input_add", input_add_tensor}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::AGGRESSIVE);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      auto fetch_node_name = "add";
      if (node.name() == fetch_node_name) {
        EXPECT_EQ("_FusedMatMul", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("filter", node.input(1));
        EXPECT_EQ(2, node.attr().at("num_args").i());
        EXPECT_EQ("bias", node.input(2));
        EXPECT_EQ("input_add", node.input(3));

        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(2, fused_ops.size());
        EXPECT_EQ("BiasAdd", fused_ops[0]);
        EXPECT_EQ("Add", fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(1, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectClose(tensors_expected[0], tensors[0], 0, 1e-6);
  }
}

class RelpaceAddWithBiasAddTest : public GrapplerTest {
 public:
  const string kAddOp = "Add";
  const string kAddV2Op = "AddV2";

 protected:
  template <DataType DTYPE>
  void RelpaceAddWithBiasAddDepthwiseConv2D(const string& add_op) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("add_op: \"" + add_op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_2(mht_2_v, 720, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "RelpaceAddWithBiasAddDepthwiseConv2D");

    using ::tensorflow::ops::Placeholder;

    for (const string& activation : {"None", "Relu", "Relu6", "Elu"}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();

      auto input_shape = Placeholder::Shape({8, 32, 32, 3});
      auto filter_shape = Placeholder::Shape({1, 1, 3, 128});
      auto bias_shape = Placeholder::Shape({128 * 3});

      auto input = Placeholder(s.WithOpName("input"), DTYPE, input_shape);
      auto filter = Placeholder(s.WithOpName("filter"), DTYPE, filter_shape);
      auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

      std::vector<int> strides = {1, 1, 1, 1};
      auto conv = ops::DepthwiseConv2dNative(s.WithOpName("depthwise_conv"),
                                             input, filter, strides, "SAME");

      Output bias_add;
      if (add_op == kAddV2Op) {
        bias_add = ops::AddV2(s.WithOpName(add_op), conv, bias);
      } else {
        bias_add = ops::Add(s.WithOpName(add_op), bias, conv);
      }

      ops::Identity fetch = [&]() -> ops::Identity {
        auto activate = s.WithOpName("activation");
        auto fetch = s.WithOpName("fetch");

        if (activation == "Relu") {
          return ops::Identity(fetch, ops::Relu(activate, bias_add));
        } else if (activation == "Relu6") {
          return ops::Identity(fetch, ops::Relu6(activate, bias_add));
        } else if (activation == "Elu") {
          return ops::Identity(fetch, ops::Elu(activate, bias_add));
        }

        return ops::Identity(fetch, bias_add);
      }();

      auto input_t = GenerateRandomTensor<DTYPE>({8, 32, 32, 3});
      auto filter_t = GenerateRandomTensor<DTYPE>({1, 1, 3, 128});
      auto bias_t = GenerateRandomTensor<DTYPE>({128 * 3});

      GrapplerItem item;
      item.fetch = {"fetch"};
      item.feed = {{"input", input_t}, {"filter", filter_t}, {"bias", bias_t}};
      TF_ASSERT_OK(s.ToGraphDef(&item.graph));

      // Place all nodes on CPU.
      for (int i = 0; i < item.graph.node_size(); ++i) {
        item.graph.mutable_node(i)->set_device("/device:CPU:0");
      }

      Remapper optimizer(RewriterConfig::AGGRESSIVE);
      GraphDef output;
      TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

      int found = 0;
      for (const NodeDef& node : output.node()) {
        if (node.name() == "activation") {
          EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "input");
          EXPECT_EQ(node.input(1), "filter");
          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), 2);
          EXPECT_EQ(fused_ops[0], "BiasAdd");
          EXPECT_EQ(fused_ops[1], activation);

          found++;
        } else if (node.name() == add_op) {
          EXPECT_EQ(node.op(), "_FusedDepthwiseConv2dNative");
          ASSERT_GE(node.input_size(), 3);
          EXPECT_EQ(node.input(0), "input");
          EXPECT_EQ(node.input(1), "filter");
          EXPECT_EQ(node.attr().at("num_args").i(), 1);
          EXPECT_EQ(node.input(2), "bias");

          const auto fused_ops = node.attr().at("fused_ops").list().s();
          ASSERT_EQ(fused_ops.size(), 1);
          EXPECT_EQ(fused_ops[0], "BiasAdd");
          found++;
        }
      }
      EXPECT_EQ(found, 1);

      auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
      ASSERT_EQ(tensors_expected.size(), 1);
      auto tensors = EvaluateNodes(output, item.fetch, item.feed);
      ASSERT_EQ(tensors.size(), 1);

      if (DTYPE == DT_BFLOAT16)
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-2, 1e-2);
      else
        test::ExpectClose(tensors[0], tensors_expected[0], 1e-6);
    }
  }
};

#define CREATE_REPLACEADDWITHBIASADD_TEST_1(ops, addop, dtype)              \
  TEST_F(RelpaceAddWithBiasAddTest, RelpaceAddWithBiasAdd##ops##_##addop) { \
    RelpaceAddWithBiasAddDepthwiseConv2D<dtype>(#addop);                    \
  }
CREATE_REPLACEADDWITHBIASADD_TEST_1(DepthConv2D, AddV2, DT_FLOAT);
CREATE_REPLACEADDWITHBIASADD_TEST_1(DepthConv2D, Add, DT_FLOAT);

class FusedMatMulBiasAddAndGeluTest : public GrapplerTest {
 public:
  template <DataType DTYPE>
  void RunTest() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_3(mht_3_v, 836, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "RunTest");

    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto lhs_shape = ops::Placeholder::Shape({8, 32});
    auto rhs_shape = ops::Placeholder::Shape({32, 64});
    auto bias_shape = ops::Placeholder::Shape({64});

    auto lhs = Placeholder(s.WithOpName("lhs"), DTYPE, lhs_shape);
    auto rhs = Placeholder(s.WithOpName("rhs"), DTYPE, rhs_shape);
    auto bias = Placeholder(s.WithOpName("bias"), DTYPE, bias_shape);

    auto matmul = ops::MatMul(s.WithOpName("matmul"), lhs, rhs);
    auto bias_add = ops::BiasAdd(s.WithOpName("bias_add"), matmul, bias);

    // Add Gelu approximate with smaller ops
    auto square_root_one_half =
        ops::Const(s.WithOpName("square_root_one_half"), {0.707106f}, {});
    auto bias_add_times_square_root_one_half =
        ops::Mul(s.WithOpName("bias_add_times_square_root_one_half"), bias_add,
                 square_root_one_half);
    auto erf =
        ops::Erf(s.WithOpName("erf"), bias_add_times_square_root_one_half);
    auto one = ops::Const(s.WithOpName("one"), {1.0f}, {});
    auto erf_plus_one = ops::AddV2(s.WithOpName("one_plus_erf"), erf, one);
    auto one_half = ops::Const(s.WithOpName("one_half"), {0.5f}, {});
    auto erf_plus_one_times_one_half = ops::Mul(
        s.WithOpName("erf_plus_one_times_one_half"), erf_plus_one, one_half);
    auto gelu = ops::Mul(s.WithOpName("fusion_output"),
                         erf_plus_one_times_one_half, bias_add);
    auto fetch = ops::Identity(s.WithOpName("fetch"), gelu);

    auto lhs_t = GenerateTensorWithSetRandom<DTYPE>({8, 32});
    auto rhs_t = GenerateTensorWithSetRandom<DTYPE>({32, 64});
    auto bias_t = GenerateTensorWithSetRandom<DTYPE>({64});

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"lhs", lhs_t}, {"rhs", rhs_t}, {"bias", bias_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef optimized_graph;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
    int found = 0;
    for (const NodeDef& node : optimized_graph.node()) {
      if (node.name() == "fusion_output") {
        EXPECT_EQ(node.op(), "_FusedMatMul");
        ASSERT_GE(node.input_size(), 3);
        EXPECT_EQ(node.input(0), "lhs");
        EXPECT_EQ(node.input(1), "rhs");
        EXPECT_EQ(node.input(2), "bias");
        EXPECT_EQ(node.attr().at("num_args").i(), 1);
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        ASSERT_EQ(fused_ops.size(), 2);
        EXPECT_EQ(fused_ops[0], "BiasAdd");
        EXPECT_EQ(fused_ops[1], "GeluExact");
        found++;
      }
    }
    EXPECT_EQ(1, found);

    // Evaluate result without remapper fusion
    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 1);

    auto tensors_evaluated =
        EvaluateNodes(optimized_graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_evaluated.size(), 1);
    test::ExpectClose(tensors_evaluated[0], tensors_expected[0], 1e-6);
  }
};

// Gelu has two implementations (1) exact and (2) approximate. Exact cannot be
// used with bfloat16 numeric since the Erf is not supported in bfloat16 yet.
// Here gelu-exact is tested for float32 numeric only. Gelu-approximate test
// is added in tensorflow/python/grappler/remapper_test.py, since the pattern is
// changed by other optimizers before the remapper optimizer.
TEST_F(FusedMatMulBiasAddAndGeluTest, Float32GeluExact) { RunTest<DT_FLOAT>(); }

class MklFusedBatchMatMul : public MklRemapperTest {
 public:
  template <typename T>
  void VerifyFused(bool adjx, bool adjy) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_4(mht_4_v, 928, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "VerifyFused");

    using ::tensorflow::ops::Placeholder;
    using normal_generator = Eigen::internal::NormalRandomGenerator<T>;

    int b0 = 2;
    int b1 = 2;
    int m = 32;
    int k = 16;
    int n = 64;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    auto input_shape =
        adjx ? TensorShape({b0, b1, k, m}) : TensorShape({b0, b1, m, k});
    auto weight_shape =
        adjy ? TensorShape({b0, b1, n, k}) : TensorShape({b0, b1, k, n});
    auto add_shape = TensorShape({b0, 1, m, n});

    auto input_placeholder_shape = ops::Placeholder::Shape(input_shape);
    auto weight_placeholder_shape = ops::Placeholder::Shape(weight_shape);
    auto add_placeholder_shape = ops::Placeholder::Shape(add_shape);

    auto input = Placeholder(s.WithOpName("input"), DataTypeToEnum<T>::v(),
                             input_placeholder_shape);
    auto weight = Placeholder(s.WithOpName("weight"), DataTypeToEnum<T>::v(),
                              weight_placeholder_shape);
    auto addend = Placeholder(s.WithOpName("addend"), DataTypeToEnum<T>::v(),
                              add_placeholder_shape);

    auto batchmatmul =
        ops::BatchMatMulV2(s.WithOpName("batchmatmul"), input, weight,
                           ops::BatchMatMulV2::Attrs().AdjX(adjx).AdjY(adjy));
    auto scale_const = ops::Const(s.WithOpName("scale_const"), {0.1f});
    auto scale =
        ops::Cast(s.WithOpName("scale"), scale_const, DataTypeToEnum<T>::v());
    auto mul = ops::Multiply(s.WithOpName("mul"), batchmatmul, scale);
    auto add = ops::AddV2(s.WithOpName("add"), mul, addend);
    auto fetch = ops::Identity(s.WithOpName("fetch"), add);

    Tensor input_t = Tensor(DataTypeToEnum<T>::v(), input_shape);
    Tensor weight_t = Tensor(DataTypeToEnum<T>::v(), weight_shape);
    Tensor add_t = Tensor(DataTypeToEnum<T>::v(), add_shape);
    input_t.flat<T>() =
        input_t.flat<T>().template setRandom<normal_generator>();
    weight_t.flat<T>() =
        weight_t.flat<T>().template setRandom<normal_generator>();
    add_t.flat<T>() = add_t.flat<T>().template setRandom<normal_generator>();

    GrapplerItem item;
    item.fetch = {"fetch"};
    item.feed = {{"input", input_t}, {"weight", weight_t}, {"addend", add_t}};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_CHECK_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "add") {
        EXPECT_EQ("_MklFusedBatchMatMulV2", node.op());
        EXPECT_EQ("input", node.input(0));
        EXPECT_EQ("weight", node.input(1));
        EXPECT_EQ("scale", node.input(2));
        EXPECT_EQ("addend", node.input(3));
        const auto fused_ops = node.attr().at("fused_ops").list().s();
        EXPECT_EQ(2, fused_ops.size());
        EXPECT_EQ("Mul", fused_ops[0]);
        found++;
        EXPECT_EQ("Add", fused_ops[1]);
        found++;
      }
    }
    EXPECT_EQ(2, found);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    std::is_same<T, float>::value
        ? test::ExpectClose(tensors_expected[0], tensors[0], 1e-6, 1e-6)
        : test::ExpectClose(tensors_expected[0], tensors[0], 1e-2, 1e-2);
  }
};

TEST_F(MklFusedBatchMatMul, MulAndAdd) {
  for (const auto adjx : {false, true})
    for (const auto adjy : {false, true}) {
      this->VerifyFused<float>(adjx, adjy);
      this->VerifyFused<bfloat16>(adjx, adjy);
    }
}

class MklRemapperSwishTest : public GrapplerTest {
 protected:
  template <DataType DTYPE>
  void RunTest() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSmkl_remapper_testDTcc mht_5(mht_5_v, 1030, "", "./tensorflow/core/grappler/optimizers/mkl_remapper_test.cc", "RunTest");

    using ::tensorflow::ops::Placeholder;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    auto mul_shape = ops::Placeholder::Shape({64, 64});

    // We will test four sitations:
    //  1. y = x * sigmoid(x)
    //  2. y = sigmoid(x) * x
    //  3. y = sigmoid(x) * sigmoid(sigmoid(x))
    //  4. y = sigmoid(sigmoid(x)) * sigmoid(x)
    auto input = Placeholder(s.WithOpName("input"), DTYPE, mul_shape);
    auto sigmoid1 = ops::Sigmoid(s.WithOpName("sigmoid1"), input);
    auto sigmoid2 = ops::Sigmoid(s.WithOpName("sigmoid2"), input);
    auto sigmoid3_1 = ops::Sigmoid(s.WithOpName("sigmoid3_1"), input);
    auto sigmoid3_2 = ops::Sigmoid(s.WithOpName("sigmoid3_2"), sigmoid3_1);
    auto sigmoid4_1 = ops::Sigmoid(s.WithOpName("sigmoid4_1"), input);
    auto sigmoid4_2 = ops::Sigmoid(s.WithOpName("sigmoid4_2"), sigmoid4_1);
    auto mul1 = ops::Mul(s.WithOpName("mul1"), input, sigmoid1);
    auto mul2 = ops::Mul(s.WithOpName("mul2"), sigmoid2, input);
    auto mul3 = ops::Mul(s.WithOpName("mul3"), sigmoid3_1, sigmoid3_2);
    auto mul4 = ops::Mul(s.WithOpName("mul4"), sigmoid4_2, sigmoid4_1);
    auto fetch1 = ops::Identity(s.WithOpName("fetch1"), mul1);
    auto fetch2 = ops::Identity(s.WithOpName("fetch2"), mul2);
    auto fetch3 = ops::Identity(s.WithOpName("fetch3"), mul3);
    auto fetch4 = ops::Identity(s.WithOpName("fetch4"), mul4);
    auto mul_t = GenerateTensorWithSetRandom<DTYPE>({64, 64});

    GrapplerItem item;
    item.fetch = {"fetch1", "fetch2", "fetch3", "fetch4"};
    item.feed = {{"input", mul_t}};
    TF_ASSERT_OK(s.ToGraphDef(&item.graph));

    // Place all nodes on CPU.
    for (int i = 0; i < item.graph.node_size(); ++i) {
      item.graph.mutable_node(i)->set_device("/device:CPU:0");
    }

    Remapper optimizer(RewriterConfig::ON);
    GraphDef output;
    TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

    int found = 0;
    for (const NodeDef& node : output.node()) {
      if (node.name() == "mul1") {
        EXPECT_EQ(node.op(), "_MklSwish");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "input");
        ++found;
      }
      if (node.name() == "mul2") {
        EXPECT_EQ(node.op(), "_MklSwish");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "input");
        ++found;
      }
      // mul3 won't be replaced by swish
      // Coz of the limitation of patternMatcher with commutative op
      if (node.name() == "mul4") {
        EXPECT_EQ(node.op(), "_MklSwish");
        ASSERT_EQ(node.input_size(), 1);
        EXPECT_EQ(node.input(0), "sigmoid4_1");
        ++found;
      }
    }
    EXPECT_EQ(found, 3);

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
    ASSERT_EQ(tensors_expected.size(), 4);
    auto tensors = EvaluateNodes(output, item.fetch, item.feed);
    ASSERT_EQ(tensors.size(), 4);
    float atol = 1e-6, rtol = 1e-6;
    if (DTYPE == DT_BFLOAT16) {
      atol = 1e-2;
      rtol = 1e-2;
    }
    test::ExpectClose(tensors[0], tensors_expected[0], atol, rtol);
    test::ExpectClose(tensors[1], tensors_expected[1], atol, rtol);
    test::ExpectClose(tensors[2], tensors_expected[2], atol, rtol);
    test::ExpectClose(tensors[3], tensors_expected[3], atol, rtol);
  }
};

TEST_F(MklRemapperSwishTest, F32) { RunTest<DT_FLOAT>(); }
TEST_F(MklRemapperSwishTest, BF16) { RunTest<DT_BFLOAT16>(); }

}  // namespace grappler
}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
