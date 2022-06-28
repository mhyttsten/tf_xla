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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc() {
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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status FoldOldBatchNorms(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);

class FoldOldBatchNormsTest : public ::testing::Test {
 protected:
  void TestFoldOldBatchNorms() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldOldBatchNorms");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // This is needed because we're trying to convert over a deprecated op which
    // should only be present in older GraphDef files. Without this we see a
    // deprecation error.
    // This is justified because we're trying to test a tool that is expected to
    // run on legacy files, to help users convert over to less problematic
    // versions.
    NodeDef batch_norm_node;
    batch_norm_node.set_op("BatchNormWithGlobalNormalization");
    batch_norm_node.set_name("output");
    AddNodeInput("conv_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("variance_epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("scale_after_normalization", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;
    original_graph_def.mutable_versions()->set_producer(8);

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("BatchNormWithGlobalNormalization", node.op());
    }
  }

  void TestFoldOldBatchNormsAfterDepthwiseConv2dNative() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_1(mht_1_v, 296, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldOldBatchNormsAfterDepthwiseConv2dNative");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = DepthwiseConv2dNative(root.WithOpName("conv_op"), input_op,
                                           weights_op, {1, 1, 1, 1}, "VALID");

    Tensor mean_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f, 30.0f, 40.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f, 0.75f, 1.0f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f, 1.1f, 1.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f, 3.0f, 4.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    NodeDef batch_norm_node;
    batch_norm_node.set_op("BatchNormWithGlobalNormalization");
    batch_norm_node.set_name("output");
    AddNodeInput("conv_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("variance_epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("scale_after_normalization", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;
    original_graph_def.mutable_versions()->set_producer(8);

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("BatchNormWithGlobalNormalization", node.op());
    }
  }

  void TestFoldFusedBatchNorms() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_2(mht_2_v, 377, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldFusedBatchNorms");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    NodeDef batch_norm_node;
    batch_norm_node.set_op("FusedBatchNorm");
    batch_norm_node.set_name("output");
    AddNodeInput("conv_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("is_training", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 2e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("FusedBatchNorm", node.op());
    }
  }

  void TestFoldFusedBatchNormsAfterDepthwiseConv2dNative() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_3(mht_3_v, 457, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldFusedBatchNormsAfterDepthwiseConv2dNative");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = DepthwiseConv2dNative(root.WithOpName("conv_op"), input_op,
                                           weights_op, {1, 1, 1, 1}, "VALID");

    Tensor mean_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f, 30.0f, 40.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f, 0.75f, 1.0f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f, 1.1f, 1.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({4}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f, 3.0f, 4.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    NodeDef batch_norm_node;
    batch_norm_node.set_op("FusedBatchNorm");
    batch_norm_node.set_name("output");
    AddNodeInput("conv_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("is_training", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectClose(original_outputs[0], fused_outputs[0], /*atol=*/2e-5,
                      /*rtol=*/2e-5);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("FusedBatchNorm", node.op());
    }
  }

  void TestFoldFusedBatchNormsWithConcat(const bool split) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_4(mht_4_v, 538, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldFusedBatchNormsWithConcat");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    // If split is true, concat two inputs at dim=3; otherwise, concat at dim 2.
    auto input_shape =
        split ? TensorShape({1, 1, 6, 2}) : TensorShape({1, 1, 12, 1});
    Tensor input_data(DT_FLOAT, input_shape);
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});

    Output input0_op =
        Const(root.WithOpName("input_op0"), Input::Initializer(input_data));
    // If split is true, concat two inputs at dim=3; otherwise, concat at dim 2.
    // The final output shape of concat is always {1, 2, 2, 2}.
    auto weight_shape =
        split ? TensorShape({1, 2, 2, 1}) : TensorShape({1, 2, 1, 2});
    Tensor weights0_data(DT_FLOAT, weight_shape);
    test::FillValues<float>(&weights0_data, {1.0f, 2.0f, 3.0f, 4.0f});
    Output weights0_op = Const(root.WithOpName("weights1_op"),
                               Input::Initializer(weights0_data));
    Output conv0_op = Conv2D(root.WithOpName("conv1_op"), input0_op,
                             weights0_op, {1, 1, 1, 1}, "VALID");

    Output input1_op =
        Const(root.WithOpName("input1_op"), Input::Initializer(input_data));
    Tensor weights1_data(DT_FLOAT, weight_shape);
    test::FillValues<float>(&weights1_data, {1.0f, 2.0f, 3.0f, 4.0f});
    Output weights1_op = Const(root.WithOpName("weights1_op"),
                               Input::Initializer(weights1_data));
    Output conv1_op = Conv2D(root.WithOpName("conv1_op"), input1_op,
                             weights1_op, {1, 1, 1, 1}, "VALID");

    Tensor shape_tensor(DT_INT32, TensorShape({}));
    // Concat at dim 3 if split; otherwise, concat at dim 2.
    int32_t concat_axis = split ? 3 : 2;
    test::FillValues<int32>(&shape_tensor, {concat_axis});
    Output shape_op =
        Const(root.WithOpName("shape_op"), Input::Initializer(shape_tensor));
    Output concat_op =
        Concat(root.WithOpName("concat_op"), {conv0_op, conv1_op}, shape_op);

    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {10.0f, 20.0f});
    Output mean_op =
        Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {0.25f, 0.5f});
    Output variance_op = Const(root.WithOpName("variance_op"),
                               Input::Initializer(variance_data));

    Tensor beta_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&beta_data, {0.1f, 0.6f});
    Output beta_op =
        Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

    Tensor gamma_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
    Output gamma_op =
        Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    NodeDef batch_norm_node;
    batch_norm_node.set_op("FusedBatchNorm");
    batch_norm_node.set_name("output");
    AddNodeInput("concat_op", &batch_norm_node);
    AddNodeInput("gamma_op", &batch_norm_node);
    AddNodeInput("beta_op", &batch_norm_node);
    AddNodeInput("mean_op", &batch_norm_node);
    AddNodeInput("variance_op", &batch_norm_node);
    SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
    SetNodeAttr("epsilon", 0.00001f, &batch_norm_node);
    SetNodeAttr("is_training", false, &batch_norm_node);
    *(original_graph_def.mutable_node()->Add()) = batch_norm_node;

    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    GraphDef fused_graph_def;
    TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                   &fused_graph_def));

    std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(fused_session->Create(fused_graph_def));
    std::vector<Tensor> fused_outputs;
    TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

    test::ExpectClose(original_outputs[0], fused_outputs[0]);

    for (const NodeDef& node : fused_graph_def.node()) {
      EXPECT_NE("FusedBatchNorm", node.op());
    }
  }
};

void TestFoldFusedBatchNormsWithBatchToSpace() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_old_batch_norms_testDTcc mht_5(mht_5_v, 642, "", "./tensorflow/tools/graph_transforms/fold_old_batch_norms_test.cc", "TestFoldFusedBatchNormsWithBatchToSpace");

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  Tensor input_data(DT_FLOAT, TensorShape({2, 1, 3, 2}));
  test::FillValues<float>(
      &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                    -5.0f, -3.0f, -6.0f});
  Output input_op =
      Const(root.WithOpName("input_op"), Input::Initializer(input_data));

  Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
  test::FillValues<float>(&weights_data,
                          {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
  Output weights_op =
      Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

  Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                          {1, 1, 1, 1}, "VALID");

  Tensor block_shape_data(DT_INT32, TensorShape({2}));
  test::FillValues<int32>(&block_shape_data, {1, 2});
  Output block_shape_op = Const(root.WithOpName("block_shape_op"),
                                Input::Initializer(block_shape_data));

  Tensor crops_data(DT_INT32, TensorShape({2, 2}));
  test::FillValues<int32>(&crops_data, {0, 0, 0, 1});
  Output crops_op =
      Const(root.WithOpName("crops_op"), Input::Initializer(crops_data));

  Output batch_to_space_op =
      BatchToSpaceND(root.WithOpName("batch_to_space_op"), conv_op,
                     block_shape_op, crops_data);

  Tensor mean_data(DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&mean_data, {10.0f, 20.0f});
  Output mean_op =
      Const(root.WithOpName("mean_op"), Input::Initializer(mean_data));

  Tensor variance_data(DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&variance_data, {0.25f, 0.5f});
  Output variance_op =
      Const(root.WithOpName("variance_op"), Input::Initializer(variance_data));

  Tensor beta_data(DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&beta_data, {0.1f, 0.6f});
  Output beta_op =
      Const(root.WithOpName("beta_op"), Input::Initializer(beta_data));

  Tensor gamma_data(DT_FLOAT, TensorShape({2}));
  test::FillValues<float>(&gamma_data, {1.0f, 2.0f});
  Output gamma_op =
      Const(root.WithOpName("gamma_op"), Input::Initializer(gamma_data));

  GraphDef original_graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

  NodeDef batch_norm_node;
  batch_norm_node.set_op("FusedBatchNorm");
  batch_norm_node.set_name("output");
  AddNodeInput("batch_to_space_op", &batch_norm_node);
  AddNodeInput("gamma_op", &batch_norm_node);
  AddNodeInput("beta_op", &batch_norm_node);
  AddNodeInput("mean_op", &batch_norm_node);
  AddNodeInput("variance_op", &batch_norm_node);
  SetNodeAttr("T", DT_FLOAT, &batch_norm_node);
  SetNodeAttr("epsilon", 0.00001f, &batch_norm_node);
  SetNodeAttr("is_training", false, &batch_norm_node);
  *(original_graph_def.mutable_node()->Add()) = batch_norm_node;

  std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
  TF_ASSERT_OK(original_session->Create(original_graph_def));
  std::vector<Tensor> original_outputs;
  TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

  GraphDef fused_graph_def;
  TF_ASSERT_OK(FoldOldBatchNorms(original_graph_def, {{}, {"output"}},
                                 &fused_graph_def));

  std::unique_ptr<Session> fused_session(NewSession(SessionOptions()));
  TF_ASSERT_OK(fused_session->Create(fused_graph_def));
  std::vector<Tensor> fused_outputs;
  TF_ASSERT_OK(fused_session->Run({}, {"output"}, {}, &fused_outputs));

  test::ExpectTensorNear<float>(original_outputs[0], fused_outputs[0], 1e-5);

  for (const NodeDef& node : fused_graph_def.node()) {
    EXPECT_NE("FusedBatchNormWithBatchToSpace", node.op());
  }
}

TEST_F(FoldOldBatchNormsTest, TestFoldOldBatchNorms) {
  TestFoldOldBatchNorms();
}

TEST_F(FoldOldBatchNormsTest, TestFoldFusedBatchNorms) {
  TestFoldFusedBatchNorms();
}

TEST_F(FoldOldBatchNormsTest, TestFoldFusedBatchNormsWithConcat) {
  // Test axis is not 3, so all weights and offsets are fused to each of inputs
  // of conv2d.
  TestFoldFusedBatchNormsWithConcat(/*split=*/true);
  // Test axis = 3, BatchNorm weights and offsets will be split before fused
  // with conv2d weights.
  TestFoldFusedBatchNormsWithConcat(/*split=*/false);
}

TEST_F(FoldOldBatchNormsTest, TestFoldFusedBatchNormsWithBatchToSpace) {
  TestFoldFusedBatchNormsWithBatchToSpace();
}

TEST_F(FoldOldBatchNormsTest, TestFoldOldBatchNormsAfterDepthwiseConv2dNative) {
  TestFoldOldBatchNormsAfterDepthwiseConv2dNative();
}

TEST_F(FoldOldBatchNormsTest,
       TestFoldFusedBatchNormsAfterDepthwiseConv2dNative) {
  TestFoldFusedBatchNormsAfterDepthwiseConv2dNative();
}

}  // namespace graph_transforms
}  // namespace tensorflow
