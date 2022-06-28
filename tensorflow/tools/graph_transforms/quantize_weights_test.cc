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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc() {
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
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status QuantizeWeights(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def);

class QuantizeWeightsTest : public ::testing::Test {
 protected:
  void BuildGraphDef(const TensorShape& input_shape,
                     std::initializer_list<float> input_values,
                     const TensorShape& weight_shape,
                     std::initializer_list<float> weight_values,
                     GraphDef* original_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/tools/graph_transforms/quantize_weights_test.cc", "BuildGraphDef");

    auto root = tensorflow::Scope::DisabledShapeInferenceScope();

    Tensor input_data(DT_FLOAT, input_shape);
    test::FillValues<float>(&input_data, input_values);
    Output input_op =
        ops::Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, weight_shape);
    test::FillValues<float>(&weights_data, weight_values);
    Output weights_op = ops::Const(root.WithOpName("weights_op"),
                                   Input::Initializer(weights_data));

    Output conv_op = ops::Conv2D(root.WithOpName("output"), input_op,
                                 weights_op, {1, 1, 1, 1}, "VALID");

    TF_ASSERT_OK(root.ToGraphDef(original_graph_def));
  }

  void TestQuantizeWeights() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc mht_1(mht_1_v, 233, "", "./tensorflow/tools/graph_transforms/quantize_weights_test.cc", "TestQuantizeWeights");

    GraphDef original_graph_def;
    BuildGraphDef({1, 1, 6, 2},
                  {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                   -5.0f, -3.0f, -6.0f},
                  {1, 2, 2, 10},
                  {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
                   3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                   0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f,
                   0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f},
                  &original_graph_def);

    TransformFuncContext context;
    context.output_names = {"output"};
    context.params["minimum_size"] = {"16"};
    GraphDef quantized_graph_def;
    TF_ASSERT_OK(
        QuantizeWeights(original_graph_def, context, &quantized_graph_def));

    // Verify the structure of the quantized graph.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(quantized_graph_def, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("input_op"));
    const NodeDef* q_input_op = node_lookup.at("input_op");
    EXPECT_EQ(DT_FLOAT, q_input_op->attr().at("dtype").type());
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    const NodeDef* q_weights_op = node_lookup.at("weights_op");
    EXPECT_EQ("Dequantize", q_weights_op->op());
    const string& weights_const_name =
        NodeNameFromInput(q_weights_op->input(0));
    EXPECT_EQ(1, node_lookup.count(weights_const_name));
    const NodeDef* q_weights_const = node_lookup.at(weights_const_name);
    EXPECT_EQ("Const", q_weights_const->op());
    EXPECT_EQ(DT_QUINT8, q_weights_const->attr().at("dtype").type());

    // Run the original graph.
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"output"}, {}, &original_outputs));

    // Run the quantized graph.
    std::unique_ptr<Session> quantized_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(quantized_session->Create(quantized_graph_def));
    std::vector<Tensor> quantized_outputs;
    TF_ASSERT_OK(
        quantized_session->Run({}, {"output"}, {}, &quantized_outputs));

    // Compare the results
    test::ExpectTensorNear<float>(original_outputs[0], quantized_outputs[0],
                                  0.5);
  }
};

TEST_F(QuantizeWeightsTest, TestQuantizeWeights) { TestQuantizeWeights(); }

TEST_F(QuantizeWeightsTest, RangesAlwaysIncludeZero) {
  GraphDef original_graph_def;
  BuildGraphDef({1, 1, 4, 4},
                {-1.0f, -4.0f, -2.0f, -5.0f, -1.0f, -4.0f, -2.0f, -5.0f, -1.0f,
                 -4.0f, -2.0f, -5.0f, -1.0f, -4.0f, -2.0f, -5.0f},
                {1, 2, 2, 10},
                {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
                 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f,
                 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f,
                 0.3f, 0.4f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f},
                &original_graph_def);
  TransformFuncContext context;
  context.output_names = {"output"};
  context.params["minimum_size"] = {"16"};
  GraphDef quantized_graph_def;
  TF_ASSERT_OK(
      QuantizeWeights(original_graph_def, context, &quantized_graph_def));

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(quantized_graph_def, &node_lookup);

  auto expected_tensor = [](float value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc mht_2(mht_2_v, 313, "", "./tensorflow/tools/graph_transforms/quantize_weights_test.cc", "lambda");

    Tensor tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&tensor, {value});
    return tensor;
  };
  auto existing_tensor = [&node_lookup](string op) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weights_testDTcc mht_3(mht_3_v, 322, "", "./tensorflow/tools/graph_transforms/quantize_weights_test.cc", "lambda");

    const NodeDef* node_def = node_lookup.at(op);
    CHECK(node_def);
    return GetNodeTensorAttr(*node_def, "value");
  };

  // The max of input_op is moved from -1.0 to 0.0.
  test::ExpectTensorNear<float>(
      expected_tensor(-5.0), existing_tensor("input_op_quantized_min"), 1e-5);
  test::ExpectTensorNear<float>(
      expected_tensor(0.0), existing_tensor("input_op_quantized_max"), 1e-5);

  // The min of weights_op is moved from 0.1 to 0.0.
  test::ExpectTensorNear<float>(
      expected_tensor(0.0), existing_tensor("weights_op_quantized_min"), 1e-5);
  test::ExpectTensorNear<float>(
      expected_tensor(4.0), existing_tensor("weights_op_quantized_max"), 1e-5);
}

}  // namespace graph_transforms
}  // namespace tensorflow
