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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"

#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace tensorrt {

struct TestParam {
  TfTrtConversionParams conv_params;
  std::vector<std::vector<int64>> input_shapes;
};

class TrtConverterTest
    : public ::testing::TestWithParam<std::tuple<TestParam, bool, bool>> {
 protected:
  TrtConverterTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "TrtConverterTest");

    param_ = std::get<0>(GetParam());
    use_variable_ = std::get<1>(GetParam());
    use_function_ = std::get<2>(GetParam());
    input_tensors_ = GetInputTensors();
  }

  // Returns the following graph: output = input * [42, 137] + input
  GraphDef GetGraphDef(PartialTensorShape input_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "GetGraphDef");

    Scope root = Scope::NewRootScope();
    Output c;
    c = ops::Const(root.WithOpName("my_const"), {{42.0f, 137.0f}});
    Output v;
    if (use_variable_) {
      Output v_handle = ops::VarHandleOp(root.WithOpName("my_var"),
                                         DataType::DT_FLOAT, {1, 2});
      v = ops::ReadVariableOp(root.WithOpName("my_var/Read/ReadVariableOp"),
                              v_handle, DataType::DT_FLOAT);
      auto v_init =
          ops::AssignVariableOp(root.WithOpName("my_var/init"), v_handle, c);
    } else {
      v = c;
    }
    const auto attrs = ops::Placeholder::Shape(input_shape);
    auto x = ops::Placeholder(root.WithOpName("input"), DT_FLOAT, attrs);
    auto y = ops::Mul(root.WithOpName("my_mul"), x, v);
    auto z = ops::Add(root.WithOpName("my_add"), x, y);
    auto q = ops::Identity(root.WithOpName("output"), z);

    GraphDef out;
    TF_CHECK_OK(root.ToGraphDef(&out));
    return out;
  }

  GraphDef GetGraphWithFunction(PartialTensorShape input_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "GetGraphWithFunction");

    using ::tensorflow::test::function::GDef;
    using ::tensorflow::test::function::NDef;
    GraphConstructorOptions opts;
    const Tensor kOne = test::AsScalar<float>(1.0f);
    TensorShapeProto value_shape_proto;
    kOne.shape().AsProto(&value_shape_proto);
    TensorShapeProto input_shape_proto;
    input_shape.AsProto(&input_shape_proto);
    NodeDef value_node;
    if (use_variable_) {
      value_node =
          NDef("my_value", "Identity", {"my_var:0"}, {{"T", DT_RESOURCE}});
    } else {
      value_node =
          NDef("my_value", "Identity", {"my_const:0"}, {{"T", DT_FLOAT}});
    }
    GraphDef gdef = GDef(
        {
            NDef("input", "Placeholder", {},
                 {{"dtype", DT_FLOAT}, {"shape", input_shape_proto}}),
            NDef("my_const", "Const", {},
                 {{"dtype", DT_FLOAT}, {"value", kOne}}),
            value_node,
            NDef("call", "StatefulPartitionedCall", {"input", "my_value"},
                 {{"Tin", DataTypeSlice{DT_FLOAT, use_variable_ ? DT_RESOURCE
                                                                : DT_FLOAT}},
                  {"Tout", DataTypeSlice{DT_FLOAT}},
                  {"f", FunctionDefHelper::FunctionRef("f", {})}}),
            NDef("output", "Identity", {"call:0"}, {{"T", DT_FLOAT}}),
        },
        {});
    FunctionDef fdef;
    if (use_variable_) {
      gdef.add_node()->CopyFrom(
          NDef("my_var", "VarHandleOp", {},
               {{"dtype", DT_FLOAT}, {"shape", value_shape_proto}}));

      gdef.add_node()->CopyFrom(NDef("my_var/init", "AssignVariableOp",
                                     {"my_var", "my_const"},
                                     {{"dtype", DT_FLOAT}}));
      gdef.add_node()->CopyFrom(NDef("my_var/Read/ReadVariableOp",
                                     "ReadVariableOp", {"my_var"},
                                     {{"dtype", DT_FLOAT}}));
      // Define function f(x, v) = x * v + x, where v is a variable.
      fdef = FunctionDefHelper::Define(
          "f",                          // Name
          {"x: float", "v: resource"},  // Args
          {"q: float"},                 // Returns
          {},                           // Attr def
          // Nodes
          {{{"my_var/Read/ReadVariableOp"},
            "ReadVariableOp",
            {"v"},
            {{"dtype", DT_FLOAT}}},
           {{"my_mul"},
            "Mul",
            {"x", "my_var/Read/ReadVariableOp"},
            {{"T", DT_FLOAT}}},
           {{"my_add"}, "AddV2", {"x", "my_mul"}, {{"T", DT_FLOAT}}},
           {{"q"}, "Identity", {"my_add"}, {{"T", DT_FLOAT}}}});
    } else {
      // Define function f(x, v) = x * v + x, where v is const value.
      fdef = FunctionDefHelper::Define(
          "f",                       // Name
          {"x: float", "v: float"},  // Args
          {"q: float"},              // Returns
          {},                        // Attr def
          // Nodes
          {{{"my_mul"}, "Mul", {"x", "v"}, {{"T", DT_FLOAT}}},
           {{"my_add"}, "AddV2", {"x", "my_mul"}, {{"T", DT_FLOAT}}},
           {{"q"}, "Identity", {"my_add"}, {{"T", DT_FLOAT}}}});
    }
    gdef.mutable_library()->add_function()->CopyFrom(fdef);

    return gdef;
  }

  // Returns the following graph: output = input * [42, 137] + input
  MetaGraphDef GetModel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_3(mht_3_v, 333, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "GetModel");

    PartialTensorShape shape({-1, 2});
    MetaGraphDef out;
    if (use_function_) {
      *(out.mutable_graph_def()) = GetGraphWithFunction(shape);
    } else {
      *(out.mutable_graph_def()) = GetGraphDef(shape);
    }
    VLOG(2) << out.graph_def().DebugString();
    TensorShapeProto shape_proto;
    shape.AsProto(&shape_proto);
    SignatureDef signature_def;
    (*signature_def.mutable_inputs())["input"].set_name("input:0");
    (*signature_def.mutable_inputs())["input"].set_dtype(DT_FLOAT);
    (*signature_def.mutable_inputs())["input"].mutable_tensor_shape()->CopyFrom(
        shape_proto);
    (*signature_def.mutable_outputs())["output"].set_name("output:0");
    (*signature_def.mutable_outputs())["output"].set_dtype(DT_FLOAT);
    (*signature_def.mutable_outputs())["output"]
        .mutable_tensor_shape()
        ->CopyFrom(shape_proto);
    (*out.mutable_signature_def())["serving_default"] = signature_def;

    VLOG(2) << signature_def.DebugString();
    return out;
  }

  Status GetSavedModelBundle(SavedModelBundle* bundle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_4(mht_4_v, 363, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "GetSavedModelBundle");

    bundle->meta_graph_def = GetModel();
    Session* session = nullptr;
    TF_RETURN_IF_ERROR(NewSession(tensorflow::SessionOptions(), &session));
    TF_RETURN_IF_ERROR(session->Create(bundle->meta_graph_def.graph_def()));
    bundle->session.reset(session);
    TF_RETURN_IF_ERROR(session->Run(/* inputs */ {}, /*outputs*/ {},
                                    /*targets*/ {"my_var/init"}, nullptr));
    return Status::OK();
  }

  // Confirms that we have a TRT node with the correct attributes.
  void CheckTrtNode(const GraphDef& converted_graph_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_5(mht_5_v, 378, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "CheckTrtNode");

    int n_trt_ops = 0;
    string op_name{"TRTEngineOp"};
    for (const auto& node : converted_graph_def.node()) {
      if (!op_name.compare(node.op())) {
        n_trt_ops++;
        const auto& attr = node.attr();
        EXPECT_EQ(attr.at("static_engine").b(),
                  param_.conv_params.convert_to_static_engine);
        if (param_.conv_params.convert_to_static_engine) {
          VLOG(2) << "Found serialized segment with size "
                  << attr.at("serialized_segment").s().size();
          EXPECT_GT(attr.at("serialized_segment").s().size(), 0);
        }
      }
    }
    EXPECT_EQ(n_trt_ops, 1);
  }

  // Creates a list of input tensors, they will be used to build the engines.
  std::vector<std::vector<Tensor>> GetInputTensors() {
    std::vector<std::vector<Tensor>> input_tensors;
    for (const std::vector<int64>& shape : param_.input_shapes) {
      Tensor tensor(DT_FLOAT, TensorShape(shape));
      test::FillIota(&tensor, 1.0f);
      input_tensors.push_back({tensor});
    }
    return input_tensors;
  }

  void RunAndCompareResults(Session* session,
                            const GraphDef& converted_graph_def) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_6(mht_6_v, 412, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "RunAndCompareResults");

    // Create a session to execute the converted graph.
    Session* p_session = nullptr;
    TF_EXPECT_OK(NewSession(SessionOptions(), &p_session));
    std::unique_ptr<tensorflow::Session> trt_session(p_session);
    TF_EXPECT_OK(trt_session->Create(converted_graph_def));

    // Run models and compare the output.
    for (const std::vector<Tensor>& input : input_tensors_) {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(
          session->Run({{"input", input.at(0)}}, {"output"}, {}, &outputs));
      std::cout << outputs.at(0).DebugString() << std::endl;

      std::vector<Tensor> trt_outputs;
      TF_EXPECT_OK(trt_session->Run({{"input", input.at(0)}}, {"output"}, {},
                                    &trt_outputs));
      std::cout << trt_outputs.at(0).DebugString() << std::endl;
      ASSERT_EQ(outputs.size(), 1);
      ASSERT_EQ(trt_outputs.size(), 1);
      tensorflow::test::ExpectEqual(outputs[0], trt_outputs[0]);
    }
  }

  void ConvertAndRunFrozenGraph() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_7(mht_7_v, 439, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "ConvertAndRunFrozenGraph");

    MetaGraphDef meta_graph_def = GetModel();

    StatusOr<GraphDef> result = tensorrt::ConvertAndBuild(
        meta_graph_def.graph_def(), {"input"}, {"output"}, input_tensors_,
        param_.conv_params);
    TF_ASSERT_OK(result.status());
    const GraphDef& converted_graph_def = result.ValueOrDie();
    CheckTrtNode(converted_graph_def);

    // Create a session to execute the original graph.
    Session* p_session = nullptr;
    TF_EXPECT_OK(NewSession(SessionOptions(), &p_session));
    std::unique_ptr<tensorflow::Session> session(p_session);
    TF_EXPECT_OK(session->Create(meta_graph_def.graph_def()));

    RunAndCompareResults(session.get(), converted_graph_def);
  }

  void ConvertAndRunSavedModel() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPStrt_convert_api_testDTcc mht_8(mht_8_v, 461, "", "./tensorflow/compiler/tf2tensorrt/trt_convert_api_test.cc", "ConvertAndRunSavedModel");

    SavedModelBundle bundle;
    TF_CHECK_OK(GetSavedModelBundle(&bundle));

    StatusOr<GraphDef> result = tensorrt::ConvertAndBuild(
        &bundle, "serving_default", input_tensors_, param_.conv_params);
    TF_ASSERT_OK(result.status());
    const GraphDef& converted_graph_def = result.ValueOrDie();
    CheckTrtNode(converted_graph_def);

    RunAndCompareResults(bundle.GetSession(), converted_graph_def);
  }

  TestParam param_;
  bool use_variable_;
  bool use_function_;
  std::vector<std::vector<Tensor>> input_tensors_;
};

INSTANTIATE_TEST_CASE_P(
    TrtConverterTestInstantiation, TrtConverterTest,
    ::testing::Combine(
        ::testing::Values(
            // Dynamic shape mode test with conver_to_static_engine=true.
            TestParam{TfTrtConversionParams{
                          1 << 20,  // max workspace size
                          TrtPrecisionMode::FP32,
                          3,      // minimum_segment_size
                          1,      // max_cached_engines
                          false,  // use_calibration
                          true,   // use_dynamic_shape
                          ProfileStrategy::kOptimal,
                          true,  // allow_build_at_runtime
                          true   // convert_to_static_engine
                      },
                      {{1, 2}, {4, 2}}},
            // Implicit batch mode test with conver_to_static_engine=true.
            TestParam{TfTrtConversionParams{
                          1 << 20,  // max workspace size
                          TrtPrecisionMode::FP16,
                          3,      // minimum_segment_size
                          1,      // max_cached_engines
                          false,  // use_calibration
                          false,  // use_dynamic_shape
                          ProfileStrategy::kRange,
                          true,  // allow_build_at_runtime
                          true   // convert_to_static_engine
                      },
                      {{1, 2}}},
            // Dynamic shape mode test convert_to_static_engine=false: we cannot
            // save the engines, therefore we do not generate profiles. A single
            // engine will be built during runtime, with profile that matches
            // the first shape ({1,2}). The second shape will run as native
            // segment.
            TestParam{TfTrtConversionParams{
                          1 << 20,  // max workspace size
                          TrtPrecisionMode::FP32,
                          3,      // minimum_segment_size
                          1,      // max_cached_engines
                          false,  // use_calibration
                          true,   // use_dynamic_shape
                          ProfileStrategy::kOptimal,
                          true,  // allow_build_at_runtime
                          false  // convert_to_static_engine
                      },
                      {{1, 2}, {4, 2}}},
            // Implicit batch mode test with convert_to_static_engine=false.
            // We will have two engines in the cache to handle the two shapes.
            TestParam{TfTrtConversionParams{
                          1 << 20,  // max workspace size
                          TrtPrecisionMode::FP16,
                          3,      // minimum_segment_size
                          2,      // max_cached_engines
                          false,  // use_calibration
                          false,  // use_dynamic_shape
                          ProfileStrategy::kRange,
                          true,  // allow_build_at_runtime
                          false  // convert_to_static_engine
                      },
                      {{1, 2}, {4, 2}}}),
        ::testing::Values(false, true),    // use_variables
        ::testing::Values(false, true)));  // use_function

TEST_P(TrtConverterTest, Basic) {
  if (use_variable_) {
    ConvertAndRunSavedModel();
  } else {
    ConvertAndRunFrozenGraph();
  }
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
