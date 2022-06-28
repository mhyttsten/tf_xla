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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/grappler/optimizers/data/split_utils.h"

#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

REGISTER_OP("TwoOutputs")
    .Input("x: float")
    .Output("y1: float")
    .Output("y2: float")
    .SetShapeFn(shape_inference::UnknownShape);

class TwoOutputsKernel : public OpKernel {
 public:
  explicit TwoOutputsKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "TwoOutputsKernel");
}
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "Compute");

    Tensor* output0;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &output0));
    CPUDevice d = ctx->eigen_device<CPUDevice>();
    output0->flat<float>().device(d) = ctx->input(0).flat<float>() + float{1};
    Tensor* output1;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {}, &output1));
    output1->flat<float>().device(d) = ctx->input(0).flat<float>() + float{2};
  }
};

REGISTER_KERNEL_BUILDER(Name("TwoOutputs").Device(DEVICE_CPU),
                        TwoOutputsKernel);

namespace grappler {
namespace split_utils {
namespace {

// Run a FunctionDef on the given input tensors.
void RunFunction(const FunctionDef& func,
                 const std::vector<Tensor>& input_tensors,
                 std::vector<Tensor>* results) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "RunFunction");

  using test::function::NDef;
  std::vector<NodeDef> nodes;
  std::vector<string> inputs;
  std::vector<std::pair<std::string, Tensor>> feeds;
  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& tensor = input_tensors[i];
    string input_name = absl::StrCat("input_", i);
    nodes.push_back(
        NDef(input_name, "Placeholder", {}, {{"dtype", tensor.dtype()}}));
    inputs.push_back(input_name);
    feeds.push_back({input_name, tensor});
  }
  nodes.push_back(NDef("f", func.signature().name(), inputs, {}));
  GraphDef graph_def = test::function::GDef(nodes, {func});
  VLOG(2) << "Original graph def:\n" << graph_def.DebugString() << "\n";

  std::unique_ptr<Session> session;
  {
    Session* new_session;
    TF_ASSERT_OK(NewSession({}, &new_session));
    session.reset(new_session);
  }
  TF_ASSERT_OK(session->Create(graph_def));
  std::vector<string> outputs;
  const int num_outputs = func.ret_size();
  for (int i = 0; i < num_outputs; i++) {
    outputs.push_back(absl::StrCat("f:", i));
  }
  TF_ASSERT_OK(session->Run(feeds, outputs, {}, results));
}

// Run two FunctionDefs in sequence. Returns second(first(*input_tensors)) in
// *results.
void RunSplitFunctions(const FunctionDef& first, const FunctionDef& second,
                       const std::vector<Tensor>& input_tensors,
                       int num_captured_inputs, std::vector<Tensor>* results) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "RunSplitFunctions");

  using test::function::NDef;
  std::vector<NodeDef> nodes;
  std::vector<string> first_inputs;
  std::vector<std::pair<std::string, Tensor>> feeds;
  for (int i = 0; i < input_tensors.size(); i++) {
    const Tensor& tensor = input_tensors[i];
    string input_name = absl::StrCat("input_", i);
    nodes.push_back(
        NDef(input_name, "Placeholder", {}, {{"dtype", tensor.dtype()}}));
    first_inputs.push_back(input_name);
    feeds.push_back({input_name, tensor});
  }
  nodes.push_back(NDef("first", first.signature().name(), first_inputs, {}));
  std::vector<string> second_inputs;
  const int num_outputs_of_first = first.ret_size();
  for (int i = 0; i < num_outputs_of_first; i++) {
    second_inputs.push_back(absl::StrCat("first:", i));
  }
  for (int i = input_tensors.size() - num_captured_inputs;
       i < input_tensors.size(); i++) {
    second_inputs.push_back(absl::StrCat("input_", i));
  }
  nodes.push_back(NDef("second", second.signature().name(), second_inputs, {}));
  GraphDef graph_def = test::function::GDef(nodes, {first, second});
  VLOG(2) << "Split graph def:\n" << graph_def.DebugString() << "\n";

  std::unique_ptr<Session> session;
  {
    Session* new_session;
    TF_ASSERT_OK(NewSession({}, &new_session));
    session.reset(new_session);
  }
  TF_ASSERT_OK(session->Create(graph_def));
  std::vector<string> second_outputs;
  const int num_outputs_of_second = second.ret_size();
  for (int i = 0; i < num_outputs_of_second; i++) {
    second_outputs.push_back(absl::StrCat("second:", i));
  }
  TF_ASSERT_OK(session->Run(feeds, second_outputs, {}, results));
}

// Tests that FunctionDef `orig` was split correctly into `first` and `second`,
// by asserting that orig(*input_tensors) returns the same tensors as
// second(first(*input_tensors)).
void CheckSplitFunctions(const FunctionDef& orig, const FunctionDef& first,
                         const FunctionDef& second,
                         const std::vector<Tensor>& input_tensors,
                         int num_captured_inputs = 0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_4(mht_4_v, 337, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "CheckSplitFunctions");

  std::vector<Tensor> orig_outputs;
  ASSERT_NO_FATAL_FAILURE(RunFunction(orig, input_tensors, &orig_outputs));
  std::vector<Tensor> second_outputs;
  ASSERT_NO_FATAL_FAILURE(RunSplitFunctions(
      first, second, input_tensors, num_captured_inputs, &second_outputs));
  ASSERT_EQ(orig_outputs.size(), second_outputs.size());
  for (int i = 0; i < orig_outputs.size() - 1; i++) {
    VLOG(1) << "Output " << i
            << ": orig output=" << orig_outputs[i].DebugString()
            << ", split output=" << second_outputs[i].DebugString();
    test::ExpectClose(orig_outputs[i], second_outputs[i], 0.001, 0.001);
  }
}

std::vector<string> GetNodeNames(const FunctionDef& function) {
  std::vector<string> node_names;
  for (const NodeDef& node : function.node_def()) {
    node_names.push_back(node.name());
  }
  return node_names;
}

const NodeDef& GetNode(const std::string& name, const FunctionDef& function) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSsplit_utils_testDTcc mht_5(mht_5_v, 364, "", "./tensorflow/core/grappler/optimizers/data/split_utils_test.cc", "GetNode");

  int i = function_utils::FindFunctionNodeWithName(name, function);
  CHECK_GE(i, 0) << "Could not find node \"" << name << "\" in FunctionDef:\n"
                 << function.DebugString();
  return function.node_def(i);
}

TEST(SplitUtilsTest, Basic) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",                // function_name
      {"a1: float", "a2: float"},  // in_def
      {"o1: float"},               // out_def
      {},                          // attr_def
      // node_def
      {
          {{"i1"}, "Identity", {"a1"}, {{"T", DT_FLOAT}}},
          {{"i2"}, "Identity", {"a2"}, {{"T", DT_FLOAT}}},
          {{"add"}, "AddV2", {"i1:output", "i2:output"}, {{"T", DT_FLOAT}}},
          {{"add2"}, "AddV2", {"add:z", "a2"}, {{"T", DT_FLOAT}}},
      },
      // ret_def
      {{"o1", "add2:z"}});

  SplitResults results =
      SplitFunction(orig, {"i1"}, /*num_captured_inputs=*/0, function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("i1"));
  ASSERT_THAT(GetNodeNames(second), UnorderedElementsAre("i2", "add", "add2"));
  CheckSplitFunctions(orig, first, second,
                      {test::AsScalar<float>(1.0), test::AsScalar<float>(2.0)});

  // Test signatures
  ASSERT_EQ(first.signature().output_arg_size(), 2);
  ASSERT_EQ(first.signature().output_arg(0).name(), "output_0");
  ASSERT_EQ(first.signature().output_arg(0).description(),
            "Output 0, corresponding to input a2");
  ASSERT_EQ(first.signature().output_arg(1).name(), "output_1");
  ASSERT_EQ(first.signature().output_arg(1).description(),
            "Output 1, corresponding to input i1:output");

  ASSERT_EQ(second.signature().input_arg_size(), 2);
  ASSERT_EQ(second.signature().input_arg(0).name(), "input_0");
  ASSERT_EQ(second.signature().input_arg(0).description(), "Input 0");
  ASSERT_EQ(second.signature().input_arg(1).name(), "input_1");
  ASSERT_EQ(second.signature().input_arg(1).description(), "Input 1");
}

TEST(SplitUtilsTest, MultiOutput) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",                             // function_name
      {"a1: float", "a2: float"},               // in_def
      {"o1: float", "o2: float", "o3: float"},  // out_def
      {},                                       // attr_def
      // node_def
      {{{"t1"}, "TwoOutputs", {"a1"}, {}},
       {{"i1"}, "Identity", {"t1:y1"}, {{"T", DT_FLOAT}}},
       {{"i2"}, "Identity", {"a2"}, {{"T", DT_FLOAT}}}},
      // ret_def
      {{"o1", "i1:output"}, {"o2", "t1:y2"}, {"o3", "i2:output"}});

  SplitResults results =
      SplitFunction(orig, {"t1"}, /*num_captured_inputs=*/0, function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("t1"));
  ASSERT_THAT(GetNodeNames(second), UnorderedElementsAre("i1", "i2"));
  CheckSplitFunctions(orig, first, second,
                      {test::AsScalar<float>(1.0), test::AsScalar<float>(2.0)});
}

// Tests case where two nodes in the second function have the same input tensor,
// where the input tensor is from the first function.
TEST(SplitUtilsTest, MultipleNodesHaveSameInput) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",                // function_name
      {"a1: float", "a2: float"},  // in_def
      {"o1: float", "o2: float"},  // out_def
      {},                          // attr_def
      // node_def
      {
          {{"i1"}, "Identity", {"a1"}, {{"T", DT_FLOAT}}},
          {{"add"}, "AddV2", {"i1:output", "a2"}, {{"T", DT_FLOAT}}},
          {{"mul"}, "Mul", {"i1:output", "a2"}, {{"T", DT_FLOAT}}},
      },
      // ret_def
      {{"o1", "add:z"}, {"o2", "mul:z"}});

  SplitResults results =
      SplitFunction(orig, {"i1"}, /*num_captured_inputs=*/0, function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("i1"));
  ASSERT_THAT(GetNodeNames(second), UnorderedElementsAre("add", "mul"));

  const NodeDef& add_node = GetNode("add", second);
  const NodeDef& mul_node = GetNode("mul", second);
  ASSERT_THAT(add_node.input(), ElementsAreArray(mul_node.input()));

  CheckSplitFunctions(orig, first, second,
                      {test::AsScalar<float>(1.0), test::AsScalar<float>(2.0)});
}

TEST(SplitUtilsTest, CapturedInputs) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",                                          // function_name
      {"a1: float", "a2: float", "a3: float", "a4: float"},  // in_def
      {"o1: float", "o2: float"},                            // out_def
      {},                                                    // attr_def
      // node_def
      {
          {{"add"}, "AddV2", {"a1", "a2"}, {{"T", DT_FLOAT}}},
          {{"add2"}, "AddV2", {"add:z", "a3"}, {{"T", DT_FLOAT}}},
      },
      // ret_def
      {{"o1", "add2:z"}, {"o2", "a4"}});

  for (int num_captured = 0; num_captured <= 4; num_captured++) {
    VLOG(1) << "Number of captured inputs: " << num_captured;
    SplitResults results =
        SplitFunction(orig, {"add"}, num_captured, function_library)
            .ValueOrDie();
    FunctionDef first = results.first_function;
    FunctionDef second = results.second_function;

    ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("add"));
    ASSERT_THAT(GetNodeNames(second), UnorderedElementsAre("add2"));
    CheckSplitFunctions(
        orig, first, second,
        {test::AsScalar<float>(1.0), test::AsScalar<float>(2.0),
         test::AsScalar<float>(3.0), test::AsScalar<float>(4.0)},
        num_captured);
  }
}

TEST(SplitUtilsTest, ControlOutputs) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",   // function_name
      {"a1: float"},  // in_def
      {"o1: float"},  // out_def
      {},             // attr_def
      // node_def
      {{{"n1"}, "NoOp", {}, {}},
       {{"n2"}, "NoOp", {}, {}},
       {{"i1"}, "Identity", {"a1", "^n1"}, {{"T", DT_FLOAT}}},

       {{"n3"}, "NoOp", {}, {}},
       {{"i2"}, "Identity", {"i1:output", "^n2"}, {{"T", DT_FLOAT}}},
       {{"i3"}, "Identity", {"i1:output", "^n3"}, {{"T", DT_FLOAT}}}},
      // ret_def
      {{"o1", "i2:output"}},
      // control_ret_def
      {{"o2", "n2"}, {"o3", "i3"}});

  SplitResults results =
      SplitFunction(orig, {"n1", "n2", "i1"}, /*num_captured_inputs=*/0,
                    function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("n1", "n2", "i1"));
  ASSERT_THAT(GetNodeNames(second), UnorderedElementsAre("n3", "i2", "i3"));
  CheckSplitFunctions(orig, first, second, {test::AsScalar<float>(1.0)});

  // Test nodes have correct control dependencies
  ASSERT_THAT(GetNode("i1", first).input(), ElementsAre("a1", "^n1"));
  ASSERT_EQ(GetNode("i2", second).input_size(), 1);
  ASSERT_THAT(GetNode("i3", second).input(), ElementsAre(_, "^n3"));

  // Test the split functions have correct control outputs.
  ASSERT_THAT(first.control_ret(), UnorderedElementsAre(Pair("o2", "n2")));
  ASSERT_THAT(second.control_ret(), UnorderedElementsAre(Pair("o3", "i3")));
}

TEST(SplitUtilsTest, Empty) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig =
      FunctionDefHelper::Create("MyFunction",                // function_name
                                {"a1: float", "a2: float"},  // in_def
                                {"o1: float", "o2: float"},  // out_def
                                {},                          // attr_def
                                // node_def
                                {},
                                // ret_def
                                {{"o1", "a1"}, {"o2", "a2"}});

  SplitResults results =
      SplitFunction(orig, {}, /*num_captured_inputs=*/1, function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_EQ(first.node_def_size(), 0);
  ASSERT_EQ(second.node_def_size(), 0);
  CheckSplitFunctions(
      orig, first, second,
      {test::AsScalar<float>(123.0), test::AsScalar<float>(234.0)},
      /*num_captured_inputs=*/1);
}

TEST(SplitUtilsTest, UniqueArgNames) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",                            // function_name
      {"input_0: float", "input_0_0: float"},  // in_def
      {"o1: float"},                           // out_def
      {},                                      // attr_def
      // node_def
      {
          {{"add"}, "AddV2", {"input_0", "input_0_0"}, {{"T", DT_FLOAT}}},
      },
      // ret_def
      {{"o1", "add:z"}});

  SplitResults results =
      SplitFunction(orig, {"add"}, /*num_captured_inputs=*/0, function_library)
          .ValueOrDie();
  FunctionDef first = results.first_function;
  FunctionDef second = results.second_function;

  ASSERT_THAT(GetNodeNames(first), UnorderedElementsAre("add"));
  ASSERT_EQ(second.node_def_size(), 0);
  CheckSplitFunctions(
      orig, first, second,
      {test::AsScalar<float>(123.0), test::AsScalar<float>(234.0)});

  ASSERT_EQ(second.signature().input_arg_size(), 1);
  ASSERT_EQ(second.signature().input_arg(0).name(), "input_0_1");
}

TEST(SplitUtilsTest, UnimplementedErrors) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());

  // Output of first function is a list
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",   // function_name
      {"a1: float"},  // in_def
      {"o1: float"},  // out_def
      {},             // attr_def
      // node_def
      {{{"split"}, "Split", {"a1"}, {{"T", DT_FLOAT}, {"num_split", 2}}},
       {{"i1"}, "Identity", {"split:output:0"}, {{"T", DT_FLOAT}}}},
      // ret_def
      {{"o1", "i1:output"}});
  ASSERT_THAT(SplitFunction(orig, {"split"}, /*num_captured_inputs=*/0,
                            function_library),
              testing::StatusIs(
                  error::UNIMPLEMENTED,
                  ::testing::HasSubstr("Splitting a function where an edge is "
                                       "a list of tensors is unsupported.")));

  // Output of first function's dtype is a placeholder AttrValue
  // The dtype of the first function's output is a placeholder AttrValue
  orig = FunctionDefHelper::Create(
      "MyFunction",    // function_name
      {"a1: float"},   // in_def
      {"o1: float"},   // out_def
      {"U: {float}"},  // attr_def
      // node_def
      {{{"i1"}, "Identity", {"a1"}, {{"T", "$U"}}},
       {{"i2"}, "Identity", {"i1:output"}, {{"T", DT_FLOAT}}}},
      // ret_def
      {{"o1", "i2:output"}});
  ASSERT_THAT(
      SplitFunction(orig, {"i1"}, /*num_captured_inputs=*/0, function_library),
      testing::StatusIs(
          error::UNIMPLEMENTED,
          ::testing::HasSubstr(
              "edge between functions has an AttrValue placeholder dtype")));
}

TEST(SplitUtilsTest, EdgeFromSecondToFirstError) {
  using test::function::NDef;
  FunctionLibraryDefinition function_library(OpRegistry::Global());

  // Output of first function is a list
  FunctionDef orig = FunctionDefHelper::Create(
      "MyFunction",   // function_name
      {"a1: float"},  // in_def
      {"o1: float"},  // out_def
      {},             // attr_def
      // node_def
      {
          {{"i1"}, "Identity", {"a1:0"}, {{"T", DT_FLOAT}}},
          {{"i2"}, "Identity", {"i1:output"}, {{"T", DT_FLOAT}}},
      },
      // ret_def
      {{"o1", "i1:output"}});

  ASSERT_THAT(
      SplitFunction(orig, {"i2"}, /*num_captured_inputs=*/0, function_library),
      testing::StatusIs(
          error::INTERNAL,
          ::testing::HasSubstr("Node i2 is in first function but has input "
                               "i1:output which is not in first function")));
}

}  // namespace
}  // namespace split_utils
}  // namespace grappler
}  // namespace tensorflow
