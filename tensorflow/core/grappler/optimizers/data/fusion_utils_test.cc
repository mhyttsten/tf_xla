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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/fusion_utils.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace fusion_utils {
namespace {

string ParseNodeConnection(const string &name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/optimizers/data/fusion_utils_test.cc", "ParseNodeConnection");

  return name.substr(0, name.find(':'));
}

void CheckUniqueNames(const FunctionDef &function) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/grappler/optimizers/data/fusion_utils_test.cc", "CheckUniqueNames");

  std::unordered_set<string> inputs;
  for (const auto &input_arg : function.signature().input_arg())
    inputs.insert(input_arg.name());
  EXPECT_EQ(inputs.size(), function.signature().input_arg_size());

  std::unordered_set<string> outputs;
  for (const auto &output_arg : function.signature().output_arg())
    outputs.insert(output_arg.name());
  EXPECT_EQ(outputs.size(), function.signature().output_arg_size());

  std::unordered_set<string> nodes;
  for (const auto &node : function.node_def()) nodes.insert(node.name());

  EXPECT_EQ(nodes.size(), function.node_def_size());
}

TEST(FusionUtilsTest, FuseFunctionsByComposition) {
  GraphDef graph;
  auto *parent_function = graph.mutable_library()->add_function();
  *parent_function = test::function::XTimesTwo();
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto *fused_function = FuseFunctions(
      *parent_function, *function, "fused_maps", fusion_utils::ComposeSignature,
      fusion_utils::ComposeInput, fusion_utils::ComposeOutput,
      fusion_utils::MergeNodes, graph.mutable_library());

  EXPECT_EQ(fused_function->signature().name(), "fused_maps");
  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 1);
  EXPECT_EQ(fused_function->ret_size(), 1);
  std::cerr << fused_function->DebugString();
  CheckUniqueNames(*fused_function);

  const NodeDef *parent_mul = nullptr, *output_mul = nullptr;
  for (const auto &fused_node : fused_function->node_def()) {
    if (fused_node.op() == "Mul") {
      if (fused_node.name() == "y")
        parent_mul = &fused_node;
      else
        output_mul = &fused_node;
    }
  }
  ASSERT_NE(parent_mul, nullptr);
  ASSERT_NE(output_mul, nullptr);
  EXPECT_EQ(ParseNodeConnection(output_mul->input(0)), parent_mul->name());

  auto output_value = fused_function->ret().at(
      fused_function->signature().output_arg(0).name());

  EXPECT_EQ(ParseNodeConnection(output_value), output_mul->name());
}

TEST(FusionUtilsTest, FuseFunctionWithPredicate) {
  GraphDef graph;
  auto *xtimes_two = graph.mutable_library()->add_function();
  *xtimes_two = test::function::XTimesTwo();
  auto *is_zero = graph.mutable_library()->add_function();
  *is_zero = test::function::IsZero();

  auto *fused_function =
      FuseFunctions(*xtimes_two, *is_zero, "fused_map_and_filter_function",
                    fusion_utils::CombineSignature, fusion_utils::ComposeInput,
                    fusion_utils::CombineOutput, fusion_utils::MergeNodes,
                    graph.mutable_library());

  EXPECT_EQ(fused_function->signature().name(),
            "fused_map_and_filter_function");

  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);

  ASSERT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("Equal", *fused_function));
  const auto &equal_node = fused_function->node_def(
      function_utils::FindFunctionNodeWithOp("Equal", *fused_function));

  EXPECT_EQ(xtimes_two->signature().output_arg(0).name(),
            fused_function->signature().output_arg(0).name());

  EXPECT_EQ(fused_function->signature().output_arg(1).name(),
            equal_node.name());

  EXPECT_EQ(ParseNodeConnection(equal_node.input(0)),
            fused_function->signature().output_arg(0).name());

  auto output_value = fused_function->ret().at(
      fused_function->signature().output_arg(1).name());
  EXPECT_EQ(ParseNodeConnection(output_value), equal_node.name());
}

TEST(FusionUtilsTest, FuseSameFunctionWithExtraOutput) {
  GraphDef graph;
  auto *parent_function = graph.mutable_library()->add_function();
  *parent_function = test::function::XTimesTwo();
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto *fused_function = FuseFunctions(
      *parent_function, *function, "fused_maps", fusion_utils::CombineSignature,
      fusion_utils::ComposeInput, fusion_utils::CombineOutput,
      fusion_utils::MergeNodes, graph.mutable_library());

  EXPECT_EQ(fused_function->signature().input_arg_size(), 1);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);
}

TEST(FusionUtilsTest, ZipFusion) {
  GraphDef graph;
  auto *function = graph.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  auto zip_signature = [](const OpDef &parent_function_signature,
                          const OpDef &function_signature,
                          OpDef *fused_function_signature) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc mht_2(mht_2_v, 333, "", "./tensorflow/core/grappler/optimizers/data/fusion_utils_test.cc", "lambda");

    *fused_function_signature = parent_function_signature;
    fused_function_signature->mutable_input_arg()->MergeFrom(
        function_signature.input_arg());
    fused_function_signature->mutable_output_arg()->MergeFrom(
        function_signature.output_arg());
  };

  auto zip_input = [](const StringCollection &parent_inputs,
                      const StringCollection &function_inputs,
                      const StringCollection &parent_outputs, int arg_num) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSfusion_utils_testDTcc mht_3(mht_3_v, 346, "", "./tensorflow/core/grappler/optimizers/data/fusion_utils_test.cc", "lambda");

    // Take corresponding parent output.
    return function_inputs.at(arg_num);
  };

  auto *fused_function =
      FuseFunctions(*function, *function, "zip_maps", zip_signature, zip_input,
                    fusion_utils::CombineOutput, fusion_utils::MergeNodes,
                    graph.mutable_library());

  EXPECT_EQ(fused_function->signature().input_arg_size(), 2);
  EXPECT_EQ(fused_function->signature().output_arg_size(), 2);
  EXPECT_EQ(fused_function->ret_size(), 2);
  CheckUniqueNames(*fused_function);
}

}  // namespace
}  // namespace fusion_utils
}  // namespace grappler
}  // namespace tensorflow
