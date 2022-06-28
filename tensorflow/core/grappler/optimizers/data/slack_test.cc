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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslack_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslack_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslack_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/data/slack.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

void SetupGrapplerItem(GrapplerItem *item) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSdataPSslack_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/optimizers/data/slack_test.cc", "SetupGrapplerItem");

  MutableGraphView graph(&item->graph);

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue(std::vector<TensorShape>({{}}), &shapes_attr);
  common_attrs[0] = std::make_pair("output_shapes", shapes_attr);
  AttrValue types_attr;
  SetAttrValue(std::vector<DataType>({DT_INT64}), &types_attr);
  common_attrs[1] = std::make_pair("output_types", types_attr);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node = graph_utils::AddNode(
      "RangeDataset", "RangeDataset", range_inputs, common_attrs, &graph);

  NodeDef *buffer_size_node =
      graph_utils::AddScalarConstNode<int64_t>(1, &graph);
  NodeDef *prefetch_node = graph_utils::AddNode(
      "PrefetchDataset", "PrefetchDataset",
      {range_node->name(), buffer_size_node->name()}, common_attrs, &graph);
  item->fetch.push_back(prefetch_node->name());
}

struct ParameterizedSlackTest
    : ::testing::TestWithParam<std::tuple<string, int>> {};

TEST_P(ParameterizedSlackTest, BasicTest) {
  GrapplerItem item;
  SetupGrapplerItem(&item);

  Slack optimizer;
  tensorflow::RewriterConfig_CustomGraphOptimizer config;
  (*config.mutable_parameter_map())["slack_period"].set_s(
      std::get<0>(GetParam()));
  TF_ASSERT_OK(optimizer.Init(&config));

  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("PrefetchDataset", output));
  NodeDef optimized_prefetch_node =
      output.node(graph_utils::FindGraphNodeWithOp("PrefetchDataset", output));
  EXPECT_EQ(optimized_prefetch_node.attr().at("slack_period").i(),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(DifferentSlackEveryValues, ParameterizedSlackTest,
                         ::testing::Values(std::make_tuple("1", 1),
                                           std::make_tuple("8", 8)));

TEST(SlackTest, TestFailWithoutInit) {
  GrapplerItem item;
  Slack optimizer;
  GraphDef output;
  Status result = optimizer.Optimize(nullptr, item, &output);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(errors::IsInvalidArgument(result));
}

TEST(SlackTest, TestFailWithInvalidSlackEveryParam) {
  GrapplerItem item;
  SetupGrapplerItem(&item);

  Slack optimizer;
  tensorflow::RewriterConfig_CustomGraphOptimizer config;
  (*config.mutable_parameter_map())["slack_period"].set_s("0");
  TF_ASSERT_OK(optimizer.Init(&config));

  GraphDef output;
  Status result = optimizer.Optimize(nullptr, item, &output);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(errors::IsInvalidArgument(result));
}

TEST(SlackTest, TestFunctionNotOptimized) {
  GrapplerFunctionItem item;
  FunctionDefLibrary lib_def;
  FunctionDef *fdef = lib_def.add_function();
  fdef->mutable_signature()->set_name("nested_function");
  auto *input_arg = fdef->mutable_signature()->add_input_arg();
  input_arg->set_name("args_0");
  input_arg->set_type(DT_INT64);
  auto *output_arg = fdef->mutable_signature()->add_output_arg();
  output_arg->set_name("identity");
  output_arg->set_type(DT_VARIANT);
  fdef->mutable_signature()->set_is_stateful(true);

  AttrValue shapes_attr;
  SetAttrValue(std::vector<TensorShape>({{}}), &shapes_attr);
  AttrValue types_attr;
  SetAttrValue(std::vector<DataType>({DT_INT64}), &types_attr);
  NodeDef *tensor_dataset_node =
      function_utils::AddNode("TensorDataset", "TensorDataset", {"args_0"},
                              {std::make_pair("output_shapes", shapes_attr),
                               std::make_pair("Toutput_types", types_attr)},
                              fdef);
  NodeDef *prefetch_node = function_utils::AddNode(
      "PrefetchDataset", "PrefetchDataset",
      {strings::StrCat(tensor_dataset_node->name(), ":handle:0"), "args_0"},
      {std::make_pair("output_shapes", shapes_attr),
       std::make_pair("output_types", types_attr)},
      fdef);

  AttrValue variant_type_attr;
  SetAttrValue(DT_VARIANT, &variant_type_attr);
  NodeDef *identity_node = function_utils::AddNode(
      "Identity", "Identity",
      {strings::StrCat(prefetch_node->name(), ":handle:0"),
       strings::StrCat("^", tensor_dataset_node->name())},
      {std::make_pair("T", variant_type_attr)}, fdef);

  (*fdef->mutable_ret())["identity"] =
      strings::StrCat(identity_node->name(), ":output:0");
  (*fdef->mutable_control_ret())[tensor_dataset_node->name()] =
      tensor_dataset_node->name();
  fdef->mutable_signature()->add_control_output(tensor_dataset_node->name());

  FunctionLibraryDefinition flib(OpRegistry::Global(), lib_def);

  TF_ASSERT_OK(
      MakeGrapplerFunctionItem(*fdef, flib, /*graph_def_version=*/27, &item));

  GraphDef output;
  Slack optimizer;
  tensorflow::RewriterConfig_CustomGraphOptimizer config;
  (*config.mutable_parameter_map())["slack_period"].set_s("8");
  TF_ASSERT_OK(optimizer.Init(&config));

  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("PrefetchDataset", output));
  NodeDef optimized_prefetch_node =
      output.node(graph_utils::FindGraphNodeWithOp("PrefetchDataset", output));
  // Should not set slack for function items.
  EXPECT_EQ(optimized_prefetch_node.attr().at("slack_period").i(), 0);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
