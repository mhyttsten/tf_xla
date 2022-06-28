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
class MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc() {
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
#include "tensorflow/core/data/rewrite_utils.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::test::AsScalar;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::testing::ElementsAre;

NodeDef GetMapNode(absl::string_view name, absl::string_view input_node_name,
                   absl::string_view function_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_0_v.push_back("input_node_name: \"" + std::string(input_node_name.data(), input_node_name.size()) + "\"");
   mht_0_v.push_back("function_name: \"" + std::string(function_name.data(), function_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/data/rewrite_utils_test.cc", "GetMapNode");

  return NDef(
      name, /*op=*/"MapDataset", {std::string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(std::string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
       {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}});
}

FunctionDef XTimesX() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/data/rewrite_utils_test.cc", "XTimesX");

  return FunctionDefHelper::Create(
      /*function_name=*/"XTimesX",
      /*in_def=*/{"x: int64"},
      /*out_def=*/{"y: int64"},
      /*attr_def=*/{},
      /*node_def=*/{{{"y"}, "Mul", {"x", "x"}, {{"T", DT_INT64}}}},
      /*ret_def=*/{{"y", "y:z:0"}});
}

GraphDef GetRangeSquareDatasetDef(const int64_t range) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSrewrite_utils_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/data/rewrite_utils_test.cc", "GetRangeSquareDatasetDef");

  return GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64_t>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       GetMapNode("map", "range", "XTimesX"),
       NDef("dataset", "_Retval", /*inputs=*/{"map"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      {XTimesX()});
}

TEST(GraphUtilTest, GetFetchNode) {
  GraphDef graph = GetRangeSquareDatasetDef(10);
  TF_ASSERT_OK_AND_ASSIGN(std::string dataset_node, GetDatasetNode(graph));
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(&graph, &dataset_node, /*add_fake_sinks=*/false);
  EXPECT_THAT(grappler_item->fetch, ElementsAre("Sink"));
}

TEST(GraphUtilTest, GetFetchNodeDef) {
  GraphDef graph = GetRangeSquareDatasetDef(10);
  TF_ASSERT_OK_AND_ASSIGN(NodeDef dataset_nodedef, GetDatasetNodeDef(graph));
  std::string dataset_node = dataset_nodedef.name();
  std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
      GetGrapplerItem(&graph, &dataset_node, /*add_fake_sinks=*/false);
  EXPECT_THAT(grappler_item->fetch, ElementsAre("Sink"));
}

struct SelectOptimizationsTestCase {
  absl::flat_hash_set<string> experiments;
  absl::flat_hash_set<tstring> optimizations_enabled;
  absl::flat_hash_set<tstring> optimizations_disabled;
  absl::flat_hash_set<tstring> optimizations_default;
  std::vector<string> expected;
};

class SelectOptimizationsTest
    : public ::testing::TestWithParam<SelectOptimizationsTestCase> {};

TEST_P(SelectOptimizationsTest, DatasetUtils) {
  const SelectOptimizationsTestCase test_case = GetParam();
  auto optimizations = SelectOptimizations(
      test_case.experiments, test_case.optimizations_enabled,
      test_case.optimizations_disabled, test_case.optimizations_default);
  EXPECT_THAT(std::vector<string>(optimizations.begin(), optimizations.end()),
              ::testing::UnorderedElementsAreArray(test_case.expected));
}

INSTANTIATE_TEST_SUITE_P(
    Test, SelectOptimizationsTest,
    ::testing::Values(
        SelectOptimizationsTestCase{
            /*experiments=*/{}, /*optimizations_enabled=*/{},
            /*optimizations_disabled=*/{}, /*optimizations_default=*/{},
            /*expected=*/{}},
        SelectOptimizationsTestCase{
            /*experiments=*/{"map_and_batch_fusion"},
            /*optimizations_enabled=*/{"bar"},
            /*optimizations_disabled=*/{}, /*optimizations_default=*/{"baz"},
            /*expected=*/{"map_and_batch_fusion", "bar", "baz"}},
        SelectOptimizationsTestCase{
            /*experiments=*/{"this_is_not_an_optimization"},
            /*optimizations_enabled=*/{"bar"},
            /*optimizations_disabled=*/{}, /*optimizations_default=*/{"baz"},
            /*expected=*/{"bar", "baz"}},
        SelectOptimizationsTestCase{/*experiments=*/{},
                                    /*optimizations_enabled=*/{"foo"},
                                    /*optimizations_disabled=*/{"baz"},
                                    /*optimizations_default=*/{"bar", "baz"},
                                    /*expected=*/{"foo", "bar"}},
        SelectOptimizationsTestCase{
            /*experiments=*/{"foo"}, /*optimizations_enabled=*/{"bar"},
            /*optimizations_disabled=*/{"foo"},
            /*optimizations_default=*/{"baz"}, /*expected=*/{"bar", "baz"}}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
