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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc() {
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

#include "tensorflow/core/grappler/utils/grappler_test.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

namespace {
void CompareGraphNodes(protobuf::RepeatedPtrField<NodeDef>* want,
                       protobuf::RepeatedPtrField<NodeDef>* got) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "CompareGraphNodes");

  auto comparator = [](const NodeDef& n1, const NodeDef& n2) -> bool {
    return n1.name() < n2.name();
  };

  std::sort(want->begin(), want->end(), comparator);
  std::sort(got->begin(), got->end(), comparator);

  ASSERT_EQ(want->size(), got->size());

  for (int i = 0; i < want->size(); ++i) {
    NodeDef& want_node = (*want)[i];
    NodeDef& got_node = (*got)[i];

    EXPECT_EQ(want_node.op(), got_node.op());
    EXPECT_EQ(want_node.name(), got_node.name());
    EXPECT_EQ(want_node.device(), got_node.device());
    ASSERT_EQ(want_node.input_size(), got_node.input_size())
        << "want_node =\n"
        << want_node.DebugString() << "\ngot_node =\n"
        << got_node.DebugString();

    // Order of control dependencies doesn't matter, so we sort them first.
    const auto is_control = [](const string& input) -> bool {
      return ParseTensorName(input).index() < 0;
    };

    auto want_inputs = want_node.mutable_input();
    auto got_inputs = got_node.mutable_input();
    std::sort(absl::c_find_if(*want_inputs, is_control), want_inputs->end());
    std::sort(absl::c_find_if(*got_inputs, is_control), got_inputs->end());

    for (int j = 0; j < want_node.input_size(); ++j) {
      const TensorId want_tensor = ParseTensorName(want_node.input(j));
      const TensorId got_tensor = ParseTensorName(got_node.input(j));
      EXPECT_EQ(want_tensor.ToString(), got_tensor.ToString());
    }
  }
}

void SetAllOptimizers(RewriterConfig* cfg, RewriterConfig::Toggle value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "SetAllOptimizers");

  cfg->set_arithmetic_optimization(value);
  cfg->set_auto_mixed_precision(value);
  cfg->set_auto_mixed_precision_mkl(value);
  cfg->set_common_subgraph_elimination(value);
  cfg->set_constant_folding(value);
  cfg->set_debug_stripper(value);
  cfg->set_dependency_optimization(value);
  cfg->set_function_optimization(value);
  cfg->set_implementation_selector(value);
  cfg->set_layout_optimizer(value);
  cfg->set_loop_optimization(value);
  cfg->set_pin_to_host_optimization(value);
  cfg->set_remapping(value);
  cfg->set_scoped_allocator_optimization(value);
  cfg->set_shape_optimization(value);
}
}  // namespace

GrapplerTest::GrapplerTest() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_2(mht_2_v, 266, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::GrapplerTest");

  // Turn off all the automatic optimizations to ensure that we run the graph
  // exactly as it is given to us. This ensures that we can compare the
  // results before and after manual optimization, without any of the
  // automatic optimizations interfering in the comparison.
  DisableAllOptimizers();
}

void GrapplerTest::DisableAllOptimizers() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::DisableAllOptimizers");

  SetAllOptimizers(
      options_.config.mutable_graph_options()->mutable_rewrite_options(),
      RewriterConfig::OFF);
}

void GrapplerTest::EnableAllOptimizers() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_4(mht_4_v, 286, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::EnableAllOptimizers");

  SetAllOptimizers(
      options_.config.mutable_graph_options()->mutable_rewrite_options(),
      RewriterConfig::ON);
}

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::EvaluateNodes");

  return EvaluateNodes(graph, node_names, {});
}

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names,
    const std::vector<std::pair<string, Tensor>>& inputs) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::EvaluateNodes");

  std::unique_ptr<tensorflow::Session> session(NewSession(options_));
  TF_CHECK_OK(session->Create(graph));
  RunOptions run_options;
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, inputs, node_names, node_names,
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

std::vector<Tensor> GrapplerTest::EvaluateFetchNodes(
    const GrapplerItem& item) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_7(mht_7_v, 320, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::EvaluateFetchNodes");

  std::unique_ptr<tensorflow::Session> session(NewSession(options_));
  TF_CHECK_OK(session->Create(item.graph));
  RunOptions run_options;
  if (!item.init_ops.empty()) {
    std::vector<Tensor> dummy;
    TF_CHECK_OK(
        session->Run(run_options, {}, {}, item.init_ops, &dummy, nullptr));
  }
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, item.feed, item.fetch, {},
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

NodeDef* GrapplerTest::AddNode(
    const string& name, const string& op, const std::vector<string>& inputs,
    const std::vector<std::pair<string, AttrValue>>& attributes,
    GraphDef* graph) const {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   mht_8_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_8(mht_8_v, 344, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::AddNode");

  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (auto attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  return node;
}

void GrapplerTest::CompareGraphs(GraphDef want, GraphDef got) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_9(mht_9_v, 360, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::CompareGraphs");

  CompareGraphNodes(want.mutable_node(), got.mutable_node());
}

void GrapplerTest::CompareFunctions(FunctionDef want, FunctionDef got) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_10(mht_10_v, 367, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::CompareFunctions");

  CompareGraphNodes(want.mutable_node_def(), got.mutable_node_def());
}

void GrapplerTest::CompareNodes(const NodeDef& want, const NodeDef& got) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_11(mht_11_v, 374, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::CompareNodes");

  EXPECT_EQ(want.name(), got.name());
  EXPECT_EQ(want.op(), got.op());

  std::vector<string> want_inputs(want.input().begin(), want.input().end());
  std::vector<string> got_inputs(got.input().begin(), got.input().end());
  EXPECT_EQ(want_inputs, got_inputs);

  const auto attr_name = [](const std::pair<const string, AttrValue>& attr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_12(mht_12_v, 385, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "lambda");

    return attr.first;
  };

  std::vector<string> want_attrs;
  std::vector<string> got_attrs;
  absl::c_transform(want.attr(), std::back_inserter(want_attrs), attr_name);
  absl::c_transform(got.attr(), std::back_inserter(got_attrs), attr_name);
  absl::c_sort(want_attrs);
  absl::c_sort(got_attrs);
  EXPECT_EQ(want_attrs, got_attrs);

  for (const string& attr : want_attrs) {
    EXPECT_TRUE(AreAttrValuesEqual(want.attr().at(attr), got.attr().at(attr)));
  }
}

bool GrapplerTest::IsNodesDirectlyConnected(const NodeMap& node_map,
                                            const string& src,
                                            const string& dst, int position) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("src: \"" + src + "\"");
   mht_13_v.push_back("dst: \"" + dst + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_13(mht_13_v, 409, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::IsNodesDirectlyConnected");

  const NodeDef* src_node = node_map.GetNode(src);
  const NodeDef* dst_node = node_map.GetNode(dst);
  EXPECT_TRUE(src_node != nullptr) << src << " node not found";
  EXPECT_TRUE(dst_node != nullptr) << dst << " node not found";
  return src_node && dst_node && dst_node->input(position) == src_node->name();
}

int GrapplerTest::CountOpNodes(const GraphDef& graph, const string& op) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgrappler_testDTcc mht_14(mht_14_v, 421, "", "./tensorflow/core/grappler/utils/grappler_test.cc", "GrapplerTest::CountOpNodes");

  return std::count_if(graph.node().begin(), graph.node().end(),
                       [&op](const NodeDef& node) { return node.op() == op; });
}

}  // namespace grappler
}  // namespace tensorflow
