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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/frame.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using GraphTypes =
    ::testing::Types<GraphDef, utils::GraphView, utils::MutableGraphView>;

template <typename T>
class FrameViewTest : public ::testing::Test {
 protected:
  NodeDef CreateNode(const string& name, const std::vector<string>& inputs) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/grappler/utils/frame_test.cc", "CreateNode");

    return CreateNode(name, "", "", inputs);
  }

  NodeDef CreateNode(const string& name, const string& op,
                     const std::vector<string>& inputs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/grappler/utils/frame_test.cc", "CreateNode");

    return CreateNode(name, op, "", inputs);
  }

  NodeDef CreateNode(const string& name, const string& op, const string& frame,
                     const std::vector<string>& inputs) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("op: \"" + op + "\"");
   mht_2_v.push_back("frame: \"" + frame + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/grappler/utils/frame_test.cc", "CreateNode");

    NodeDef node;
    node.set_name(name);
    if (!op.empty()) {
      node.set_op(op);
    }
    if (!frame.empty()) {
      AttrValue frame_name;
      frame_name.set_s(frame);
      node.mutable_attr()->insert({"frame_name", frame_name});
    }
    for (const string& input : inputs) {
      node.add_input(input);
    }
    return node;
  }
};

TYPED_TEST_SUITE(FrameViewTest, GraphTypes);

template <typename T>
void InferFromGraph(FrameView* frame_view, GraphDef* graph, bool valid) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/grappler/utils/frame_test.cc", "InferFromGraph");

  Status status;
  T graph_view(graph, &status);
  TF_ASSERT_OK(status);
  status = frame_view->InferFromGraphView(graph_view);
  if (valid) {
    TF_ASSERT_OK(status);
  } else {
    ASSERT_FALSE(status.ok());
  }
}

template <>
void InferFromGraph<GraphDef>(FrameView* frame_view, GraphDef* graph,
                              bool valid) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSframe_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/grappler/utils/frame_test.cc", "InferFromGraph<GraphDef>");

  Status status = frame_view->InferFromGraph(*graph);
  if (valid) {
    TF_ASSERT_OK(status);
  } else {
    ASSERT_FALSE(status.ok());
  }
}

TYPED_TEST(FrameViewTest, NestedLoop) {
  GraphDef graph;
  // Create a two-level nested loop
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context1", {"0"});
  *graph.add_node() = this->CreateNode("2", {"1"});
  *graph.add_node() = this->CreateNode("3", "Merge", {"2", "14"});
  *graph.add_node() = this->CreateNode("4", {"3"});
  *graph.add_node() = this->CreateNode("5", "Switch", {"4"});
  *graph.add_node() = this->CreateNode("6", {"5"});
  *graph.add_node() = this->CreateNode("7", "Enter", "while/context2", {"6"});
  *graph.add_node() = this->CreateNode("8", {"7"});
  *graph.add_node() = this->CreateNode("9", "Merge", {"8", "12"});
  *graph.add_node() = this->CreateNode("10", {"9"});
  *graph.add_node() = this->CreateNode("11", "Switch", {"10"});
  *graph.add_node() = this->CreateNode("12", "NextIteration", {"11"});
  *graph.add_node() = this->CreateNode("13", "Exit", {"11"});
  *graph.add_node() = this->CreateNode("14", "NextIteration", {"13"});
  *graph.add_node() = this->CreateNode("15", {"5"});
  *graph.add_node() = this->CreateNode("16", "Exit", {"15"});
  *graph.add_node() = this->CreateNode("17", {"16"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}},      {"1", {0}},     {"2", {0}},     {"3", {0}},
      {"4", {0}},     {"5", {0}},     {"6", {0}},     {"7", {0, 1}},
      {"8", {0, 1}},  {"9", {0, 1}},  {"10", {0, 1}}, {"11", {0, 1}},
      {"12", {0, 1}}, {"13", {0, 1}}, {"14", {0}},    {"15", {0}},
      {"16", {0}},    {"17", {}}};

  EXPECT_EQ(frame_view.num_frames(), 2);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, MultipleInputsToEnter) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", {});
  *graph.add_node() =
      this->CreateNode("2", "Enter", "while/context", {"0", "1"});
  *graph.add_node() = this->CreateNode("3", "Exit", {"2"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {}}, {"2", {0}}, {"3", {0}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, ExitOutput) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context", {"0"});
  *graph.add_node() = this->CreateNode("2", "Exit", {"1"});
  *graph.add_node() = this->CreateNode("3", {});
  *graph.add_node() = this->CreateNode("4", {"2", "3"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {}}, {"4", {}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, MultipleEnterNodes) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context", {"0"});
  *graph.add_node() = this->CreateNode("2", {"1"});
  *graph.add_node() = this->CreateNode("5", {});
  *graph.add_node() = this->CreateNode("4", "Enter", "while/context", {"5"});
  *graph.add_node() = this->CreateNode("3", {"4", "2"});
  *graph.add_node() = this->CreateNode("6", "Merge", {"3", "8"});
  *graph.add_node() = this->CreateNode("7", "Switch", {"6"});
  *graph.add_node() = this->CreateNode("8", "NextIteration", {"7"});
  *graph.add_node() = this->CreateNode("9", "Exit", {"7"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/true);

  std::unordered_map<string, std::vector<int>> expected = {
      {"0", {}}, {"1", {0}}, {"2", {0}}, {"3", {0}}, {"4", {0}},
      {"5", {}}, {"6", {0}}, {"7", {0}}, {"8", {0}}, {"9", {0}}};

  EXPECT_EQ(frame_view.num_frames(), 1);
  for (const NodeDef& node : graph.node()) {
    std::vector<int> expected_frames = expected[node.name()];
    std::vector<int> node_frames = frame_view.Frames(node);
    EXPECT_EQ(expected_frames, node_frames);
  }
}

TYPED_TEST(FrameViewTest, ConflictingFrames) {
  GraphDef graph;
  *graph.add_node() = this->CreateNode("0", {});
  *graph.add_node() = this->CreateNode("1", "Enter", "while/context1", {"0"});
  *graph.add_node() = this->CreateNode("2", "Enter", "while/context2", {"1"});
  *graph.add_node() = this->CreateNode("3", {"1", "2"});

  FrameView frame_view;
  InferFromGraph<TypeParam>(&frame_view, &graph, /*valid=*/false);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
