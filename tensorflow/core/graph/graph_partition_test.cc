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
class MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc() {
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

#include "tensorflow/core/graph/graph_partition.h"

#include <unordered_map>
#include <utility>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

// from graph_partition.cc
extern Status TopologicalSortNodesWithTimePriority(
    const GraphDef* gdef,
    std::vector<std::pair<const NodeDef*, int64_t>>* nodes,
    std::unordered_map<const NodeDef*, int64_t>* node_to_start_time_out);

namespace {

using ops::_Recv;
using ops::_Send;
using ops::Const;
using ops::Identity;
using ops::LoopCond;
using ops::NextIteration;

const char gpu_device[] = "/job:a/replica:0/task:0/device:GPU:0";

string SplitByDevice(const Node* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_0(mht_0_v, 233, "", "./tensorflow/core/graph/graph_partition_test.cc", "SplitByDevice");
 return node->assigned_device_name(); }

string DeviceName(const Node* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/graph/graph_partition_test.cc", "DeviceName");

  char first = node->name()[0];
  if (first == 'G') {
    return gpu_device;
  } else {
    const string cpu_prefix = "/job:a/replica:0/task:0/cpu:";
    int index = first - 'A';
    return strings::StrCat(cpu_prefix, index);
  }
}

void Partition(const GraphDef& graph_def,
               std::unordered_map<string, GraphDef>* partitions) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/graph/graph_partition_test.cc", "Partition");

  Graph g(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, &g));

  // Assigns devices to each node. Uses 1st letter of the node name as the
  // device index if no device is specified.
  for (Node* node : g.nodes()) {
    string device_name = !node->requested_device().empty()
                             ? node->requested_device()
                             : DeviceName(node);
    node->set_assigned_device_name(device_name);
  }

  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [&g](const string& prefix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");
 return g.NewName(prefix); };
  popts.get_incarnation = [](const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_4(mht_4_v, 278, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");

    return (name[0] - 'A') + 100;
  };
  Status s = Partition(popts, &g, partitions);
  CHECK(s.ok()) << s;

  // Check versions.
  EXPECT_EQ(graph_def.versions().producer(), TF_GRAPH_DEF_VERSION);
  // Partitions must inherit the versions of the original graph.
  for (auto& it : *partitions) {
    EXPECT_EQ(graph_def.versions().producer(), it.second.versions().producer());
    EXPECT_EQ(graph_def.versions().min_consumer(),
              it.second.versions().min_consumer());
  }
}

void CheckLoopConstruction(const GraphDef& graph_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_5(mht_5_v, 297, "", "./tensorflow/core/graph/graph_partition_test.cc", "CheckLoopConstruction");

  std::unordered_map<string, GraphDef> partitions;
  Partition(graph_def, &partitions);
  for (const auto& kv : partitions) {
    const GraphDef& gdef = kv.second;
    bool has_control_enter = false;
    bool has_control_merge = false;
    bool has_control_switch = false;
    bool has_control_next = false;
    for (const NodeDef& ndef : gdef.node()) {
      // _recvs must have a control input
      if (ndef.op() == "_Recv") {
        bool has_control = false;
        for (const string& input_name : ndef.input()) {
          if (absl::StartsWith(input_name, "^")) {
            has_control = true;
            break;
          }
        }
        EXPECT_TRUE(has_control);
      }
      // Must have a control loop
      if (absl::StartsWith(ndef.name(), "_cloop")) {
        if (ndef.op() == "Enter") {
          has_control_enter = true;
        }
        if (ndef.op() == "Merge") {
          has_control_merge = true;
        }
        if (ndef.op() == "Switch") {
          has_control_switch = true;
        }
        if (ndef.op() == "NextIteration") {
          has_control_next = true;
        }
      }
    }
    EXPECT_TRUE(has_control_enter);
    EXPECT_TRUE(has_control_merge);
    EXPECT_TRUE(has_control_switch);
    EXPECT_TRUE(has_control_next);
  }
}

REGISTER_OP("FloatInput")
    .Output("o: float")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("BoolInput")
    .Output("o: bool")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("Combine")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn(shape_inference::UnknownShape);

Output ConstructOp(const Scope& scope, const string& op_type,
                   const gtl::ArraySlice<Input>& inputs) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("op_type: \"" + op_type + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_6(mht_6_v, 358, "", "./tensorflow/core/graph/graph_partition_test.cc", "ConstructOp");

  if (!scope.ok()) return Output();
  const string unique_name = scope.GetUniqueNameForOp(op_type);
  auto builder =
      NodeBuilder(unique_name, op_type, scope.graph()->op_registry());
  for (auto const& input : inputs) {
    builder.Input(ops::NodeOut(input.node(), input.index()));
  }
  scope.UpdateBuilder(&builder);
  Node* ret;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return Output();
  scope.UpdateStatus(scope.DoShapeInference(ret));
  if (!scope.ok()) return Output();
  return Output(ret);
}

Output FloatInput(const Scope& scope) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_7(mht_7_v, 378, "", "./tensorflow/core/graph/graph_partition_test.cc", "FloatInput");

  return ConstructOp(scope, "FloatInput", {});
}

Output BoolInput(const Scope& scope) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/graph/graph_partition_test.cc", "BoolInput");

  return ConstructOp(scope, "BoolInput", {});
}

Output Combine(const Scope& scope, Input a, Input b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_9(mht_9_v, 392, "", "./tensorflow/core/graph/graph_partition_test.cc", "Combine");

  return ConstructOp(scope, "Combine", {std::move(a), std::move(b)});
}

class GraphPartitionTest : public ::testing::Test {
 protected:
  GraphPartitionTest()
      : in_(Scope::NewRootScope().ExitOnError()),
        scope_a_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:0")),
        scope_b_(Scope::NewRootScope().ExitOnError().WithDevice(
            "/job:a/replica:0/task:0/cpu:1")) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_10(mht_10_v, 406, "", "./tensorflow/core/graph/graph_partition_test.cc", "GraphPartitionTest");
}

  const GraphDef& ToGraphDef() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_11(mht_11_v, 411, "", "./tensorflow/core/graph/graph_partition_test.cc", "ToGraphDef");

    TF_EXPECT_OK(in_.ToGraphDef(&in_graph_def_));
    return in_graph_def_;
  }

  void ExpectMatchA() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_12(mht_12_v, 419, "", "./tensorflow/core/graph/graph_partition_test.cc", "ExpectMatchA");

    GraphDef graph_def;
    TF_EXPECT_OK(scope_a_.ToGraphDef(&graph_def));
    string a = "/job:a/replica:0/task:0/cpu:0";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[a]);
  }

  void ExpectMatchB() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_13(mht_13_v, 429, "", "./tensorflow/core/graph/graph_partition_test.cc", "ExpectMatchB");

    GraphDef graph_def;
    TF_EXPECT_OK(scope_b_.ToGraphDef(&graph_def));
    string b = "/job:a/replica:0/task:0/cpu:1";
    TF_EXPECT_GRAPH_EQ(graph_def, partitions_[b]);
  }

  void ExpectFunctions(const FunctionDefLibrary& library,
                       const std::set<string>& expected_names) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_14(mht_14_v, 440, "", "./tensorflow/core/graph/graph_partition_test.cc", "ExpectFunctions");

    std::set<string> actual_names;
    for (const FunctionDef& fdef : library.function()) {
      actual_names.insert(fdef.signature().name());
    }
    EXPECT_EQ(actual_names, expected_names);
  }

  Scope in_;
  GraphDef in_graph_def_;
  Scope scope_a_;
  Scope scope_b_;
  std::unordered_map<string, GraphDef> partitions_;
};

TEST_F(GraphPartitionTest, SingleDevice) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  Combine(in_.WithOpName("A2"), a1, a1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(1, partitions_.size());

  a1 = FloatInput(scope_a_.WithOpName("A1"));
  Combine(scope_a_.WithOpName("A2"), a1, a1);
  ExpectMatchA();
}

TEST_F(GraphPartitionTest, CrossDeviceData) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  ExpectMatchA();

  b1 = FloatInput(scope_b_.WithOpName("B1"));
  auto recv =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  Combine(scope_b_.WithOpName("B2"), recv, b1);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c = Const(scope_a_.WithOpName("A1/_0").WithControlDependencies(a1), {});
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id = Identity(scope_b_.WithOpName("A1/_3"), recv);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2").WithControlDependencies(id), b1, b1);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceData_MultiUse) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);
  Combine(in_.WithOpName("B3"), a1, a1);

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2"), recv, b1);
  Combine(scope_b_.WithOpName("B3"), recv, recv);
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceControl_MultiUse) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2").WithControlDependencies(a1), b1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  auto c = Const(scope_a_.WithOpName("A1/_0").WithControlDependencies(a1), {});
  _Send(scope_a_.WithOpName("A1/_1"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv =
      _Recv(scope_b_.WithOpName("A1/_2"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id = Identity(scope_b_.WithOpName("A1/_3"), recv);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2").WithControlDependencies(id), b1, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDevice_DataControl) {
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  Combine(in_.WithOpName("B2"), a1, b1);
  FloatInput(in_.WithOpName("B3").WithControlDependencies(a1));

  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";
  a1 = FloatInput(scope_a_.WithOpName("A1"));
  _Send(scope_a_.WithOpName("A1/_0"), a1, "edge_1_A1", a, 82, b);
  auto c = Const(scope_a_.WithOpName("A1/_2").WithControlDependencies(a1), {});
  // NOTE: Send 0 A1/_1 -> A1/_2 is not necessarily needed. We could
  // use A1/_0 -> A1/_4 as the control as a minor optimization.
  _Send(scope_a_.WithOpName("A1/_3"), c, "edge_3_A1", a, 82, b);
  ExpectMatchA();

  auto recv1 =
      _Recv(scope_b_.WithOpName("A1/_4"), DT_FLOAT, "edge_3_A1", a, 82, b);
  auto id1 = Identity(scope_b_.WithOpName("A1/_5"), recv1);
  auto recv2 =
      _Recv(scope_b_.WithOpName("A1/_1"), DT_FLOAT, "edge_1_A1", a, 82, b);
  b1 = FloatInput(scope_b_.WithOpName("B1"));
  Combine(scope_b_.WithOpName("B2"), recv2, b1);
  FloatInput(scope_b_.WithOpName("B3").WithControlDependencies(id1));
  ExpectMatchB();
}

TEST_F(GraphPartitionTest, CrossDeviceLoopSimple) {
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = ::tensorflow::ops::internal::Enter(in_.WithOpName("A2"), a1, "foo");
  auto a3 = ::tensorflow::ops::Merge(in_.WithOpName("A3"),
                                     {a2, Input("A5", 0, DT_BOOL)})
                .output;
  LoopCond(in_.WithOpName("A4"), a3);
  auto b1 = Identity(in_.WithOpName("B1"), a3);
  NextIteration(in_.WithOpName("A5"), b1);

  CheckLoopConstruction(ToGraphDef());
}

TEST_F(GraphPartitionTest, CrossDeviceLoopSimple1) {
  auto a1 = BoolInput(in_.WithOpName("A1"));
  auto a2 = ::tensorflow::ops::internal::Enter(in_.WithOpName("B2"), a1, "foo");
  auto a3 = ::tensorflow::ops::Merge(in_.WithOpName("A3"),
                                     {a2, Input("B5", 0, DT_BOOL)})
                .output;
  LoopCond(in_.WithOpName("A4"), a3);
  auto b1 = Identity(in_.WithOpName("B1"), a3);
  NextIteration(in_.WithOpName("B5"), b1);

  std::unordered_map<string, GraphDef> partitions;
  Partition(ToGraphDef(), &partitions);
  for (const auto& kv : partitions) {
    const GraphDef& gdef = kv.second;
    for (const NodeDef& ndef : gdef.node()) {
      if (ndef.name() == "A3") {
        // A3, B2, and B5 are on the same device.
        EXPECT_EQ(ndef.input(0), "B2");
        EXPECT_EQ(ndef.input(1), "B5");
      }
    }
  }
}

TEST_F(GraphPartitionTest, CrossDeviceLoopFull) {
  Scope cpu0 = in_.WithDevice("/job:a/replica:0/task:0/cpu:0");
  auto p1 = ops::Placeholder(cpu0, DT_INT32);
  auto p2 = ops::Placeholder(cpu0, DT_INT32);
  OutputList outputs;
  // while i1 < 10: i1 += i2
  TF_ASSERT_OK(ops::BuildWhileLoop(
      cpu0, {p1, p2},
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_15(mht_15_v, 637, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");

        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_16(mht_16_v, 645, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");

        Scope cpu1 = s.WithDevice("/job:a/replica:0/task:0/cpu:1");
        outputs->push_back(ops::AddN(cpu1, {inputs[0], inputs[1]}));
        outputs->push_back(inputs[1]);
        return s.status();
      },
      "test_loop", &outputs));
  CheckLoopConstruction(ToGraphDef());
}

TEST_F(GraphPartitionTest, PartitionIncompleteGraph) {
  NodeDef ndef;
  Graph g(OpRegistry::Global());
  // Invalid graph since the Combine node requires an input.
  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      name: "N"
      op: "Combine"
      )EOF",
      &ndef);
  ASSERT_TRUE(parsed);
  Status status;
  g.AddNode(ndef, &status);
  TF_ASSERT_OK(status);

  PartitionOptions popts;
  popts.node_to_loc = SplitByDevice;
  popts.new_name = [&g](const string& prefix) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_17(mht_17_v, 676, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");
 return g.NewName(prefix); };
  popts.get_incarnation = [](const string&) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraph_partition_testDTcc mht_18(mht_18_v, 680, "", "./tensorflow/core/graph/graph_partition_test.cc", "lambda");
 return 1; };

  std::unordered_map<string, GraphDef> partitions;
  status = Partition(popts, &g, &partitions);
  // Partitioning should fail, but not crash like it did before the
  // changes that accompanied the addition of this test.
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code()) << status;
}

TEST_F(GraphPartitionTest, Functions) {
  FunctionDefLibrary fdef_lib;
  *fdef_lib.add_function() = test::function::XTimesTwo();
  *fdef_lib.add_function() = test::function::XTimesFour();
  TF_ASSERT_OK(in_.graph()->AddFunctionLibrary(fdef_lib));

  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
  auto a1 = FloatInput(in_.WithOpName("A1"));
  auto b1 = FloatInput(in_.WithOpName("B1"));
  ConstructOp(in_.WithOpName("A2"), "XTimesTwo", {a1});
  ConstructOp(in_.WithOpName("B2"), "XTimesFour", {b1});

  // The `Partition()` helper function uses the first letter of the op name ('A'
  // or 'B') to choose a device for each node.
  Partition(ToGraphDef(), &partitions_);
  EXPECT_EQ(2, partitions_.size());

  // Test that partition graphs inherit function library from original graph.
  string a = "/job:a/replica:0/task:0/cpu:0";
  string b = "/job:a/replica:0/task:0/cpu:1";

  // Node "A2" is placed in part `a`, and uses only "XTimesTwo".
  ExpectFunctions(partitions_[a].library(), {"XTimesTwo"});
  // Node "B2" is placed in part `b`, and uses both "XTimesFour" directly,
  // and "XTimesTwo" in the body of "XTimesFour".
  ExpectFunctions(partitions_[b].library(), {"XTimesTwo", "XTimesFour"});
}

TEST_F(GraphPartitionTest, SetIncarnation) {
  GraphDef gdef;
  const char* const kSendRecvAttrs = R"proto(
  attr { key: 'T' value { type: DT_FLOAT  }  }
  attr { key: 'client_terminated' value {  b: false } }
  attr { key: 'recv_device' value { s: 'B' } }
  attr { key: 'send_device' value { s: 'A' } }
  attr { key: 'send_device_incarnation' value { i: 0 }  }
  attr { key: 'tensor_name' value { s: 'test' } }
)proto";
  CHECK(protobuf::TextFormat::ParseFromString(
      strings::StrCat(
          "node { name: 'A/Pi' op: 'Const' ",
          "  attr { key: 'dtype' value { type: DT_FLOAT } } ",
          "  attr { key: 'value' value { tensor { ",
          "    dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }",
          "node { name: 'A' op: '_Send' input: 'A/Pi' ", kSendRecvAttrs, "}",
          "node { name: 'B' op: '_Recv' ", kSendRecvAttrs,
          "  attr { key: 'tensor_type' value { type:DT_FLOAT}}}"),
      &gdef));
  gdef.mutable_versions()->set_producer(TF_GRAPH_DEF_VERSION);
  Partition(gdef, &partitions_);
  EXPECT_EQ(2, partitions_.size());

  for (const auto& kv : partitions_) {
    const GraphDef& gdef = kv.second;
    for (const NodeDef& ndef : gdef.node()) {
      if (ndef.name() == "A" || ndef.name() == "B") {
        int64_t val;
        TF_CHECK_OK(GetNodeAttr(ndef, "send_device_incarnation", &val));
        EXPECT_EQ(val, 100);  // Send device is "A".
      }
    }
  }
}

TEST(TopologicalSortNodesWithTimePriorityTest, NoDependencies) {
  // Create placeholders, shuffle them so the order in the graph is not strictly
  // increasing.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  for (int i = 0; i < 20; ++i) {
    indexes.push_back((i + 2001) % 20);
  }
  std::vector<ops::Placeholder> placeholders;
  for (int i : indexes) {
    placeholders.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                              DT_FLOAT);
    placeholders.back().node()->AddAttr("_start_time", i + 1);
  }

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_EQ(nodes.size(), 20);
  for (int i = 0; i < nodes.size(); ++i) {
    EXPECT_EQ(strings::StrCat("p", i), nodes[i].first->name());
    EXPECT_EQ(i + 1, nodes[i].second);
  }
}

TEST(TopologicalSortNodesWithTimePriority, Dependencies) {
  // Create placeholders, shuffle them so the order in the graph is not strictly
  // increasing.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  std::vector<ops::Placeholder> placeholders_in_order;
  const int num_leaves = 20;
  for (int i = 0; i < num_leaves; ++i) {
    indexes.push_back((i + 2001) % num_leaves);
    placeholders_in_order.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                                       DT_FLOAT);
    placeholders_in_order.back().node()->AddAttr("_start_time", i + 1);
  }
  std::vector<ops::Placeholder> placeholders;
  for (int i : indexes) {
    placeholders.push_back(placeholders_in_order[i]);
  }

  // Create ops that depend on the placeholders. We give start times to these
  // that are in descending order (e.g., the op that depends on the first
  // placeholder runs last).
  std::vector<ops::Square> squares;
  for (int i : indexes) {
    squares.emplace_back(root.WithOpName(strings::StrCat("s", i)),
                         placeholders[i]);
    squares.back().node()->AddAttr("_start_time", 50 - (i + 1));
  }

  // Create addn to sum all squares.
  std::vector<Input> inputs;
  for (const auto& s : squares) inputs.push_back(s);
  ops::AddN addn = ops::AddN(root.WithOpName("addn"),
                             tensorflow::gtl::ArraySlice<Input>(inputs));
  // Start times is actually listed earlier than the nodes it depends on.
  // But because of dependency ordering, it is last in the list.
  addn.node()->AddAttr("_start_time", 1);

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_EQ(1 + squares.size() + placeholders.size(), nodes.size());
  for (int i = 0; i < placeholders.size(); ++i) {
    const NodeDef* node = nodes[i].first;
    EXPECT_EQ(strings::StrCat("p", i), node->name());
    EXPECT_EQ(i + 1, nodes[i].second);
    EXPECT_EQ(i + 1, node_to_start_time[node]);
  }
  for (int i = 0; i < squares.size(); ++i) {
    int node_index = placeholders.size() + i;
    int square_index = num_leaves - 1 - i;
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("s", square_index), node->name());
    EXPECT_EQ(50 - (square_index + 1), nodes[node_index].second);
    EXPECT_EQ(50 - (square_index + 1), node_to_start_time[node]);
  }
  EXPECT_EQ("addn", nodes.back().first->name());
  EXPECT_EQ(50, nodes.back().second);
  EXPECT_EQ(50, node_to_start_time[nodes.back().first]);
}

TEST(TopologicalSortNodesWithTimePriority, WhileLoop) {
  using namespace ::tensorflow::ops;            // NOLINT(build/namespaces)
  using namespace ::tensorflow::ops::internal;  // NOLINT(build/namespaces)

  // Create placeholders.
  Scope root = Scope::NewRootScope().ExitOnError();
  std::vector<int> indexes;
  std::vector<Placeholder> placeholders_in_order;
  const int num_leaves = 20;
  for (int i = 0; i < num_leaves; ++i) {
    indexes.push_back((i + 2001) % num_leaves);
    placeholders_in_order.emplace_back(root.WithOpName(strings::StrCat("p", i)),
                                       DT_FLOAT);
    placeholders_in_order.back().node()->AddAttr("_start_time", i + 1);
  }
  std::vector<Placeholder> placeholders;
  placeholders.reserve(indexes.size());
  for (int i : indexes) {
    placeholders.push_back(placeholders_in_order[i]);
  }

  // Add a while loop above each placeholder.
  std::vector<Exit> while_exits;
  const int nodes_per_loop = 8;
  for (int i : indexes) {
    Scope scope = root.NewSubScope(strings::StrCat("while", i));
    auto dummy = Placeholder(scope, DT_FLOAT);

    Enter enter(scope, placeholders[i], strings::StrCat("frame", i));
    Merge merge(scope, std::initializer_list<Input>{enter, dummy});
    auto cv = Const(scope.WithControlDependencies({merge.output}), false);
    LoopCond loop_cond(scope, cv);
    Switch switch_node(scope, merge.output, loop_cond);
    Identity identity(scope, switch_node.output_true);
    NextIteration next_iteration(scope, identity);
    while_exits.emplace_back(scope.WithOpName("exit"),
                             switch_node.output_false);

    // Complete loop by removing dummy node and attaching NextIteration to
    // that input of the merge node.
    scope.graph()->RemoveNode(dummy.node());
    scope.graph()->AddEdge(next_iteration.node(), 0, merge.output.node(), 1);

    int base_start_time = i * 10 + 100;
    for (const auto& op : std::initializer_list<Output>{
             enter, merge.output, cv, loop_cond, switch_node.output_false,
             identity, next_iteration, while_exits.back()}) {
      op.node()->AddAttr("_start_time", base_start_time++);
    }
  }

  // Create ops that depend on the loop exits.
  std::vector<Square> squares;
  squares.reserve(indexes.size());
  for (int i : indexes) {
    squares.emplace_back(root.WithOpName(strings::StrCat("s", i)),
                         while_exits[i]);
    squares.back().node()->AddAttr("_start_time", 500 - (i + 1));
  }

  GraphDef gdef;
  TF_EXPECT_OK(root.ToGraphDef(&gdef));

  // Run the sort. The while loop nodes do not appear in the output <nodes>.
  std::vector<std::pair<const NodeDef*, int64_t>> nodes;
  std::unordered_map<const NodeDef*, int64_t> node_to_start_time;
  TF_CHECK_OK(
      TopologicalSortNodesWithTimePriority(&gdef, &nodes, &node_to_start_time));
  ASSERT_LT(while_exits.size() + squares.size() + placeholders.size(),
            nodes.size());
  int node_index = 0;
  for (int i = 0; i < placeholders.size(); ++i, ++node_index) {
    const NodeDef* node = nodes[i].first;
    EXPECT_EQ(strings::StrCat("p", i), node->name());
    EXPECT_EQ(i + 1, nodes[i].second);
    EXPECT_EQ(i + 1, node_to_start_time[node]);
  }
  for (int i = 0; i < while_exits.size(); ++i, node_index += nodes_per_loop) {
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("while", i, "/Enter"), node->name());
    EXPECT_EQ(100 + i * 10, nodes[node_index].second);
    EXPECT_EQ(100 + i * 10, node_to_start_time[node]);
  }
  for (int i = 0; i < squares.size(); ++i, ++node_index) {
    int square_index = num_leaves - 1 - i;
    const NodeDef* node = nodes[node_index].first;
    EXPECT_EQ(strings::StrCat("s", square_index), node->name());
    EXPECT_EQ(500 - (square_index + 1), nodes[node_index].second);
    EXPECT_EQ(500 - (square_index + 1), node_to_start_time[node]);
  }
}

}  // namespace
}  // namespace tensorflow
