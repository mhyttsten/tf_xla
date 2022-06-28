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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/replicate_per_replica_nodes.h"

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GraphHelper {
 public:
  explicit GraphHelper(const Graph& graph) : graph_(graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes_test.cc", "GraphHelper");

    for (Node* node : graph.nodes()) {
      nodes_by_name_[node->name()] = node;
    }
  }

  Node* GetNodeByName(const string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes_test.cc", "GetNodeByName");

    const auto it = nodes_by_name_.find(name);
    if (it != nodes_by_name_.end()) {
      return it->second;
    }
    for (const auto& entry : nodes_by_name_) {
      if (absl::StartsWith(entry.first, name)) {
        return entry.second;
      }
    }
    return nullptr;
  }

  void SetAssignedDevice(const string& node_name, const string& device_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("node_name: \"" + node_name + "\"");
   mht_2_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes_test.cc", "SetAssignedDevice");

    CHECK_NOTNULL(GetNodeByName(node_name))
        ->set_assigned_device_name(device_name);
  }

  void CheckArgNum(const int expected_num) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes_test.cc", "CheckArgNum");

    int arg_num = 0;
    for (Node* node : graph_.op_nodes()) {
      if (node->IsArg()) {
        arg_num++;
      }
    }
    EXPECT_EQ(arg_num, expected_num);
  }

  void CheckAssignedDevice(const string& node_name,
                           const string& expected_device_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("node_name: \"" + node_name + "\"");
   mht_4_v.push_back("expected_device_name: \"" + expected_device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSreplicate_per_replica_nodes_testDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/common_runtime/replicate_per_replica_nodes_test.cc", "CheckAssignedDevice");

    EXPECT_EQ(expected_device_name,
              CHECK_NOTNULL(GetNodeByName(node_name))->assigned_device_name());
  }

 private:
  const Graph& graph_;
  // Maps from a node name to a Node* in the graph.
  absl::flat_hash_map<string, Node*> nodes_by_name_;
};

TEST(ReplicatePerReplicaNodesTest, SingleCompositeDevice) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto one = ops::Const<int32>(scope.WithOpName("one"), 1);
  auto write = ops::AssignVariableOp(scope.WithOpName("write"), arg, one);
  auto ret = ops::_Retval(
      scope.WithOpName("ret").WithControlDependencies({write}), read, 0);

  const std::vector<string> underlying_devices = {"TPU:0", "TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0);
    // Const(CPU:0) -> AssignVariableOp(TPU_COMPOSITE:0);
    // ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 5);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "TPU:0");
    helper.SetAssignedDevice("one", "CPU:0");
    helper.SetAssignedDevice("write", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("ret", "CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0, TPU:1) -> ReadVariableOp(TPU:0);
    // Const(CPU:0) -> AssignVariableOp(TPU:0, TPU:1);
    // ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 7);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevice("arg/R0", "TPU:0");
    helper.CheckAssignedDevice("arg/R1", "TPU:1");
    helper.CheckAssignedDevice("read", "TPU:0");
    helper.CheckAssignedDevice("one", "CPU:0");
    helper.CheckAssignedDevice("write/R0", "TPU:0");
    helper.CheckAssignedDevice("write/R1", "TPU:1");
    helper.CheckAssignedDevice("ret", "CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, SingleCompositeDeviceToSingleDevice) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);

  const std::vector<string> underlying_devices = {"TPU:0"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "TPU:0");
    helper.SetAssignedDevice("ret", "CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.CheckArgNum(1);
    helper.CheckAssignedDevice("arg", "TPU:0");
    helper.CheckAssignedDevice("read", "TPU:0");
    helper.CheckAssignedDevice("ret", "CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, MultipleCompositeDevices) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_RESOURCE, 0);
  Output arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 0);
  auto read0 = ops::ReadVariableOp(scope.WithOpName("read0"), arg0, DT_INT32);
  auto read1 = ops::ReadVariableOp(scope.WithOpName("read1"), arg1, DT_INT32);
  auto identity0 = ops::Identity(scope.WithOpName("identity0"), read0);
  auto identity1 = ops::Identity(scope.WithOpName("identity1"), read1);
  auto add = ops::Add(scope.WithOpName("add"), identity0, identity1);
  auto ret = ops::_Retval(scope.WithOpName("ret"), add, 0);

  const std::vector<string> underlying_devices_0 = {"TPU:0", "TPU:1"};
  const std::vector<string> underlying_devices_1 = {"TPU:2", "TPU:3"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"TPU_COMPOSITE:0", &underlying_devices_0},
                           {"TPU_COMPOSITE:1", &underlying_devices_1}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU_COMPOSITE:0) ->
    // Identity(TPU:1)
    // _Arg(TPU_COMPOSITE:1) -> ReadVariableOp(TPU_COMPOSITE:1)
    // -> Identity(TPU:3)
    // Identity(TPU:1), Identity(TPU:3) -> Add(TPU:0)-> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 8);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg0", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read0", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("identity0", "TPU:1");
    helper.SetAssignedDevice("arg1", "TPU_COMPOSITE:1");
    helper.SetAssignedDevice("read1", "TPU_COMPOSITE:1");
    helper.SetAssignedDevice("identity1", "TPU:3");
    helper.SetAssignedDevice("add", "TPU:0");
    helper.SetAssignedDevice("ret", "CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0, TPU:3) -> ReadVariableOp(TPU:1, TPU:3) -> Identity(TPU:1,
    // TPU:3) -> Add(TPU:0)-> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 8);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevice("arg0/R1", "TPU:1");
    helper.CheckAssignedDevice("arg1/R1", "TPU:3");
    helper.CheckAssignedDevice("read0/R1", "TPU:1");
    helper.CheckAssignedDevice("read1/R1", "TPU:3");
    helper.CheckAssignedDevice("identity0", "TPU:1");
    helper.CheckAssignedDevice("identity1", "TPU:3");
    helper.CheckAssignedDevice("add", "TPU:0");
    helper.CheckAssignedDevice("ret", "CPU:0");
  }
}

TEST(ReplicatePerReplicaNodesTest, NestedFunctions) {
  const std::vector<string> underlying_devices = {"TPU:0", "TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"TPU_COMPOSITE:0", &underlying_devices}};

  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
    auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
    auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);
    Graph graph(OpRegistry::Global());
    TF_ASSERT_OK(scope.ToGraph(&graph));
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "TPU:0");
    helper.SetAssignedDevice("ret", "CPU:0");
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(graph, "Func", &fdef));
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));
  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("func", "Func", &flib_def)
                   .Input(arg.name(), 0, DT_RESOURCE)
                   .Finalize(&def));
  Status status;
  Node* func = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(arg.node(), 0, func, 0);
  auto ret = ops::_Retval(scope.WithOpName("ret"), Output(func), 0);
  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> Func(CPU:0) -> _Retval(CPU:0)
    GraphHelper helper(graph);
    EXPECT_EQ(graph.num_op_nodes(), 3);
    helper.SetAssignedDevice("arg", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("func", "CPU:0");
    helper.SetAssignedDevice("ret", "CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0), _Arg(TPU:1) -> Pack(CPU:0) -> Func(CPU:0) -> _Retval(CPU:0)
    EXPECT_EQ(graph.num_op_nodes(), 5);
    GraphHelper helper(graph);
    helper.CheckArgNum(2);
    helper.CheckAssignedDevice("arg/R0", "TPU:0");
    helper.CheckAssignedDevice("arg/R1", "TPU:1");
    helper.CheckAssignedDevice("arg/Packed", "CPU:0");
    helper.CheckAssignedDevice("func", "CPU:0");
    helper.CheckAssignedDevice("ret", "CPU:0");
    const EdgeSet& packed_in_edges =
        helper.GetNodeByName("arg/Packed")->in_edges();
    EXPECT_EQ(packed_in_edges.size(), 2);
    auto it = packed_in_edges.begin();
    EXPECT_EQ(helper.GetNodeByName("arg/R0"), (*it++)->src());
    EXPECT_EQ(helper.GetNodeByName("arg/R1"), (*it)->src());
    const EdgeSet& func_in_edges = helper.GetNodeByName("func")->in_edges();
    EXPECT_EQ(func_in_edges.size(), 1);
    EXPECT_EQ(helper.GetNodeByName("arg/Packed"),
              (*func_in_edges.begin())->src());
  }
}

TEST(ReplicatePerReplicaNodesTest, DeadArgNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto read = ops::ReadVariableOp(scope.WithOpName("read"), arg, DT_INT32);
  auto ret = ops::_Retval(scope.WithOpName("ret"), read, 0);

  const std::vector<string> underlying_devices = {"TPU:0", "TPU:1"};
  const absl::flat_hash_map<string, const std::vector<string>*>
      composite_devices = {{"TPU_COMPOSITE:0", &underlying_devices}};

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  {
    // _Arg(TPU_COMPOSITE:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    ASSERT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.SetAssignedDevice("arg", "TPU_COMPOSITE:0");
    helper.SetAssignedDevice("read", "TPU:0");
    helper.SetAssignedDevice("ret", "CPU:0");
  }

  TF_EXPECT_OK(
      ReplicatePerReplicaNodesInFunctionGraph(composite_devices, &graph));

  {
    // _Arg(TPU:0) -> ReadVariableOp(TPU:0) -> _Retval(CPU:0)
    // "arg/R1" is a dead node, so gets removed.
    EXPECT_EQ(graph.num_op_nodes(), 3);
    GraphHelper helper(graph);
    helper.CheckArgNum(1);
    helper.CheckAssignedDevice("arg/R0", "TPU:0");
    helper.CheckAssignedDevice("read", "TPU:0");
    helper.CheckAssignedDevice("ret", "CPU:0");
  }
}

}  // namespace
}  // namespace tensorflow
