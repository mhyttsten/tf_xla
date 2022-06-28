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
class MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc() {
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

#include "tensorflow/compiler/jit/build_xla_ops_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_subgraphs_pass.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

class BuildXlaOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "SetUp");

    // This is needed to register the XLA_* devices.
    CHECK(DeviceFactory::AddDevices(
              SessionOptions(), "/job:localhost/replica:0/task:0", &devices_)
              .ok());
  }

 private:
  std::vector<std::unique_ptr<Device>> devices_;
};

using ::tensorflow::testing::FindNodeByName;
using ::tensorflow::testing::matchers::Attr;
using ::tensorflow::testing::matchers::CtrlDeps;
using ::tensorflow::testing::matchers::Inputs;
using ::tensorflow::testing::matchers::NodeWith;
using ::tensorflow::testing::matchers::Op;
using ::tensorflow::testing::matchers::Out;
using ::testing::_;

Status BuildXlaOps(const Scope& s, const FunctionDefLibrary& fdef_lib,
                   std::unique_ptr<Graph>* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "BuildXlaOps");

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  FunctionLibraryDefinition flib_def(graph->op_registry(), fdef_lib);

  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : graph->nodes()) {
    if (n->requested_device().empty()) {
      n->set_assigned_device_name(kCpuDevice);
    } else {
      n->set_assigned_device_name(n->requested_device());
    }
  }

  FixupSourceAndSinkEdges(graph.get());

  GraphOptimizationPassWrapper wrapper;
  GraphOptimizationPassOptions opt_options =
      wrapper.CreateGraphOptimizationPassOptions(&graph);
  opt_options.flib_def = &flib_def;

  BuildXlaOpsPass pass(/*enable_lazy_compilation=*/true);
  TF_RETURN_IF_ERROR(pass.Run(opt_options));
  VLOG(3) << graph->ToGraphDefDebug().DebugString();
  *result = std::move(graph);
  return Status::OK();
}

Status MakeXlaCompiledKernel(Graph* graph, const string& callee_name,
                             const string& node_name, int num_constant_args,
                             int num_resource_args, Node** result) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("callee_name: \"" + callee_name + "\"");
   mht_2_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_2(mht_2_v, 267, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "MakeXlaCompiledKernel");

  NodeDef call_node;
  call_node.set_name(node_name);
  call_node.set_op(callee_name);
  AddNodeAttr(kXlaCompiledKernelAttr, true, &call_node);
  AddNodeAttr(kXlaNumConstantArgsAttr, num_constant_args, &call_node);
  AddNodeAttr(kXlaNumResourceArgsAttr, num_resource_args, &call_node);
  TF_ASSIGN_OR_RETURN(*result, graph->AddNode(call_node));
  return Status::OK();
}

Status MakeXlaCompiledKernel(Graph* graph, const string& callee_name,
                             const string& node_name, Node** result) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("callee_name: \"" + callee_name + "\"");
   mht_3_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "MakeXlaCompiledKernel");

  return MakeXlaCompiledKernel(graph, callee_name, node_name,
                               /*num_constant_args=*/0, /*num_resource_args=*/0,
                               result);
}

Node* MakeWrite(const Scope& scope, Output value_to_write, const string& id) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_4(mht_4_v, 294, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "MakeWrite");

  Output var_handle = ops::VarHandleOp(scope.WithOpName("Var_" + id), DT_FLOAT,
                                       TensorShape({}));
  ops::AssignVariableOp assign_op(scope.WithOpName("Assignee_" + id),
                                  var_handle, value_to_write);
  return assign_op.operation.node();
}

Node* MakeWrite(const Scope& scope, const string& id) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "MakeWrite");

  return MakeWrite(
      scope, ops::Const(scope.WithOpName("ValueToAssign" + id), 1.0f), id);
}

FunctionDefLibrary CreateFunctionDefLibWithConstFunction(const string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSbuild_xla_ops_pass_testDTcc mht_6(mht_6_v, 315, "", "./tensorflow/compiler/jit/build_xla_ops_pass_test.cc", "CreateFunctionDefLibWithConstFunction");

  FunctionDefLibrary fdef_lib;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{}, /*out_def=*/{"out: float"},
      /*attr_def*/
      {}, /*node_def=*/{FunctionDefHelper::Const("one", 1.0f)},
      /*ret_def=*/{{"out", "out:output:0"}});
  *fdef_lib.add_function() = std::move(func);
  return fdef_lib;
}

TEST_F(BuildXlaOpsTest, ControlDepsPreserved) {
  const char* kXlaDeviceName = "/job:worker/replica:0/task:0/device:XLA_CPU:0";
  Scope root = Scope::NewRootScope().WithDevice(kXlaDeviceName).ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));
  Node* call;
  TF_ASSERT_OK(MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", &call));
  call->AddAttr(kXlaHasReferenceVarsAttr, false);
  call->set_requested_device(kXlaDeviceName);
  Node* write_op = MakeWrite(root, "write");
  write_op->AddAttr(kXlaHasReferenceVarsAttr, false);
  root.graph()->AddControlEdge(call, write_op);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, fdef_lib, &graph));

  Node* write_op_new = FindNodeByName(graph.get(), write_op->name());
  ASSERT_NE(write_op_new, nullptr);
  EXPECT_THAT(write_op_new, NodeWith(CtrlDeps(NodeWith(Op("_XlaRun")))));
}

TEST_F(BuildXlaOpsTest, CleanFailureOnBogusAttr) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));

  Node* call;
  TF_ASSERT_OK(
      MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", 100, 100, &call));

  Node* write_op = MakeWrite(root, "write");
  root.graph()->AddControlEdge(call, write_op);

  std::unique_ptr<Graph> graph;
  Status failure_status = BuildXlaOps(root, fdef_lib, &graph);
  ASSERT_FALSE(failure_status.ok());
  EXPECT_EQ(failure_status.code(), error::INVALID_ARGUMENT);
}

TEST_F(BuildXlaOpsTest, OnNonXlaDevice) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));

  Node* call;
  TF_ASSERT_OK(MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", &call));
  TF_ASSERT_OK(root.DoShapeInference(call));
  call->AddAttr(kXlaHasReferenceVarsAttr, false);

  Node* write_op = MakeWrite(root, Output(call), "write_result");
  write_op->AddAttr(kXlaHasReferenceVarsAttr, false);

  auto xla_compile = NodeWith(Op("_XlaCompile"), Attr("must_compile", false));
  auto predicated_compilation_key =
      NodeWith(Op("Switch"), Inputs(Out(0, xla_compile), Out(1, xla_compile)));
  auto xla_run =
      NodeWith(Op("_XlaRun"), Inputs(Out(1, predicated_compilation_key)));
  auto tf_call =
      NodeWith(Op("StatefulPartitionedCall"),
               CtrlDeps(NodeWith(Op("Identity"),
                                 Inputs(Out(0, predicated_compilation_key)))));
  auto merge = NodeWith(Op("_XlaMerge"), Inputs(Out(tf_call), Out(xla_run)));
  auto assign_var = NodeWith(Op("AssignVariableOp"), Inputs(_, Out(merge)));

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, fdef_lib, &graph));

  Node* write_op_new = FindNodeByName(graph.get(), write_op->name());
  ASSERT_NE(write_op_new, nullptr);
  EXPECT_THAT(write_op_new, assign_var);
}

TEST_F(BuildXlaOpsTest, OnXlaDevice) {
  const char* kXlaDeviceName = "/job:worker/replica:0/task:0/device:XLA_CPU:0";
  Scope root = Scope::NewRootScope().WithDevice(kXlaDeviceName).ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));

  Node* call;
  TF_ASSERT_OK(MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", &call));
  call->set_requested_device(kXlaDeviceName);
  TF_ASSERT_OK(root.DoShapeInference(call));
  call->AddAttr(kXlaHasReferenceVarsAttr, false);

  Node* write_op = MakeWrite(root, Output(call), "write_result");
  write_op->AddAttr(kXlaHasReferenceVarsAttr, false);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, fdef_lib, &graph));

  auto xla_op =
      NodeWith(Op("_XlaRun"), Inputs(Out(NodeWith(Op("_XlaCompile")))));
  auto assign_var =
      NodeWith(Op("AssignVariableOp"), Inputs(Out(NodeWith()), Out(xla_op)));

  Node* write_op_new = FindNodeByName(graph.get(), write_op->name());
  ASSERT_NE(write_op_new, nullptr);
  EXPECT_THAT(write_op_new, assign_var);
}

TEST_F(BuildXlaOpsTest, NoExtraMergeForEdgeToSink) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));
  Node* call;
  TF_ASSERT_OK(MakeXlaCompiledKernel(root.graph(), "cluster_0", "C", &call));
  call->AddAttr(kXlaHasReferenceVarsAttr, false);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, fdef_lib, &graph));

  Node* sink_node = graph->sink_node();
  EXPECT_THAT(sink_node,
              NodeWith(CtrlDeps(NodeWith(Op("_XlaRun")),
                                NodeWith(Op("StatefulPartitionedCall")),
                                NodeWith(Op("NoOp")))));
}

#ifdef GOOGLE_CUDA
FunctionDefLibrary CreateFunctionDefLibWithInt32Input(const string& name) {
  FunctionDefLibrary fdef_lib;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{"in: int32"},
      /*out_def=*/{"out: int32"},
      /*attr_def=*/{}, /*node_def=*/{{{"out"}, "Identity", {"in"}}},
      /*ret_def=*/{{"out", "out:output:0"}});
  *fdef_lib.add_function() = std::move(func);
  return fdef_lib;
}

// This tests a rewrite that only makes sense and is active in a CUDA-enabled
// build.  Specifically we check that we insert an IdentityN op to avoid extra
// device-to-host copies.
TEST_F(BuildXlaOpsTest, NoDeviceToHostCopiesForClustersWithInt32Inputs) {
  const char* kXlaDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";
  Scope root = Scope::NewRootScope()
                   .WithDevice(kXlaDeviceName)
                   .WithAssignedDevice(kXlaDeviceName)
                   .ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithInt32Input("cluster_int32");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));
  Node* call;
  TF_ASSERT_OK(
      MakeXlaCompiledKernel(root.graph(), "cluster_int32", "C", &call));
  call->set_requested_device(kXlaDeviceName);
  call->AddAttr(kXlaHasReferenceVarsAttr, false);

  auto var =
      ops::VarHandleOp(root.WithOpName("var"), DT_INT32, TensorShape({}));
  auto int32_on_device =
      ops::ReadVariableOp(root.WithOpName("int32_on_device"), var, DT_INT32);

  root.graph()->AddEdge(int32_on_device.node(), 0, call, 0);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildXlaOps(root, fdef_lib, &graph));

  Node* stateful_partitioned_call_op = nullptr;
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == "StatefulPartitionedCall") {
      ASSERT_EQ(stateful_partitioned_call_op, nullptr);
      stateful_partitioned_call_op = n;
    }
  }

  ASSERT_NE(stateful_partitioned_call_op, nullptr);
  auto xla_compile = NodeWith(Op("_XlaCompile"));
  auto switch_on_compilation_pred =
      NodeWith(Op("Switch"), Inputs(Out(0, xla_compile), Out(1, xla_compile)));
  auto ctrl_dep =
      NodeWith(Op("Identity"), Inputs(Out(0, switch_on_compilation_pred)));
  // Check that we pipe int32 inputs through an IdentityN to avoid extra D2H
  // copies.
  EXPECT_THAT(
      stateful_partitioned_call_op,
      NodeWith(Inputs(Out(NodeWith(Op("IdentityN"), CtrlDeps(ctrl_dep))))));
}
#endif

}  // namespace
}  // namespace tensorflow
