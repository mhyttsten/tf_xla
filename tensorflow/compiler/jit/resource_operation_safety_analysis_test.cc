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
class MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc() {
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

#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Node* MakeRead(const Scope& scope, const string& id) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "MakeRead");

  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output read =
      ops::ReadVariableOp(scope.WithOpName("Read" + id), var_handle, DT_FLOAT);
  return read.node();
}

Node* MakeWrite(const Scope& scope, const string& id) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "MakeWrite");

  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output value_to_write =
      ops::Const(scope.WithOpName("ValueToAssign" + id), 1.0f);
  ops::AssignVariableOp assign_op(scope.WithOpName("Assignee" + id), var_handle,
                                  value_to_write);
  return assign_op.operation.node();
}

Node* MakeModify(const Scope& scope, const string& id) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "MakeModify");

  Output var_handle =
      ops::VarHandleOp(scope.WithOpName("Var" + id), DT_FLOAT, TensorShape({}));
  Output value_to_write = ops::Const(scope.WithOpName("Increment" + id), 1.0f);
  ops::AssignAddVariableOp assign_add_op(scope.WithOpName("Increment" + id),
                                         var_handle, value_to_write);
  return assign_add_op.operation.node();
}

Node* MakeNeutral(const Scope& scope, const string& id) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("id: \"" + id + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "MakeNeutral");

  return ops::Const(scope.WithOpName("Const" + id), 42.0f).node();
}

Status ComputeIncompatiblePairs(Graph* g,
                                std::vector<std::pair<int, int>>* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_4(mht_4_v, 258, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "ComputeIncompatiblePairs");

  FixupSourceAndSinkEdges(g);
  return ComputeIncompatibleResourceOperationPairs(*g, &g->flib_def(), {},
                                                   result);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(read, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadWriteNoEdges) {
  Scope root = Scope::NewRootScope().ExitOnError();

  MakeRead(root, "R");
  MakeWrite(root, "W");

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");

  root.graph()->AddControlEdge(read, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, ModifyRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");

  root.graph()->AddControlEdge(modify, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> modify_read_pair = {modify->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], modify_read_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ModifyWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(modify, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  EXPECT_EQ(incompatible_pairs[0], write_modify_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadModifyWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(read, modify);
  root.graph()->AddControlEdge(modify, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteModifyRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, modify);
  root.graph()->AddControlEdge(modify, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 3);

  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  std::pair<int, int> modify_read_pair = {modify->id(), read->id()};
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], modify_read_pair);
  EXPECT_EQ(incompatible_pairs[1], write_read_pair);
  EXPECT_EQ(incompatible_pairs[2], write_modify_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteReadModify) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* read = MakeRead(root, "R");
  Node* modify = MakeModify(root, "M");
  Node* write = MakeWrite(root, "W");

  root.graph()->AddControlEdge(write, read);
  root.graph()->AddControlEdge(read, modify);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 2);

  std::pair<int, int> write_modify_pair = {write->id(), modify->id()};
  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
  EXPECT_EQ(incompatible_pairs[1], write_modify_pair);
}

FunctionDefLibrary CreateFunctionDefLibWithConstFunction(const string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_5(mht_5_v, 430, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "CreateFunctionDefLibWithConstFunction");

  FunctionDefLibrary flib_def;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{}, /*out_def=*/{"out: float"},
      /*attr_def*/
      {}, /*node_def=*/{FunctionDefHelper::Const("one", 1.0f)},
      /*ret_def=*/{{"out", "out:output:0"}});
  *flib_def.add_function() = std::move(func);
  return flib_def;
}

Node* MakeCall(Graph* graph, const string& callee_name, const string& node_name,
               Status* status) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("callee_name: \"" + callee_name + "\"");
   mht_6_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_6(mht_6_v, 447, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "MakeCall");

  NodeDef call_node;
  call_node.set_name(node_name);
  call_node.set_op(callee_name);
  return graph->AddNode(call_node, status);
}

TEST(ResourceOperationSafetyAnalysisTest, CallRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(call, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> call_read_edge = {call->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], call_read_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, ReadCall) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(read, call);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, CallWrite) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(call, write);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  EXPECT_EQ(incompatible_pairs.size(), 0);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteCall) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  Status status;
  Node* call = MakeCall(root.graph(), "Const_func", "C", &status);
  TF_ASSERT_OK(status);

  root.graph()->AddControlEdge(write, call);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_call_edge = {write->id(), call->id()};
  EXPECT_EQ(incompatible_pairs[0], write_call_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, SymbolicGradientRead) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* read = MakeRead(root, "R");
  NameAttrList fn;
  fn.set_name("Const_func");
  Node* symbolic_gradient =
      ops::SymbolicGradient(root, /*input=*/{ops::Const(root, 1.0f)},
                            /*Tout=*/{DT_FLOAT}, fn)
          .output[0]
          .node();

  root.graph()->AddControlEdge(symbolic_gradient, read);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> symbolic_gradient_read_edge = {symbolic_gradient->id(),
                                                     read->id()};
  EXPECT_EQ(incompatible_pairs[0], symbolic_gradient_read_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, WriteSymbolicGradient) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary flib_def =
      CreateFunctionDefLibWithConstFunction("Const_func");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(flib_def));

  Node* write = MakeWrite(root, "W");
  NameAttrList fn;
  fn.set_name("Const_func");
  Node* symbolic_gradient =
      ops::SymbolicGradient(root, /*input=*/{ops::Const(root, 1.0f)},
                            /*Tout=*/{DT_FLOAT}, fn)
          .output[0]
          .node();

  root.graph()->AddControlEdge(write, symbolic_gradient);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);
  std::pair<int, int> write_symbolic_gradient_edge = {write->id(),
                                                      symbolic_gradient->id()};
  EXPECT_EQ(incompatible_pairs[0], write_symbolic_gradient_edge);
}

TEST(ResourceOperationSafetyAnalysisTest, ChainOfOps) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* neutral_0 = MakeNeutral(root, "N0");
  Node* read_0 = MakeRead(root, "R0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral_1 = MakeNeutral(root, "N1");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral_0);
  root.graph()->AddControlEdge(neutral_0, read_0);
  root.graph()->AddControlEdge(read_0, write_1);
  root.graph()->AddControlEdge(write_1, neutral_1);
  root.graph()->AddControlEdge(neutral_1, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 3);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, DagOfOps) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral = MakeNeutral(root, "N");
  Node* read_0 = MakeRead(root, "R0");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral);
  root.graph()->AddControlEdge(write_1, neutral);
  root.graph()->AddControlEdge(neutral, read_0);
  root.graph()->AddControlEdge(neutral, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 4);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_0_pair = {write_1->id(), read_0->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_0_pair);
  EXPECT_EQ(incompatible_pairs[3], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, DagOfOpsWithRepeatedPaths) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Node* write_0 = MakeWrite(root, "W0");
  Node* write_1 = MakeWrite(root, "W1");
  Node* neutral = MakeNeutral(root, "N");
  Node* read_0 = MakeRead(root, "R0");
  Node* read_1 = MakeRead(root, "R1");

  root.graph()->AddControlEdge(write_0, neutral);
  root.graph()->AddControlEdge(write_1, neutral);
  root.graph()->AddControlEdge(neutral, read_0);
  root.graph()->AddControlEdge(neutral, read_1);
  root.graph()->AddControlEdge(write_1, read_1);

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 4);
  std::pair<int, int> write_0_read_0_pair = {write_0->id(), read_0->id()};
  std::pair<int, int> write_0_read_1_pair = {write_0->id(), read_1->id()};
  std::pair<int, int> write_1_read_0_pair = {write_1->id(), read_0->id()};
  std::pair<int, int> write_1_read_1_pair = {write_1->id(), read_1->id()};

  EXPECT_EQ(incompatible_pairs[0], write_0_read_0_pair);
  EXPECT_EQ(incompatible_pairs[1], write_0_read_1_pair);
  EXPECT_EQ(incompatible_pairs[2], write_1_read_0_pair);
  EXPECT_EQ(incompatible_pairs[3], write_1_read_1_pair);
}

TEST(ResourceOperationSafetyAnalysisTest, Loop) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output init_value = ops::Placeholder(root.WithOpName("init"), DT_FLOAT);
  Output loop_cond = ops::Placeholder(root.WithOpName("init"), DT_BOOL);
  Output enter_value =
      ops::internal::Enter(root.WithOpName("enter"), init_value, "fr");
  ops::Merge iv(root.WithOpName("iv"), {enter_value, enter_value});
  ops::Switch latch(root.WithOpName("latch"), iv.output, loop_cond);
  ops::internal::Exit exit(root.WithOpName("exit"), iv.output);
  Output next_iteration =
      ops::NextIteration(root.WithOpName("next_iteration"), latch.output_true);
  TF_ASSERT_OK(
      root.graph()->UpdateEdge(next_iteration.node(), 0, iv.output.node(), 1));

  Node* write = MakeWrite(root, "W");
  Node* read = MakeRead(root, "R");

  root.graph()->AddControlEdge(iv.output.node(), write);
  root.graph()->AddControlEdge(write, read);
  root.graph()->AddControlEdge(read, next_iteration.node());

  std::vector<std::pair<int, int>> incompatible_pairs;
  TF_ASSERT_OK(ComputeIncompatiblePairs(root.graph(), &incompatible_pairs));

  ASSERT_EQ(incompatible_pairs.size(), 1);

  std::pair<int, int> write_read_pair = {write->id(), read->id()};
  EXPECT_EQ(incompatible_pairs[0], write_read_pair);
}

bool IsResourceArgDef(const OpDef::ArgDef& arg_def) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSresource_operation_safety_analysis_testDTcc mht_7(mht_7_v, 714, "", "./tensorflow/compiler/jit/resource_operation_safety_analysis_test.cc", "IsResourceArgDef");

  return arg_def.type() == DT_RESOURCE;
}
}  // namespace
}  // namespace tensorflow
