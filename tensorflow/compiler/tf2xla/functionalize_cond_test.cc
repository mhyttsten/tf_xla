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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc() {
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

// Tests for the backward const analysis.

#include "tensorflow/compiler/tf2xla/functionalize_cond.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace functionalize_cond {

class FunctionalizeCondTest : public ::testing::Test {
 protected:
  FunctionalizeCondTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/tf2xla/functionalize_cond_test.cc", "FunctionalizeCondTest");

    graph_.reset(new Graph(OpRegistry::Global()));
    flib_def_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib_));
    fc_.reset(new functionalize_cond::FunctionalizeCond(
        graph_.get(), flib_def_.get(), NodeFilter{}));
  }

  StateMap::CondId GetUniqueId(const StateMap::StateMap::CondState& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/functionalize_cond_test.cc", "GetUniqueId");

    return fc_->state_map_.GetCondId(state);
  }

  string GetString(const StateMap::StateMap::CondId id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSfunctionalize_cond_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/tf2xla/functionalize_cond_test.cc", "GetString");

    return fc_->state_map_.CondStateToString(id);
  }

  StatusOr<StateMap::CondId> JoinCondStatesNonMerge(StateMap::CondId src,
                                                    StateMap::CondId dst) {
    return fc_->JoinCondStatesNonMerge(src, dst);
  }

  StatusOr<StateMap::CondId> JoinCondStatesMerge(Node* n, StateMap::CondId src,
                                                 StateMap::CondId dst) {
    return fc_->JoinCondStatesMerge(n, src, dst);
  }

  FunctionDefLibrary fdef_lib_;
  std::unique_ptr<functionalize_cond::FunctionalizeCond> fc_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<Graph> graph_;
};

namespace {

TEST_F(FunctionalizeCondTest, JoinCondStates) {
  Tensor pred_tensor(DT_BOOL, TensorShape());
  pred_tensor.flat<bool>().setZero();
  Node* pred = test::graph::Constant(graph_.get(), pred_tensor, "pred");
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* m = test::graph::Merge(graph_.get(), val, val);

  StateMap::CondId then_branch;
  {
    StateMap::CondState ss;
    ss.insert(std::make_pair(OutputTensor(pred, 0), BranchType::kThenBranch));
    then_branch = GetUniqueId(ss);
  }
  StateMap::CondId else_branch;
  {
    StateMap::CondState ss;
    ss.insert(std::make_pair(OutputTensor(pred, 0), BranchType::kElseBranch));
    else_branch = GetUniqueId(ss);
  }

  // An non-merge op with inputs from then and else branch.
  Status status = JoinCondStatesNonMerge(then_branch, else_branch).status();
  EXPECT_TRUE(errors::IsInvalidArgument(status));

  // Merge between then and else branch.
  auto joined_or = JoinCondStatesMerge(m, then_branch, else_branch);
  TF_EXPECT_OK(joined_or.status());
  StateMap::CondId joined = joined_or.ValueOrDie();

  // Merge between then branch and both branch.
  auto t = JoinCondStatesNonMerge(then_branch, joined);
  // Note: this is OK in terms of constraint predication, but
  TF_EXPECT_OK(t.status());
}

TEST_F(FunctionalizeCondTest, JoinCondStatesMergeWithInputNotInCondContext) {
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* m = test::graph::Merge(graph_.get(), val, val);

  StateMap::CondState cond_state;
  auto joined_or = JoinCondStatesMerge(m, /*src=*/nullptr, &cond_state);
  EXPECT_FALSE(joined_or.ok());
}

TEST(FunctionalizeCond, DuplicateConstNodes) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto const_op = ops::Const(root.WithOpName("const"), 1);
  auto arg_0_op = ops::_Arg(root.WithOpName("arg_0"), DT_BOOL, 0);
  auto arg_1_op = ops::_Arg(root.WithOpName("arg_1"), DT_INT32, 1);
  auto switch_op = ops::Switch(root.WithOpName("switch"), arg_1_op, arg_0_op);
  auto identity_n_false_op =
      ops::IdentityN(root.WithOpName("identity_n_0"),
                     {switch_op.output_false, const_op, const_op});
  auto identity_n_true_op =
      ops::IdentityN(root.WithOpName("identity_n_1"),
                     {switch_op.output_true, const_op, const_op});
  auto merge_op = ops::Merge(
      root.WithOpName("merge"),
      {identity_n_false_op.output.front(), identity_n_true_op.output.front()});
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, &graph));

  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);

  auto status = tensorflow::FunctionalizeCond(&graph, &flib_def);
  TF_ASSERT_OK(status);

  FunctionDefLibrary flib_def_proto = flib_def.ToProto();
  for (const auto& fdef : flib_def_proto.function()) {
    absl::flat_hash_set<absl::string_view> node_names;
    for (const auto& node : fdef.node_def()) {
      EXPECT_TRUE(node_names.insert(node.name()).second)
          << node.op() << " with duplicate node name '" << node.name()
          << "' found.";
    }
  }
}

}  // namespace
}  // namespace functionalize_cond
}  // namespace tensorflow
