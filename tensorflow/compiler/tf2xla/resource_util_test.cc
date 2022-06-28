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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_util_testDTcc() {
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

#include "tensorflow/compiler/tf2xla/resource_util.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {
ResourceUsageAnalysis::NodeInfo node_info_from_string(absl::string_view s) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + std::string(s.data(), s.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_util_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/resource_util_test.cc", "node_info_from_string");

  std::vector<std::string> tokens = absl::StrSplit(s, ':');
  EXPECT_EQ(tokens.size(), 3);

  ResourceUsageAnalysis::NodeInfo node_info;
  if (tokens[0].empty()) {
    node_info.function_name_ = absl::nullopt;
  } else {
    node_info.function_name_ = std::move(tokens[0]);
  }
  node_info.node_name_ = std::move(tokens[1]);
  node_info.op_ = std::move(tokens[2]);
  return node_info;
}

void AnalyzeAndVerify(
    const GraphDef& graphdef, FunctionLibraryDefinition* flib_def,
    const absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>>&
        expected) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSresource_util_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/tf2xla/resource_util_test.cc", "AnalyzeAndVerify");

  auto graph = absl::make_unique<Graph>(flib_def);
  TF_EXPECT_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), graphdef, graph.get()));

  auto pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      nullptr, Env::Default(), /*config=*/nullptr, TF_GRAPH_DEF_VERSION,
      flib_def, OptimizerOptions());
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                      absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
      source_to_path;
  TF_EXPECT_OK(ResourceUsageAnalysis::Analyze(graph.get(), lib_runtime,
                                              &source_to_path));

  absl::flat_hash_map<ResourceUsageAnalysis::NodeInfo,
                      absl::flat_hash_set<ResourceUsageAnalysis::NodeInfo>>
      expected_source_to_path;
  for (auto it : expected) {
    auto src_node_info = node_info_from_string(it.first);
    for (const std::string& user : it.second) {
      expected_source_to_path[src_node_info].emplace(
          node_info_from_string(user));
    }
  }

  EXPECT_EQ(source_to_path, expected_source_to_path);
}

}  // anonymous namespace

TEST(ResourceOpAnalyzerTest, SingleResourceSingleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] =
      absl::flat_hash_set<std::string>({":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceSingleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> resource_identity -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_identity_builder("resource_identity", "Identity",
                                          op_reg);
    resource_identity_builder.Input(stack_op);
    Node* resource_identity = opts.FinalizeBuilder(&resource_identity_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_identity:Identity", ":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceMultipleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                        stack_close0
     *                       /
     * stack_size -> stack_op
     *                       \
     *                        stack_close1
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op);
    opts.FinalizeBuilder(&stack_close1_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, SingleResourceMultipleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                                              stack_close0
     *                                             /
     * stack_size -> stack_op -> resource_identity
     *                                             \
     *                                              stack_close1
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_identity_builder("resource_identity", "Identity",
                                          op_reg);
    resource_identity_builder.Input(stack_op);
    Node* resource_identity = opts.FinalizeBuilder(&resource_identity_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(resource_identity);
    opts.FinalizeBuilder(&stack_close1_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_identity:Identity", ":stack_close0:StackCloseV2",
       ":stack_close1:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, MultipleResourceMultipleUserNoPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     *                        stack_close0
     *                       /
     *               stack_op0
     *             /         \
     *            /           stack_close1
     * stack_size
     *            \           stack_close2
     *             \         /
     *               stack_op1
     *                       \
     *                         stack_close3
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op0_builder("stack_op0", "StackV2", op_reg);
    stack_op0_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op0 = opts.FinalizeBuilder(&stack_op0_builder);

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close1_builder);

    NodeBuilder stack_op1_builder("stack_op1", "StackV2", op_reg);
    stack_op1_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op1 = opts.FinalizeBuilder(&stack_op1_builder);

    NodeBuilder stack_close2_builder("stack_close2", "StackCloseV2", op_reg);
    stack_close2_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close2_builder);

    NodeBuilder stack_close3_builder("stack_close3", "StackCloseV2", op_reg);
    stack_close3_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close3_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op0:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  expected[":stack_op1:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close2:StackCloseV2", ":stack_close3:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, MultipleResourceMultipleUserWithPassThrough) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*                                I
     *               stack_op0  ----> d  --->  stack_close0
     *             /                  e
     *            /                   n
     * stack_size ------------------> t
     *            \                   i
     *             \                  t
     *               stack_op1  ----> y  --->  stack_close0
     *                                N
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op0_builder("stack_op0", "StackV2", op_reg);
    stack_op0_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op0 = opts.FinalizeBuilder(&stack_op0_builder);

    NodeBuilder stack_op1_builder("stack_op1", "StackV2", op_reg);
    stack_op1_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op1 = opts.FinalizeBuilder(&stack_op1_builder);

    NodeBuilder identity_n_builder("identity_n", "IdentityN", op_reg);
    identity_n_builder.Input({stack_op0, stack_size_placeholder, stack_op1});

    NodeBuilder stack_close0_builder("stack_close0", "StackCloseV2", op_reg);
    stack_close0_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close0_builder);

    NodeBuilder stack_close1_builder("stack_close1", "StackCloseV2", op_reg);
    stack_close1_builder.Input(stack_op0);
    opts.FinalizeBuilder(&stack_close1_builder);

    NodeBuilder stack_close2_builder("stack_close2", "StackCloseV2", op_reg);
    stack_close2_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close2_builder);

    NodeBuilder stack_close3_builder("stack_close3", "StackCloseV2", op_reg);
    stack_close3_builder.Input(stack_op1);
    opts.FinalizeBuilder(&stack_close3_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op0:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close0:StackCloseV2", ":stack_close1:StackCloseV2"});
  expected[":stack_op1:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close2:StackCloseV2", ":stack_close3:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourcePassThroughFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  pass_through_function:
   *
   *  _Arg -> Identity -> _Retval
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"pass_through_function",
      /*arg_def=*/{"in: resource"},
      /*ret_def=*/{"out: resource"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"out"}, "Identity", {"in"}, {{"T", DataType::DT_RESOURCE}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> pass_through_function -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder pass_through_fn_builder("pass_through_fn",
                                        "pass_through_function", op_reg);
    pass_through_fn_builder.Input(stack_op);
    Node* pass_through_fn = opts.FinalizeBuilder(&pass_through_fn_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(pass_through_fn);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":stack_close:StackCloseV2", ":pass_through_fn:pass_through_function",
       "pass_through_function:out:Identity"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourceUserInFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  resource_user_function:
   *
   *  _Arg -> Identity -> StackCloseV2
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"resource_user_function",
      /*arg_def=*/{"in: resource"},
      /*ret_def=*/{},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"stack_close"},
        "StackCloseV2",
        {"in"},
        {{"T", DataType::DT_RESOURCE}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> stack_op -> resource_user_function
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder stack_op_builder("stack_op", "StackV2", op_reg);
    stack_op_builder.Input(stack_size_placeholder).Attr("elem_type", DT_FLOAT);
    Node* stack_op = opts.FinalizeBuilder(&stack_op_builder);

    NodeBuilder resource_user_fn_builder("resource_user_function",
                                         "resource_user_function", op_reg);
    resource_user_fn_builder.Input(stack_op);
    opts.FinalizeBuilder(&resource_user_fn_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected[":stack_op:StackV2"] = absl::flat_hash_set<std::string>(
      {":resource_user_function:resource_user_function",
       "resource_user_function:stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

TEST(ResourceOpAnalyzerTest, ResourceSourceInFunction) {
  auto library = absl::make_unique<FunctionDefLibrary>();
  /*
   *  resource_source_function:
   *
   *  _Arg -> StackV2 -> _Retval
   */
  *library->add_function() = FunctionDefHelper::Define(
      /*function_name=*/"resource_source_function",
      /*arg_def=*/{"in: int32"},
      /*ret_def=*/{"out: resource"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"out"}, "StackV2", {"in"}, {{"elem_type", DataType::DT_FLOAT}}}});

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), *library);
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
  auto opts = builder.opts();
  auto op_reg = opts.op_registry();

  {
    /*
     * stack_size -> resource_source_function -> stack_close
     */
    NodeBuilder stack_size_placeholder_builder("stack_size", "Placeholder",
                                               op_reg);
    stack_size_placeholder_builder.Attr("dtype", DT_INT32);
    Node* stack_size_placeholder =
        opts.FinalizeBuilder(&stack_size_placeholder_builder);

    NodeBuilder resource_source_fn_builder("resource_source_function",
                                           "resource_source_function", op_reg);
    resource_source_fn_builder.Input(stack_size_placeholder);
    Node* resource_source_function =
        opts.FinalizeBuilder(&resource_source_fn_builder);

    NodeBuilder stack_close_builder("stack_close", "StackCloseV2", op_reg);
    stack_close_builder.Input(resource_source_function);
    opts.FinalizeBuilder(&stack_close_builder);
  }

  GraphDef graphdef;
  TF_EXPECT_OK(builder.ToGraphDef(&graphdef));

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> expected;
  expected["resource_source_function:out:StackV2"] =
      absl::flat_hash_set<std::string>({":stack_close:StackCloseV2"});
  AnalyzeAndVerify(graphdef, &flib_def, expected);
}

}  // namespace tensorflow
