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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc() {
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

#include "tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass.h"

#include <map>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using FDH = ::tensorflow::FunctionDefHelper;

// Returns void so that we can call TF_ASSERT_OK inside it.
static void RunPass(const GraphDef& original, GraphDef* rewritten,
                    FunctionLibraryDefinition* flib_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass_test.cc", "RunPass");

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.add_default_attributes = false;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, original, graph.get()));
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.flib_def = flib_def;
  IsolatePlacerInspectionRequiredOpsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  graph->ToGraphDef(rewritten);
}
static void RunPass(const GraphDef& original, GraphDef* rewritten) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass_test.cc", "RunPass");

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), original.library());
  RunPass(original, rewritten, &flib_def);
}

void RunPassAndCompare(const GraphDef& original, const GraphDef& expected) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass_test.cc", "RunPassAndCompare");

  GraphDef rewritten;
  RunPass(original, &rewritten);
  TF_EXPECT_GRAPH_EQ(expected, rewritten);
}

void RunPassAndCompare(const GraphDef& original,
                       const std::vector<GraphDef>& expected_alternatives) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSisolate_placer_inspection_required_ops_pass_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass_test.cc", "RunPassAndCompare");

  GraphDef rewritten;
  RunPass(original, &rewritten);

  std::vector<string> errors;
  errors.push_back(absl::StrCat("Graphs did not match.\n  Rewritten graph:\n",
                                SummarizeGraphDef(rewritten)));
  for (const GraphDef& alternative : expected_alternatives) {
    string diff;
    bool graphs_equal = EqualGraphDef(rewritten, alternative, &diff);
    if (graphs_equal) {
      return;
    }
    errors.push_back(absl::StrCat("  Expected alternative:\n",
                                  SummarizeGraphDef(alternative)));
  }
  EXPECT_TRUE(false) << absl::StrJoin(errors, "\n");
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, Basic) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp: ResourceIdentity)
   *                   |
   *                   v
   *                y (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::ResourceIdentity();
  GraphDef original = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE}},
                {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("x_f", "Identity", {"x"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE}},
                {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
          NDef("f_y", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("y", "_Retval", {"f_y:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, FunctionDefinitionNotInGraph) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp: ResourceIdentity)
   *                   |
   *                   v
   *                y (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::ResourceIdentity();
  GraphDef original = GDef({
      NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
      NDef("f", "PartitionedCall", {"x"},
           {{"Tin", DataTypeSlice{DT_RESOURCE}},
            {"Tout", DataTypeSlice{DT_RESOURCE}},
            {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
      NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
  });

  GraphDef expected = GDef({
      NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
      NDef("x_f", "Identity", {"x"}, {{"T", DT_RESOURCE}}),
      NDef("f", "PartitionedCall", {"x_f"},
           {{"Tin", DataTypeSlice{DT_RESOURCE}},
            {"Tout", DataTypeSlice{DT_RESOURCE}},
            {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
      NDef("f_y", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
      NDef("y", "_Retval", {"f_y:0"}, {{"T", DT_RESOURCE}}),
  });

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
  TF_ASSERT_OK(flib_def.AddFunctionDef(func));
  GraphDef rewritten;
  RunPass(original, &rewritten, &flib_def);
  TF_EXPECT_GRAPH_EQ(expected, rewritten);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, MultipleInputsAndOutputs) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    r2 (_Retval, DT_RESOURCE)
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, UnusedOutput) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    <unused>
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          // Identity is created for output that was not used.
          NDef("f_0", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, OutputsConsumedBySameOp) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |     |
   *                   |     |
   *                   v     v
   *                add (Add, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("add", "Add", {"f:0", "f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  // There are two possible namings for outputs depending on map
  // iteration order.
  GraphDef expected1 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_add", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("f_add_0", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("add", "Add", {"f_add", "f_add_0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected2 = GDef(
      {
          // Same as above
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          // Different from above
          NDef("f_add", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("f_add_0", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("add", "Add", {"f_add_0", "f_add"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, {expected1, expected2});
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, IdenticalInputs) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |      |
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    r2 (_Retval, DT_RESOURCE)
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "a"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  // There are two possible namings for outputs depending on map
  // iteration order.
  GraphDef expected1 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("a_f_0", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "a_f_0"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected2 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("a_f_0", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall",
               {"a_f_0", "a_f"},  // the only different line from above
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, {expected1, expected2});
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, DirectCallsAreNotIsolated) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (direct function call to ResourceIdentity)
   *                   |
   *                   v
   *                y (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::ResourceIdentity();
  GraphDef original = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "ResourceIdentity", {"x"}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, original);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest,
     FunctionsNotReturningResourcesAreNotIsolated) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp, ReadResourceVariable)
   *                   |
   *                   v
   *                y (_Retval, DT_FLOAT)
   */
  FunctionDef func = test::function::ReadResourceVariable();
  GraphDef original = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_FLOAT}},
                {"f", FDH::FunctionRef("ReadResourceVariable", {})}}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_FLOAT}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, original);
}

}  // namespace tensorflow
