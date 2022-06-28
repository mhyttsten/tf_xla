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
class MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc() {
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

#include "tensorflow/compiler/jit/compilability_check_util.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/graphcycles/graphcycles.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

constexpr char kXlaOutsideCompilationAttr[] = "_xla_outside_compilation";

bool HasResourceInput(const Node& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_0(mht_0_v, 239, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "HasResourceInput");

  return absl::c_count(node.input_types(), DT_RESOURCE) != 0;
}

void LogNotCompilable(const Node& node, absl::string_view reason = "") {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "LogNotCompilable");

  VLOG(3) << "Found uncompilable node " << node.name() << " (op "
          << node.type_string() << ")" << (reason.empty() ? "" : ": ")
          << reason;
}

bool IsInOutsideCompilationCluster(const Node& n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "IsInOutsideCompilationCluster");

  return n.attrs().Find(kXlaOutsideCompilationAttr) != nullptr;
}

Status MakeCallNodeFromAttribute(const Node& node, const std::string& attr_name,
                                 NodeDef* node_def) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "MakeCallNodeFromAttribute");

  const NameAttrList* name_attr;
  TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), attr_name, &name_attr));
  node_def->set_op(name_attr->name());
  *(node_def->mutable_attr()) = name_attr->attr();
  return Status::OK();
}

StatusOr<std::vector<NodeDef>> MakeCallNodesFromAttribute(
    const Node& node, absl::string_view attr_name,
    absl::string_view call_name) {
  std::vector<NameAttrList> attr_lists;
  TF_RETURN_IF_ERROR(GetNodeAttr(node.attrs(), attr_name, &attr_lists));

  std::vector<NodeDef> out;
  out.reserve(attr_lists.size());
  for (int i = 0; i < attr_lists.size(); i++) {
    out.emplace_back();
    NodeDef& inserted = out.back();
    inserted.set_name(absl::StrCat(call_name, "_", i));
    inserted.set_op(attr_lists[i].name());
    *inserted.mutable_attr() = attr_lists[i].attr();
  }
  return out;
}

// Utility which searches for values in a sorted list by scanning over it once.
// No matter how many times ScanForValue is called, the list is scanned at most
// once. However, if a call to ScanForValue skips over a value, that value is
// not revisited in future calls to ScanForValue, so callers must take
// care to order their calls.
//
// Useful for merging multiple sorted lists in O(n) time.
class SinglePassSearch {
 public:
  // Creates a SinglePassSearch object that can be used to search in `values`.
  // Does not take ownership of `values`. `values` must outlive this.
  // `values` must be sorted.
  explicit SinglePassSearch(absl::Span<int const> values)
      : current_index_(0), values_(values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_4(mht_4_v, 306, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "SinglePassSearch");
}

  // Scans forward in the vector looking for "value", updating the internal
  // position in to the vector.
  // Returns true iff the vector contains the given value at or after current
  // position.
  // Not thread-safe.
  bool ScanForValue(int value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_5(mht_5_v, 316, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "ScanForValue");

    while (current_index_ < values_.size() &&
           values_[current_index_] <= value) {
      if (values_[current_index_] == value) {
        current_index_++;
        return true;
      }
      current_index_++;
    }
    return false;
  }

 private:
  int current_index_;
  const absl::Span<int const> values_;
};

}  // anonymous namespace

RecursiveCompilabilityChecker::UncompilableNodesMap
RecursiveCompilabilityChecker::FindUncompilableNodes(
    const Node& node, FunctionLibraryRuntime* lib_runtime,
    const std::vector<RecursiveCompilabilityChecker::StackFrame>*
        node_stack_trace) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_6(mht_6_v, 342, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::FindUncompilableNodes");

  std::vector<StackFrameView> stack_trace;
  // If `node_stack_trace` is provided, that means `node` is inside
  // a function body, and therefore, arg nodes and retval nodes are
  // not considered uncompilable.
  if (node_stack_trace != nullptr) {
    for (const auto& frame : *node_stack_trace) {
      stack_trace.emplace_back(
          StackFrameView{frame.name, frame.function_name, frame.stack_trace});
    }
  }
  stack_trace.emplace_back(
      StackFrameView{node.name(), "", node.GetStackTrace()});

  RecursiveCompilabilityChecker::UncompilableNodesMap uncompilable_nodes;
  IsCompilableNode(node, lib_runtime, &stack_trace,
                   /*encapsulating_function=*/nullptr, &uncompilable_nodes);
  return uncompilable_nodes;
}

bool RecursiveCompilabilityChecker::HasXLAKernel(
    const Node& node, string* uncompilable_reason) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_7(mht_7_v, 366, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::HasXLAKernel");

  // There is a SymbolicGradient kernel on the XLA_JIT device, but the gradient
  // is really a kind of function call and will be handled by
  // IsCompilableCall().
  if (node.type_string() == "SymbolicGradient") {
    *uncompilable_reason =
        "SymbolicGradient should be handled by IsCompilableCall().";
    return false;
  }

  if (node.type_string() == "Const") {
    const AttrValue* attr = node.attrs().Find("dtype");
    if (!op_filter_.allow_string_consts && attr != nullptr &&
        attr->type() == DT_STRING) {
      *uncompilable_reason =
          "Const op with type DT_STRING is not supported by XLA.";
      return false;
    }
  }

  // XLA does not offer guaranteed aliasing between the input and output of the
  // XLA cluster so it can't implement the forward-tensor-ref semantic.  Leave
  // such nodes out of XLA clusters.
  if (HasForwardedRefInput(node)) {
    VLOG(2) << "Rejecting " << node.name() << ": Identity with unsafe cast.";
    *uncompilable_reason = "Identity with unsafe cast.";
    return false;
  }

  Status s = FindKernelDef(jit_device_type_, node.def(), nullptr, nullptr);
  if (!s.ok()) {
    *uncompilable_reason = s.error_message();
    return false;
  }
  return true;
}

// Tests whether 'if_node' is compilable. Every operator in the then_branch and
// else_branch functions must be compilable for 'if_node' to be compilable.
bool RecursiveCompilabilityChecker::IsCompilableIf(
    const Node& if_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_8(mht_8_v, 413, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::IsCompilableIf");

  bool is_compilable = true;
  is_compilable &= ExtractNodeDefAndCheckCompilability(
      if_node, "then_branch", "if_then", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);
  if (!uncompilable_nodes && !is_compilable) return is_compilable;

  is_compilable &= ExtractNodeDefAndCheckCompilability(
      if_node, "else_branch", "if_else", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  return is_compilable;
}

bool RecursiveCompilabilityChecker::IsCompilableCase(
    const Node& case_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_9(mht_9_v, 435, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::IsCompilableCase");

  StatusOr<std::vector<NodeDef>> calls =
      MakeCallNodesFromAttribute(case_node, "branches", "branch");
  if (!calls.ok()) {
    VLOG(2) << "Rejecting node " << case_node.name() << ": "
            << "missing attribute 'branches'";
    return false;
  }

  bool is_compilable = true;

  for (const NodeDef& call : *calls) {
    is_compilable &=
        IsCompilableCall(call, lib_runtime, stack_trace, encapsulating_function,
                         uncompilable_nodes);
  }
  return is_compilable;
}

// Tests whether 'while_node' is a completely compilable loop.
// Every operator in the condition and body functions must be compilable for a
// while loop to be compilable.
bool RecursiveCompilabilityChecker::IsCompilableWhile(
    const Node& while_node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_10(mht_10_v, 465, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::IsCompilableWhile");

  bool is_compilable = true;
  is_compilable &= ExtractNodeDefAndCheckCompilability(
      while_node, "cond", "while_cond", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  if (!uncompilable_nodes && !is_compilable) return is_compilable;

  is_compilable &= ExtractNodeDefAndCheckCompilability(
      while_node, "body", "while_body", encapsulating_function, lib_runtime,
      stack_trace, uncompilable_nodes);

  return is_compilable;
}

bool RecursiveCompilabilityChecker::ExtractNodeDefAndCheckCompilability(
    const Node& node, const std::string& attr_name,
    const std::string& call_name, NameAttrList* encapsulating_function,
    FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_11_v.push_back("call_name: \"" + call_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_11(mht_11_v, 491, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::ExtractNodeDefAndCheckCompilability");

  NodeDef call;
  call.set_name(call_name);
  if (!MakeCallNodeFromAttribute(node, attr_name, &call).ok()) {
    const auto uncompilable_reason = absl::StrCat(
        "missing '", attr_name, "' attribute from node", node.name());
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting node " << node.name() << ": " << uncompilable_reason
            << ".";
    return false;
  }
  if (!IsCompilableCall(call, lib_runtime, stack_trace, encapsulating_function,
                        uncompilable_nodes)) {
    VLOG(2) << "Rejecting node " << node.name()
            << ": can't compile : " << call.op();
    return false;
  }
  return true;
}

// Tests whether 'call_def' is a call to a completely compilable function.
// Every operator in the function must be compilable for a function to be
// compilable.
bool RecursiveCompilabilityChecker::IsCompilableCall(
    const NodeDef& call_def, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_12(mht_12_v, 523, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::IsCompilableCall");

  if (stack_trace->size() > kMaxRecursionDepth) {
    std::string uncompilable_reason = "function depth limit exceeded";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting " << call_def.op() << ": " << uncompilable_reason
            << ".";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status s;
  NameAttrList function;
  s = NameAndAttrsFromFunctionCall(call_def, &function);
  if (s.ok()) {
    s = lib_runtime->Instantiate(function.name(), AttrSlice(&function.attr()),
                                 &handle);
  }
  if (!s.ok()) {
    std::string uncompilable_reason =
        absl::StrCat("could not instantiate call: '", function.name(), "'");
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    VLOG(2) << "Rejecting " << call_def.DebugString() << ": "
            << uncompilable_reason << " : " << s;
    return false;
  }

  auto release_handle_on_return = gtl::MakeCleanup(
      [&] { TF_CHECK_OK(lib_runtime->ReleaseHandle(handle)); });
  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  bool is_compilable = true;
  for (const Node* node : fbody->graph->op_nodes()) {
    stack_trace->emplace_back(
        StackFrameView{node->name(), function.name(), node->GetStackTrace()});
    is_compilable &= IsCompilableNode(*node, lib_runtime, stack_trace,
                                      &function, uncompilable_nodes);
    stack_trace->pop_back();
    if (!uncompilable_nodes && !is_compilable) return is_compilable;
  }

  return is_compilable;
}

bool RecursiveCompilabilityChecker::OpIsInaccurate(const Node& node) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_13(mht_13_v, 570, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::OpIsInaccurate");

  // b/127344411: SelfAdjointEigV2 and Svd precision issues.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd";
}

bool RecursiveCompilabilityChecker::OpIsSlow(const Node& node) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_14(mht_14_v, 579, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::OpIsSlow");

  // b/128001705: SelfAdjointEigV2 and Svd performance issues.
  // b/135640736: MatrixInverse performance issues.
  // b/111271662: MatrixSolve performance issues.
  // https://github.com/tensorflow/tensorflow/pull/31012:
  //    ResizeNearestNeighbor, ResizeBilinear, and ResizeBilinearGrad sometimes
  //    create convolutions too large for CuDNN to handle.
  return node.type_string() == "SelfAdjointEigV2" ||
         node.type_string() == "Svd" || node.type_string() == "Qr" ||
         node.type_string() == "MatrixInverse" ||
         node.type_string() == "MatrixSolve" ||
         node.type_string() == "ResizeBilinearGrad";
}

bool RecursiveCompilabilityChecker::IsCompilableNode(
    const Node& node, FunctionLibraryRuntime* lib_runtime,
    std::vector<StackFrameView>* stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes)
    const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_15(mht_15_v, 601, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::IsCompilableNode");

  auto stack_depth = stack_trace->size();

  if (op_filter_.allow_outside_compiled && IsInOutsideCompilationCluster(node))
    return true;

  if (node.IsSource() || node.IsSink()) {
    absl::string_view uncompilable_reason = "source or sink node";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  // _Arg nodes in a top-level function represent feeds and _Retval nodes in a
  // top-level function represent fetches.
  if (stack_depth == 1 &&
      (node.type_string() == "_Arg" || node.type_string() == "_Retval")) {
    absl::string_view uncompilable_reason = "top level _Arg or _Retval";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.attrs().Find("_scoped_allocator") ||
      node.attrs().Find("_forward_from")) {
    // TODO(b/128858118): XLA does not support _scoped_allocator and
    // _forward_from.
    absl::string_view uncompilable_reason =
        "_scoped_allocator or _forward_from attribute";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  string uncompilable_reason;
  if (IsFunctionCall(*lib_runtime->GetFunctionLibraryDefinition(), node)) {
    if (!IsCompilableCall(node.def(), lib_runtime, stack_trace,
                          encapsulating_function, uncompilable_nodes)) {
      LogNotCompilable(node, "unsupported function");
      return false;
    }
  } else if (!HasXLAKernel(node, &uncompilable_reason)) {
    MaybeMarkUncompilableNode(
        absl::StrCat("unsupported op: ", uncompilable_reason), *stack_trace,
        encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (node.IsWhileNode() &&
      !IsCompilableWhile(node, lib_runtime, stack_trace, encapsulating_function,
                         uncompilable_nodes)) {
    LogNotCompilable(node, "unsupported while");
    return false;
  }

  if (node.IsIfNode() &&
      !IsCompilableIf(node, lib_runtime, stack_trace, encapsulating_function,
                      uncompilable_nodes)) {
    LogNotCompilable(node, "unsupported if");
    return false;
  }

  if (op_filter_.require_always_compilable && node.IsCaseNode() &&
      !IsCompilableCase(node, lib_runtime, stack_trace, encapsulating_function,
                        uncompilable_nodes)) {
    LogNotCompilable(node, "unsupported case");
    return false;
  }

  if (!op_filter_.allow_stateful_rng_ops &&
      IsStatefulRandomOp(node.type_string())) {
    absl::string_view uncompilable_reason = "stateful random op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_control_trigger && node.IsControlTrigger()) {
    absl::string_view uncompilable_reason = "not allowed control trigger";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_eliding_assert_and_checknumerics_ops &&
      IsAssertOrCheckNumerics(node.type_string())) {
    absl::string_view uncompilable_reason = "Assert or CheckNumerics";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_collective_reduce_v2 &&
      node.type_string() == "CollectiveReduceV2") {
    absl::string_view uncompilable_reason = "Collective op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_unique_op && node.type_string() == "Unique") {
    absl::string_view uncompilable_reason = "Unique op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_ops_producing_or_consuming_variant &&
      OpProducesOrConsumesVariant(node)) {
    absl::string_view uncompilable_reason = "DT_VARIANT producer/consumer";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_stack_ops && IsStackOp(node)) {
    absl::string_view uncompilable_reason = "Stack op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_tensor_array_ops && IsTensorArrayOp(node)) {
    absl::string_view uncompilable_reason = "TensorArray op";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_resource_ops_in_called_functions && stack_depth > 1 &&
      HasResourceInput(node)) {
    absl::string_view uncompilable_reason =
        "resource variable op in called function";
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_inaccurate_ops && OpIsInaccurate(node)) {
    absl::string_view uncompilable_reason =
        "operation with numerical accuracy issues";
    BroadcastOptimizationRemark(XlaOptimizationRemark::INACCURATE_OPERATION,
                                node.DebugString())
        .IgnoreError();
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  if (!op_filter_.allow_slow_ops && OpIsSlow(node)) {
    absl::string_view uncompilable_reason = "slow operation";
    BroadcastOptimizationRemark(XlaOptimizationRemark::SLOW_OPERATION,
                                node.DebugString())
        .IgnoreError();
    MaybeMarkUncompilableNode(uncompilable_reason, *stack_trace,
                              encapsulating_function, uncompilable_nodes);
    LogNotCompilable(node, uncompilable_reason);
    return false;
  }

  return true;
}

RecursiveCompilabilityChecker::OperationFilter CreateOperationFilter(
    const XlaOpRegistry::DeviceRegistration& registration) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_16(mht_16_v, 782, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "CreateOperationFilter");

  RecursiveCompilabilityChecker::OperationFilter op_filter;
  op_filter.allow_resource_ops_in_called_functions =
      registration.cluster_resource_variable_ops_unsafely;
  op_filter.allow_stack_ops = registration.cluster_stack_ops;
  op_filter.allow_tensor_array_ops = registration.cluster_tensor_array_ops;
  op_filter.allow_stateful_rng_ops = registration.cluster_stateful_rng_ops;
  op_filter.allow_control_trigger = registration.cluster_control_trigger;
  op_filter.allow_eliding_assert_and_checknumerics_ops =
      registration.elide_assert_and_checknumerics;
  op_filter.allow_ops_producing_or_consuming_variant =
      registration.cluster_variant_ops;
  op_filter.allow_slow_ops = registration.cluster_slow_ops;
  op_filter.allow_inaccurate_ops = registration.cluster_inaccurate_ops;
  return op_filter;
}

/*static*/ void RecursiveCompilabilityChecker::MaybeMarkUncompilableNode(
    const absl::string_view reason,
    const std::vector<StackFrameView>& stack_trace,
    NameAttrList* encapsulating_function,
    RecursiveCompilabilityChecker::UncompilableNodesMap* uncompilable_nodes) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("reason: \"" + std::string(reason.data(), reason.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_17(mht_17_v, 807, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "RecursiveCompilabilityChecker::MaybeMarkUncompilableNode");

  if (!uncompilable_nodes) return;

  UncompilableNodeInfo node_info;
  node_info.uncompilable_reason = std::string(reason);
  absl::c_transform(stack_trace, std::back_inserter(node_info.stack_trace),
                    [](const StackFrameView& stack_element) {
                      return StackFrame{
                          std::string(stack_element.name),
                          std::string(stack_element.function_name),
                          stack_element.stack_trace};
                    });

  node_info.name = std::string(stack_trace.back().name);
  auto function =
      encapsulating_function ? *encapsulating_function : NameAttrList();
  auto function_identifier = function.ShortDebugString();

  auto it = uncompilable_nodes->find(function_identifier);
  if (it == uncompilable_nodes->end()) {
    std::vector<RecursiveCompilabilityChecker::UncompilableNodeInfo>
        uncompilable_node_info{std::move(node_info)};
    uncompilable_nodes->emplace(
        std::move(function_identifier),
        std::make_pair(function, std::move(uncompilable_node_info)));
  } else {
    it->second.second.emplace_back(std::move(node_info));
  }
}

// Returns `true` iff node has a given `attr` set to `true`. Returns `false`
// both for the missing attr, and the attr set to `false`.
static bool HasBoolAttr(const NodeDef& node, const char* attr) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("attr: \"" + (attr == nullptr ? std::string("nullptr") : std::string((char*)attr)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_18(mht_18_v, 843, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "HasBoolAttr");

  const auto& it = node.attr().find(attr);
  return it != node.attr().end() && it->second.b();
}

bool CanCreateXlaKernel(const NodeDef& node_def) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_19(mht_19_v, 851, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "CanCreateXlaKernel");

  return HasBoolAttr(node_def, kXlaMustCompileAttr);
}

Status GetBodyAndConstantsAndResources(FunctionLibraryRuntime* flr,
                                       const NameAttrList& function,
                                       const FunctionBody** fbody,
                                       std::vector<int>* constant_arg_indices,
                                       std::vector<int>* resource_arg_indices) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_20(mht_20_v, 862, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "GetBodyAndConstantsAndResources");

  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(function.name(), AttrSlice(&function.attr()), &handle));
  *fbody = flr->GetFunctionBody(handle);
  CHECK(*fbody);  // Can't be nullptr since we just instantiated it.
  const DataTypeVector& arg_types = (*fbody)->arg_types;
  std::vector<bool> const_args(arg_types.size());
  // If we can't analyze the const args. Bail out.
  TF_RETURN_IF_ERROR(
      BackwardsConstAnalysis(*((*fbody)->graph), &const_args,
                             /*compile_time_const_nodes=*/nullptr, flr));

  for (size_t i = 0; i < const_args.size(); ++i) {
    if (const_args[i]) {
      constant_arg_indices->push_back(i);
    }
  }

  // There can be hundreds of resource variables. Reserve the space for them.
  // We don't reserve for constants above as they are usually few.
  resource_arg_indices->reserve(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i] == DT_RESOURCE) {
      resource_arg_indices->push_back(i);
    }
  }

  return Status::OK();
}

tensorflow::MemoryTypeVector GetInputMemoryTypes(
    const tensorflow::FunctionBody* fbody,
    absl::Span<int const> constant_arg_indices,
    absl::Span<int const> resource_arg_indices) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_21(mht_21_v, 899, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "GetInputMemoryTypes");

  // Set input and output memory types.
  tensorflow::MemoryTypeVector input_memory_types(fbody->arg_types.size(),
                                                  tensorflow::DEVICE_MEMORY);
  // These indices are used only for optimization purposes. They allow us
  // to loop over constant_arg_indices and resource_arg_indices only once
  // while iterating over all the function arguments checking if it is a
  // resource or a constant.
  // The reason we optimized this code is because functions can have a lot of
  // captured arguments. For example, the backward pass of ResNet50 takes in all
  // 214 variables and a similar number of activations.
  SinglePassSearch constants_search(constant_arg_indices);
  SinglePassSearch resources_search(resource_arg_indices);
  for (size_t i = 0; i < fbody->arg_types.size(); ++i) {
    if (resources_search.ScanForValue(i) || constants_search.ScanForValue(i)) {
      // Compile-time constants and resource handles are expected to be in
      // host memory.
      input_memory_types[i] = tensorflow::HOST_MEMORY;
    }
  }
  return input_memory_types;
}

tensorflow::MemoryTypeVector GetOutputMemoryTypes(
    const tensorflow::FunctionBody* fbody) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_22(mht_22_v, 926, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "GetOutputMemoryTypes");

  tensorflow::MemoryTypeVector output_memory_types(fbody->ret_types.size(),
                                                   tensorflow::DEVICE_MEMORY);
  for (size_t i = 0; i < fbody->ret_types.size(); ++i) {
    if (fbody->ret_types[i] == tensorflow::DT_RESOURCE) {
      output_memory_types[i] = tensorflow::HOST_MEMORY;
    }
  }
  return output_memory_types;
}

static auto const ops_triggering_xla_compilation =
    new absl::flat_hash_set<std::string>{"XlaBroadcastHelper",
                                         "XlaConv",
                                         "XlaConvV2",
                                         "XlaDequantize",
                                         "XlaDot",
                                         "XlaDotV2",
                                         "XlaDynamicSlice",
                                         "XlaDynamicUpdateSlice",
                                         "XlaEinsum",
                                         "XlaGather",
                                         "XlaIf",
                                         "XlaKeyValueSort",
                                         "XlaPad",
                                         "XlaRecv",
                                         "XlaReduce",
                                         "XlaReduceWindow",
                                         "XlaReplicaId",
                                         "XlaRngBitGenerator",
                                         "XlaScatter",
                                         "XlaSelectAndScatter",
                                         "XlaSelfAdjointEig",
                                         "XlaSend",
                                         "XlaSharding",
                                         "XlaSort",
                                         "XlaSpmdFullToShardShape",
                                         "XlaSpmdShardToFullShape",
                                         "XlaSvd",
                                         "XlaVariadicReduceV2",
                                         "XlaVariadicSort",
                                         "XlaWhile"};

static bool NodeCanTriggerXlaCompilation(const NodeDef& node) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_23(mht_23_v, 972, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "NodeCanTriggerXlaCompilation");

  return node.attr().find(kXlaClusterIdAttr) != node.attr().end() ||
         HasBoolAttr(node, kXlaMustCompileAttr) ||
         HasBoolAttr(node, kXlaCompileAttr) ||
         HasBoolAttr(node, kXlaScopeAttr) ||
         HasBoolAttr(node, kXlaInternalScopeAttr) ||
         ops_triggering_xla_compilation->count(node.op());
}

bool CanTriggerXlaCompilation(const GraphDef& graph) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSjitPScompilability_check_utilDTcc mht_24(mht_24_v, 984, "", "./tensorflow/compiler/jit/compilability_check_util.cc", "CanTriggerXlaCompilation");

  for (const FunctionDef& function : graph.library().function()) {
    for (const NodeDef& node : function.node_def()) {
      if (NodeCanTriggerXlaCompilation(node)) {
        return true;
      }
    }
  }

  for (const NodeDef& node : graph.node()) {
    if (NodeCanTriggerXlaCompilation(node)) {
      return true;
    }
  }

  return false;
}

}  // namespace tensorflow
