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
class MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc() {
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

#include "tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.h"
#include <iterator>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

// StatusOrOptional<T> instances hold
//
//  - A non-OK Status to indicate an error that needs to be propagated out of
//    this pass (e.g. the Graph is malformed).
//
//  - A nullopt to indicate the function that created the instance failed to do
//    what it set out to do but this is not actually an error
//    (e.g. TryToGetTensorFromConstOp was passed a non-Const node).
//
//  - A T to indicate a successful operation.
template <class T>
using StatusOrOptional = StatusOr<absl::optional<T>>;

StatusOrOptional<Tensor> TryToGetTensorFromConstOp(Node* n) {
  if (n->type_string() != "Const") {
    return {absl::nullopt};
  }

  const TensorProto* proto = nullptr;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "value", &proto));
  Tensor tensor(proto->dtype());
  TF_RET_CHECK(tensor.FromProto(*proto));
  return {tensor};
}

struct SliceInputs {
  Output slice_op;
  Output input;
  Output begin;
  Output size;

  // The size of the TF slice operation as a std::vector.  We can always compute
  // this because we only manipulate slices with a Const size.
  std::vector<int64_t> size_as_vector;
};

std::vector<int64_t> IntTensorAsVector(const Tensor& t) {
  DCHECK(t.dtype() == DT_INT32 || t.dtype() == DT_INT64);
  std::vector<int64_t> result;
  result.reserve(t.NumElements());
  for (int i = 0; i < t.NumElements(); i++) {
    int64_t element = t.dtype() == DT_INT32
                          ? static_cast<int64_t>(t.flat<int32>()(i))
                          : t.flat<int64_t>()(i);
    result.push_back(element);
  }
  return result;
}

// Packages up the inputs to a Slice operation into an instance of
// `SliceInputs`.
StatusOrOptional<SliceInputs> GetSliceInputs(Node* slice) {
  const int kSliceInputIndex = 0;
  const int kSliceBeginIndex = 1;
  const int kSliceSizeIndex = 2;

  const Edge* slice_input_edge;
  TF_RETURN_IF_ERROR(slice->input_edge(kSliceInputIndex, &slice_input_edge));
  const Edge* slice_size_edge;
  TF_RETURN_IF_ERROR(slice->input_edge(kSliceSizeIndex, &slice_size_edge));
  const Edge* slice_begin_edge;
  TF_RETURN_IF_ERROR(slice->input_edge(kSliceBeginIndex, &slice_begin_edge));

  SliceInputs slice_inputs;
  slice_inputs.input =
      Output(slice_input_edge->src(), slice_input_edge->src_output());
  slice_inputs.begin =
      Output(slice_begin_edge->src(), slice_begin_edge->src_output());
  slice_inputs.size =
      Output(slice_size_edge->src(), slice_size_edge->src_output());

  TF_ASSIGN_OR_RETURN(absl::optional<Tensor> tf_slice_size,
                      TryToGetTensorFromConstOp(slice_inputs.size.node()));
  if (!tf_slice_size.has_value()) {
    return {absl::nullopt};
  }

  if (tf_slice_size->dims() != 1) {
    return {absl::nullopt};
  }

  slice_inputs.size_as_vector = IntTensorAsVector(*tf_slice_size);
  return {slice_inputs};
}

// Casts `x` to a DT_INT64 if it isn't one already.
Output MakeInt64(const Scope& host_scope, absl::string_view name,
                 const Output& x) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_0(mht_0_v, 297, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "MakeInt64");

  return x.type() == DT_INT64
             ? x
             : ops::Cast(host_scope.WithOpName(name, "_s64"), x, DT_INT64);
}

// Returns `slice_inputs` with the index and size inputs cast to DT_INT64.
SliceInputs MakeSliceIndexAndSizeInt64(const Scope& host_scope,
                                       const SliceInputs& slice_inputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_1(mht_1_v, 308, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "MakeSliceIndexAndSizeInt64");

  SliceInputs result;
  result.input = slice_inputs.input;
  result.begin = MakeInt64(host_scope, "begin", slice_inputs.begin);
  result.size = MakeInt64(host_scope, "size", slice_inputs.size);
  result.size_as_vector = slice_inputs.size_as_vector;
  return result;
}

// This class caches emitted constants to avoid creating multiple nodes for the
// same constant value.  This helps make the generated GraphDef more readable.
class ConstantCache {
 public:
  explicit ConstantCache(const Scope& s,
                         const std::vector<const Edge*>& control_deps)
      : scope_(s), control_deps_(control_deps) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_2(mht_2_v, 326, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "ConstantCache");
}

  Output Get1DHostConstant(int64_t constant) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_3(mht_3_v, 331, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "Get1DHostConstant");

    auto it = cache_.find(constant);
    if (it == cache_.end()) {
      Output new_const =
          ops::Const(scope_.WithOpName("const_", constant), {constant});
      it = cache_.insert({constant, new_const}).first;
      for (const Edge* e : control_deps_) {
        scope_.graph()->AddControlEdge(e->src(), new_const.node());
      }
    }
    return it->second;
  }

 private:
  Scope scope_;
  std::unordered_map<int, Output> cache_;
  std::vector<const Edge*> control_deps_;
};

// Returns a node computing the size of the Slice op with inputs `slice_inputs`.
Status ComputeSliceSize(const Scope& host_scope,
                        const SliceInputs& slice_inputs,
                        std::vector<const Edge*> control_deps, Output* size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_4(mht_4_v, 356, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "ComputeSliceSize");

  // If slice_size[i] >= 0 then slice_size[i] = slice_size[i].
  //
  // If slice_size[i] == -1 then slice_size[i] = input_size[i] -
  // begin[i].
  //
  // If slice_size[i] < -1 then executing the slice will throw an error, and we
  // don't do anything here.  We've already filtered these cases out in
  // IsRewritableSlice.

  if (absl::c_all_of(slice_inputs.size_as_vector,
                     [](int64_t i) { return i >= 0; })) {
    *size = slice_inputs.size;
    return Status::OK();
  }

  Output input_shape =
      ops::Shape(host_scope.WithOpName("input_shape"), slice_inputs.input,
                 ops::Shape::OutType(DT_INT64));

  ConstantCache constant_pool(host_scope, control_deps);

  std::vector<Output> slice_size;
  for (int i = 0, end = slice_inputs.size_as_vector.size(); i < end; i++) {
    if (slice_inputs.size_as_vector[i] >= 0) {
      slice_size.push_back(
          constant_pool.Get1DHostConstant(slice_inputs.size_as_vector[i]));
      continue;
    }

    DCHECK_EQ(slice_inputs.size_as_vector[i], -1);

    Output begin_i = ops::Slice(
        host_scope.WithOpName("begin_", i), slice_inputs.begin,
        constant_pool.Get1DHostConstant(i), constant_pool.Get1DHostConstant(1));

    Output input_shape_i = ops::Slice(
        host_scope.WithOpName("input_shape_", i), input_shape,
        constant_pool.Get1DHostConstant(i), constant_pool.Get1DHostConstant(1));

    slice_size.push_back(ops::Sub(host_scope.WithOpName("slice_size_", i),
                                  input_shape_i, begin_i));
    DCHECK_EQ(slice_size.back().type(), DT_INT64);
  }

  // Trivial ConcatV2 nodes (with exactly one input) are disallowed.
  if (slice_size.size() == 1) {
    *size = slice_size[0];
  } else {
    auto concat_axis = ops::Const(host_scope.WithOpName("concat_axis"), 0);
    for (const Edge* e : control_deps) {
      host_scope.graph()->AddControlEdge(e->src(), concat_axis.node());
    }
    *size = ops::Concat(host_scope.WithOpName("slice_size"), slice_size,
                        concat_axis);
  }
  return Status::OK();
}

// Terminology: "static sized" slice is a slice with the
// _XlaCompileTimeConstantInputs attribute set to {2}.  The output shape of
// these slices can be solely determined by their "size" input.
Status ConvertTensorFlowSliceToStaticShapedSlice(
    Graph* g, Node* slice, const SliceInputs& slice_inputs,
    absl::string_view cluster_name, Node** result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("cluster_name: \"" + std::string(cluster_name.data(), cluster_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_5(mht_5_v, 424, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "ConvertTensorFlowSliceToStaticShapedSlice");

  string host_name;
  TF_RETURN_IF_ERROR(DeviceNameUtils::DeviceNameToCpuDeviceName(
      slice->assigned_device_name(), &host_name));

  Status status;
  Scope main_scope =
      NewInternalScope(g, &status, /*refiner=*/nullptr)
          .WithXlaCluster(string(cluster_name))
          .NewSubScope(absl::StrCat(slice->name(), "/static_shaped_slice"));
  Scope host_scope = main_scope.WithAssignedDevice(host_name);

  // In the future we may want to be clever here and avoid the extra Cast ops.
  SliceInputs slice_inputs_int64 =
      MakeSliceIndexAndSizeInt64(host_scope, slice_inputs);

  // Create a list of all control dependencies to be copied when possibly
  // replacing nodes related to slice_size.
  Node* old_size;
  std::vector<const Edge*> old_size_ctrl_deps;
  TF_RETURN_IF_ERROR(slice->input_node(2, &old_size));
  absl::c_copy_if(old_size->in_edges(), std::back_inserter(old_size_ctrl_deps),
                  [](const Edge* e) { return e->IsControlEdge(); });

  Output slice_size;
  TF_RETURN_IF_ERROR(ComputeSliceSize(host_scope, slice_inputs_int64,
                                      old_size_ctrl_deps, &slice_size));

  *result =
      ops::Slice(main_scope.WithAssignedDevice(slice->assigned_device_name())
                     .WithOpName("static_shaped_slice"),
                 slice_inputs_int64.input, slice_inputs_int64.begin, slice_size)
          .node();

  TF_RETURN_IF_ERROR(main_scope.status());

  std::vector<string> compile_time_const_inputs;
  compile_time_const_inputs.push_back("size");
  (*result)->AddAttr(kXlaCompileTimeConstantInputsAttr,
                     compile_time_const_inputs);
  return status;
}

void ReplaceTensorFlowSliceWithStaticShapedSlice(Graph* g, Node* slice,
                                                 Node* static_shaped_slice) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_6(mht_6_v, 471, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "ReplaceTensorFlowSliceWithStaticShapedSlice");

  absl::InlinedVector<const Edge*, 6> edges_to_remove;
  std::vector<const Edge*> slice_out_edges;
  absl::c_copy(slice->out_edges(), std::back_inserter(slice_out_edges));
  for (const Edge* e : slice_out_edges) {
    DCHECK(e->src_output() == 0 || e->src_output() == Graph::kControlSlot);

    int src_output = e->src_output();
    int dst_input = e->dst_input();
    Node* dst = e->dst();
    g->RemoveEdge(e);
    g->AddEdge(static_shaped_slice, src_output, dst, dst_input);
  }

  for (const Edge* e : slice->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), static_shaped_slice);
    }
  }

  g->RemoveNode(slice);
}

Status RewriteSlice(Graph* g, Node* slice, const SliceInputs& slice_inputs,
                    absl::string_view cluster_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("cluster_name: \"" + std::string(cluster_name.data(), cluster_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_7(mht_7_v, 499, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "RewriteSlice");

  VLOG(3) << "Rewriting slice " << slice->name()
          << " to a \"static shaped\" Slice";
  Node* static_shaped_slice;
  TF_RETURN_IF_ERROR(ConvertTensorFlowSliceToStaticShapedSlice(
      g, slice, slice_inputs, cluster_name, &static_shaped_slice));
  ReplaceTensorFlowSliceWithStaticShapedSlice(g, slice, static_shaped_slice);
  return Status::OK();
}

// Return true if `n` is a slice we should rewrite to have a static shape
// (i.e. have the output shape only depend on the "size" input).
StatusOr<bool> ShouldRewriteSlice(Node* n) {
  if (n->type_string() != "Slice") {
    return false;
  }

  if (!GetXlaClusterForNode(*n).has_value()) {
    // There is no need to change slice ops outside XLA clusters.
    return false;
  }

  TF_ASSIGN_OR_RETURN(absl::optional<SliceInputs> slice_inputs,
                      GetSliceInputs(n));
  if (!slice_inputs.has_value()) {
    return false;
  }

  // If slice_size[i] < -1 for any i then executing the slice will throw an
  // error, and we don't do anything here.
  bool slice_size_has_error =
      absl::c_all_of(slice_inputs->size_as_vector,
                     [](int64_t size_i) { return size_i >= -1; });
  if (!slice_size_has_error) {
    return false;
  }

  // No point in rewriting slices that have both size and begin as constants.
  return !slice_inputs->begin.node()->IsConstant();
}

Status FindAndRewriteSlices(Graph* g, bool* changed) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_8(mht_8_v, 543, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "FindAndRewriteSlices");

  std::vector<Node*> slices_to_rewrite;
  for (Node* n : g->nodes()) {
    TF_ASSIGN_OR_RETURN(bool is_rewritable, ShouldRewriteSlice(n));
    if (is_rewritable) {
      slices_to_rewrite.push_back(n);
    }
  }

  for (Node* n : slices_to_rewrite) {
    TF_ASSIGN_OR_RETURN(absl::optional<SliceInputs> slice_inputs,
                        GetSliceInputs(n));
    TF_RET_CHECK(slice_inputs.has_value());
    TF_RETURN_IF_ERROR(
        RewriteSlice(g, n, *slice_inputs, *GetXlaClusterForNode(*n)));
  }

  if (!slices_to_rewrite.empty()) {
    // We've added constants to the graph; hook them up to _SOURCE.
    FixupSourceAndSinkEdges(g);
  }

  *changed = !slices_to_rewrite.empty();

  return Status::OK();
}
}  // namespace

Status IncreaseDynamismForAutoJitPass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSincrease_dynamism_for_auto_jit_passDTcc mht_9(mht_9_v, 575, "", "./tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.cc", "IncreaseDynamismForAutoJitPass::Run");

  MarkForCompilationPassFlags* flags = GetMarkForCompilationPassFlags();
  if (flags->tf_xla_clustering_debug) {
    DumpGraphToFile("before_increase_dynamism_for_auto_jit_pass",
                    **options.graph, options.flib_def);
  }

  bool changed;
  TF_RETURN_IF_ERROR(FindAndRewriteSlices(options.graph->get(), &changed));
  if (changed && flags->tf_xla_clustering_debug) {
    DumpGraphToFile("increase_dynamism_for_auto_jit_pass", **options.graph,
                    options.flib_def);
  }

  return Status::OK();
}

}  // namespace tensorflow
