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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc() {
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

#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/canonicalizer.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/traversal.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/strided_slice_op.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

// Mark nodes created or optimized by a stage with a tag.
constexpr char kAddOpsRewriteTag[] =
    "_grappler_ArithmeticOptimizer_AddOpsRewriteStage";
constexpr char kMinimizeBroadcastsTag[] =
    "_grappler_ArithmeticOptimizer_MinimizeBroadcasts";

// Extract values from a Const op to `values`. Returns true if succeeds.
template <typename T>
bool ValuesFromConstNode(const NodeDef& node, std::vector<T>* values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_0(mht_0_v, 244, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ValuesFromConstNode");

  if (node.op() != "Const") {
    return false;
  }

  if (node.attr().count("dtype") == 0 || node.attr().count("value") == 0 ||
      node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
    return false;
  }

  // TensorProto represents the content of the tensor in either <type>_val or
  // tensor_content.
  const TensorProto& tensor = node.attr().at("value").tensor();
  typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
      checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

  if (!tensor_values->empty() && tensor.has_tensor_shape()) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    const TensorShapeProto& shape = tensor.tensor_shape();
    if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size()) {
      values->insert(values->end(), tensor_values->begin(),
                     tensor_values->end());
      return true;
    }
  }

  const auto tensor_content_size = tensor.tensor_content().size();
  if (tensor_content_size > 0) {
    CHECK_EQ(0, tensor_content_size % sizeof(T))
        << "tensor_content_size (" << tensor_content_size
        << ") is not a multiple of " << sizeof(T);
    values->resize(tensor_content_size / sizeof(T));
    port::CopyToArray(tensor.tensor_content(),
                      reinterpret_cast<char*>(values->data()));
    return true;
  }

  return false;
}

bool MaybeAddControlInput(const string& new_input, NodeDef* node,
                          GraphDef* graph, NodeMap* node_map) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("new_input: \"" + new_input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_1(mht_1_v, 291, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "MaybeAddControlInput");

  bool already_exists = false;
  for (const string& input : node->input()) {
    if (input == new_input || AsControlDependency(input) == new_input) {
      already_exists = true;
      break;
    }
  }
  if (!already_exists) {
    const string ctrl_dep =
        ConstantFolding::AddControlDependency(new_input, graph, node_map);
    node->add_input(ctrl_dep);
    node_map->AddOutput(NodeName(new_input), node->name());
  }
  return !already_exists;
}

void SetDataTypeToAttr(DataType dtype, const string& attr_name, NodeDef* node) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_2(mht_2_v, 312, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SetDataTypeToAttr");

  (*node->mutable_attr())[attr_name].set_type(dtype);
}

NodeDef* GetTailOfValuePreservingChain(
    const NodeDef& node, const NodeMap& node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetTailOfValuePreservingChain");

  auto is_value_preserving_non_branching = [&](const NodeDef& node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_4(mht_4_v, 325, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

    return nodes_to_preserve.find(node.name()) == nodes_to_preserve.end() &&
           IsValuePreserving(node) && NumNonControlOutputs(node, node_map) == 1;
  };
  return GetTailOfChain(node, node_map, /*follow_control_input=*/false,
                        is_value_preserving_non_branching);
}

NodeDef* GetTailOfIdempotentChain(
    const NodeDef& node, const NodeMap& node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_5(mht_5_v, 338, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetTailOfIdempotentChain");

  auto is_idempotent_non_branching = [&](const NodeDef& node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_6(mht_6_v, 342, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

    return nodes_to_preserve.find(node.name()) == nodes_to_preserve.end() &&
           IsIdempotent(node) && NumNonControlOutputs(node, node_map) == 1;
  };
  return GetTailOfChain(node, node_map, /*follow_control_input=*/false,
                        is_idempotent_non_branching);
}

// GetElementUnexhaustive tries to get the value of an element in a tensor and
// turn it into complex128 type. It only check for a limited number of data
// types, so it's unexhaustive.
bool GetElementUnexhaustive(const Tensor& t, int i, const std::set<int>& dtypes,
                            complex128* element) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_7(mht_7_v, 357, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetElementUnexhaustive");

  if (dtypes.find(t.dtype()) == dtypes.end()) return false;
  switch (t.dtype()) {
    case DT_BFLOAT16:
      *element = complex128(t.flat<bfloat16>()(i));
      return true;
    case DT_HALF:
      *element = complex128(static_cast<double>(t.flat<Eigen::half>()(i)), 0);
      return true;
    case DT_INT32:
      *element = complex128(t.flat<int32>()(i));
      return true;
    case DT_INT64:
      *element = complex128(t.flat<int64_t>()(i));
      return true;
    case DT_FLOAT:
      *element = complex128(t.flat<float>()(i));
      return true;
    case DT_DOUBLE:
      *element = complex128(t.flat<double>()(i));
      return true;
    case DT_COMPLEX64:
      *element = complex128(t.flat<complex64>()(i));
      return true;
    case DT_COMPLEX128:
      *element = t.flat<complex128>()(i);
      return true;
    default:
      return false;
  }
}

bool NodeIsOnCpu(const NodeDef& node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_8(mht_8_v, 392, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "NodeIsOnCpu");

  string task;
  string device;
  return DeviceNameUtils::SplitDeviceName(node.device(), &task, &device) &&
         absl::StrContains(device, DEVICE_CPU);
}

// True if all regular (non-control) inputs reference the same node or if there
// are no non-control inputs
bool AllRegularInputsEqual(const NodeDef& node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_9(mht_9_v, 404, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AllRegularInputsEqual");

  if (!HasRegularInputs(node)) return true;
  for (int i = 1; i < node.input_size(); ++i) {
    if (IsControlInput(node.input(i))) {
      break;
    }
    if (node.input(0) != node.input(i)) {
      return false;
    }
  }
  return true;
}

// Replace a node with NoOp and reset shape inference results for it..
void ReplaceWithNoOp(NodeDef* node, const GraphOptimizerContext& ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_10(mht_10_v, 421, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReplaceWithNoOp");

  ctx.node_map->RemoveInputs(node->name());
  ctx.graph_properties->ClearInputProperties(node->name());
  ctx.graph_properties->ClearOutputProperties(node->name());
  EraseRegularNodeAttributes(node);
  node->set_op("NoOp");
  node->clear_input();
}

// Graph optimizer context extension specific to ArithmeticOptimizer.
struct ArithmeticOptimizerContext {
  explicit ArithmeticOptimizerContext(SetVector<NodeDef*>* nodes_to_simplify)
      : nodes_to_simplify(nodes_to_simplify) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_11(mht_11_v, 436, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ArithmeticOptimizerContext");
}
  SetVector<NodeDef*>* nodes_to_simplify;
};

// Base class for single arithmetic optimization: e.g. Bitcast optimization,
// AddOps optimization, etc...
class ArithmeticOptimizerStage : public GraphOptimizerStage<string> {
 public:
  explicit ArithmeticOptimizerStage(const string& name,
                                    const GraphOptimizerContext& ctx,
                                    const ArithmeticOptimizerContext ctx_ext)
      : GraphOptimizerStage("ArithmeticOptimizer", name, ctx),
        ctx_ext_(ctx_ext) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_12(mht_12_v, 452, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ArithmeticOptimizerStage");
}
  ~ArithmeticOptimizerStage() override = default;

 protected:
  // Simplification graph rewrite can create additional nodes that are inputs
  // to final simplified node, they can be also added to the arithmetic
  // optimizer queue for further optimization.
  void AddToOptimizationQueue(NodeDef* node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_13(mht_13_v, 462, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddToOptimizationQueue");

    ctx_ext_.nodes_to_simplify->PushBack(node);
  }

  // Update consumers of node to take new_input as input instead.
  Status UpdateConsumers(NodeDef* node, const string& new_input) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("new_input: \"" + new_input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_14(mht_14_v, 471, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "UpdateConsumers");

    const auto consumers = ctx().node_map->GetOutputs(node->name());
    if (consumers.empty()) return Status::OK();
    const TensorId new_tensor = ParseTensorName(new_input);
    for (NodeDef* consumer : consumers) {
      if (consumer->name() == new_tensor.node()) continue;
      bool updated = false;
      for (int i = 0; i < consumer->input_size(); ++i) {
        const TensorId input_tensor = ParseTensorName(consumer->input(i));
        if (input_tensor.node() == node->name()) {
          if (new_tensor.index() < 0 && input_tensor.index() >= 0) {
            // Overwriting a data input with a control input will make the graph
            // invalid.
            return errors::InvalidArgument(
                "Cannot override data input ", input_tensor.ToString(),
                " with control input ", new_tensor.ToString());
          }
          consumer->set_input(i, input_tensor.index() < 0
                                     ? absl::StrCat("^", new_tensor.node())
                                     : new_input);
          ctx().node_map->UpdateInput(consumer->name(), node->name(),
                                      new_input);
          updated = true;
        }
      }
      if (updated) {
        DedupControlInputs(consumer);
        AddToOptimizationQueue(consumer);
      }
    }
    return Status::OK();
  }

  // TODO(ezhulenev): remove this method from ArithmeticOptimizer when all
  // optimizations will be migrated to stages
  void ForwardControlDependencies(
      NodeDef* target_node, const std::vector<const NodeDef*>& src_nodes) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_15(mht_15_v, 510, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ForwardControlDependencies");

    for (const auto& src : src_nodes) {
      for (int i = src->input_size() - 1; i >= 0; --i) {
        if (IsControlInput(src->input(i))) {
          *target_node->add_input() = src->input(i);
          ctx().node_map->AddOutput(NodeName(src->input(i)),
                                    target_node->name());
        } else {
          break;
        }
      }
    }
    DedupControlInputs(target_node);
  }

  bool IsReallyConstant(const NodeDef& node) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_16(mht_16_v, 528, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsReallyConstant");

    if (!IsConstant(node)) {
      return false;
    }
    // If the node is fed it's not constant anymore.
    return ctx().feed_nodes->find(node.name()) == ctx().feed_nodes->end();
  }

  bool IsInPreserveSet(const NodeDef& node) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_17(mht_17_v, 539, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsInPreserveSet");

    return ctx().nodes_to_preserve->find(node.name()) !=
           ctx().nodes_to_preserve->end();
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool IsDrivenByControlDependency(const NodeDef& node) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_18(mht_18_v, 548, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsDrivenByControlDependency");

    return std::any_of(
        node.input().begin(), node.input().end(),
        [](const string& input) { return IsControlInput(input); });
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool DrivesControlDependency(const NodeDef& node) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_19(mht_19_v, 558, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "DrivesControlDependency");

    for (const NodeDef* output : ctx().node_map->GetOutputs(node.name())) {
      for (int i = 0; i < output->input_size(); ++i) {
        const TensorId tensor = ParseTensorName(output->input(i));
        if (tensor.node() == node.name() && tensor.index() < 0) {
          return true;
        }
      }
    }
    return false;
  }

  bool GetTensorFromConstNode(const string& node_name_or_input,
                              Tensor* tensor) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("node_name_or_input: \"" + node_name_or_input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_20(mht_20_v, 575, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetTensorFromConstNode");

    const NodeDef* node = ctx().node_map->GetNode(node_name_or_input);
    return node != nullptr && IsReallyConstant(*node) &&
           CheckAttrExists(*node, "value").ok() &&
           tensor->FromProto(node->attr().at("value").tensor());
  }

 private:
  // Extended context required for ArithmeticOptimizer.
  const ArithmeticOptimizerContext ctx_ext_;
};

// Subtype of ArithmeticOptimizerStage that does optimization by rewriting a
// group of nodes from the optimized graph.
//
// * AddOpsRewrite:
//   Rewrite a group of Add/AddN with compact Add/AddN tree
//
// * MinimizeBroadcasts:
//   Rewrite a group of binary associative ops, reordering
//   inputs, to minimize the cost of broadcast
class ArithmeticNodesGroupOptimizerStage : public ArithmeticOptimizerStage {
 public:
  explicit ArithmeticNodesGroupOptimizerStage(
      const string& name, const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext ctx_ext)
      : ArithmeticOptimizerStage(name, ctx, ctx_ext) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_21(mht_21_v, 605, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ArithmeticNodesGroupOptimizerStage");
}
  ~ArithmeticNodesGroupOptimizerStage() override = default;

  // Input name with a statically inferred shape from GraphProperties
  struct InputAndShape {
    InputAndShape(const string& input, const TensorShapeProto& shape)
        : input(input), shape(shape) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_22(mht_22_v, 615, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "InputAndShape");
}
    string input;
    TensorShapeProto shape;
  };

  // Subgraph (subtree) of nodes, that we want to optimize in "one shot" (e.g.
  // all the Add nodes that we plan to rewrite with a single AddN). Subgraph is
  // obtained by graph traversal, starting from a root node.
  struct OptimizedNodesGroup {
    NodeDef* root_node;
    TensorShapeProto root_shape;
    // Optimized nodes that will be updated or removed by rewrite
    std::vector<NodeDef*> optimized_nodes;
    // Inputs to optimized nodes
    std::vector<InputAndShape> inputs;
  };

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_23(mht_23_v, 635, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    OptimizedNodesGroup group;
    TF_RETURN_IF_ERROR(CreateOptimizedNodesGroup(node, &group));

    if (!group.optimized_nodes.empty()) {
      *simplified_node_name = RewriteOptimizedNodesGroup(group);
    }

    return Status::OK();
  }

 protected:
  // Modify the optimized graph after nodes group was successfully identified
  virtual string RewriteOptimizedNodesGroup(
      const OptimizedNodesGroup& group) = 0;

  // Check if input can become a part of current optimized nodes group.
  virtual bool IsAbsorbableByOptimizedNodesGroup(
      const OptimizedNodesGroup& group, const NodeDef& node) const = 0;

  Status AbsorbInputByOptimizedNodesGroup(const string& input,
                                          OptimizedNodesGroup* group) const {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_24(mht_24_v, 662, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AbsorbInputByOptimizedNodesGroup");

    std::deque<const string*> input_tensors;
    input_tensors.push_front(&input);

    while (!input_tensors.empty()) {
      const string* input_tensor = input_tensors.front();
      input_tensors.pop_front();

      // Get a node for the input tensor.
      NodeDef* input_node;
      TF_RETURN_IF_ERROR(GetInputNode(*input_tensor, &input_node));

      if (IsAbsorbableByOptimizedNodesGroup(*group, *input_node)) {
        group->optimized_nodes.push_back(input_node);
        for (int i = input_node->input_size() - 1; i >= 0; --i) {
          const string& absorbed_node_input = input_node->input(i);
          // TODO(ezhulenev): support control inputs
          if (IsControlInput(absorbed_node_input)) continue;
          input_tensors.push_front(&absorbed_node_input);
        }
      } else {
        // If input node can't be absorbed, add it to OptimizedNodesGroup input.
        const OpInfo::TensorProperties* properties;
        TF_RETURN_IF_ERROR(GetTensorProperties(*input_tensor, &properties));
        group->inputs.emplace_back(*input_tensor, properties->shape());
      }
    }

    return Status::OK();
  }

  Status CreateOptimizedNodesGroup(NodeDef* root_node,
                                   OptimizedNodesGroup* group) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_25(mht_25_v, 697, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CreateOptimizedNodesGroup");

    const OpInfo::TensorProperties* root_node_output_properties;
    TF_RETURN_IF_ERROR(
        GetTensorProperties(root_node->name(), &root_node_output_properties));

    group->root_node = root_node;
    group->root_shape = root_node_output_properties->shape();

    group->optimized_nodes.reserve(root_node->input_size());
    for (int i = 0; i < root_node->input_size(); ++i) {
      const string& input_i = root_node->input(i);
      // TODO(ezhulenev): add support for control inputs
      if (IsControlInput(input_i)) continue;
      TF_RETURN_IF_ERROR(AbsorbInputByOptimizedNodesGroup(input_i, group));
    }

    return Status::OK();
  }

  // Check if all inputs can be broadcasted to the same shape
  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool HasAllInputsBroadcastableToShape(
      const NodeDef& node, const OpInfo::TensorProperties& properties) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_26(mht_26_v, 722, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HasAllInputsBroadcastableToShape");

    auto is_broadcastable = [this, &properties](const string& input) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_27(mht_27_v, 727, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

      const OpInfo::TensorProperties* input_props;
      Status has_input_properties = GetTensorProperties(input, &input_props);
      return has_input_properties.ok() &&
             ShapesBroadcastable(properties, *input_props);
    };
    return std::all_of(node.input().begin(), node.input().end(),
                       is_broadcastable);
  }

  string ShapeSignature(const TensorShapeProto& shape) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_28(mht_28_v, 740, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ShapeSignature");

    string signature = strings::StrCat("rank:", shape.dim_size(), ":dim");
    for (int i = 0; i < shape.dim_size(); ++i)
      strings::StrAppend(&signature, ":", shape.dim(i).size());
    return signature;
  }

  void MarkWithTag(const StringPiece tag, NodeDef* node) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_29(mht_29_v, 750, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "MarkWithTag");

    AddNodeAttr(tag, true, node);
  }

  void MarkAllMembersWithTag(const OptimizedNodesGroup& group,
                             const StringPiece tag) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_30(mht_30_v, 758, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "MarkAllMembersWithTag");

    AddNodeAttr(tag, true, group.root_node);
    for (NodeDef* optimized_node : group.optimized_nodes) {
      AddNodeAttr(tag, true, optimized_node);
    }
  }

  bool IsOnTheSameDevice(const OptimizedNodesGroup& group,
                         const NodeDef& node) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_31(mht_31_v, 769, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsOnTheSameDevice");

    return group.root_node->device() == node.device();
  }

  bool IsInPreserveSet(const NodeDef& node) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_32(mht_32_v, 776, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsInPreserveSet");

    return ctx().nodes_to_preserve->find(node.name()) !=
           ctx().nodes_to_preserve->end();
  }

  bool IsMarkedWithTag(const NodeDef& node, const StringPiece tag) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_33(mht_33_v, 784, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsMarkedWithTag");

    return HasNodeAttr(node, tag);
  }

  bool IsMarkedWithAnyTag(const NodeDef& node, const StringPiece tag1,
                          const StringPiece tag2) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_34(mht_34_v, 792, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsMarkedWithAnyTag");

    return IsMarkedWithTag(node, tag1) || IsMarkedWithTag(node, tag2);
  }
};

// Rewrite a tree of Add/AddN with a single AddN operation, consuming all the
// original inputs of absorbed nodes.
//
// 1) All nodes must have the same device placement.
//
// 2) If All nodes in a Add/AddN subgraph have symbolically equal shape, tree is
//    optimized to a single AddN node.
//
//                AddN_1
//             /    |    \
//          Add_1   z   Add_2       -> AddN(x, y, z, w, q, e)
//          /  \        /  \
//         x    y      w    Add_3
//                          / \
//                         q   e
//
// 3) If some nodes have different shape (it needs to be broadcastable to the
//    shape of a "root), tree is optimized to AddNs for symbolically equal
//    shapes, and a tree of Add ops, that minimize broadcasts.
//
//                AddN_1                                 Add
//             /    |    \                              /  \
//          Add_1   z   Add_2       ->               Add    w
//          /  \        /  \                        /   \
//         x    y      w    Add_3      AddN(x, y, q, e)  z
//                          / \
//                         q   e
class AddOpsRewriteStage : public ArithmeticNodesGroupOptimizerStage {
 public:
  explicit AddOpsRewriteStage(const GraphOptimizerContext& ctx,
                              const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticNodesGroupOptimizerStage("AddOpsRewrite", ctx, ctx_ext) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_35(mht_35_v, 831, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddOpsRewriteStage");
}
  ~AddOpsRewriteStage() override = default;

  // Check if a node can become a root of AddOpsGroup
  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_36(mht_36_v, 838, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    if (!CanOptimize(*node)) return false;

    // shape must be symbolically defined and all inputs compatible with it
    const OpInfo::TensorProperties* properties;
    Status has_properties = GetTensorProperties(node->name(), &properties);
    return has_properties.ok() && ShapeIsSymbolicallyDefined(*properties) &&
           HasAllInputsBroadcastableToShape(*node, *properties);
  }

 protected:
  // Check if a node can be absorbed by current OptimizedNodesGroup
  bool IsAbsorbableByOptimizedNodesGroup(const OptimizedNodesGroup& group,
                                         const NodeDef& node) const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_37(mht_37_v, 854, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsAbsorbableByOptimizedNodesGroup");

    if (!CanOptimize(node)) return false;

    if (!IsOnTheSameDevice(group, node)) {
      return false;
    }
    // with a single output data consumer (presumably if we reach this node from
    // previously absorbed or a root node, it means that this node is not used
    // as an input to any other op, outside of the group)
    if (NumNonControlDataOutputs(node, *ctx().node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    const OpInfo::TensorProperties* properties;
    Status has_properties = GetTensorProperties(node.name(), &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(node, *properties);
  }

  // Node requirements both for a root node and an absorbed node
  bool CanOptimize(const NodeDef& node) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_38(mht_38_v, 877, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CanOptimize");

    // TODO(ezhulenev): check if AccumulateNV2 can be supported too
    if (!IsAdd(node) && !IsAddN(node)) {
      return false;
    }
    if (IsInPreserveSet(node) || IsMarkedWithTag(node, kAddOpsRewriteTag)) {
      return false;
    }
    // TODO(ezhulenev): relax this condition for root node
    return !(IsDrivenByControlDependency(node) ||
             DrivesControlDependency(node));
  }

  // Rewrite a group of add ops into a single AddN if all input shapes are
  // symbolically equal. If not, create AddN for equal shapes first, and then
  // build an Add tree, minimizing the cost of broadcasts.
  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_39(mht_39_v, 896, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RewriteOptimizedNodesGroup");

    VLOG(2) << "Collapse Add/AddN: root=" << group.root_node->name()
            << " op=" << group.root_node->op()
            << " num_optimized_nodes=" << group.optimized_nodes.size()
            << " num_inputs=" << group.inputs.size();

    // Do not optimize any of the nodes that are part of this group.
    MarkAllMembersWithTag(group, kAddOpsRewriteTag);

    // All new nodes will be placed under the scope of a root node.
    auto root_scope_and_name = ParseNodeScopeAndName(group.root_node->name());

    // Find what shapes are present in the inputs of absorbed nodes.
    std::unordered_map<string, std::vector<InputAndShape>> shape_sig_to_inputs;
    for (const auto& input : group.inputs) {
      shape_sig_to_inputs[ShapeSignature(input.shape)].push_back(input);
    }

    using SigKV = decltype(shape_sig_to_inputs)::value_type;
    VLOG(3) << "Add/AddN group has " << shape_sig_to_inputs.size()
            << " unique shapes: "
            << absl::StrJoin(shape_sig_to_inputs, ", ",
                             [](string* out, SigKV p) {
                               strings::StrAppend(out, p.first);
                             });

    // Collect all the shapes from representative elements.
    std::vector<TensorShapeProto> shapes;
    shapes.reserve(shape_sig_to_inputs.size());
    for (const auto& el : shape_sig_to_inputs)
      shapes.push_back(el.second[0].shape);

    // If all inputs have the same shape, rewrite whole group with a single AddN
    if (shapes.size() == 1) {
      string node_name = UniqueOptimizedNodeName(root_scope_and_name);
      AddInputsOfSymbolicallyEqualShape(*group.root_node, node_name,
                                        group.inputs);
      return node_name;
    }

    // For inputs of different shapes:
    // 1. Rewrite inputs of the same shape using AddN (leaf nodes)
    // 2. Build a tree of Add nodes, minimizing cost of broadcast
    std::sort(shapes.begin(), shapes.end(),
              [](const TensorShapeProto& left, const TensorShapeProto& right) {
                return CompareSymbolicallyShapedTensorSizes(left, right);
              });

    // optimized name for leaf AddN nodes
    auto leaf_node_name = [&root_scope_and_name, this](int i) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_40(mht_40_v, 948, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

      return UniqueOptimizedNodeName(root_scope_and_name,
                                     strings::StrCat("Leaf_", i));
    };
    // optimized name for internal nodes of a tree built up from AddN leaves
    auto internal_node_name = [&root_scope_and_name, this](int i) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_41(mht_41_v, 956, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

      return UniqueOptimizedNodeName(root_scope_and_name,
                                     strings::StrCat("Internal_", i));
    };

    // Add/AddN nodes that must be added to the tree
    std::deque<InputAndShape> add_ops;

    // Prepare leaf AddN nodes for inputs of equal shape
    for (int i = 0, end = shapes.size(); i < end; ++i) {
      const auto node_name = leaf_node_name(i);
      const auto& inputs = shape_sig_to_inputs[ShapeSignature(shapes[i])];
      add_ops.push_back(AddInputsOfSymbolicallyEqualShape(*group.root_node,
                                                          node_name, inputs));
    }

    // Build up a tree of Add ops
    int internal_nodes = 0;
    do {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops.front();
      add_ops.pop_front();
      string name = add_ops.empty()
                        ? UniqueOptimizedNodeName(root_scope_and_name)
                        : internal_node_name(internal_nodes++);
      InputAndShape add = AddAggregatedInputs(*group.root_node, name, lhs, rhs);
      add_ops.push_front(add);
    } while (add_ops.size() > 1);

    InputAndShape optimized_root_node = add_ops.front();
    return optimized_root_node.input;
  }

  // Add 'AddN' node to aggregate inputs of symbolically equal shape
  InputAndShape AddInputsOfSymbolicallyEqualShape(
      const NodeDef& root_node, const string& node_name,
      const std::vector<InputAndShape>& inputs) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_42(mht_42_v, 997, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddInputsOfSymbolicallyEqualShape");

    CHECK(!inputs.empty()) << "Inputs must be non-empty";

    // Do not create redundant AddN nodes
    if (inputs.size() == 1 || root_node.attr().count("T") == 0) {
      return inputs[0];
    }

    // get shape from representative element
    auto shape = inputs[0].shape;

    // copy attributes from a root node
    DataType dtype = root_node.attr().at("T").type();

    // add new AddN node
    NodeDef* node = AddEmptyNode(node_name);
    node->set_op("AddN");
    node->set_device(root_node.device());
    (*node->mutable_attr())["T"].set_type(dtype);
    (*node->mutable_attr())["N"].set_i(inputs.size());

    for (const auto& inputAndShape : inputs) {
      ctx().node_map->AddOutput(inputAndShape.input, node_name);
      node->add_input(inputAndShape.input);
    }

    MarkWithTag(kAddOpsRewriteTag, node);
    return InputAndShape(node_name, shape);
  }

  // Add a single 'Add' node to sum two inputs
  InputAndShape AddAggregatedInputs(const NodeDef& root_node,
                                    const string& node_name,
                                    const InputAndShape& left,
                                    const InputAndShape& right) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_43(mht_43_v, 1035, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddAggregatedInputs");

    // copy attributes from a root node
    DataType dtype = root_node.attr().at("T").type();

    // add new Add node
    NodeDef* node = AddEmptyNode(node_name);
    node->set_op((dtype == DT_STRING || dtype == DT_STRING_REF) ? "Add"
                                                                : "AddV2");
    node->set_device(root_node.device());
    (*node->mutable_attr())["T"].set_type(dtype);
    node->add_input(left.input);
    node->add_input(right.input);

    ctx().node_map->AddOutput(left.input, node_name);
    ctx().node_map->AddOutput(right.input, node_name);

    MarkWithTag(kAddOpsRewriteTag, node);
    return InputAndShape(
        node_name, TensorShapeProto());  // shape is not important at this point
  }
};

// Use the distributive property of multiplication and division over addition,
// along with commutativity of the former, to hoist common factors/denominators
// out of aggregate nodes where ALL the inputs are Mul/Div nodes.
// This pattern occurs frequently in regularization terms for the gradients
// during training.
//
// For example, we can rewrite an expression of the form:
//   AddN(Mul(x, y1), Mul(y2, x), Mul(x, y3), ... Mul(x, yn))
// to the following:
//   Mul(x, AddN(y1, y2, y3, ... yn))
// For division, we can rewrite
//   AddN(Div(y1, x), Div(y2, x), Div(y3, x), ... Div(yn, x))
// to:
//   Div(AddN(y1, y2, y3, ... yn), x)
class HoistCommonFactorOutOfAggregation : public ArithmeticOptimizerStage {
 public:
  explicit HoistCommonFactorOutOfAggregation(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("HoistCommonFactor", ctx, ctx_ext) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_44(mht_44_v, 1079, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HoistCommonFactorOutOfAggregation");
}
  ~HoistCommonFactorOutOfAggregation() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_45(mht_45_v, 1085, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAggregate(*node) && NumNonControlInputs(*node) > 1 &&
           !IsRewritten(node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_46(mht_46_v, 1093, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    bool common_factor_is_denominator = false;
    std::set<string> common_factors;
    std::vector<string> ctrl_deps;
    TF_RETURN_IF_ERROR(GetCommonFactors(
        node, &common_factors, &common_factor_is_denominator, &ctrl_deps));

    if (common_factors.size() == 1) {
      const string& common_factor = *common_factors.begin();

      // Gather up the non-shared factors
      bool shapes_match = true;
      std::vector<string> unique_factors;
      TF_RETURN_IF_ERROR(GetUniqueFactors(node, common_factor,
                                          common_factor_is_denominator,
                                          &shapes_match, &unique_factors));

      if (shapes_match) {
        NodeDef* input_0;
        TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input_0));

        // Use a copy of the first node for the outer multiplication/division.
        NodeDef* new_outer_node = AddCopyNode(
            OuterNodeName(node, common_factor_is_denominator), input_0);
        // And a copy of aggregation node as one of the inner operands
        NodeDef* new_add_node = AddCopyNode(InnerAddNodeName(node), node);

        new_outer_node->set_device(node->device());
        if (common_factor_is_denominator) {
          new_outer_node->set_input(0, new_add_node->name());
          new_outer_node->set_input(1, common_factor);
        } else {
          new_outer_node->set_input(0, common_factor);
          new_outer_node->set_input(1, new_add_node->name());
        }

        ctx().node_map->AddOutput(common_factor, new_outer_node->name());
        ctx().node_map->AddOutput(new_add_node->name(), new_outer_node->name());

        // Hoist non-shared factors up into the new AddN node.
        for (int i = 0, end = unique_factors.size(); i < end; ++i) {
          const string& unique_factor_i = unique_factors[i];
          new_add_node->set_input(i, unique_factor_i);
          ctx().node_map->AddOutput(unique_factor_i, new_add_node->name());
        }

        // Add control deps on add node
        for (const string& ctrl_dep : ctrl_deps) {
          *new_add_node->add_input() = ctrl_dep;
          ctx().node_map->AddOutput(NodeName(ctrl_dep), new_add_node->name());
        }

        // optimize new inner aggregation node
        AddToOptimizationQueue(new_add_node);
        // do not optimize the same node twice
        rewritten_nodes_.insert(node->name());
        *simplified_node_name = new_outer_node->name();
      }
    }
    return Status::OK();
  }

 private:
  // Get a name for new outer node
  string OuterNodeName(const NodeDef* node, bool is_div) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_47(mht_47_v, 1162, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "OuterNodeName");

    auto scope_and_name = ParseNodeScopeAndName(node->name());
    return is_div ? OptimizedNodeName(scope_and_name, "Div")
                  : OptimizedNodeName(scope_and_name, "Mul");
  }

  // Get a name new inner Add node
  string InnerAddNodeName(const NodeDef* node) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_48(mht_48_v, 1172, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "InnerAddNodeName");

    auto scope_and_name = ParseNodeScopeAndName(node->name());
    return OptimizedNodeName(scope_and_name, "AddV2");
  }

  // Determine the set of common factors if the input nodes are all Mul or
  // Div nodes.
  Status GetCommonFactors(const NodeDef* node, std::set<string>* common_factors,
                          bool* common_factor_is_denominator,
                          std::vector<string>* ctrl_deps) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_49(mht_49_v, 1184, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetCommonFactors");

    CHECK(common_factors->empty());
    CHECK_NOTNULL(common_factor_is_denominator);
    *common_factor_is_denominator = false;

    bool has_mul = false;
    bool has_div = false;
    for (int i = 0; i < node->input_size(); ++i) {
      if (i > 0 && common_factors->empty()) break;
      if (IsControlInput(node->input(i))) {
        ctrl_deps->push_back(node->input(i));
        continue;
      }
      NodeDef* input;
      TF_RETURN_IF_ERROR(GetInputNode(node->input(i), &input));

      if ((!IsMul(*input) && !IsAnyDiv(*input)) || (IsMul(*input) && has_div) ||
          (IsAnyDiv(*input) && has_mul)) {
        // Break if input is neither a Mul or Div, or if there are both Mul &
        // Div Ops.
        common_factors->clear();
        break;
      } else if (IsAnyDiv(*input)) {
        has_div = true;
        // In case of possible common dividers, we avoid hoisting out if any
        // input is not float/double, since integer division is not distributive
        // over addition.
        const OpInfo::TensorProperties* properties0;
        const OpInfo::TensorProperties* properties1;
        TF_RETURN_IF_ERROR(GetTensorProperties(input->input(0), &properties0));
        TF_RETURN_IF_ERROR(GetTensorProperties(input->input(1), &properties1));
        if (properties0->dtype() != DT_FLOAT &&
            properties0->dtype() != DT_DOUBLE &&
            properties1->dtype() != DT_FLOAT &&
            properties1->dtype() != DT_DOUBLE) {
          common_factors->clear();
          break;
        }
      } else if (IsMul(*input)) {
        has_mul = true;
      }

      // We only focus on common factors from denominators if any Op is a
      // Div.
      std::set<string> factors_i =
          has_mul ? std::set<string>{input->input(0), input->input(1)}
                  : std::set<string>{input->input(1)};
      if (i == 0) {
        std::swap(*common_factors, factors_i);
      } else {
        std::set<string> intersection;
        std::set_intersection(
            factors_i.begin(), factors_i.end(), common_factors->begin(),
            common_factors->end(),
            std::inserter(intersection, intersection.begin()));
        std::swap(*common_factors, intersection);
      }
      for (int i = 2; i < input->input_size(); ++i) {
        ctrl_deps->push_back(input->input(i));
      }
    }

    *common_factor_is_denominator = has_div;
    return Status::OK();
  }

  // Gather up the non-shared factors (the y's in the example).
  // Unless the aggregation is Add, we have to make sure that all the y's
  // have the same shape since the other aggregation ops do not support
  // broadcasting.
  Status GetUniqueFactors(const NodeDef* node, const string& common_factor,
                          const bool common_factor_is_denominator,
                          bool* shapes_match,
                          std::vector<string>* unique_factors) const {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("common_factor: \"" + common_factor + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_50(mht_50_v, 1261, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetUniqueFactors");

    *shapes_match = true;
    unique_factors->reserve(node->input_size());

    for (int i = 0; i < node->input_size() && *shapes_match; ++i) {
      const string& input = node->input(i);
      if (IsControlInput(input)) {
        break;
      }
      NodeDef* inner_node;
      TF_RETURN_IF_ERROR(GetInputNode(input, &inner_node));
      const int unique_factor_index =
          common_factor_is_denominator
              ? 0
              : (inner_node->input(0) == common_factor ? 1 : 0);
      unique_factors->push_back(inner_node->input(unique_factor_index));
      if (i > 0 && !IsAdd(*node)) {
        const OpInfo::TensorProperties* lhs;
        const OpInfo::TensorProperties* rhs;
        TF_RETURN_IF_ERROR(GetTensorProperties(unique_factors->front(), &lhs));
        TF_RETURN_IF_ERROR(GetTensorProperties(unique_factors->back(), &rhs));
        *shapes_match = ShapesSymbolicallyEqual(*lhs, *rhs);
      }
    }
    return Status::OK();
  }

  bool IsRewritten(const NodeDef* node) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_51(mht_51_v, 1291, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsRewritten");

    // if graph rewrite happens in multiple passes without graph pruning between
    // them, it's possible that rewritten node already exists in a graph
    return rewritten_nodes_.find(node->name()) != rewritten_nodes_.end() ||
           ctx().node_map->NodeExists(OuterNodeName(node, false)) ||
           ctx().node_map->NodeExists(OuterNodeName(node, true)) ||
           ctx().node_map->NodeExists(InnerAddNodeName(node));
  }

  // keep names of the nodes that were optimized by this stage
  std::unordered_set<string> rewritten_nodes_;
};

// Binary associative ops can be re-ordered to minimize the number of broadcasts
// and the size of a temporary tensors.
//
// Example: [a, c] - scalars, [b, d] - matrices
//   @ - binary associative op (Add or Mul)
//   @* - broadcast
//
//           @                      @*
//        /     \                /      \
//      @*       @*      ->     @        @
//    /   \    /   \          /   \    /   \
//   a     b  c     d        a     c  b     d
class MinimizeBroadcasts : public ArithmeticNodesGroupOptimizerStage {
 public:
  explicit MinimizeBroadcasts(const GraphOptimizerContext& ctx,
                              const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticNodesGroupOptimizerStage("MinimizeBroadcasts", ctx, ctx_ext) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_52(mht_52_v, 1323, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "MinimizeBroadcasts");

  }
  ~MinimizeBroadcasts() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_53(mht_53_v, 1330, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    if (!IsBinaryAssociative(*node)) return false;

    if (IsMarkedWithAnyTag(*node, kMinimizeBroadcastsTag, kAddOpsRewriteTag))
      return false;

    // has a symbolically defined shape with broadcastable inputs
    const OpInfo::TensorProperties* properties;
    Status has_properties = GetTensorProperties(node->name(), &properties);
    return has_properties.ok() && ShapeIsSymbolicallyDefined(*properties) &&
           HasAllInputsBroadcastableToShape(*node, *properties);
  }

 protected:
  bool IsBinaryAssociative(const NodeDef& node) const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_54(mht_54_v, 1347, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsBinaryAssociative");

    return IsMul(node) || IsAdd(node);
  }

  bool IsSameOp(const OptimizedNodesGroup& group, const NodeDef& node) const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_55(mht_55_v, 1354, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSameOp");

    return group.root_node->op() == node.op();
  }

  // Check if a node can be absorbed by current OptimizedNodesGroup
  bool IsAbsorbableByOptimizedNodesGroup(const OptimizedNodesGroup& group,
                                         const NodeDef& node) const override {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_56(mht_56_v, 1363, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsAbsorbableByOptimizedNodesGroup");

    if (!IsSameOp(group, node)) {
      return false;
    }
    if (IsInPreserveSet(node)) {
      return false;
    }
    // Nodes optimized by AddOpsRewrite already have optimal broadcasts.
    if (IsMarkedWithAnyTag(node, kMinimizeBroadcastsTag, kAddOpsRewriteTag)) {
      return false;
    }
    if (IsDrivenByControlDependency(node) || DrivesControlDependency(node)) {
      return false;
    }
    if (!IsOnTheSameDevice(group, node)) {
      return false;
    }
    // Optimized nodes updated in place, and that would break the graph, if the
    // node has multiple output consumers
    if (NumNonControlOutputs(node, *ctx().node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    const OpInfo::TensorProperties* properties;
    Status has_properties = GetTensorProperties(node.name(), &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(node, *properties);
  }

  std::size_t CountUniqueShapes(const std::vector<InputAndShape>& inputs) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_57(mht_57_v, 1395, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CountUniqueShapes");

    std::set<string> sigs;
    for (const auto& ias : inputs) {
      sigs.insert(ShapeSignature(ias.shape));
    }
    return sigs.size();
  }

  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_58(mht_58_v, 1406, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RewriteOptimizedNodesGroup");

    VLOG(2) << "Minimize broadcast: root=" << group.root_node->name()
            << " op=" << group.root_node->op()
            << " num_optimized_nodes=" << group.optimized_nodes.size();

    // Do not optimize any of the nodes that are part of this group.
    MarkAllMembersWithTag(group, kMinimizeBroadcastsTag);

    if (CountUniqueShapes(group.inputs) <= 1) {
      VLOG(3) << "Skip min-bcast group with single unique shape";
      // nothing to optimize when all shapes are the same
      return group.root_node->name();
    }

    auto num_nodes = /*root*/ 1 + group.optimized_nodes.size();
    auto num_inputs = group.inputs.size();
    CHECK_EQ(num_nodes, num_inputs - 1)
        << "Can't build a tree with " << num_inputs << " inputs, using "
        << num_nodes << "binary op nodes.";

    std::deque<InputAndShape> add_ops(group.inputs.begin(), group.inputs.end());
    std::deque<NodeDef*> optimized_nodes(group.optimized_nodes.begin(),
                                         group.optimized_nodes.end());

    // sort inputs by it's shape from smallest to largest
    std::stable_sort(add_ops.begin(), add_ops.end(),
                     [](const InputAndShape& lhs, const InputAndShape& rhs) {
                       return CompareSymbolicallyShapedTensorSizes(lhs.shape,
                                                                   rhs.shape);
                     });

    // If there is an odd number of inputs, last one is the largest, and we want
    // to attach it to the root node, to build a well balanced tree.
    std::deque<InputAndShape> add_ops_leftover;
    if (add_ops.size() % 2 != 0) {
      add_ops_leftover.push_back(add_ops.back());
      add_ops.pop_back();
    }

    // At this point it's guaranteed that add_ops have even number of inputs.
    do {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops.front();
      add_ops.pop_front();

      NodeDef* node;
      if (!optimized_nodes.empty()) {
        // re-purpose optimized nodes to build a new tree
        node = optimized_nodes.back();
        optimized_nodes.pop_back();
      } else {
        // or use root node if none optimized nodes left
        node = group.root_node;
      }
      InputAndShape updated_node = UpdateInputs(lhs.input, rhs.input, node);

      // Pushing updated node to the back of a deque will create a wide and
      // short tree, pushing to the front will create a tall tree. We prefer to
      // get a wide tree, it minimizes the potential number of temporary tensors
      // required to keep in memory, though sometimes we can go up to prevent
      // propagating a broadcast from leaves to the root. Example:
      //
      // inputs: [s, s, s, M] (s - scalar, M - matrix)
      // @* - op with broadcast
      //
      //  (only push_back)           @*     (push_front first op)
      //                            /  \
      //       @*                  @    M
      //     /   \                / \
      //    @     @*      ->     @   s
      //   / \   / \            / \
      //  s   s s   M          s   s
      if (add_ops.size() >= 2 &&
          CompareSymbolicallyShapedTensorSizes(add_ops.at(0).shape,
                                               add_ops.at(1).shape)) {
        add_ops.push_front(updated_node);
      } else {
        add_ops.push_back(updated_node);
      }
    } while (add_ops.size() > 1);
    CHECK_EQ(1, add_ops.size());

    // attach the largest tensor to the root op
    if (!add_ops_leftover.empty()) {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops_leftover.front();
      InputAndShape updated_node =
          UpdateInputs(lhs.input, rhs.input, group.root_node);
      add_ops.push_back(updated_node);
    }

    return add_ops.front().input;
  }

  InputAndShape UpdateInputs(const string& input_0, const string& input_1,
                             NodeDef* node) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("input_0: \"" + input_0 + "\"");
   mht_59_v.push_back("input_1: \"" + input_1 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_59(mht_59_v, 1508, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "UpdateInputs");

    string old_input_0 = node->input(0);
    string old_input_1 = node->input(1);

    // Update inputs only if they changed
    if (old_input_0 != input_0 || old_input_1 != input_1) {
      node->set_input(0, input_0);
      node->set_input(1, input_1);
      // Invalidate node properties (shape)
      ctx().graph_properties->ClearOutputProperties(node->name());
      ctx().graph_properties->ClearInputProperties(node->name());
      // Update the node map
      ctx().node_map->RemoveOutput(NodeName(old_input_0), node->name());
      ctx().node_map->RemoveOutput(NodeName(old_input_1), node->name());
      ctx().node_map->AddOutput(NodeName(input_0), node->name());
      ctx().node_map->AddOutput(NodeName(input_1), node->name());
      // Add updated node to optimization queue
      AddToOptimizationQueue(node);
    }

    TensorShapeProto shape;  // shape is not important at this point
    return InputAndShape(node->name(), shape);
  }
};

// Removes inverse transpose nodes
class RemoveIdentityTranspose : public ArithmeticOptimizerStage {
 public:
  explicit RemoveIdentityTranspose(const GraphOptimizerContext& ctx,
                                   const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveIdentityTranspose", ctx, ctx_ext) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_60(mht_60_v, 1541, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveIdentityTranspose");
}
  ~RemoveIdentityTranspose() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_61(mht_61_v, 1547, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsTranspose(*node) || IsConjugateTranspose(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_62(mht_62_v, 1554, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));
    NodeDef* tail = node;
    tail = GetTailOfIdempotentChain(*tail, *ctx().node_map,
                                    *ctx().nodes_to_preserve);
    NodeDef* first_transpose;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &first_transpose));

    NodeDef* node_perm;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &node_perm));
    if (!IsConstant(*node_perm)) {
      return Status::OK();
    }
    std::vector<int64_t> node_perm_values;
    TF_RETURN_IF_ERROR(GetPermutation(*node_perm, &node_perm_values));
    if (first_transpose->op() == node->op()) {
      // Remove pairs of transposes that cancel each other.
      NodeDef* first_transpose_perm;
      TF_RETURN_IF_ERROR(
          GetInputNode(first_transpose->input(1), &first_transpose_perm));
      if (!IsConstant(*first_transpose_perm)) {
        return Status::OK();
      }
      std::vector<int64_t> first_transpose_perm_values;
      TF_RETURN_IF_ERROR(
          GetPermutation(*first_transpose_perm, &first_transpose_perm_values));
      if (AreInversePermutations(node_perm_values,
                                 first_transpose_perm_values)) {
        if (tail == node) {
          // Bypass adjacent pair.
          *simplified_node_name = first_transpose->input(0);
        } else {
          // Bypass pair connected through chain.
          tail->set_input(0, first_transpose->input(0));
          ctx().node_map->UpdateInput(tail->name(), first_transpose->name(),
                                      first_transpose->input(0));
          ForwardControlDependencies(tail, {first_transpose});
          *simplified_node_name = node->input(0);
        }
      }
    } else {
      // Remove simple identity transposes.
      if (IsIdentityPermutation(node_perm_values)) {
        if (IsConjugateTranspose(*node)) {
          const NodeScopeAndName transpose =
              ParseNodeScopeAndName(node->name());
          const string optimized_node_name = OptimizedNodeName(transpose);
          NodeDef* new_op = AddCopyNode(optimized_node_name, node);
          new_op->set_op("Conj");
          new_op->mutable_input()->RemoveLast();
          new_op->mutable_attr()->erase("Tperm");
          ForwardControlDependencies(new_op, {node});
          *simplified_node_name = new_op->name();
        } else {
          *simplified_node_name = node->input(0);
        }
      }
    }
    return Status::OK();
  }

 private:
  Status GetPermutation(const NodeDef& node_perm,
                        std::vector<int64_t>* perm64) const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_63(mht_63_v, 1620, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetPermutation");

    std::vector<int> perm32;
    if (ValuesFromConstNode(node_perm, &perm32)) {
      perm64->reserve(perm32.size());
      for (int val : perm32) {
        perm64->push_back(static_cast<int64_t>(val));
      }
      return Status::OK();
    }
    if (ValuesFromConstNode(node_perm, perm64)) {
      return Status::OK();
    }
    return errors::InvalidArgument("Couldn't extract permutation from ",
                                   node_perm.name());
  }

  bool AreInversePermutations(const std::vector<int64_t>& a,
                              const std::vector<int64_t>& b) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_64(mht_64_v, 1640, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AreInversePermutations");

    if (a.size() != b.size()) {
      return false;
    }
    for (int i = 0, end = a.size(); i < end; ++i) {
      if (a[b[i]] != i) {
        return false;
      }
    }
    return true;
  }

  bool IsIdentityPermutation(const std::vector<int64_t>& perm) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_65(mht_65_v, 1655, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsIdentityPermutation");

    for (int64_t i = 0, end = perm.size(); i < end; ++i) {
      if (i != perm[i]) {
        return false;
      }
    }
    return true;
  }
};

// An involution is an element-wise function f(x) that is its own inverse,
// i.e. f(f(x)) = x. If we can find a chain of ops
//   f->op1->op2->...opn->f
// where op1 through opn preserve the values of their inputs, we can remove
// the two instances of the involution from the graph, since they cancel
// each other.
class RemoveInvolution : public ArithmeticOptimizerStage {
 public:
  explicit RemoveInvolution(const GraphOptimizerContext& ctx,
                            const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveInvolution", ctx, ctx_ext) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_66(mht_66_v, 1678, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveInvolution");
}
  ~RemoveInvolution() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_67(mht_67_v, 1684, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsInvolution(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_68(mht_68_v, 1691, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* tail = GetTailOfValuePreservingChain(*node, *ctx().node_map,
                                                  *ctx().nodes_to_preserve);

    NodeDef* involution;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &involution));

    if (involution->op() == node->op()) {
      // Skip both *node and *involution since they cancel each other.
      if (tail == node) {
        // The two nodes to eliminate are adjacent.
        *simplified_node_name = involution->input(0);
      } else {
        tail->set_input(0, involution->input(0));
        ctx().node_map->UpdateInput(tail->name(), involution->name(),
                                    involution->input(0));
        *simplified_node_name = node->input(0);
      }
    }

    return Status::OK();
  }
};

// Remove redundant Bitcasts.
// 1) Remove Bitcast whose source type and destination type are equal
// 2) Rewrite Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
class RemoveRedundantBitcastStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantBitcastStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantBitcast", ctx, ctx_ext) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_69(mht_69_v, 1726, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveRedundantBitcastStage");
}
  ~RemoveRedundantBitcastStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_70(mht_70_v, 1732, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsBitcast(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_71(mht_71_v, 1739, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    // Bypass Bitcast whose source type and destination type are equal.
    AttrSlice attrs(*node);
    DataType input_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &input_type));
    DataType output_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "type", &output_type));
    if ((input_type == output_type) && !IsInPreserveSet(*node)) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    NodeDef* bitcast;
    TF_RETURN_IF_ERROR(GetInputNode(node->name(), &bitcast));
    NodeDef* operand;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &operand));

    if (IsBitcast(*operand) && !IsInPreserveSet(*operand)) {
      AttrSlice operand_attrs(*operand);
      DataType operand_input_type;
      TF_RETURN_IF_ERROR(GetNodeAttr(operand_attrs, "T", &operand_input_type));
      // Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
      bitcast->set_input(0, operand->input(0));
      SetDataTypeToAttr(operand_input_type, "T", bitcast);
      ctx().node_map->UpdateInput(bitcast->name(), bitcast->input(0),
                                  operand->input(0));
      AddToOptimizationQueue(bitcast);
      *simplified_node_name = bitcast->name();
    }

    return Status::OK();
  }
};

// Remove Casts whose source type and destination type are equal.
class RemoveRedundantCastStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantCastStage(const GraphOptimizerContext& ctx,
                                    const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantCast", ctx, ctx_ext) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_72(mht_72_v, 1783, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveRedundantCastStage");
}
  ~RemoveRedundantCastStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_73(mht_73_v, 1789, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsCast(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_74(mht_74_v, 1796, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    // Bypass Cast whose source type and destination type are equal.
    AttrSlice attrs(*node);
    DataType input_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "SrcT", &input_type));
    DataType output_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "DstT", &output_type));
    if (input_type == output_type) {
      *simplified_node_name = node->input(0);
    }
    return Status::OK();
  }
};

class RemoveNegationStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveNegationStage(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveNegation", ctx, ctx_ext) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_75(mht_75_v, 1819, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveNegationStage");
}
  ~RemoveNegationStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_76(mht_76_v, 1825, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return (IsAdd(*node) || IsSub(*node)) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_77(mht_77_v, 1832, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* x;
    NodeDef* y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    bool updated = false;
    if (IsNeg(*y)) {
      // a - (-b) = a + b or  a + (-b) = a - b
      ForwardControlDependencies(node, {y});
      ctx().node_map->UpdateInput(node->name(), node->input(1), y->input(0));
      node->set_op(IsAdd(*node) ? "Sub" : "AddV2");
      node->set_input(1, y->input(0));
      updated = true;
    } else if (IsAdd(*node) && IsNeg(*x)) {
      // (-a) + b = b - a
      ForwardControlDependencies(node, {x});
      ctx().node_map->UpdateInput(node->name(), node->input(0), x->input(0));
      node->set_op("Sub");
      node->mutable_input()->SwapElements(0, 1);
      node->set_input(1, x->input(0));
      updated = true;
    }
    if (updated) {
      AddToOptimizationQueue(node);
    }
    return Status::OK();
  }
};

class RemoveLogicalNotStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveLogicalNotStage(const GraphOptimizerContext& ctx,
                                 const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveLogicalNot", ctx, ctx_ext) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_78(mht_78_v, 1868, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveLogicalNotStage");
}
  ~RemoveLogicalNotStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_79(mht_79_v, 1874, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsLogicalNot(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_80(mht_80_v, 1881, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    const string node_name = node->name();
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (IsInPreserveSet(*input) ||
        NumNonControlOutputs(*input, *ctx().node_map) > 1) {
      return Status::OK();
    }
    string new_op;
    if (IsEqual(*input)) {
      new_op = "NotEqual";
    } else if (IsNotEqual(*input)) {
      new_op = "Equal";
    } else if (IsLess(*input)) {
      new_op = "GreaterEqual";
    } else if (IsLessEqual(*input)) {
      new_op = "Greater";
    } else if (IsGreater(*input)) {
      new_op = "LessEqual";
    } else if (IsGreaterEqual(*input)) {
      new_op = "Less";
    }
    if (!new_op.empty()) {
      input->set_op(new_op);
      *simplified_node_name = input->name();
    }
    return Status::OK();
  }
};

// This optimization hoists the common prefix of unary ops of the inputs to
// concat out of the concat, for example:
//    Concat([Exp(Sin(x)), Exp(Sin(y)), Exp(Sin(z))])
// becomes
//    Exp(Sin(Concat([x, y, z]))).
// Similarly, it will hoist the common postfix of unary ops into Split or
// SplitV nodes, for example:
//    [Exp(Sin(y)) for y in Split(x)]
// becomes
//    [y for y in Split(Exp(Sin(x))]
//
// TODO(rmlarsen): Support casting. We would have to change the type attribute
// on the concat/split node.
// TODO(rmlarsen): Handle Enter/Exit.
class HoistCWiseUnaryChainsStage : public ArithmeticOptimizerStage {
 public:
  explicit HoistCWiseUnaryChainsStage(const GraphOptimizerContext& ctx,
                                      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("", ctx, ctx_ext) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_81(mht_81_v, 1932, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HoistCWiseUnaryChainsStage");
}

  ~HoistCWiseUnaryChainsStage() override = default;

  struct ChainLink {
    ChainLink() = default;
    ChainLink(NodeDef* _node, int _port_origin)
        : node(_node), port_origin(_port_origin) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_82(mht_82_v, 1942, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ChainLink");
}
    NodeDef* node;    // Node in a chain.
    int port_origin;  // Port on concat/split node from which this chain
                      // originates.

    bool operator<(const ChainLink& other) const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_83(mht_83_v, 1950, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "operator<");

      if (port_origin < other.port_origin) {
        return true;
      } else if (port_origin > other.port_origin) {
        return false;
      } else {
        return node->name() < other.node->name();
      }
    }
  };

  // We use an ordinary set sorted on port and node name, so the order, and
  // hence the node name used for the hoisted chain, will be deterministic.
  using ChainLinkSet = std::set<ChainLink>;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_84(mht_84_v, 1968, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    if (IsInPreserveSet(*node)) return false;
    if (IsConcat(*node) && node->attr().count("N") != 0) {
      const int n = node->attr().at("N").i();
      return n > 1 && FirstNInputsAreUnique(*node, n);
    } else if ((IsSplit(*node) || IsSplitV(*node)) &&
               node->attr().count("num_split") != 0) {
      const int num_split = node->attr().at("num_split").i();
      if (NumNonControlOutputs(*node, *ctx().node_map) > num_split) {
        // TODO(rmlarsen): Remove this constraint when we have optimizations
        // in place for merging slices into splits.
        return false;
      }
      if (NumControlOutputs(*node, *ctx().node_map) > 0) {
        // TODO(ezhulenev): Unary ops after Split might have a control path to
        // the Split node, and we currently do not properly handle cycles.
        return false;
      }
      return num_split > 1 && !IsAlreadyOptimized(*node);
    }
    return false;
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_85(mht_85_v, 1994, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    node_is_concat_ = IsConcat(*node);
    int prefix_length;
    std::set<string> ctrl_inputs;
    ChainLinkSet tails;
    TF_RETURN_IF_ERROR(
        FindCommonUnaryOpChain(*node, &prefix_length, &tails, &ctrl_inputs));
    if (prefix_length > 0 && !tails.empty()) {
      TF_RETURN_IF_ERROR(
          HoistUnaryOpChain(prefix_length, tails, &ctrl_inputs, node));
    }
    return Status::OK();
  }

 private:
  bool FirstNInputsAreUnique(const NodeDef& node, int n) const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_86(mht_86_v, 2012, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FirstNInputsAreUnique");

    if (n > node.input_size()) return false;
    absl::flat_hash_set<string> unique_inputs;
    const int start = node.op() == "Concat" ? 1 : 0;
    const int end = start + n;
    for (int i = start; i < end; ++i) {
      unique_inputs.insert(node.input(i));
    }
    int unique_input_size = unique_inputs.size();
    return unique_input_size == n;
  }

  // Returns the length of the common unary chain of ops that can be
  // hoisted to the other side of concat or split.
  Status FindCommonUnaryOpChain(const NodeDef& root_node, int* prefix_length,
                                ChainLinkSet* tails,
                                std::set<string>* ctrl_inputs) const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_87(mht_87_v, 2031, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FindCommonUnaryOpChain");

    *prefix_length = 0;
    // Follow the chains starting at each concat input or split output as long
    // as all the following conditions hold:
    //   1. The ops in all chains are the same.
    //   2. The ops are unary elementwise op.
    //   3. The op output has only a single consumer (concat only).
    ChainLinkSet cur_tails;
    TF_RETURN_IF_ERROR(InitializeChains(root_node, &cur_tails));
    if (cur_tails.size() < 2) {
      return Status::OK();
    }
    ctrl_inputs->clear();
    bool stop = false;
    while (!stop && !cur_tails.empty() &&
           OpsAreSafeToHoist(root_node, cur_tails)) {
      // We found one more link that can be hoisted.
      ++(*prefix_length);
      tails->swap(cur_tails);
      GatherControlInputs(ctrl_inputs, *tails);

      // Advance tail pointers to the next level.
      TF_RETURN_IF_ERROR(AdvanceTails(*tails, &cur_tails, &stop));
    }
    return Status::OK();
  }

  // Hoists the chains to the other side of concat or split and attaches the
  // control inputs gathered from them to the concat or split node.
  Status HoistUnaryOpChain(const int prefix_length, const ChainLinkSet& tails,
                           std::set<string>* ctrl_inputs, NodeDef* root_node) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_88(mht_88_v, 2064, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HoistUnaryOpChain");

    VLOG(3) << "Hoist unary op chain:"
            << " root=" << root_node->DebugString()
            << " prefix_length=" << prefix_length << " ctrl_inputs=["
            << absl::StrJoin(*ctrl_inputs, ", ") << "]";

    if (tails.empty()) {
      return Status::OK();
    }
    AddToOptimizationQueue(root_node);
    optimized_nodes_.insert(root_node->name());
    if (node_is_concat_) {
      AddControlInputs(ctrl_inputs, root_node);
      return HoistChainForConcat(prefix_length, tails, root_node);
    } else {
      return HoistChainForSplit(prefix_length, tails, ctrl_inputs, root_node);
    }
  }

  void GatherControlInputs(std::set<string>* ctrl_inputs,
                           const ChainLinkSet& ops) const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_89(mht_89_v, 2087, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GatherControlInputs");

    for (const auto& link : ops) {
      const NodeDef* node = link.node;
      for (int i = node->input_size() - 1; i >= 0; --i) {
        const string& input = node->input(i);
        if (!IsControlInput(input)) break;
        ctrl_inputs->insert(input);
      }
    }
  }

  void AddControlInputs(std::set<string>* new_ctrl_inputs,
                        NodeDef* node) const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_90(mht_90_v, 2102, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddControlInputs");

    for (int i = node->input_size() - 1; i >= 0; --i) {
      const string& existing_input = node->input(i);
      if (!IsControlInput(existing_input)) break;
      new_ctrl_inputs->erase(existing_input);
    }
    for (const string& new_input : *new_ctrl_inputs) {
      ctx().node_map->AddOutput(NodeName(new_input), node->name());
      node->add_input(new_input);
    }
  }

  Status InitializeChains(const NodeDef& node, ChainLinkSet* tails) const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_91(mht_91_v, 2117, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "InitializeChains");

    if (node_is_concat_) {
      // Handle concat nodes by looking backwards in the graph.
      TF_RETURN_IF_ERROR(CheckAttrExists(node, "N"));
      const int n = node.attr().at("N").i();
      const int start = node.op() == "Concat" ? 1 : 0;
      const int end = start + n;
      if (end > node.input_size()) {
        return errors::FailedPrecondition("Got attr N=", n,
                                          " without enough inputs.");
      }
      // Set up tail pointers to point to the immediate inputs to Concat.
      for (int input_port = start; input_port < end; ++input_port) {
        if (IsControlInput(node.input(input_port))) {
          return errors::FailedPrecondition(
              "Got control input ", node.input(input_port),
              " where normal input was expected.");
        }
        NodeDef* tail;
        TF_RETURN_IF_ERROR(GetInputNode(node.input(input_port), &tail));
        tails->insert(ChainLink(tail, input_port));
      }
      return Status::OK();
    } else {
      // Handle split nodes by looking forwards in the graph.
      const auto& outputs = ctx().node_map->GetOutputs(node.name());
      for (NodeDef* output : outputs) {
        if (output->input_size() == 0 || IsControlInput(output->input(0))) {
          continue;
        }
        TensorId tensor_id = ParseTensorName(output->input(0));
        if (tensor_id.node() == node.name()) {
          tails->insert(ChainLink(output, tensor_id.index()));
        } else {
          // This output node has a non-control input other than the split node,
          // abort.
          tails->clear();
          return Status::OK();
        }
      }
    }
    return Status::OK();
  }

  bool OpsAreSafeToHoist(const NodeDef& root_node,
                         const ChainLinkSet& ops) const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_92(mht_92_v, 2165, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "OpsAreSafeToHoist");

    if (ops.empty()) return true;
    const NodeDef* op0 = ops.begin()->node;
    if (ModifiesFrameInfo(*op0) || !IsUnaryElementWise(*op0)) return false;
    for (const auto& link : ops) {
      const NodeDef* op = link.node;
      if (op->device() != root_node.device() || op->op() != op0->op() ||
          IsInPreserveSet(*op)) {
        return false;
      }
      if (ctx().node_map->GetOutputs(op->name()).size() > 1) {
        // TODO(rmlarsen): Allow outgoing control edges.
        return false;
      }
      // Do not hoist Relu if it can be fused with its predecessors. This is
      // important because remapping runs after arithmetic.
      if (IsRelu(*op) || IsRelu6(*op)) {
        NodeDef* operand = nullptr;
        if (!GetInputNode(op->input(0), &operand).ok()) {
          return false;
        }
        if (IsFusedBatchNorm(*operand) || IsBiasAdd(*operand)) {
          return false;
        }
      }
    }
    return true;
  }

  Status AdvanceTails(const ChainLinkSet& tails, ChainLinkSet* new_tails,
                      bool* stop) const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_93(mht_93_v, 2198, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AdvanceTails");

    *stop = true;
    new_tails->clear();
    for (const auto& link : tails) {
      const NodeDef* tail = link.node;
      if (node_is_concat_) {
        if (tail->input_size() == 0 || IsControlInput(tail->input(0))) {
          return Status::OK();
        }
        NodeDef* new_tail;
        TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &new_tail));
        // Remember original port.
        new_tails->insert(ChainLink(new_tail, link.port_origin));
      } else {
        for (NodeDef* new_tail : ctx().node_map->GetOutputs(tail->name())) {
          const TensorId tensor = ParseTensorName(new_tail->input(0));
          if (tensor.node() != tail->name()) {
            return Status::OK();
          }
          // Skip control outputs.
          if (tensor.index() >= 0) {
            // Remember original port.
            new_tails->insert(ChainLink(new_tail, link.port_origin));
          }
        }
      }
    }
    *stop = false;
    return Status::OK();
  }

  Status HoistChainForConcat(const int prefix_length, const ChainLinkSet& tails,
                             NodeDef* concat_node) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_94(mht_94_v, 2233, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HoistChainForConcat");

    const string& concat_name = concat_node->name();
    const int first_input = concat_node->op() == "Concat" ? 1 : 0;
    for (const auto& link : tails) {
      NodeDef* tail = CHECK_NOTNULL(link.node);
      const int concat_port = link.port_origin;
      CHECK_GE(concat_port, 0);
      CHECK_LT(concat_port, concat_node->input_size());
      const string concat_input = concat_node->input(concat_port);
      // Hook the node following tail directly into the concat node.
      const string tail_input = tail->input(0);
      concat_node->set_input(concat_port, tail_input);
      ctx().node_map->UpdateInput(concat_name, concat_input, tail_input);

      if (concat_port == first_input) {
        // Update the consumers of concat to consume the end of the chain
        // instead.
        TF_RETURN_IF_ERROR(UpdateConsumers(concat_node, concat_input));
        // Reuse nodes in the first chain to process output of concat.
        tail->set_input(0, concat_name);
        ctx().node_map->UpdateInput(tail->name(), tail_input, concat_name);
      }
    }
    return Status::OK();
  }

  Status HoistChainForSplit(const int prefix_length, const ChainLinkSet& tails,
                            std::set<string>* ctrl_inputs,
                            NodeDef* split_node) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_95(mht_95_v, 2264, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "HoistChainForSplit");

    // Create a new chain before the split node to process the input tensor.
    const string& split_name = split_node->name();
    auto root_scope_and_name = ParseNodeScopeAndName(split_name);

    // We use the first tail node in the set as a template to get the list of
    // ops to apply (starting from the end).
    NodeDef* cur_tail = tails.begin()->node;
    NodeDef* cur_copy = AddCopyNode(
        OptimizedNodeName(root_scope_and_name, cur_tail->name()), cur_tail);
    cur_copy->clear_input();

    // Update the split to take its input from the tail of the new chain.
    const int value_slot = split_node->op() == "SplitV" ? 0 : 1;
    const string orig_input = split_node->input(value_slot);
    split_node->set_input(value_slot, cur_copy->name());
    ctx().node_map->UpdateInput(split_node->name(), orig_input,
                                cur_copy->name());
    TF_RETURN_IF_ERROR(GetInputNode(cur_tail->input(0), &cur_tail));

    // Now walk backwards creating the rest of the chain.
    while (cur_tail != split_node) {
      NodeDef* new_copy = AddCopyNode(
          OptimizedNodeName(root_scope_and_name, cur_tail->name()), cur_tail);
      new_copy->clear_input();
      cur_copy->add_input(new_copy->name());
      ctx().node_map->AddOutput(new_copy->name(), cur_copy->name());
      cur_copy = new_copy;
      TF_RETURN_IF_ERROR(GetInputNode(cur_tail->input(0), &cur_tail));
    }
    // Connect the original input to the head of the new chain.
    cur_copy->add_input(orig_input);
    ctx().node_map->UpdateOutput(NodeName(orig_input), split_name,
                                 cur_copy->name());
    // Make sure all the control inputs are satisfied before running the first
    // node in the new chain.
    AddControlInputs(ctrl_inputs, cur_copy);

    // Connect all consumers of the tail nodes directly to the
    // output port of Split from which the chain started.
    for (const auto& link : tails) {
      TF_RETURN_IF_ERROR(UpdateConsumers(
          link.node, link.port_origin == 0
                         ? split_name
                         : strings::StrCat(split_name, ":", link.port_origin)));
    }
    return Status::OK();
  }

  bool IsAlreadyOptimized(const NodeDef& node) const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_96(mht_96_v, 2316, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsAlreadyOptimized");

    return optimized_nodes_.find(node.name()) != optimized_nodes_.end();
  }

 private:
  bool node_is_concat_;
  std::unordered_set<string> optimized_nodes_;
};

class RemoveIdempotentStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveIdempotentStage(const GraphOptimizerContext& ctx,
                                 const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveIdempotent", ctx, ctx_ext) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_97(mht_97_v, 2332, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveIdempotentStage");
}
  ~RemoveIdempotentStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_98(mht_98_v, 2338, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return node->input_size() == 1 && IsIdempotent(*node) &&
           !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_99(mht_99_v, 2346, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (input->op() == node->op() && input->device() == node->device()) {
      *simplified_node_name = node->input(0);
    }
    return Status::OK();
  }
};

// Performs the conversion:
// Div(x, Sqrt(y)) => Mul(x, Rsqrt(y))
// TODO(srjoglekar): Generalize to optimize cases like (x / pow(y, z)).
class SqrtDivToRsqrtMulStage : public ArithmeticOptimizerStage {
 public:
  explicit SqrtDivToRsqrtMulStage(const GraphOptimizerContext& ctx,
                                  const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("SqrtDivToRsqrtMul", ctx, ctx_ext) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_100(mht_100_v, 2366, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SqrtDivToRsqrtMulStage");
}
  ~SqrtDivToRsqrtMulStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_101(mht_101_v, 2372, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    // Note: div_no_nan(a, sqrt(b)) => mul_no_nan(a, rsqrt(b))
    // for b == 0 would result in a / Inf instead of 0.
    return IsAnyDiv(*node) && !IsDivNoNan(*node) && !IsFloorDiv(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_102(mht_102_v, 2381, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    // Optimize only if divisor is a Sqrt whose output is not being consumed
    // elsewhere.
    if (IsSqrt(*y) && !IsInPreserveSet(*y) &&
        (NumNonControlOutputs(*y, *ctx().node_map) == 1)) {
      if (IsXdivy(*node)) {
        // xdivy(a, sqrt(b)) => mul_no_nan(rsqrt(b), a)
        node->set_op("MulNoNan");
        node->mutable_input()->SwapElements(0, 1);
      } else {
        // div(a, sqrt(b)) => mul(a, rsqrt(b))
        node->set_op("Mul");
      }
      y->set_op("Rsqrt");
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    }
    return Status::OK();
  }
};

// Performs the following conversion for real types:
//   Square(Sub(x, y)) => Identity(SquaredDifference(x, y) )
class FuseSquaredDiffStage : public ArithmeticOptimizerStage {
 public:
  explicit FuseSquaredDiffStage(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FuseSquaredDiffStage", ctx, ctx_ext) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_103(mht_103_v, 2413, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FuseSquaredDiffStage");
}
  ~FuseSquaredDiffStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_104(mht_104_v, 2419, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsSquare(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_105(mht_105_v, 2426, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* b;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &b));
    // Optimize only if base is a Sub whose output is not being consumed
    // elsewhere.
    if (IsSub(*b) && !IsInPreserveSet(*b) &&
        (NumNonControlOutputs(*b, *ctx().node_map) == 1)) {
      // For complex, SquaredDiff computes conj(x-y)*(x-y), so this rewrite is
      // invalid.
      const DataType type = GetDataTypeFromAttr(*b, "T");
      if ((type == DT_COMPLEX64) || (type == DT_COMPLEX128))
        return Status::OK();
      node->set_op("Identity");
      b->set_op("SquaredDifference");
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(b);
    }
    return Status::OK();
  }
};

// Performs the conversion:
// Log(Softmax(x)) => LogSoftmax(x)
class LogSoftmaxStage : public ArithmeticOptimizerStage {
 public:
  explicit LogSoftmaxStage(const GraphOptimizerContext& ctx,
                           const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("LogSoftmaxStage", ctx, ctx_ext) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_106(mht_106_v, 2456, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "LogSoftmaxStage");
}
  ~LogSoftmaxStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_107(mht_107_v, 2462, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");
 return IsLog(*node); }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_108(mht_108_v, 2467, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* x;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
    // Optimize only if arg is a Softmax whose output is not being consumed
    // elsewhere.
    if (IsSoftmax(*x) && !IsInPreserveSet(*x) &&
        (NumNonControlOutputs(*x, *ctx().node_map) == 1)) {
      // Log(Softmax(x)) => LogSoftmax(Identity(x))
      node->set_op("LogSoftmax");
      x->set_op("Identity");
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(x);
    }
    return Status::OK();
  }
};

// Bypass redundant reshape nodes:
//
//   Reshape                    Reshape  <-+
//      ^                                  |
//      |                                  |
//   Reshape       becomes      Reshape    |
//      ^                                  |
//      |                                  |
//    input                      input  ---+
//
// Additionally,  Reshape and BroadcastTo nodes where the
// input and target shapes are equal are bypassed.
//
class RemoveRedundantReshapeOrBroadcastTo : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantReshapeOrBroadcastTo(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantReshapeOrBroadcastTo", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_109(mht_109_v, 2506, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveRedundantReshapeOrBroadcastTo");
}
  ~RemoveRedundantReshapeOrBroadcastTo() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_110(mht_110_v, 2512, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsReshape(*node) || IsBroadcastTo(*node);
  }

  // TODO(rmlarsen): Handle unary ops with multiple outputs.
  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_111(mht_111_v, 2520, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    // 1. If the reshape is a no-op, forward its input to its consumers, unless
    // it anchors a control dependency since we want to make sure that control
    // dependency is triggered.
    if (!IsInPreserveSet(*node) && InputMatchesTargetShape(*node) &&
        !HasControlInputs(*node)) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    // 2. Bypass reshape followed by reshape, possibly separated by a simple
    // chain of unary elementwise ops that are not outputs.
    if (IsReshape(*node)) {
      bool skip = false;
      gtl::InlinedVector<const NodeDef*, 4> nodes_in_chain;
      const auto predicate_fn = [this, node, &skip,
                                 &nodes_in_chain](const NodeDef& input) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_112(mht_112_v, 2539, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

        nodes_in_chain.push_back(&input);
        if ((input.name() != node->name() &&
             NumNonControlOutputs(input, *ctx().node_map) > 1) ||
            IsInPreserveSet(input) || ModifiesFrameInfo(input)) {
          skip = true;
          return false;
        }
        return IsUnaryElementWise(input);
      };

      // Walk up the input chain until we find a node that is not unary
      // element-wise. If it is another Reshape node, we can bypass it.
      NodeDef* tail =
          GetTailOfChain(*node, *ctx().node_map,
                         /*follow_control_input*/ false, predicate_fn);

      if (!skip && tail != nullptr && !IsInPreserveSet(*tail)) {
        NodeDef* reshape_to_bypass;
        TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &reshape_to_bypass));
        if (reshape_to_bypass == nullptr ||
            (!IsReshape(*reshape_to_bypass) ||
             NumNonControlOutputs(*reshape_to_bypass, *ctx().node_map) > 1 ||
             IsInPreserveSet(*reshape_to_bypass))) {
          return Status::OK();
        }
        // Clearing invalid shape inference results of nodes in chain.
        for (const NodeDef* node_in_chain : nodes_in_chain) {
          ctx().graph_properties->ClearInputProperties(node_in_chain->name());
          if (node_in_chain != node) {
            ctx().graph_properties->ClearOutputProperties(
                node_in_chain->name());
          }
        }
        // We now have
        //    reshape_to_bypass -> tail -> ... -> node
        // where tail maybe equal to node.
        TF_RETURN_IF_ERROR(
            UpdateConsumers(reshape_to_bypass, reshape_to_bypass->input(0)));
        ForwardControlDependencies(tail, {reshape_to_bypass});
        // Change the bypassed reshape to NoOp.
        ReplaceWithNoOp(reshape_to_bypass, ctx());
        *simplified_node_name = node->name();
        return Status::OK();
      }
    }

    return Status::OK();
  }

 private:
  // Returns whether `reshape` is an identity op.
  bool InputMatchesTargetShape(const NodeDef& reshape) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_113(mht_113_v, 2594, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "InputMatchesTargetShape");

    const OpInfo::TensorProperties* reshape_props;
    const OpInfo::TensorProperties* input_props;
    if (!GetTensorProperties(reshape.name(), &reshape_props).ok() ||
        !GetTensorProperties(reshape.input(0), &input_props).ok()) {
      return false;
    }

    return ShapesSymbolicallyEqual(input_props->shape(),
                                   reshape_props->shape());
  }
};

// Reorder casting and value-preserving ops if beneficial.
//
// Original motivation: A common pattern after the layout optimizer is
// casting an uint8 NHWC image to float before transposing it to NCHW. It
// is beneficial to reorder the cast and the transpose to make the transpose
// process smaller amount of data. More generally, this optimization converts
//   Op(Cast(tensor, dst_type))
// to
//   Cast(Op(tensor), dst_type)
// when sizeof(tensor.type) < sizeof(dst_type), and Op is any value-preserving
// Op, i.e. an op that only reorders the elements in its first input. Similarly,
// this optimization converts
//   Cast(Op(tensor), dst_type)
// to
//   Op(Cast(tensor, dst_type))
// when sizeof(tensor.type) > sizeof(dst_type)
//
class ReorderCastLikeAndValuePreserving : public ArithmeticOptimizerStage {
 public:
  explicit ReorderCastLikeAndValuePreserving(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReorderCastLikeAndValuePreserving", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_114(mht_114_v, 2633, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReorderCastLikeAndValuePreserving");
}
  ~ReorderCastLikeAndValuePreserving() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_115(mht_115_v, 2639, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return (IsValuePreserving(*node) || IsCastLike(*node)) &&
           !IsCheckNumerics(*node) && NodeIsOnCpuOrGpu(node) &&
           !IsControlFlow(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* consumer, string* simplified_node_name) override {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_116(mht_116_v, 2648, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* producer;

    if (consumer->input_size() < 1) {
      return errors::FailedPrecondition("Node ", simplified_node_name,
                                        " lacks inputs");
    }

    TF_RETURN_IF_ERROR(GetInputNode(consumer->input(0), &producer));
    const bool producer_is_cast = IsCastLike(*producer);
    const bool can_optimize =
        !IsCheckNumerics(*producer) &&
        ((producer_is_cast && IsValuePreserving(*consumer)) ||
         (IsValuePreserving(*producer) && IsCastLike(*consumer)));
    if (!can_optimize || IsControlFlow(*producer) ||
        IsInPreserveSet(*producer) ||
        producer->device() != consumer->device()) {
      return Status::OK();
    }

    const NodeDef* cast_like_node = producer_is_cast ? producer : consumer;
    const OpDef* cast_like_op_def = nullptr;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(cast_like_node->op(),
                                                         &cast_like_op_def));
    DataType cast_src_type;
    TF_RETURN_IF_ERROR(InputTypeForNode(*cast_like_node, *cast_like_op_def, 0,
                                        &cast_src_type));
    DataType cast_dst_type;
    TF_RETURN_IF_ERROR(OutputTypeForNode(*cast_like_node, *cast_like_op_def, 0,
                                         &cast_dst_type));
    if (!IsFixedSizeType(cast_src_type) || !IsFixedSizeType(cast_dst_type)) {
      return Status::OK();
    } else if (producer_is_cast &&
               DataTypeSize(cast_dst_type) <= DataTypeSize(cast_src_type)) {
      return Status::OK();
    } else if (!producer_is_cast &&
               DataTypeSize(cast_dst_type) >= DataTypeSize(cast_src_type)) {
      return Status::OK();
    }

    // Check that nodes were not already optimized.
    const string optimized_producer_name = OptimizedNodeName(
        ParseNodeScopeAndName(producer->name()), DataTypeString(cast_dst_type));
    const string optimized_consumer_name = OptimizedNodeName(
        ParseNodeScopeAndName(consumer->name()), DataTypeString(cast_src_type));
    const bool is_already_optimized =
        ctx().node_map->NodeExists(optimized_consumer_name) ||
        ctx().node_map->NodeExists(optimized_producer_name);
    if (is_already_optimized) {
      return Status::OK();
    }

    // Add copies of consumer and producer in reverse order.
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(producer->input(0), &input));
    // Create new producer node.
    NodeDef* new_producer = AddCopyNode(optimized_consumer_name, consumer);
    new_producer->set_input(0, producer->input(0));
    ctx().node_map->AddOutput(input->name(), new_producer->name());

    // Create new consumer node.
    NodeDef* new_consumer = AddCopyNode(optimized_producer_name, producer);
    new_consumer->set_input(0, new_producer->name());

    NodeDef* new_value_preserving =
        producer_is_cast ? new_producer : new_consumer;
    const DataType new_input_type =
        producer_is_cast ? cast_src_type : cast_dst_type;
    // Update the input type of the value-preserving node. The input and
    // output types of the cast-like nodes remain the same.
    TF_RETURN_IF_ERROR(SetInputType(new_input_type, new_value_preserving));
    // Make sure there is a kernel registered for the value preserving op
    // with the new input type.
    TF_RETURN_IF_ERROR(IsKernelRegisteredForNode(*new_value_preserving));
    ctx().node_map->AddOutput(new_producer->name(), new_consumer->name());

    AddToOptimizationQueue(new_producer);
    *simplified_node_name = new_consumer->name();

    return Status::OK();
  }

 private:
  // Sets the type of the first input to dtype.
  Status SetInputType(DataType dtype, NodeDef* node) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_117(mht_117_v, 2735, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SetInputType");

    const OpDef* op_def = nullptr;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node->op(), &op_def));
    const OpDef::ArgDef& input_arg = op_def->input_arg(0);
    const string& type_attr_name = input_arg.type_attr();
    if (type_attr_name.empty()) {
      if (input_arg.type() == DT_INVALID || input_arg.type() != dtype) {
        return errors::InvalidArgument("Could not set input type of ",
                                       node->op(), " op to ",
                                       DataTypeString(dtype));
      } else {
        // Op has fixed input type that already matches dtype.
        return Status::OK();
      }
    }
    SetDataTypeToAttr(dtype, type_attr_name, node);
    return Status::OK();
  }
  // This optimization can be dangerous on devices other than CPU and
  // GPU. The transpose might not be implemented for image.type, or
  // might be slower with image.type than with cast_dst_type.
  bool NodeIsOnCpuOrGpu(const NodeDef* node) const {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_118(mht_118_v, 2759, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "NodeIsOnCpuOrGpu");

    using absl::StrContains;

    string task;
    string device;

    return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
           (StrContains(device, DEVICE_CPU) || StrContains(device, DEVICE_GPU));
  }

  bool IsFixedSizeType(DataType dtype) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_119(mht_119_v, 2772, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsFixedSizeType");

    return dtype != DT_STRING && dtype != DT_VARIANT && dtype != DT_RESOURCE &&
           !kQuantizedTypes.Contains(dtype);
  }
};

// Fold a multiply of a scalar into the following convolution. This folding
// can jump across nodes that merely reorders data (such as reshape and
// transpose). For example, we can optimize
//
//
//         Conv2D                             Conv2D
//        /      \                           /      \
//    Transpose  weights*       ->     Transpose    Mul
//       |                                |        /   \
//      Mul                               |    weights  scale
//     /   \                              |
//   input  scale**                     input
//
//  *) weights must be a const
// **) scale must be a const scalar
//
// When `weights` and `scale` are constant, `Mul` in the optimized graph can be
// constant-folded, also weights tend to be smaller than the activations.
//
// TODO(jingyue): Fold scalar multiplies to Conv?DBackpropFilter and
// Conv?DBackpropInput.
class FoldMultiplyIntoConv : public ArithmeticOptimizerStage {
 public:
  explicit FoldMultiplyIntoConv(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldMultiplyIntoConv", ctx, ctx_ext) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_120(mht_120_v, 2806, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FoldMultiplyIntoConv");
}
  ~FoldMultiplyIntoConv() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_121(mht_121_v, 2812, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsConv2D(*node) || IsConv3D(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_122(mht_122_v, 2819, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

#define TF_RETURN_IF_TRUE(...) \
  if ((__VA_ARGS__)) return Status::OK()

    NodeDef* conv = node;

    NodeDef* weights;
    TF_RETURN_IF_ERROR(GetInputNode(conv->input(1), &weights));

    // Fold the multiply to conv only when the weights are constant, so the
    // multiply can be constant-folded.
    //
    // TODO(jingyue): When the weights aren't constant, this should also help
    // performance a bit and memory usage a lot, since the weights tend to be
    // smaller than the activations.
    TF_RETURN_IF_TRUE(!IsConstant(*weights));

    // Verify that this node was not already optimized.
    const string scaled_weights_node_name =
        OptimizedNodeName(ParseNodeScopeAndName(weights->name()),
                          strings::StrCat("scaled", "_", conv->name()));

    TF_RETURN_IF_TRUE(ctx().node_map->NodeExists(scaled_weights_node_name));

    // Find the tail of value preserving chain entering the Conv node.
    NodeDef* tail = GetTailOfValuePreservingChain(*conv, *ctx().node_map,
                                                  *ctx().nodes_to_preserve);

    NodeDef* source;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &source));

    // Check that value preserving chain is the only consumer of the Mul output.
    TF_RETURN_IF_TRUE(!IsAnyMul(*source));
    TF_RETURN_IF_TRUE(NumNonControlOutputs(*source, *ctx().node_map) != 1);
    // And that Mul is not in the preserve set.
    TF_RETURN_IF_TRUE(IsInPreserveSet(*source));

    const NodeDef* mul = source;
    int input_idx = 0;
    int scale_idx = 1;
    NodeDef* scale;  // scalar multiplier for the input tensor
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(mul->input(scale_idx), &scale));
    TF_RETURN_IF_ERROR(GetInputNode(mul->input(input_idx), &input));
    if (!IsConstant(*scale) && IsConstant(*input)) {
      VLOG(3) << "Swapped inputs to mul";
      std::swap(scale_idx, input_idx);
      std::swap(scale, input);
    }
    TF_RETURN_IF_TRUE(!IsConstant(*scale));

    // Check that one of the inputs to mul is a constant scalar.
    const TensorProto& scale_tensor = scale->attr().at("value").tensor();
    bool scale_is_a_scalar = scale_tensor.has_tensor_shape() &&
                             scale_tensor.tensor_shape().dim_size() == 0;
    TF_RETURN_IF_TRUE(!scale_is_a_scalar);

    // Check that 'scale * weight' can be const folded.
    TF_RETURN_IF_TRUE(!IsConstant(*scale));
    TF_RETURN_IF_ERROR(CheckAttrsExist(*scale, {"dtype"}));
    TF_RETURN_IF_ERROR(CheckAttrExists(*weights, "dtype"));
    TF_RETURN_IF_TRUE(scale->attr().at("dtype").type() !=
                      weights->attr().at("dtype").type());

    // At this point all preconditions are met, and we safely do the rewrite.
    VLOG(3) << "Fold multiply into conv: conv=" << conv->name()
            << " mul=" << mul->name() << " weights=" << weights->name();

    // Create new node `scaled_weights`.
    NodeDef* scaled_weights = AddEmptyNode(scaled_weights_node_name);
    scaled_weights->set_op(source->op());
    scaled_weights->set_device(weights->device());
    (*scaled_weights->mutable_attr())["T"] = weights->attr().at("dtype");
    AddToOptimizationQueue(scaled_weights);

    // Link in its inputs.
    scaled_weights->add_input(conv->input(1));
    ctx().node_map->AddOutput(weights->name(), scaled_weights->name());
    scaled_weights->add_input(mul->input(scale_idx));
    ctx().node_map->AddOutput(scale->name(), scaled_weights->name());
    ForwardControlDependencies(scaled_weights, {source});

    // Update `conv`'s weights to `scaled_weights`.
    conv->set_input(1, scaled_weights->name());
    ctx().node_map->UpdateInput(conv->name(), weights->name(),
                                scaled_weights->name());
    AddToOptimizationQueue(conv);

    // Update `tail` node to bypass `mul` because it's folded to the weights.
    tail->set_input(0, mul->input(input_idx));
    ctx().node_map->UpdateInput(tail->name(), mul->name(), input->name());
    AddToOptimizationQueue(tail);
    *simplified_node_name = conv->name();

    return Status::OK();
#undef TF_RETURN_IF_TRUE
  }
};

// Fold Transpose into matrix multiplication.
class FoldTransposeIntoMatMul : public ArithmeticOptimizerStage {
 public:
  explicit FoldTransposeIntoMatMul(const GraphOptimizerContext& ctx,
                                   const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldTransposeIntoMatMul", ctx, ctx_ext) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_123(mht_123_v, 2926, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FoldTransposeIntoMatMul");
}
  ~FoldTransposeIntoMatMul() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_124(mht_124_v, 2932, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAnyMatMul(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_125(mht_125_v, 2939, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    const NodeScopeAndName matmul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(matmul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    NodeDef* a;
    NodeDef* b;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &a));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &b));

    bool is_complex = false;
    if (node->op() != "SparseMatMul") {
      const DataType type = GetDataTypeFromAttr(*node, "T");
      is_complex = (type == DT_COMPLEX64) || (type == DT_COMPLEX128);
    }

    const std::set<string> foldable_transpose_ops =
        !is_complex
            ? std::set<string>{"ConjugateTranspose", "Transpose"}
            : (IsAnyBatchMatMul(*node) ? std::set<string>{"ConjugateTranspose"}
                                       : std::set<string>{"Transpose"});

    const bool a_is_foldable = foldable_transpose_ops.count(a->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*a, ctx().node_map);
    const bool b_is_foldable = foldable_transpose_ops.count(b->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*b, ctx().node_map);
    if (!a_is_foldable && !b_is_foldable) return Status::OK();

    NodeDef* new_op = AddCopyNode(optimized_node_name, node);

    if (a_is_foldable) {
      const string attr_a = IsAnyBatchMatMul(*node) ? "adj_x" : "transpose_a";
      FlipBooleanAttr(attr_a, new_op);
      new_op->set_input(0, a->input(0));
      ctx().node_map->UpdateInput(new_op->name(), a->name(), a->input(0));
    } else {
      ctx().node_map->UpdateOutput(a->name(), node->name(), new_op->name());
    }

    if (b_is_foldable) {
      const string attr_b = IsAnyBatchMatMul(*node) ? "adj_y" : "transpose_b";
      FlipBooleanAttr(attr_b, new_op);
      new_op->set_input(1, b->input(0));
      ctx().node_map->UpdateInput(new_op->name(), b->name(), b->input(0));
    } else {
      ctx().node_map->UpdateOutput(b->name(), node->name(), new_op->name());
    }

    std::vector<const NodeDef*> deps_to_forward = {node};
    if (a_is_foldable) deps_to_forward.push_back(a);
    if (b_is_foldable) deps_to_forward.push_back(b);
    ForwardControlDependencies(new_op, deps_to_forward);
    *simplified_node_name = new_op->name();

    return Status::OK();
  }

 private:
  void FlipBooleanAttr(const string& attr_name, NodeDef* node) {
   std::vector<std::string> mht_126_v;
   mht_126_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_126(mht_126_v, 3001, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FlipBooleanAttr");

    const bool old_value =
        !node->attr().count(attr_name) ? false : node->attr().at(attr_name).b();
    (*node->mutable_attr())[attr_name].set_b(!old_value);
  }

  template <typename T>
  bool IsInnerMatrixTranspose(const std::vector<T>& perm) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_127(mht_127_v, 3011, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsInnerMatrixTranspose");

    const T n = perm.size();
    if (n < 2) {
      return false;
    }
    for (T i = 0; i < n - 2; ++i) {
      if (perm[i] != i) {
        return false;
      }
    }
    return perm[n - 1] == n - 2 && perm[n - 2] == n - 1;
  }

  bool IsInnerMatrixTransposeNode(const NodeDef& transpose_node,
                                  const NodeMap* node_map) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_128(mht_128_v, 3028, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsInnerMatrixTransposeNode");

    if (transpose_node.op() != "Transpose" &&
        transpose_node.op() != "ConjugateTranspose") {
      return false;
    }
    const NodeDef* perm_node = node_map->GetNode(transpose_node.input(1));
    std::vector<int> perm32;
    if (ValuesFromConstNode(*perm_node, &perm32)) {
      return IsInnerMatrixTranspose(perm32);
    }
    std::vector<int64_t> perm64;
    if (ValuesFromConstNode(*perm_node, &perm64)) {
      return IsInnerMatrixTranspose(perm64);
    }
    return false;
  }
};

class FoldConjugateIntoTranspose : public ArithmeticOptimizerStage {
 public:
  explicit FoldConjugateIntoTranspose(const GraphOptimizerContext& ctx,
                                      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldConjugateIntoTranspose", ctx, ctx_ext) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_129(mht_129_v, 3053, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FoldConjugateIntoTranspose");
}
  ~FoldConjugateIntoTranspose() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_130(mht_130_v, 3059, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsConj(*node) || IsTranspose(*node) || IsConjugateTranspose(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_131(mht_131_v, 3066, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    const NodeScopeAndName matmul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(matmul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));

    const NodeDef* transpose_op = node->op() == "Conj" ? input : node;
    const NodeDef* conj_op = node->op() == "Conj" ? node : input;

    if ((IsTranspose(*transpose_op) || IsConjugateTranspose(*transpose_op)) &&
        IsConj(*conj_op)) {
      NodeDef* new_op = AddCopyNode(optimized_node_name, transpose_op);

      // Flip the type of transpose op to absorb the conjugation.
      new_op->set_op(transpose_op->op() == "Transpose" ? "ConjugateTranspose"
                                                       : "Transpose");
      new_op->set_input(0, input->input(0));
      ctx().node_map->UpdateInput(new_op->name(), node->name(),
                                  input->input(0));
      ForwardControlDependencies(new_op, {node, input});
      *simplified_node_name = new_op->name();
    }

    return Status::OK();
  }
};

// Replace Mul node with identical inputs with a Square.
class ReplaceMulWithSquare : public ArithmeticOptimizerStage {
 public:
  explicit ReplaceMulWithSquare(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReplaceMulWithSquare", ctx, ctx_ext) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_132(mht_132_v, 3103, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReplaceMulWithSquare");
}
  ~ReplaceMulWithSquare() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_133(mht_133_v, 3109, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    if (!node || node->input_size() < 2) {
      // Invalid node
      return false;
    }

    return IsAnyMul(*node) && node->input(0) == node->input(1);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_134(mht_134_v, 3121, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    const NodeScopeAndName mul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(mul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    const DataType type = GetDataTypeFromAttr(*node, "T");
    bool is_complex = (type == DT_COMPLEX64) || (type == DT_COMPLEX128);

    if (!is_complex || NodeIsOnCpu(*node)) {
      NodeDef* new_square_node = AddCopyNode(optimized_node_name, node);
      new_square_node->set_op("Square");
      for (int i = 1; i < new_square_node->input_size(); ++i) {
        new_square_node->set_input(i - 1, new_square_node->input(i));
      }
      new_square_node->mutable_input()->RemoveLast();
      for (const string& input : new_square_node->input()) {
        ctx().node_map->AddOutput(NodeName(input), new_square_node->name());
      }
      *simplified_node_name = new_square_node->name();
    }

    return Status::OK();
  }
};

// Replace a combination of Mul with broadcasting by Tile. E.g. replace
//
// input(1x22x1x48x1x64) -> Mul (1x22x2x48x2x64) -> output
// Ones (1x22x2x48x2x64) -^
//
// with
//
// input -> Tile(1x22x2x48x2x64) -> output
class ReplaceMulWithBroadcastByTile : public ArithmeticOptimizerStage {
 public:
  explicit ReplaceMulWithBroadcastByTile(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReplaceMulWithBroadcastByTile", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_135(mht_135_v, 3163, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReplaceMulWithBroadcastByTile");
}
  ~ReplaceMulWithBroadcastByTile() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_136(mht_136_v, 3169, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsMul(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_137(mht_137_v, 3176, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef *input, *ones;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &ones));
    if (IsInPreserveSet(*node) || IsInPreserveSet(*input) ||
        IsInPreserveSet(*ones)) {
      return Status::OK();
    }

    // TODO(kkiningh): Generalize using IsOnes from constant_folding.cc
    if (IsConstant(*input) || !IsOnes(*ones)) return Status::OK();

    // Avoid optimizing the same node twice
    const NodeScopeAndName scope_and_name = ParseNodeScopeAndName(node->name());
    const string tile_node_name = OptimizedNodeName(scope_and_name, "Tile");
    const string const_node_name = OptimizedNodeName(scope_and_name, "Const");
    if (ctx().node_map->NodeExists(tile_node_name) ||
        ctx().node_map->NodeExists(const_node_name)) {
      return Status::OK();
    }

    const std::vector<OpInfo::TensorProperties>& props =
        ctx().graph_properties->GetInputProperties(node->name());
    if (props.size() != 2) return Status::OK();

    // Ignore ops where the shape doesn't change
    const TensorShapeProto& input_shape = props[0].shape();
    const TensorShapeProto& ones_shape = props[1].shape();
    TensorShapeProto output_shape;
    if (!ShapeAfterBroadcast(input_shape, ones_shape, &output_shape)) {
      return Status::OK();
    }
    if (ShapesSymbolicallyEqual(input_shape, output_shape)) {
      return Status::OK();
    }

    // All inputs must have same input/output dimensions
    if (input_shape.dim_size() != output_shape.dim_size() ||
        ones_shape.dim_size() != output_shape.dim_size())
      return Status::OK();

    // At this point all preconditions are met. Can proceed with rewrite.
    VLOG(3) << "Simplify multiply with all ones input: node=" << node->name()
            << "@" << output_shape << " ones=" << ones->name() << "@"
            << ones_shape << " input=" << input->name() << "@" << input_shape;

    // 1. Create constant node with correct tile multiples
    Tensor multiples(DT_INT32, TensorShape({output_shape.dim_size()}));
    for (int i = 0; i < output_shape.dim_size(); ++i) {
      int64_t size = output_shape.dim(i).size() / input_shape.dim(i).size();
      if (TF_PREDICT_FALSE(size >= INT_MAX)) {
        return Status(error::OUT_OF_RANGE, "int32 overflow");
      }
      multiples.flat<int32>()(i) = static_cast<int32>(size);
    }

    NodeDef* const_node = AddEmptyNode(const_node_name);
    TF_RETURN_IF_ERROR(ConstantFolding::CreateNodeDef(
        const_node->name(), TensorValue(&multiples), const_node));
    const_node->set_device(node->device());
    ForwardControlDependencies(const_node, {ones});
    AddToOptimizationQueue(const_node);

    // 2. Replace multiply node with Tile(Const, input);
    const DataType type = GetDataTypeFromAttr(*node, "T");
    NodeDef* tile_node = AddEmptyNode(tile_node_name);
    tile_node->set_op("Tile");
    tile_node->set_device(node->device());
    SetDataTypeToAttr(type, "T", tile_node);
    SetDataTypeToAttr(DT_INT32, "Tmultiples", tile_node);
    tile_node->add_input(input->name());
    tile_node->add_input(const_node->name());

    ForwardControlDependencies(tile_node, {node});
    *simplified_node_name = tile_node->name();

    return Status::OK();
  }

 protected:
  bool IsOnes(const NodeDef& node) const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_138(mht_138_v, 3259, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsOnes");

    if (!IsReallyConstant(node)) return false;
    if (node.attr().at("dtype").type() != DT_FLOAT) return false;

    Tensor tensor;
    if (!tensor.FromProto(node.attr().at("value").tensor())) {
      return false;
    }

    auto values = tensor.flat<float>();
    for (int i = 0; i < tensor.NumElements(); ++i) {
      if (values(i) != 1.0f) {
        return false;
      }
    }

    return true;
  }
};

// Image upsampling often produces an unnecessary reshape that is difficult to
// eliminate in other stages. This stage reduces the number of dimensions
// involved allowing the reshape to be removed.
//
// For example, given
//   B,W,H,C -> Reshape(B,W,1,H,1,C) -> Tile(1,1,2,1,2,1) -> Reshape(B,2W,2H,C)
// this pass converts the sequence to
//   B,W,H,C -> Reshape(B,W,H,C) -> Tile(1,1,2,2) -> Reshape(B,2W,2H,C)
//
// The first reshape is now redundant and can be removed in a later pass.
//
// Note: This only optimizes the simple (but extremely common) case of 2D
// upsampling.
//
// TODO(kkiningh): Generalize to more complex upsampling patterns.
class ReduceUpsamplingDims : public ArithmeticOptimizerStage {
 public:
  explicit ReduceUpsamplingDims(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReduceUpsamplingDims", ctx, ctx_ext) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_139(mht_139_v, 3301, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReduceUpsamplingDims");
}
  ~ReduceUpsamplingDims() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_140(mht_140_v, 3307, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsReshape(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_141(mht_141_v, 3314, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* tile;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &tile));
    if (!IsTile(*tile) || IsInPreserveSet(*tile)) {
      return Status::OK();
    }

    if (NumNonControlOutputs(*tile, *ctx().node_map) != 1) {
      // Optimization is only worthwile when there is a single output from Tile.
      // Otherwise, we need to insert additional Reshape ops that can't be
      // easily removed.
      return Status::OK();
    }

    NodeDef* reshape;
    TF_RETURN_IF_ERROR(GetInputNode(tile->input(0), &reshape));
    if (!IsReshape(*reshape) || IsInPreserveSet(*reshape)) {
      return Status::OK();
    }

    NodeDef* multiples;
    TF_RETURN_IF_ERROR(GetInputNode(tile->input(1), &multiples));

    NodeDef* shape;
    TF_RETURN_IF_ERROR(GetInputNode(reshape->input(1), &shape));

    // Avoid optimizing the same nodes twice
    const NodeScopeAndName scope_and_name = ParseNodeScopeAndName(node->name());
    const string new_reshape_name =
        OptimizedNodeName(scope_and_name, "Reshape");
    const string new_tile_name = OptimizedNodeName(scope_and_name, "Tile");
    const string new_multiples_name =
        OptimizedNodeName(scope_and_name, "Multiples");
    const string new_shape_name = OptimizedNodeName(scope_and_name, "Shape");
    if (ctx().node_map->NodeExists(new_reshape_name) ||
        ctx().node_map->NodeExists(new_tile_name) ||
        ctx().node_map->NodeExists(new_shape_name) ||
        ctx().node_map->NodeExists(new_multiples_name)) {
      return Status::OK();
    }

    // Compuate updated multiples/shape values.
    AttrValue new_multiples_attr;
    if (!CreateUpdatedMultiplesProto(multiples,
                                     new_multiples_attr.mutable_tensor())) {
      return Status::OK();
    }
    AttrValue new_shape_attr;
    if (!CreateUpdatedShapeProto(shape, new_shape_attr.mutable_tensor())) {
      return Status::OK();
    }

    // At this point the graph is validated and can be updated
    // Note: We can assume shape/multiples are DT_INT32 only at this point since
    // they're checked in CreateUpdated*Proto()

    // 1. Create the constant nodes used by the new Reshape/Tile nodes
    NodeDef* new_multiples = AddEmptyNode(new_multiples_name);
    new_multiples->set_op("Const");
    SetDataTypeToAttr(DT_INT32, "dtype", new_multiples);
    new_multiples->mutable_attr()->insert({"value", new_multiples_attr});
    new_multiples->set_device(multiples->device());

    NodeDef* new_shape = AddEmptyNode(new_shape_name);
    new_shape->set_op("Const");
    SetDataTypeToAttr(DT_INT32, "dtype", new_shape);
    new_shape->mutable_attr()->insert({"value", new_shape_attr});
    new_shape->set_device(shape->device());

    // 2. Create the new Reshape/Tile nodes
    NodeDef* new_reshape = AddEmptyNode(new_reshape_name);
    CopyReshapeWithInput(reshape, new_reshape, /*input=*/reshape->input(0),
                         /*shape=*/new_shape->name());
    NodeDef* new_tile = AddEmptyNode(new_tile_name);
    CopyTileWithInput(tile, new_tile, /*input=*/new_reshape->name(),
                      /*multiples=*/new_multiples->name());

    // 3. Update consumer of original Tile node and add control
    node->set_input(0, new_tile->name());
    ctx().node_map->UpdateInput(node->name(), tile->name(), new_tile->name());

    ForwardControlDependencies(new_tile, {tile});
    ForwardControlDependencies(new_multiples, {multiples});
    ForwardControlDependencies(new_reshape, {reshape});
    ForwardControlDependencies(new_shape, {shape});

    *simplified_node_name = node->name();
    return Status::OK();
  }

 private:
  bool CreateUpdatedMultiplesProto(const NodeDef* node, TensorProto* proto) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_142(mht_142_v, 3408, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CreateUpdatedMultiplesProto");

    Tensor multiples;
    if (!GetTensorFromConstNode(node->name(), &multiples)) {
      return false;
    }

    // Dimensions should be [X, Y, N, 1, M, 1]
    if (multiples.dtype() != DT_INT32 || multiples.NumElements() != 6) {
      return false;
    }

    const auto& multiples_values = multiples.flat<int32>();
    if (multiples_values(3) != 1 || multiples_values(5) != 1) {
      return false;
    }

    // Convert to [X, Y, N, M]
    Tensor new_multiples(DT_INT32, {4});
    new_multiples.flat<int32>()(0) = multiples_values(0);
    new_multiples.flat<int32>()(1) = multiples_values(1);
    new_multiples.flat<int32>()(2) = multiples_values(2);
    new_multiples.flat<int32>()(3) = multiples_values(4);

    new_multiples.AsProtoTensorContent(proto);
    return true;
  }

  bool CreateUpdatedShapeProto(const NodeDef* node, TensorProto* proto) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_143(mht_143_v, 3438, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CreateUpdatedShapeProto");

    Tensor shape;
    if (!GetTensorFromConstNode(node->name(), &shape)) {
      return false;
    }

    // Dimensions should be [B, W, 1, H, 1, C]
    if (shape.dtype() != DT_INT32 || shape.NumElements() != 6) {
      return false;
    }

    const auto& shape_values = shape.flat<int32>();
    if (shape_values(2) != 1 || shape_values(4) != 1) {
      return false;
    }

    // Convert to [B, W, H, C]
    Tensor new_shape(DT_INT32, {4});
    new_shape.flat<int32>()(0) = shape_values(0);
    new_shape.flat<int32>()(1) = shape_values(1);
    new_shape.flat<int32>()(2) = shape_values(3);
    new_shape.flat<int32>()(3) = shape_values(5);

    new_shape.AsProtoTensorContent(proto);
    return true;
  }

  void CopyReshapeWithInput(const NodeDef* reshape, NodeDef* new_reshape,
                            const string& input, const string& shape) {
   std::vector<std::string> mht_144_v;
   mht_144_v.push_back("input: \"" + input + "\"");
   mht_144_v.push_back("shape: \"" + shape + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_144(mht_144_v, 3471, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CopyReshapeWithInput");

    new_reshape->set_op("Reshape");
    new_reshape->set_device(reshape->device());
    SetDataTypeToAttr(GetDataTypeFromAttr(*reshape, "T"), "T", new_reshape);
    SetDataTypeToAttr(GetDataTypeFromAttr(*reshape, "Tshape"), "Tshape",
                      new_reshape);

    new_reshape->add_input(input);
    ctx().node_map->AddOutput(NodeName(input), new_reshape->name());
    new_reshape->add_input(shape);
    ctx().node_map->AddOutput(NodeName(shape), new_reshape->name());

    AddToOptimizationQueue(new_reshape);
  }

  void CopyTileWithInput(const NodeDef* tile, NodeDef* new_tile,
                         const string& input, const string& multiples) {
   std::vector<std::string> mht_145_v;
   mht_145_v.push_back("input: \"" + input + "\"");
   mht_145_v.push_back("multiples: \"" + multiples + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_145(mht_145_v, 3492, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CopyTileWithInput");

    new_tile->set_op("Tile");
    new_tile->set_device(tile->device());
    SetDataTypeToAttr(GetDataTypeFromAttr(*tile, "T"), "T", new_tile);
    SetDataTypeToAttr(GetDataTypeFromAttr(*tile, "Tmultiples"), "Tmultiples",
                      new_tile);

    new_tile->add_input(input);
    ctx().node_map->AddOutput(NodeName(input), new_tile->name());
    new_tile->add_input(multiples);
    ctx().node_map->AddOutput(NodeName(multiples), new_tile->name());

    AddToOptimizationQueue(new_tile);
  }
};

// Replace a sequence of Pack nodes with identical inputs with Tile
// For example, given a Tensor X with shape (I,J,K)
// Let P(x, n) = Pack([x, x], axis=n)
//
// P(P(X, 2), 1)
//   = Tile(Reshape(Tile(Reshape(x,
//              [I,    J, 1, K]), [1,    1, 2, 1]),
//              [I, 1, J, 2, K]), [1, 2, 1, 1, 1]))
//   = Tile(Reshape(x,
//              [I, 1, J, 1, K]), [1, 2, 1, 2, 1])
//   = Reshape(Tile(x, [1, 2, 2]), [I, 2, J, 2, K])
//
// The outermost reshape is often redundant and can be removed in another pass
class ReplacePackWithTileReshape : public ArithmeticOptimizerStage {
 public:
  explicit ReplacePackWithTileReshape(const GraphOptimizerContext& ctx,
                                      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReplacePackWithTileReshape", ctx, ctx_ext) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_146(mht_146_v, 3528, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ReplacePackWithTileReshape");
}
  ~ReplacePackWithTileReshape() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_147(mht_147_v, 3534, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsPack(*node) && NumNonControlInputs(*node) > 1 &&
           !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_148(mht_148_v, 3542, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    // 1. traverse the chain of Pack ops to get the original input
    NodeDef* input = node;
    std::vector<const NodeDef*> chain;
    while (IsPack(*input) && NumNonControlInputs(*node) > 1 &&
           !IsInPreserveSet(*input)) {
      // Only pack operations with all identical inputs are supported
      if (!AllRegularInputsEqual(*input)) {
        break;
      }
      chain.push_back(input);
      TF_RETURN_IF_ERROR(GetInputNode(input->input(0), &input));
    }

    // Must be at least two Pack operations to consider for replacement
    if (chain.empty()) {
      return Status::OK();
    }

    // Avoid optimizing the same node twice
    const NodeScopeAndName node_scope_and_name =
        ParseNodeScopeAndName(node->name());
    const string new_const_name =
        OptimizedNodeName(node_scope_and_name, "Multiples");
    const string new_tile_name = OptimizedNodeName(node_scope_and_name, "Tile");
    const string new_shape_name =
        OptimizedNodeName(node_scope_and_name, "Shape");
    const string new_reshape_name =
        OptimizedNodeName(node_scope_and_name, "Reshape");
    if (ctx().node_map->NodeExists(new_const_name) ||
        ctx().node_map->NodeExists(new_tile_name) ||
        ctx().node_map->NodeExists(new_shape_name) ||
        ctx().node_map->NodeExists(new_reshape_name)) {
      return Status::OK();
    }

    // 2. Calculate the multiples and shape tensor using the chain
    const OpInfo::TensorProperties* input_props;
    TF_RETURN_IF_ERROR(GetTensorProperties(input->name(), &input_props));
    const TensorShapeProto& input_shape = input_props->shape();
    if (!PartialTensorShape(input_shape).IsFullyDefined()) {
      return Status::OK();
    }
    Tensor multiples(DT_INT32, TensorShape({input_shape.dim_size()}));
    TF_RETURN_IF_ERROR(CalculateMultiplesFromChain(chain, &multiples));

    const OpInfo::TensorProperties* output_props;
    TF_RETURN_IF_ERROR(GetTensorProperties(node->name(), &output_props));
    const TensorShapeProto& output_shape = output_props->shape();
    if (!PartialTensorShape(output_shape).IsFullyDefined()) {
      return Status::OK();
    }
    Tensor output_shape_tensor(DT_INT32,
                               TensorShape({output_shape.dim_size()}));
    for (int i = 0; i < output_shape.dim_size(); ++i) {
      output_shape_tensor.flat<int32>()(i) = output_shape.dim(i).size();
    }

    // 3. Create constant node with correct multiples value
    NodeDef* new_const_node = AddEmptyNode(new_const_name);
    TF_RETURN_IF_ERROR(ConstantFolding::CreateNodeDef(
        new_const_node->name(), TensorValue(&multiples), new_const_node));
    new_const_node->set_device(node->device());
    MaybeAddControlInput(input->name(), new_const_node, ctx().optimized_graph,
                         ctx().node_map);
    AddToOptimizationQueue(new_const_node);

    // 4. Replace the Pack node with Tile(Const(N), input);
    DataType dtype = GetDataTypeFromAttr(*node, "T");
    NodeDef* new_tile_node = AddEmptyNode(new_tile_name);
    new_tile_node->set_op("Tile");
    new_tile_node->set_device(node->device());
    SetDataTypeToAttr(dtype, "T", new_tile_node);
    SetDataTypeToAttr(DT_INT32, "Tmultiples", new_tile_node);
    new_tile_node->add_input(input->name());
    ctx().node_map->AddOutput(input->name(), new_tile_node->name());
    new_tile_node->add_input(new_const_node->name());
    ctx().node_map->AddOutput(new_const_node->name(), new_tile_node->name());

    // Tile inherits all control dependencies from the original pack chain
    ForwardControlDependencies(new_tile_node, chain);
    AddToOptimizationQueue(new_tile_node);

    // 5. Add a new Reshape node to preserve the existing shape
    NodeDef* new_shape_node = AddEmptyNode(new_shape_name);
    TF_RETURN_IF_ERROR(ConstantFolding::CreateNodeDef(
        new_shape_node->name(), TensorValue(&output_shape_tensor),
        new_shape_node));
    new_shape_node->set_device(node->device());
    MaybeAddControlInput(input->name(), new_shape_node, ctx().optimized_graph,
                         ctx().node_map);
    AddToOptimizationQueue(new_shape_node);

    NodeDef* new_reshape_node = AddEmptyNode(new_reshape_name);
    new_reshape_node->set_op("Reshape");
    new_reshape_node->set_device(node->device());
    SetDataTypeToAttr(dtype, "T", new_reshape_node);
    SetDataTypeToAttr(DT_INT32, "Tshape", new_reshape_node);
    new_reshape_node->add_input(new_tile_node->name());
    ctx().node_map->AddOutput(new_tile_node->name(), new_reshape_node->name());
    new_reshape_node->add_input(new_shape_node->name());
    ctx().node_map->AddOutput(new_shape_node->name(), new_reshape_node->name());

    *simplified_node_name = new_reshape_node->name();

    return Status::OK();
  }

 protected:
  Status CalculateMultiplesFromChain(const std::vector<const NodeDef*>& chain,
                                     Tensor* multiples) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_149(mht_149_v, 3655, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CalculateMultiplesFromChain");

    // Keep track of how the multiples correspond to each shape dimension.
    // For example, given Stack([x, x], axis=1) with rank(x) = 3, we start with
    //    multiples=[1, 1, 1] , dims=[0, 1, 2]
    // After processing the stack op
    //    multiples=[1, 2, 1] , dims=[0, 1, 1, 2]
    std::vector<int32> dims(multiples->NumElements());
    std::iota(dims.begin(), dims.end(), 0);

    for (int i = 0; i < multiples->NumElements(); ++i) {
      multiples->flat<int32>()(i) = 1;
    }

    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      AttrSlice attrs(**it);
      int64_t axis, n;
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "axis", &axis));
      TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "N", &n));

      if (axis >= dims.size()) {
        // We don't handle the case where Pack is performed on the last axis,
        // e.g. Pack([x, x], axis=3) where rank(x) == 3
        return Status(error::OUT_OF_RANGE, "axis value out of range of dims");
      }

      int64_t m = multiples->flat<int32>()(dims[axis]) * n;
      if (TF_PREDICT_FALSE(m > INT_MAX)) {
        return Status(error::OUT_OF_RANGE, "int32 overflow");
      }
      multiples->flat<int32>()(dims[axis]) = static_cast<int32>(m);

      // Copy index from immediate right of inserted axis
      dims.insert(dims.begin() + axis, dims[axis]);
    }

    return Status::OK();
  }
};

// Simplify aggregation (e.g. AddN) nodes:
//
// 1. Discard aggregate nodes with a single input and no control dependencies.
//
// 2. Try to rewrite aggregations of N >= 2 identical terms (possibly due to
//    deduping or other rewrites) so we can get rid of the sum entirely.
//
//    The expression (using AddN as an example of an aggregate op):
//      AddN(x, x, x, ... ,x)
//           <-- N terms -->
//    can be rewritten to:
//      Mul(Const(N), x))
//
class SimplifyAggregation : public ArithmeticOptimizerStage {
 public:
  explicit SimplifyAggregation(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("SimplifyAggregation", ctx, ctx_ext) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_150(mht_150_v, 3714, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SimplifyAggregation");
}
  ~SimplifyAggregation() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_151(mht_151_v, 3720, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAggregate(*node) && HasRegularInputs(*node) &&
           GetDataTypeFromAttr(*node, "T") !=
               DT_VARIANT;  // TODO(b/119787146): Enable for variants.
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_152(mht_152_v, 3729, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    // 1. Discard aggregate nodes with a single input and no control deps.
    if (node->input_size() == 1) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    // 2. Rewrite aggregations of N >= 2 identical terms.

    // All non-control inputs must be identical.
    bool all_equal = true;
    int num_inputs = 1;
    for (int i = 1; i < node->input_size(); ++i) {
      if (IsControlInput(node->input(i))) break;
      ++num_inputs;
      if (node->input(i) != node->input(0)) {
        all_equal = false;
        break;
      }
    }
    if (!all_equal) return Status::OK();

    // And node should not be optimized earlier.
    const NodeScopeAndName node_scope_and_name =
        ParseNodeScopeAndName(node->name());
    const string optimized_const_name =
        OptimizedNodeName(node_scope_and_name, "Const");
    const string optimized_mul_name =
        OptimizedNodeName(node_scope_and_name, "Mul");

    bool is_already_optimized =
        ctx().node_map->NodeExists(optimized_const_name) ||
        ctx().node_map->NodeExists(optimized_mul_name);

    if (is_already_optimized) return Status::OK();

    // At this point all preconditions are met, and we safely do the rewrite.
    VLOG(3) << "Simplify aggregation with identical inputs: node="
            << node->name() << " num_inputs=" << num_inputs;

    // 1. Create constant node with value N.
    const auto type = GetDataTypeFromAttr(*node, "T");
    Tensor t(type, TensorShape({}));
    Status status = SetTensorValue(type, num_inputs, &t);
    if (!status.ok()) {
      return errors::Internal("Failed to create const node: ",
                              status.error_message());
    }

    TensorValue value(&t);
    NodeDef* new_const_node = AddEmptyNode(optimized_const_name);
    status = ConstantFolding::CreateNodeDef(new_const_node->name(), value,
                                            new_const_node);
    if (!status.ok()) {
      return errors::Internal("Failed to create const node: ",
                              status.error_message());
    }
    new_const_node->set_device(node->device());
    MaybeAddControlInput(NodeName(node->input(0)), new_const_node,
                         ctx().optimized_graph, ctx().node_map);
    AddToOptimizationQueue(new_const_node);

    // 2. Replace the aggregate node with Mul(Const(N), x).
    NodeDef* new_mul_node = AddEmptyNode(optimized_mul_name);
    new_mul_node->set_op("Mul");
    new_mul_node->set_device(node->device());
    SetDataTypeToAttr(type, "T", new_mul_node);
    new_mul_node->add_input(new_const_node->name());
    ctx().node_map->AddOutput(new_const_node->name(), new_mul_node->name());
    new_mul_node->add_input(node->input(0));
    ctx().node_map->AddOutput(node->input(0), new_mul_node->name());

    ForwardControlDependencies(new_mul_node, {node});
    *simplified_node_name = new_mul_node->name();

    return Status::OK();
  }
};

class ConvertPowStage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertPowStage(const GraphOptimizerContext& ctx,
                           const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertPow", ctx, ctx_ext) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_153(mht_153_v, 3815, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ConvertPowStage");
}

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_154(mht_154_v, 3820, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsPow(*node) &&
           ctx().graph_properties->HasOutputProperties(node->name()) &&
           ctx().graph_properties->HasInputProperties(node->name());
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_155(mht_155_v, 3829, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    Tensor pow;
    if (!GetTensorFromConstNode(node->input(1), &pow)) return Status::OK();
    complex128 prev, curr;
    for (int i = 0; i < pow.NumElements(); ++i) {
      if (!GetElementUnexhaustive(pow, i, {pow.dtype()}, &curr)) {
        // input data type is not supported by Pow. Skip.
        return Status::OK();
      }
      if (i != 0 && curr != prev) {
        // pow has different values on different elements. Skip.
        return Status::OK();
      }
      prev = curr;
    }
    NodeDef *x, *y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    const auto& value_props =
        ctx().graph_properties->GetInputProperties(node->name())[0];
    const TensorShapeProto& output_shape =
        ctx().graph_properties->GetOutputProperties(node->name())[0].shape();
    if (curr == complex128(2, 0)) {
      node->set_op("Square");
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    } else if (curr == complex128(3, 0)) {
      // TODO(courbet): Use 'Cube' when it's added to TF ops.
      if (NodeIsOnCpu(*node)) {
        // We create an inner square node: inner_square = square(x)
        const NodeScopeAndName scope_and_name =
            ParseNodeScopeAndName(node->name());
        const string inner_square_name =
            OptimizedNodeName(scope_and_name, "_inner");
        NodeDef* inner_square_node = ctx().node_map->GetNode(inner_square_name);
        if (inner_square_node == nullptr) {
          inner_square_node = AddCopyNode(inner_square_name, node);
          inner_square_node->set_op("Square");
          inner_square_node->mutable_input()->RemoveLast();
        }
        ctx().node_map->AddOutput(x->name(), inner_square_node->name());
        // We modify `node`: node = mul(x, inner_square);
        node->set_op("Mul");
        node->set_input(1, inner_square_node->name());
        node->add_input(AsControlDependency(y->name()));

        AddToOptimizationQueue(node);
        AddToOptimizationQueue(inner_square_node);
        AddToOptimizationQueue(y);
      }
    } else if (curr == complex128(1, 0) &&
               ShapesSymbolicallyEqual(value_props.shape(), output_shape)) {
      // Pow could be used to broadcast, so make sure the shapes of the two
      // arguments are identical before replacing Pow with Identity.
      node->set_op("Identity");
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    } else if (curr == complex128(0.5, 0)) {
      node->set_op("Sqrt");
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    } else if (curr == complex128(0, 0) &&
               ShapesSymbolicallyEqual(value_props.shape(), output_shape) &&
               PartialTensorShape(output_shape).IsFullyDefined()) {
      const auto dtype = node->attr().at("T").type();
      Tensor ones(dtype, output_shape);
      for (int i = 0; i < ones.NumElements(); ++i) {
        TF_RETURN_IF_ERROR(SetElementToOne(i, &ones));
      }
      node->set_op("Const");
      (*node->mutable_attr())["dtype"].set_type(dtype);
      node->mutable_attr()->erase("T");
      ones.AsProtoTensorContent(
          (*node->mutable_attr())["value"].mutable_tensor());
      node->set_input(0, AsControlDependency(x->name()));
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(x);
      AddToOptimizationQueue(y);
    } else if (curr == complex128(-0.5, 0)) {
      node->set_op("Rsqrt");
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    } else if (curr == complex128(-1, 0)) {
      node->set_op("Reciprocal");
      node->set_input(1, AsControlDependency(y->name()));
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    }
    return Status::OK();
  }

 private:
  Status SetElementToOne(int i, Tensor* t) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_156(mht_156_v, 3929, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SetElementToOne");

    switch (t->dtype()) {
      case DT_INT32:
        t->flat<int32>()(i) = 1;
        return Status::OK();
      case DT_INT64:
        t->flat<int64_t>()(i) = 1L;
        return Status::OK();
      case DT_FLOAT:
        t->flat<float>()(i) = 1.0f;
        return Status::OK();
      case DT_DOUBLE:
        t->flat<double>()(i) = 1.0;
        return Status::OK();
      case DT_COMPLEX64:
        t->flat<complex64>()(i) = complex64(1);
        return Status::OK();
      case DT_COMPLEX128:
        t->flat<complex128>()(i) = complex128(1);
        return Status::OK();
      default:
        return errors::InvalidArgument("Invalid data type: ", t->dtype());
    }
  }
};

class ConvertLog1pStage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertLog1pStage(const GraphOptimizerContext& ctx,
                             const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertLog1p", ctx, ctx_ext) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_157(mht_157_v, 3962, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ConvertLog1pStage");
}
  ~ConvertLog1pStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_158(mht_158_v, 3968, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");
 return IsLog(*node); }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_159(mht_159_v, 3973, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (!IsAdd(*input)) {
      return Status::OK();
    }

    if (ctx().graph_properties->GetInputProperties(input->name()).size() < 2) {
      return Status::OK();
    }

    bool modified = false;
    TF_RETURN_IF_ERROR(TrySimplifyInternal(node, input, 0, 1, &modified));
    if (!modified) {
      TF_RETURN_IF_ERROR(TrySimplifyInternal(node, input, 1, 0, &modified));
    }
    if (modified) {
      *simplified_node_name = node->name();
    }
    return Status::OK();
  }

 private:
  Status TrySimplifyInternal(NodeDef* node, NodeDef* add_node, int i, int j,
                             bool* modified) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_160(mht_160_v, 4000, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplifyInternal");

    const auto& t =
        ctx().graph_properties->GetInputProperties(add_node->name())[i];
    const auto& c =
        ctx().graph_properties->GetInputProperties(add_node->name())[j];
    for (int k = 0; k < c.shape().dim_size(); ++k) {
      // Skip if c shape is not fully determined.
      if (c.shape().dim(k).size() < 0) {
        return Status::OK();
      }
    }
    TensorShapeProto broadcast_shape;
    if (!ShapeAfterBroadcast(t.shape(), c.shape(), &broadcast_shape)) {
      return Status::OK();
    }
    if (!ShapesSymbolicallyEqual(t.shape(), broadcast_shape)) {
      // skip if the non-constant tensor doesn't have the same shape after
      // broadcast.
      return Status::OK();
    }
    Tensor constant;
    if (GetTensorFromConstNode(add_node->input(j), &constant)) {
      complex128 element;
      // TODO(rmlarsen): Refactor the more general IsOnes from
      // constant_folding.cc and use it here. Perhaps also convert log(x - (-1))
      // or (preferably) add a passes to canonicalize Sub(x, -1) to Add(x, 1),
      // and Neg(-1) to 1.
      for (int k = 0; k < constant.NumElements(); ++k) {
        if (!GetElementUnexhaustive(constant, k,
                                    {DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE,
                                     DT_COMPLEX64, DT_COMPLEX128},
                                    &element)) {
          // input data type is not supported by log1p. Skip.
          return Status::OK();
        }
        if (element != complex128(1)) {
          // current element is not 1. Skip.
          return Status::OK();
        }
      }
      NodeDef *x, *y;
      TF_RETURN_IF_ERROR(GetInputNode(add_node->input(i), &x));
      TF_RETURN_IF_ERROR(GetInputNode(add_node->input(j), &y));
      node->set_op("Log1p");
      node->set_input(0, add_node->input(i));
      node->add_input(AsControlDependency(y->name()));
      ForwardControlDependencies(node, {add_node});

      AddToOptimizationQueue(node);
      AddToOptimizationQueue(add_node);
      AddToOptimizationQueue(x);
      AddToOptimizationQueue(y);
      *modified = true;
    }
    return Status::OK();
  }
};

class ConvertExpm1Stage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertExpm1Stage(const GraphOptimizerContext& ctx,
                             const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertExpm1", ctx, ctx_ext) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_161(mht_161_v, 4065, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ConvertExpm1Stage");
}
  ~ConvertExpm1Stage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_162(mht_162_v, 4071, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    if (!IsSub(*node)) return false;

    NodeDef* input;
    if (!GetInputNode(node->input(0), &input).ok()) return false;

    return IsExp(*input);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_163(mht_163_v, 4083, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    if (ctx().graph_properties->GetInputProperties(node->name()).size() < 2) {
      return Status::OK();
    }
    const auto& t = ctx().graph_properties->GetInputProperties(node->name())[0];
    const auto& c = ctx().graph_properties->GetInputProperties(node->name())[1];
    TensorShapeProto broadcast_shape;
    if (!ShapeAfterBroadcast(t.shape(), c.shape(), &broadcast_shape)) {
      return Status::OK();
    }
    if (!ShapesSymbolicallyEqual(t.shape(), broadcast_shape)) {
      // skip if the non-constant tensor doesn't have the same shape after
      // broadcast.
      return Status::OK();
    }
    Tensor constant;
    if (!GetTensorFromConstNode(node->input(1), &constant)) return Status::OK();
    // TODO(rmlarsen): Use the more general IsOnes helper here.
    complex128 element;
    for (int k = 0; k < constant.NumElements(); ++k) {
      if (!GetElementUnexhaustive(constant, k,
                                  {DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE,
                                   DT_COMPLEX64, DT_COMPLEX128},
                                  &element)) {
        // input data type is not supported by expm1. Skip.
        return Status::OK();
      }
      if (element != complex128(1)) {
        // current element is not 1. Skip.
        return Status::OK();
      }
    }
    NodeDef* exp;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &exp));
    NodeDef *exp_input, *ones;
    TF_RETURN_IF_ERROR(GetInputNode(exp->input(0), &exp_input));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &ones));
    node->set_op("Expm1");
    node->set_input(0, exp->input(0));
    node->set_input(1, AsControlDependency(ones->name()));
    ForwardControlDependencies(node, {exp});

    AddToOptimizationQueue(node);
    AddToOptimizationQueue(exp);
    AddToOptimizationQueue(exp_input);
    AddToOptimizationQueue(ones);
    *simplified_node_name = node->name();
    return Status::OK();
  }
};

// Performs conversions like:
// Max(Sqrt(x)) => Sqrt(Max(x))
// Checks for a max/min reduction over element-wise monotonic functions, such
// as Sqrt, Sigmoid, Tanh, etc.
class OptimizeMaxOrMinOfMonotonicStage : public ArithmeticOptimizerStage {
 public:
  explicit OptimizeMaxOrMinOfMonotonicStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("OptimizeMaxOrMinOfMonotonicStage", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_164(mht_164_v, 4147, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "OptimizeMaxOrMinOfMonotonicStage");
}
  ~OptimizeMaxOrMinOfMonotonicStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_165(mht_165_v, 4153, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAnyMax(*node) || IsAnyMin(*node) || IsAnyMaxPool(*node) ||
           IsArgMax(*node) || IsArgMin(*node);
  }

  Status TrySimplify(NodeDef* reduction_node,
                     string* simplified_node_name) override {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_166(mht_166_v, 4162, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    if (IsInPreserveSet(*reduction_node)) {
      return Status::OK();
    }

    NodeDef* inner_function;
    TF_RETURN_IF_ERROR(GetInputNode(reduction_node->input(0), &inner_function));

    NodeDef* inner_function_input = nullptr;
    if (inner_function->input_size() > 0) {
      TF_RETURN_IF_ERROR(
          GetInputNode(inner_function->input(0), &inner_function_input));
    }

    // Optimize only if:
    // 0. inner_function is not in the preserve set,
    // 1. inner_function's Op is element-wise monotonic
    // 2. inner_function's output is not being consumed elsewhere.
    // 3. is monotonic increasing if reduction_node is a pooling operation
    //    since we don't have MinPool operations.
    // 4. inner_functions is not a Relu node with an input from FusedBatchNorm
    //    or BiasAdd. This pattern will be fused later by remapper.
    auto can_be_fused_by_remapper = [](const NodeDef& consumer,
                                       const NodeDef& producer) -> bool {
      if (IsRelu(consumer) || IsRelu6(consumer)) {
        if (IsFusedBatchNorm(producer) || IsBiasAdd(producer)) {
          return true;
        }
      }
      return false;
    };
    bool is_non_decreasing = false;
    if (!IsInPreserveSet(*inner_function) &&
        IsElementWiseMonotonic(*inner_function, &is_non_decreasing) &&
        ctx().node_map->GetOutputs(inner_function->name()).size() == 1 &&
        (is_non_decreasing || !IsAnyMaxPool(*reduction_node)) &&
        !can_be_fused_by_remapper(*inner_function, *inner_function_input)) {
      // Swap the first inputs of the inner function Op & the reduction Op.
      NodeDef* inner_input;
      TF_RETURN_IF_ERROR(GetInputNode(inner_function->input(0), &inner_input));
      reduction_node->set_input(0, inner_input->name());
      ctx().node_map->UpdateInput(reduction_node->name(),
                                  inner_function->name(), inner_input->name());
      inner_function->set_input(0, reduction_node->name());
      TF_RETURN_IF_ERROR(
          UpdateConsumers(reduction_node, inner_function->name()));
      ctx().node_map->UpdateInput(inner_function->name(), inner_input->name(),
                                  reduction_node->name());
      if (!is_non_decreasing) {
        // Flip Min<->Max if the function is non-increasing, e.g.
        // Max(Neg(x)) = Neg(Min(x)).
        const string opposite = FlipMinMax(*reduction_node);
        reduction_node->set_op(opposite);
      }

      if (IsArgMax(*reduction_node) || IsArgMin(*reduction_node)) {
        // ArgMax(Sqrt(x)) = ArgMax(x)
        inner_function->set_op("Identity");
      }

      AddToOptimizationQueue(reduction_node);
      AddToOptimizationQueue(inner_function);
      AddToOptimizationQueue(inner_input);
    }
    return Status::OK();
  }

 private:
  string FlipMinMax(const NodeDef& node) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_167(mht_167_v, 4233, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "FlipMinMax");

    const string& op = node.op();
    if (IsAnyMax(node) || IsArgMax(node)) {
      return str_util::StringReplace(op, "Max", "Min", false);
    } else {
      return str_util::StringReplace(op, "Min", "Max", false);
    }
  }
};

// Replace a chain of type&shape preserving unary ops with a
// '_UnaryOpsComposition' node.
// TODO(ezhulenev): It should be a part of remapper optimizer because it doesn't
// have to do much with arithmetic (together with FoldMultiplyIntoConv stage?).
class UnaryOpsComposition : public ArithmeticOptimizerStage {
 public:
  explicit UnaryOpsComposition(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("UnaryOpsComposition", ctx, ctx_ext) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_168(mht_168_v, 4254, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "UnaryOpsComposition");

    // WARN: This should be consistent with unary_ops_composition.cc.
    // clang-format off
    supported_ops_ = {// Ops defined via Eigen scalar ops.
                      {"Abs",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Acos",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Acosh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Asin",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Asinh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Atan",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Atanh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Ceil",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Cos",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Cosh",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Expm1",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Exp",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Floor",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Inv",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Log",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Log1p",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Neg",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Reciprocal", {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Rint",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Round",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Rsqrt",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sigmoid",    {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sin",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sinh",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Sqrt",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Square",     {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Tan",        {DT_FLOAT,          DT_DOUBLE}},
                      {"Tanh",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      // Additional ops that are not part of the Eigen.
                      {"Elu",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Relu",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Relu6",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Selu",       {DT_FLOAT, DT_HALF, DT_DOUBLE}}};
    // clang-format on
  }
  ~UnaryOpsComposition() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_169(mht_169_v, 4298, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return CanOptimize(*node) &&
           // Check that this node was not already a root of a fused chain. If
           // graph optimization runs twice without pruning in between,
           // fused_nodes_ will not have this information.
           !ctx().node_map->NodeExists(OptimizedNodeName(*node));
  }

  Status TrySimplify(NodeDef* root, string* simplified_node_name) override {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_170(mht_170_v, 4309, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    TF_RETURN_IF_ERROR(CheckAttrExists(*root, "T"));
    DataType dtype = root->attr().at("T").type();

    // Keep a trace of all supported input nodes that can be fused together.
    std::vector<string> op_nodes = {root->name()};
    std::vector<string> op_names = {root->op()};

    // Check if we should follow input(0) while building an op composition.
    const auto predicate_fn = [&](const NodeDef& input) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_171(mht_171_v, 4321, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

      if (input.name() == root->name()) return true;

      bool follow_input_node =
          dtype == GetDataTypeFromAttr(input, "T") &&
          NumNonControlDataOutputs(input, *ctx().node_map) == 1 &&
          CanOptimize(input);

      if (follow_input_node) {
        op_nodes.push_back(input.name());
        op_names.push_back(input.op());
      }

      return follow_input_node;
    };

    NodeDef* last_op = GetTailOfChain(
        *root, *ctx().node_map, /*follow_control_input*/ false, predicate_fn);

    // We were not able to find a chain that can be replaced.
    if (op_names.size() == 1) return Status::OK();

    // Do not add fused nodes to any other chain.
    std::for_each(op_nodes.begin(), op_nodes.end(),
                  [this](const string& name) { AddToFusedNodes(name); });

    // Reverse the trace to get correct composition computation order.
    std::reverse(op_names.begin(), op_names.end());

    VLOG(2) << "Fuse unary ops: root=" << root->name() << " op_names=["
            << absl::StrJoin(op_names, ", ") << "]";

    NodeDef* composition_node = ctx().optimized_graph->add_node();
    composition_node->set_name(OptimizedNodeName(*root));
    composition_node->set_op("_UnaryOpsComposition");
    composition_node->add_input(last_op->input(0));
    composition_node->set_device(root->device());

    auto attr = composition_node->mutable_attr();
    SetAttrValue(dtype, &(*attr)["T"]);
    SetAttrValue(op_names, &(*attr)["op_names"]);

    ctx().node_map->AddNode(composition_node->name(), composition_node);
    ctx().node_map->AddOutput(NodeName(last_op->input(0)),
                              composition_node->name());

    *simplified_node_name = composition_node->name();

    return Status::OK();
  }

 private:
  bool CanOptimize(const NodeDef& node) const {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_172(mht_172_v, 4376, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CanOptimize");

    DataType dtype = GetDataTypeFromAttr(node, "T");
    if (!IsSupported(node.op(), dtype)) {
      return false;
    }
    if (IsInPreserveSet(node)) {
      return false;
    }
    if (!NodeIsOnCpu(node)) {
      return false;
    }
    if (NodeIsAlreadyFused(node)) {
      return false;
    }
    return !(IsDrivenByControlDependency(node) ||
             DrivesControlDependency(node));
  }

  bool NodeIsAlreadyFused(const NodeDef& node) const {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_173(mht_173_v, 4397, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "NodeIsAlreadyFused");

    return fused_nodes_.count(node.name()) > 0;
  }

  string OptimizedNodeName(const NodeDef& node) const {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_174(mht_174_v, 4404, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "OptimizedNodeName");

    return strings::StrCat(node.name(), "/unary_ops_composition");
  }

  void AddToFusedNodes(const string& name) {
   std::vector<std::string> mht_175_v;
   mht_175_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_175(mht_175_v, 4412, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "AddToFusedNodes");
 fused_nodes_.insert(name); }

  // Check if an op is supported by the _UnaryOpsComposition for the given type.
  bool IsSupported(const string& op_name, DataType dtype) const {
   std::vector<std::string> mht_176_v;
   mht_176_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_176(mht_176_v, 4419, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    const auto it = supported_ops_.find(op_name);
    return it != supported_ops_.end() && it->second.count(dtype) > 0;
  }

  std::unordered_map<string, std::set<DataType>> supported_ops_;
  std::unordered_set<string> fused_nodes_;
};

// Replace operations of the form:
//    x = stack((a_0, a_1, ..., a_{n-1}), axis=k)[:,...,i,...]
// with
//    a_i
// when the strided slice index `i` is applied in the k'th axis.
//
// Similarly, replace operations of the form:
//    x = stack((a_0, a_1, ..., a_{n-1}), axis=k)[:,...,i:i+1,...]
// with
//    expand_dims(a_i, axis=k)
// where the slice operator can be StridedSlice or Slice.
//
// TODO(ebrevdo): Extend to also replace operations of the form
//    concat((a_0, a_1, ..., ), axis=k)[:, ..., s_i:s_{i+1}, ...]
// with
//    a_i,
// when
//    s_i = cumsum(shape(a)[k] for a in (a_0, ...,))[i]
// and slicing is in the k'th axis.
class RemoveStackSliceSameAxis : public ArithmeticOptimizerStage {
 public:
  explicit RemoveStackSliceSameAxis(const GraphOptimizerContext& ctx,
                                    const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveStackStridedSliceSameAxis", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_177(mht_177_v, 4455, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveStackSliceSameAxis");
}
  ~RemoveStackSliceSameAxis() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_178(mht_178_v, 4461, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return (IsStridedSlice(*node) || IsSlice(*node)) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_179(mht_179_v, 4468, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    // *node is a StridedSlice NodeDef.
    NodeDef* pack;

    // Get the input and see if it's a Pack op.
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &pack));
    if (!IsPack(*pack)) return Status::OK();

    bool return_early;
    PartialTensorShape pack_output_shape;
    int pack_axis;
    TF_RETURN_IF_ERROR(
        CheckInputs(node, pack, &pack_output_shape, &pack_axis, &return_early));
    if (return_early) return Status::OK();

    int64_t slice_start_value;
    bool found;
    bool must_expand_dims;
    TF_RETURN_IF_ERROR(GetSliceAxis(node, pack, pack_output_shape, pack_axis,
                                    &slice_start_value, &found,
                                    &must_expand_dims));
    if (!found) return Status::OK();

    return RewriteGraph(node, pack, slice_start_value, pack_axis,
                        must_expand_dims, simplified_node_name);
  }

 protected:
  Status CheckInputs(const NodeDef* node, const NodeDef* pack,
                     PartialTensorShape* pack_output_shape, int* pack_axis,
                     bool* return_early) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_180(mht_180_v, 4501, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "CheckInputs");

    *return_early = true;
    TF_RETURN_IF_ERROR(CheckAttrExists(*pack, "axis"));

    *pack_axis = pack->attr().at("axis").i();
    auto slice_properties =
        ctx().graph_properties->GetInputProperties(node->name());
    if (slice_properties.empty() ||
        slice_properties[0].shape().unknown_rank()) {
      return Status::OK();
    }
    *pack_output_shape = slice_properties[0].shape();
    const int pack_output_rank = pack_output_shape->dims();
    if (*pack_axis < 0) {
      *pack_axis += pack_output_rank;
    }
    if (*pack_axis < 0 || *pack_axis >= pack_output_rank) {
      return errors::InvalidArgument(
          "Pack node (", pack->name(),
          ") axis attribute is out of bounds: ", pack->attr().at("axis").i());
    }
    *return_early = false;
    return Status::OK();
  }

  Status GetSliceAxis(const NodeDef* node, const NodeDef* pack,
                      const PartialTensorShape& pack_output_shape,
                      int pack_axis, int64_t* slice_start_value, bool* found,
                      bool* must_expand_dims) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_181(mht_181_v, 4532, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetSliceAxis");

    *found = false;
    if (IsSlice(*node)) {
      *must_expand_dims = true;
      return GetSimpleSliceAxis(node, pack, pack_output_shape, pack_axis,
                                slice_start_value, found);
    } else {
      return GetStridedSliceAxis(node, pack, pack_output_shape, pack_axis,
                                 slice_start_value, found, must_expand_dims);
    }
  }

  Status GetSimpleSliceAxis(const NodeDef* node, const NodeDef* pack,
                            const PartialTensorShape& pack_output_shape,
                            int pack_axis, int64_t* slice_start_value,
                            bool* found) {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_182(mht_182_v, 4550, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetSimpleSliceAxis");

    NodeDef* slice_begin;
    NodeDef* slice_size;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &slice_begin));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(2), &slice_size));
    for (const auto* n : {slice_begin, slice_size}) {
      if (!IsReallyConstant(*n)) return Status::OK();
    }

    Tensor slice_begin_t;
    Tensor slice_size_t;
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_begin, "value"));
    if (!slice_begin_t.FromProto(slice_begin->attr().at("value").tensor())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_size, "value"));
    if (!slice_size_t.FromProto(slice_size->attr().at("value").tensor())) {
      return Status::OK();
    }

    auto copy_tensor_values_to_vector =
        [node](const Tensor& t, gtl::InlinedVector<int64, 4>* vec) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_183(mht_183_v, 4574, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");

          if (t.dtype() == DT_INT32) {
            auto t_flat = t.flat<int32>();
            vec->assign(&t_flat(0), &t_flat(t.NumElements()));
          } else if (t.dtype() == DT_INT64) {
            auto t_flat = t.flat<int64_t>();
            vec->assign(&t_flat(0), &t_flat(t.NumElements()));
          } else {
            return errors::InvalidArgument("Node ", node->name(),
                                           " has invalid type for Index attr: ",
                                           DataTypeString(t.dtype()));
          }
          return Status::OK();
        };

    gtl::InlinedVector<int64_t, 4> slice_begin_vec;
    gtl::InlinedVector<int64_t, 4> slice_size_vec;
    TF_RETURN_IF_ERROR(
        copy_tensor_values_to_vector(slice_begin_t, &slice_begin_vec));
    TF_RETURN_IF_ERROR(
        copy_tensor_values_to_vector(slice_size_t, &slice_size_vec));

    if (slice_begin_vec.size() != slice_size_vec.size()) {
      return errors::InvalidArgument("Node ", node->name(),
                                     " has mismatched lengths for begin (",
                                     slice_begin_vec.size(), ") and size (",
                                     slice_size_vec.size(), ") vectors.");
    }
    int slice_begin_vec_size = slice_begin_vec.size();
    if (!pack_output_shape.unknown_rank() &&
        slice_begin_vec_size != pack_output_shape.dims()) {
      return Status::OK();
    }
    if (pack_axis >= slice_begin_vec_size) {
      return errors::InvalidArgument(
          "Input to node ", node->name(), " had pack_axis ", pack_axis,
          " but rank was ", slice_begin_vec_size, ".");
    }

    *slice_start_value = slice_begin_vec[pack_axis];
    if (slice_size_vec[pack_axis] != 1) {
      // Not slicing a single value out.
      return Status::OK();
    }

    for (int i = 0; i < slice_begin_vec_size; ++i) {
      if (i != pack_axis) {
        if (slice_begin_vec[i] != 0 ||
            !(slice_size_vec[i] == -1 ||
              slice_size_vec[i] == pack_output_shape.dim_size(i))) {
          // Not slicing on the same axis as the Pack op.
          return Status::OK();
        }
      }
    }

    if (*slice_start_value < 0 || *slice_start_value >= pack->input_size()) {
      return errors::InvalidArgument(
          "Node ", node->name(), " requested invalid slice index ",
          *slice_start_value, " on axis ", pack_axis,
          " from tensor of shape: ", pack_output_shape.DebugString());
    }

    *found = true;  // slice_start_value is valid.
    return Status::OK();
  }

  Status GetStridedSliceAxis(const NodeDef* node, const NodeDef* pack,
                             const PartialTensorShape& pack_output_shape,
                             int pack_axis, int64_t* slice_start_value,
                             bool* found, bool* must_expand_dims) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_184(mht_184_v, 4647, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "GetStridedSliceAxis");

    TF_RETURN_IF_ERROR(
        CheckAttrsExist(*node, {"begin_mask", "end_mask", "ellipsis_mask",
                                "new_axis_mask", "shrink_axis_mask"}));

    const int begin_mask = node->attr().at("begin_mask").i();
    const int end_mask = node->attr().at("end_mask").i();
    const int ellipsis_mask = node->attr().at("ellipsis_mask").i();
    const int new_axis_mask = node->attr().at("new_axis_mask").i();
    const int shrink_axis_mask = node->attr().at("shrink_axis_mask").i();

    // Check that the StridedSlice is one of these at pack_axis:
    //   [..., i, ...]
    //   [..., i:i+1, ...]
    //   [..., :1, ...]
    //   [..., -1:, ...]
    ///  [..., s_{pack_axis}-1:, ...]
    NodeDef* slice_begin;
    NodeDef* slice_end;
    NodeDef* slice_strides;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &slice_begin));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(2), &slice_end));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(3), &slice_strides));

    for (const auto* n : {slice_begin, slice_end, slice_strides}) {
      if (!IsReallyConstant(*n)) return Status::OK();
    }

    Tensor slice_begin_t;
    Tensor slice_end_t;
    Tensor slice_strides_t;

    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_begin, "value"));
    if (!slice_begin_t.FromProto(slice_begin->attr().at("value").tensor())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_end, "value"));
    if (!slice_end_t.FromProto(slice_end->attr().at("value").tensor())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_strides, "value"));
    if (!slice_strides_t.FromProto(
            slice_strides->attr().at("value").tensor())) {
      return Status::OK();
    }
    TensorShape processing_shape;
    TensorShape final_shape;
    bool is_identity;
    bool is_simple_slice;
    bool slice_dim0;
    gtl::InlinedVector<int64_t, 4> slice_begin_vec;
    gtl::InlinedVector<int64_t, 4> slice_end_vec;
    gtl::InlinedVector<int64_t, 4> slice_strides_vec;
    TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
        &slice_begin_t, &slice_end_t, slice_strides_t, pack_output_shape,
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
        &processing_shape, &final_shape, &is_identity, &is_simple_slice,
        &slice_dim0, &slice_begin_vec, &slice_end_vec, &slice_strides_vec));

    if (!is_simple_slice) return Status::OK();

    int begin_index = -1;
    int64_t begin_value = 0;
    for (int i = 0, end = slice_begin_vec.size(); i < end; ++i) {
      const int64_t v = slice_begin_vec[i];
      if (v != 0) {
        if (begin_index != -1) {
          // At least two start values that are nonzero.
          return Status::OK();
        }
        begin_index = i;
        begin_value = v;
      }
    }

    int end_index = -1;
    int64_t end_value = 0;
    for (int i = 0, end = slice_begin_vec.size(); i < end; ++i) {
      const int64_t v = slice_end_vec[i];
      if (v != pack_output_shape.dim_size(i)) {
        if (end_index != -1) {
          // At least two end values that are nonzero.
          return Status::OK();
        }
        end_index = i;
        end_value = v;
      }
    }

    if (begin_index == -1 && end_index == -1) return Status::OK();
    if (begin_index != -1 && end_index != -1 && begin_index != end_index) {
      // Somehow received different axes for begin/end slicing
      return Status::OK();
    }
    const int slice_axis = (begin_index == -1) ? end_index : begin_index;
    if (slice_axis != pack_axis) {
      // Not slicing on the same axis as the Pack op.
      return Status::OK();
    }
    *slice_start_value = (begin_index == -1) ? 0 : begin_value;
    const int64_t slice_end_value =
        (end_index == -1) ? pack_output_shape.dim_size(slice_axis) : end_value;
    if (slice_end_value != *slice_start_value + 1) {
      // Not slicing a single value out.
      return Status::OK();
    }

    if (*slice_start_value < 0 || *slice_start_value >= pack->input_size()) {
      return errors::InvalidArgument(
          "Node ", node->name(), " requested invalid slice index ",
          *slice_start_value, " on axis ", slice_axis,
          " from tensor of shape: ", pack_output_shape.DebugString());
    }

    if (shrink_axis_mask == 0) {
      *must_expand_dims = true;
    } else if (shrink_axis_mask == (1 << slice_axis)) {
      *must_expand_dims = false;
    } else {
      // Shrinking on a different axis from the one that we are slicing on.
      return Status::OK();
    }

    *found = true;  // slice_start_value is valid.
    return Status::OK();
  }

  Status RewriteGraph(const NodeDef* node, const NodeDef* pack,
                      int64_t slice_start_value, int pack_axis,
                      bool must_expand_dims, string* simplified_node_name) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_185(mht_185_v, 4779, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RewriteGraph");

    const string& input_slice = pack->input(slice_start_value);

    const OpInfo::TensorProperties* input_slice_properties;
    TF_RETURN_IF_ERROR(GetTensorProperties(pack->input(slice_start_value),
                                           &input_slice_properties));
    PartialTensorShape input_slice_shape(input_slice_properties->shape());

    const OpInfo::TensorProperties* output_properties;
    TF_RETURN_IF_ERROR(GetTensorProperties(
        strings::StrCat(node->name(), ":", 0), &output_properties));
    PartialTensorShape output_shape(output_properties->shape());
    NodeDef* output =
        AddEmptyNode(OptimizedNodeName(ParseNodeScopeAndName(node->name())));
    if (!must_expand_dims) {
      output->set_op("Identity");
      output->set_device(node->device());
      SetDataTypeToAttr(output_properties->dtype(), "T", output);
      output->add_input(input_slice);
    } else {
      NodeDef* axis = AddEmptyNode(
          OptimizedNodeName(ParseNodeScopeAndName(node->name()), "Axis"));
      axis->set_op("Const");
      axis->set_device(node->device());
      // We need to add a control edge from input slice to guarantee that axis
      // constant will be executed in the same frame as `input_slice`, otherwise
      // ExpandDims might have mismatched input frames.
      axis->add_input(absl::StrCat("^", ParseTensorName(input_slice).node()));
      auto axis_attr = axis->mutable_attr();
      SetDataTypeToAttr(DT_INT32, "dtype", axis);
      auto* axis_t = (*axis_attr)["value"].mutable_tensor();
      axis_t->set_dtype(DT_INT32);
      axis_t->add_int_val(pack_axis);
      AddToOptimizationQueue(axis);
      output->set_op("ExpandDims");
      output->set_device(node->device());
      SetDataTypeToAttr(output_properties->dtype(), "T", output);
      SetDataTypeToAttr(DT_INT32, "Tdim", output);
      output->add_input(input_slice);
      output->add_input(axis->name());
    }

    // Copy dependencies over.
    ForwardControlDependencies(output, {node, pack});
    AddToOptimizationQueue(output);
    *simplified_node_name = output->name();

    return Status::OK();
  }
};

// Eliminates unnecessary copies during sparse embedding lookup operations.
//
// For non-partitioned variables, the `tf.nn.embedding_lookup_sparse()` function
// generates code of the form:
//
//     embeddings = <a 2D Tensor>
//     sparse_ids = <a tf.int64 SparseTensor>
//     segment_ids = sparse_ids.indices[:, 0]
//     ids, idx = tf.unique(sparse_ids.values)
//     gathered_rows = tf.gather(params, ids)
//     result = tf.sparse.segment_<combiner>(gathered_rows, idx, segment_ids)
//
// In this case, all of the work in `tf.unique()` and `tf.gather()`
// can be avoided by passing the full embeddings to
// `tf.sparse.segment_<combiner>()` and performing the same amount of
// computation (but fewer copies and allocations) as follows:
//
//     embeddings = <a 2D Tensor>
//     sparse_ids = <a tf.int64 SparseTensor>
//     segment_ids = sparse_ids.indices[:, 0]
//     result = tf.sparse.segment_<combiner>(
//          embeddings, sparse_ids.values, segment_ids)
class SimplifyEmbeddingLookupStage : public ArithmeticOptimizerStage {
 public:
  explicit SimplifyEmbeddingLookupStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("SimplifyEmbeddingLookupStage", ctx, ctx_ext) {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_186(mht_186_v, 4860, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "SimplifyEmbeddingLookupStage");

  }
  ~SimplifyEmbeddingLookupStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_187(mht_187_v, 4867, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAnySparseSegmentReduction(*node);
  }

  Status TrySimplify(NodeDef* reduction_node,
                     string* simplified_node_name) override {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_188(mht_188_v, 4875, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    if (IsInPreserveSet(*reduction_node)) return Status::OK();

    // Input 0 (data) of the reduction node must be a tf.gather() on the 0th
    // axis.
    NodeDef* gather_node = nullptr;
    TF_RETURN_IF_ERROR(GetInputNode(reduction_node->input(0), &gather_node));
    if (!IsGather(*gather_node) || IsInPreserveSet(*gather_node) ||
        gather_node->device() != reduction_node->device())
      return Status::OK();
    if (gather_node->op() == "GatherV2" && !IsAxis0(*gather_node, 2))
      return Status::OK();

    // Input 1 (indices) of the gather node must be a tf.unique() on the 0th
    // axis.
    NodeDef* unique_node = nullptr;
    TF_RETURN_IF_ERROR(GetInputNode(gather_node->input(1), &unique_node));
    if (!IsUnique(*unique_node) || IsInPreserveSet(*unique_node) ||
        unique_node->device() != gather_node->device())
      return Status::OK();
    if (unique_node->op() == "UniqueV2" && !IsAxis0(*unique_node, 1))
      return Status::OK();

    DataType unique_element_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(*unique_node, "T", &unique_element_type));

    // Input 1 (indices) of the reduction node must be output 1 of the unique
    // node.
    const TensorId idx_tensor = ParseTensorName(reduction_node->input(1));
    if (idx_tensor != TensorId(unique_node->name(), 1)) return Status::OK();

    // Input 1 (indices) of the reduction node becomes input 0 (x) of the unique
    // node.
    reduction_node->set_input(1, unique_node->input(0));
    ctx().node_map->UpdateInput(reduction_node->name(),
                                reduction_node->input(1),
                                unique_node->input(0));
    SetDataTypeToAttr(unique_element_type, "Tidx", reduction_node);

    // Input 0 (data) of the reduction node becomes input 1 (params) of the
    // gather node.
    const OpInfo::TensorProperties* gather_input_properties;
    TF_RETURN_IF_ERROR(
        GetTensorProperties(gather_node->input(0), &gather_input_properties));
    if (gather_input_properties->dtype() == DT_RESOURCE) {
      // If the input is a ResourceGather, we need to add
      // ReadVariableOp.
      NodeDef* variable_node = nullptr;
      TF_RETURN_IF_ERROR(GetInputNode(gather_node->input(0), &variable_node));
      NodeDef* read_var_node = ctx().optimized_graph->add_node();
      read_var_node->set_name(OptimizedNodeName(
          ParseNodeScopeAndName(reduction_node->name()), "ReadVar"));
      read_var_node->set_op("ReadVariableOp");
      read_var_node->add_input(gather_node->input(0));
      read_var_node->set_device(variable_node->device());

      // The Variable and the Gather node should have the same
      // dtype, but it might not be set on both nodes.
      auto attr = read_var_node->mutable_attr();
      if (variable_node->attr().count("dtype")) {
        SetAttrValue(variable_node->attr().at("dtype").type(),
                     &(*attr)["dtype"]);
      }
      if (gather_node->attr().count("dtype")) {
        SetAttrValue(gather_node->attr().at("dtype").type(), &(*attr)["dtype"]);
      }
      // Copy the _class attr from the Gather node should it exist in case
      // of location constraints with the variable.
      if (gather_node->attr().count("_class")) {
        (*attr)["_class"] = gather_node->attr().at("_class");
      }
      if (variable_node->attr().count("shape")) {
        SetAttrValue(variable_node->attr().at("shape").shape(),
                     &(*attr)["_output_shapes"]);
      }

      ctx().node_map->AddNode(read_var_node->name(), read_var_node);
      reduction_node->set_input(0, read_var_node->name());
      ctx().node_map->UpdateInput(reduction_node->name(),
                                  reduction_node->input(0),
                                  read_var_node->name());
    } else {
      reduction_node->set_input(0, gather_node->input(0));
      ctx().node_map->UpdateInput(reduction_node->name(),
                                  reduction_node->input(0),
                                  gather_node->input(0));
    }
    *simplified_node_name = reduction_node->name();
    return Status::OK();
  }

 private:
  bool IsAxis0(const NodeDef& node, int axis_input) {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_189(mht_189_v, 4970, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsAxis0");

    Tensor axis_tensor;
    if (!GetTensorFromConstNode(node.input(axis_input), &axis_tensor))
      return false;
    if (axis_tensor.NumElements() != 1) return false;
    if (axis_tensor.dtype() == DT_INT32) {
      return axis_tensor.flat<int32>()(0) == 0;
    } else if (axis_tensor.dtype() == DT_INT64) {
      return axis_tensor.flat<int64_t>()(0) == 0;
    } else {
      return false;
    }
  }
};

// Eliminates unnecessary casts before sparse segment reduction operations.
//
// Existing graphs and library code would often insert a cast from DT_INT64 to
// DT_INT32 on the indices and/or segment_ids inputs to "SparseSegment*" ops.
// Support for for DT_INT64 indices and/or segment_ids now exists, so we can
// pass the input directly without a cast.
class RemoveCastIntoSegmentReductionStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveCastIntoSegmentReductionStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveCastIntoSegmentReductionStage", ctx,
                                 ctx_ext) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_190(mht_190_v, 5000, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "RemoveCastIntoSegmentReductionStage");
}
  ~RemoveCastIntoSegmentReductionStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_191(mht_191_v, 5006, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsSupported");

    return IsAnySparseSegmentReduction(*node);
  }

  Status TrySimplify(NodeDef* reduction_node,
                     string* simplified_node_name) override {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_192(mht_192_v, 5014, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "TrySimplify");

    if (IsInPreserveSet(*reduction_node)) return Status::OK();

    bool optimized = false;

    // Inputs 1 (indices) and 2 (segment_ids) can be either DT_INT32 or
    // DT_INT64.
    std::array<std::pair<int, string>, 2> input_details = {
        std::make_pair(1, "Tidx"), std::make_pair(2, "Tsegmentids")};

    for (const auto& input : input_details) {
      int input_index = input.first;
      const string& type_attr_name = input.second;
      NodeDef* cast_node = nullptr;
      TF_RETURN_IF_ERROR(
          GetInputNode(reduction_node->input(input_index), &cast_node));
      DataType original_index_type;
      if (IsCastFromSupportedType(*cast_node, &original_index_type)) {
        reduction_node->set_input(input_index, cast_node->input(0));
        ctx().node_map->UpdateInput(reduction_node->name(),
                                    reduction_node->input(1),
                                    cast_node->input(0));
        SetDataTypeToAttr(original_index_type, type_attr_name, reduction_node);
        optimized = true;
      }
    }

    if (optimized) *simplified_node_name = reduction_node->name();
    return Status::OK();
  }

 private:
  bool IsCastFromSupportedType(const NodeDef& node, DataType* out_input_type) {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_193(mht_193_v, 5049, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "IsCastFromSupportedType");

    if (!IsCast(node)) return false;
    if (!GetNodeAttr(node, "SrcT", out_input_type).ok()) return false;
    return *out_input_type == DT_INT32 || *out_input_type == DT_INT64;
  }
};

}  // namespace

Status ArithmeticOptimizer::SimplifyArithmeticOps(bool can_use_shapes) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_194(mht_194_v, 5061, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ArithmeticOptimizer::SimplifyArithmeticOps");

  SetVector<NodeDef*> nodes_to_simplify;
  nodes_to_simplify.Reserve(optimized_graph_->node_size());
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph_->mutable_node(i));
  }

  const GraphOptimizerContext ctx(&nodes_to_preserve_, optimized_graph_,
                                  graph_properties_.get(), node_map_.get(),
                                  &feed_nodes_, opt_level_);
  const ArithmeticOptimizerContext ctx_ext(&nodes_to_simplify);

  // Stop pipeline after first stage returning non-empty simplified tensor
  // name.
  const auto stop = [](const string& result) {
   std::vector<std::string> mht_195_v;
   mht_195_v.push_back("result: \"" + result + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_195(mht_195_v, 5079, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "lambda");
 return !result.empty(); };
  GraphOptimizerStagePipeline<string> pipeline(stop);
  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;

  if (options_.combine_add_to_addn && can_use_shapes)
    pipeline.AddStage<AddOpsRewriteStage>(ctx, ctx_ext);
  if (options_.fold_conjugate_into_transpose)
    pipeline.AddStage<FoldConjugateIntoTranspose>(ctx, ctx_ext);
  if (options_.fold_multiply_into_conv)
    pipeline.AddStage<FoldMultiplyIntoConv>(ctx, ctx_ext);
  if (options_.fold_transpose_into_matmul)
    pipeline.AddStage<FoldTransposeIntoMatMul>(ctx, ctx_ext);
  if (is_aggressive && options_.hoist_common_factor_out_of_aggregation &&
      can_use_shapes)
    pipeline.AddStage<HoistCommonFactorOutOfAggregation>(ctx, ctx_ext);
  if (options_.minimize_broadcasts && can_use_shapes)
    pipeline.AddStage<MinimizeBroadcasts>(ctx, ctx_ext);
  if (options_.remove_identity_transpose && can_use_shapes)
    pipeline.AddStage<RemoveIdentityTranspose>(ctx, ctx_ext);
  if (options_.remove_involution)
    pipeline.AddStage<RemoveInvolution>(ctx, ctx_ext);
  if (options_.remove_redundant_bitcast)
    pipeline.AddStage<RemoveRedundantBitcastStage>(ctx, ctx_ext);
  if (options_.remove_redundant_cast)
    pipeline.AddStage<RemoveRedundantCastStage>(ctx, ctx_ext);
  if (options_.replace_pack_with_tile_reshape)
    pipeline.AddStage<ReplacePackWithTileReshape>(ctx, ctx_ext);
  if (options_.replace_mul_with_tile && can_use_shapes)
    pipeline.AddStage<ReplaceMulWithBroadcastByTile>(ctx, ctx_ext);
  if (options_.reduce_upsampling_dims)
    pipeline.AddStage<ReduceUpsamplingDims>(ctx, ctx_ext);
  if (options_.remove_redundant_reshape && can_use_shapes)
    pipeline.AddStage<RemoveRedundantReshapeOrBroadcastTo>(ctx, ctx_ext);
  if (options_.remove_negation)
    pipeline.AddStage<RemoveNegationStage>(ctx, ctx_ext);
  if (options_.replace_mul_with_square)
    pipeline.AddStage<ReplaceMulWithSquare>(ctx, ctx_ext);
  if (options_.remove_logical_not)
    pipeline.AddStage<RemoveLogicalNotStage>(ctx, ctx_ext);
  if (options_.reorder_cast_like_and_value_preserving)
    pipeline.AddStage<ReorderCastLikeAndValuePreserving>(ctx, ctx_ext);
  if (options_.simplify_aggregation)
    pipeline.AddStage<SimplifyAggregation>(ctx, ctx_ext);
  if (options_.hoist_cwise_unary_chains)
    pipeline.AddStage<HoistCWiseUnaryChainsStage>(ctx, ctx_ext);
  if (options_.convert_sqrt_div_to_rsqrt_mul)
    pipeline.AddStage<SqrtDivToRsqrtMulStage>(ctx, ctx_ext);
  if (options_.remove_idempotent)
    pipeline.AddStage<RemoveIdempotentStage>(ctx, ctx_ext);
  if (options_.convert_pow) pipeline.AddStage<ConvertPowStage>(ctx, ctx_ext);
  if (options_.convert_log1p)
    pipeline.AddStage<ConvertLog1pStage>(ctx, ctx_ext);
  if (options_.convert_log_softmax)
    pipeline.AddStage<LogSoftmaxStage>(ctx, ctx_ext);
  if (options_.optimize_max_or_min_of_monotonic)
    pipeline.AddStage<OptimizeMaxOrMinOfMonotonicStage>(ctx, ctx_ext);
  if (options_.convert_expm1)
    pipeline.AddStage<ConvertExpm1Stage>(ctx, ctx_ext);
  if (options_.unary_ops_composition)
    pipeline.AddStage<UnaryOpsComposition>(ctx, ctx_ext);
  if (options_.remove_stack_slice_same_axis)
    pipeline.AddStage<RemoveStackSliceSameAxis>(ctx, ctx_ext);
  if (options_.simplify_embedding_lookup)
    pipeline.AddStage<SimplifyEmbeddingLookupStage>(ctx, ctx_ext);
  if (options_.remove_cast_into_segment_reduction)
    pipeline.AddStage<RemoveCastIntoSegmentReductionStage>(ctx, ctx_ext);
  if (options_.fuse_squared_diff)
    pipeline.AddStage<FuseSquaredDiffStage>(ctx, ctx_ext);

  VLOG(1) << "Run " << pipeline.NumStages() << " arithmetic optimizer stages: "
          << absl::StrJoin(pipeline.StageNames(), ", ");

  while (!nodes_to_simplify.Empty()) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = nodes_to_simplify.PopBack();

    string simplified_tensor = "";
    bool optimized = pipeline.PassThroughAllStages(node, &simplified_tensor);

    // If the node was not optimized by any of the stages, go to the next one.
    if (!optimized) continue;

    // re-wire consumers of an old node to the new one
    if (NodeName(simplified_tensor) != node->name()) {
      // Always consider simplified_tensor for further optimizations.
      NodeDef* simplified_node = node_map_->GetNode(simplified_tensor);
      if (simplified_node != nullptr) {
        nodes_to_simplify.PushBack(simplified_node);
      }
      // When `node` is simplified to another node rather than in-place, the
      // consumers of `node` are already redirected to `simplified_tensor`.
      // Re-push the consumers into `nodes_to_simplify` for further
      // optimizations.
      const std::vector<NodeDef*> consumers =
          node_map_->GetOutputsOrderedByNodeName(node->name());
      for (NodeDef* consumer : consumers) {
        // Update `consumer`'s use of `node` to `input`'s operand.
        for (int i = 0; i < consumer->input_size(); ++i) {
          int operand_pos;
          string operand_node_name =
              ParseNodeName(consumer->input(i), &operand_pos);
          if (operand_node_name == node->name()) {
            *consumer->mutable_input(i) =
                (operand_pos < 0
                     ? AsControlDependency(NodeName(simplified_tensor))
                     : simplified_tensor);
          }
        }
        node_map_->UpdateInput(consumer->name(), node->name(),
                               simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
  }
  return Status::OK();
}

Status ArithmeticOptimizer::Optimize(Cluster* /*cluster*/,
                                     const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSarithmetic_optimizerDTcc mht_196(mht_196_v, 5201, "", "./tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc", "ArithmeticOptimizer::Optimize");

  // Set up helper data structures.
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  GrapplerItem optimized_item(item);
  optimized_graph_ = &optimized_item.graph;

  node_map_.reset(new NodeMap(optimized_graph_));
  for (const auto& feed : item.feed) {
    feed_nodes_.insert(NodeName(feed.first));
  }

  // // Disable restricted graph rewrites.
  options_.unary_ops_composition &=
      item.optimization_options().allow_non_differentiable_rewrites;

  // Perform topological sort on the graph in order to help DedupComputations
  // and AddOpsRewrite to optimize larger subgraphs starting from the roots
  // with more inputs.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph_));
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  graph_properties_.reset(new GraphProperties(optimized_item));
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  const Status status =
      graph_properties_->InferStatically(assume_valid_feeds,
                                         /*aggressive_shape_inference=*/false,
                                         /*include_tensor_values=*/false);
  const bool can_use_shapes = status.ok();
  if (!can_use_shapes) {
    VLOG(1) << "Shape inference failed." << status.error_message();
  }

  // Perform the optimizations.
  TF_RETURN_IF_ERROR(SimplifyArithmeticOps(can_use_shapes));
  *optimized_graph = std::move(*optimized_graph_);
  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
