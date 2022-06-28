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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc() {
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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kNHWC[] = "NHWC";
constexpr char kNCHW[] = "NCHW";
constexpr float kVoltaGPURatioThreshold = 0.5;
constexpr float kConvGPUFP16Threshold = 0.5;

struct MutableNodeViewFormatter {
  void operator()(std::string* out, utils::MutableNodeView* node_view) const {
    absl::StrAppend(out, node_view->node()->name());
  }
};

inline std::pair<int, int> GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  int num_volta = 0;
  for (const auto& device : devices) {
    if (device.second.type() != kGPU) {
      continue;
    }
    num_gpus++;
    auto compute_capability_it =
        device.second.environment().find("architecture");
    if (compute_capability_it == device.second.environment().end()) {
      continue;
    }
    double compute_capability = 0.0;
    if (absl::SimpleAtod(compute_capability_it->second, &compute_capability) &&
        compute_capability >= 7.0) {
      num_volta++;
    }
  }
  return {num_gpus, num_volta};
}

inline bool NumConvOnDeviceWithDataTypeOverThreshold(
    const TransposeContext& context, absl::string_view device,
    const DataType& data_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_0(mht_0_v, 244, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "NumConvOnDeviceWithDataTypeOverThreshold");

  int num_conv_gpu = 0;
  int num_conv_gpu_fp16 = 0;

  for (const auto& node : context.graph_view->GetNodes()) {
    const auto* node_def = node.node();
    if (!IsConv2D(*node_def) && !IsConv3D(*node_def)) {
      continue;
    }
    const string& device_name =
        GetDeviceName(context.virtual_placer.get(), *node_def);
    string device_type;
    string task;
    if (!DeviceNameUtils::SplitDeviceName(device_name, &task, &device_type) ||
        !absl::StrContains(absl::AsciiStrToLower(device_type),
                           absl::AsciiStrToLower(device))) {
      continue;
    }
    num_conv_gpu++;
    const auto* t_attr = node.GetAttr("T");
    if (t_attr == nullptr) {
      continue;
    }
    if (t_attr->type() == data_type) {
      num_conv_gpu_fp16++;
    }
  }

  if (num_conv_gpu == 0) return false;

  return (static_cast<float>(num_conv_gpu_fp16) /
          static_cast<float>(num_conv_gpu)) >= kConvGPUFP16Threshold;
}

inline std::pair<string, string> GetSrcAndDstDataFormats(
    const TransposeContext& context, int num_gpus, int num_voltas) {
  string src_format = kNHWC;
  string dst_format = kNCHW;

  const bool is_NHWC_enforced =
      (!context.enforced_layout.empty() && context.enforced_layout == "NHWC");
  const bool should_swap =
      ((static_cast<float>(num_voltas) / static_cast<float>(num_gpus)) >=
       kVoltaGPURatioThreshold) &&
      NumConvOnDeviceWithDataTypeOverThreshold(context, kGPU, DT_HALF);
  // We swap only if NHWC is enforced or no layout is enforced and the devices
  // config meet the thresholds
  if (is_NHWC_enforced || (context.enforced_layout.empty() && should_swap)) {
    std::swap(src_format, dst_format);
  }

  return {src_format, dst_format};
}

Status ExpandLayoutSensitiveOp(TransposeContext* context,
                               TransposerFactory* transposer_factory) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_1(mht_1_v, 302, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "ExpandLayoutSensitiveOp");

  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutSensitiveOp(*node_def)) {
      std::shared_ptr<Transposer> transposer =
          transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            error::NOT_FOUND,
            absl::StrCat(
                "Layout sensitive operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
}

Status ExpandLayoutAgnosticOp(TransposeContext* context,
                              TransposerFactory* transposer_factory) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_2(mht_2_v, 327, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "ExpandLayoutAgnosticOp");

  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutAgnosticOp(*node_def)) {
      const auto& transposer = transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            error::NOT_FOUND,
            absl::StrCat(
                "Layout agnostic operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
}

inline bool IsCancellableConstPermTransposeNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_3(mht_3_v, 352, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "IsCancellableConstPermTransposeNodePair");

  Tensor fanout_tensor;
  if (!GetValueAttrFromConstInputNode(fanout_transpose, IsTranspose, 1,
                                      &fanout_tensor)) {
    return false;
  }
  Tensor fanin_tensor;
  if (!GetValueAttrFromConstInputNode(fanin_transpose, IsTranspose, 1,
                                      &fanin_tensor)) {
    return false;
  }
  if (fanout_tensor.NumElements() != fanin_tensor.NumElements()) {
    return false;
  }

  // Using dst->src to permute on src->dst will result in
  // seq(0, ..., num_elements - 1) if they are cancellable.
  const auto& fanout_tensor_data = fanout_tensor.unaligned_flat<int32>();
  const auto& fanin_tensor_data = fanin_tensor.unaligned_flat<int32>();
  const int num_elements = fanout_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    if (fanout_tensor_data(fanin_tensor_data(i)) != i) {
      return false;
    }
  }
  return true;
}

inline bool IsCancellableDataFormatNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_4(mht_4_v, 385, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "IsCancellableDataFormatNodePair");

  if (!IsDataFormatOp(fanout_transpose) || !IsDataFormatOp(fanin_transpose)) {
    return false;
  }

  auto src_dst_match = [](const utils::MutableNodeView& src,
                          const utils::MutableNodeView& dst) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_5(mht_5_v, 394, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "lambda");

    const auto* src_format = src.GetAttr(kAttrSrcFormat);
    if (src_format == nullptr) {
      return false;
    }
    const auto* dst_format = dst.GetAttr(kAttrDstFormat);
    if (dst_format == nullptr) {
      return false;
    }
    return src_format->s() == dst_format->s();
  };

  // If src_format node A is equal to dst_format of node B and dst_format of
  // node A is equal to src_format of node B, then they are cancellable.
  return src_dst_match(fanin_transpose, fanout_transpose) &&
         src_dst_match(fanout_transpose, fanin_transpose);
}

inline bool IsCancellableNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_6(mht_6_v, 417, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "IsCancellableNodePair");

  return IsCancellableConstPermTransposeNodePair(fanout_transpose,
                                                 fanin_transpose) ||
         IsCancellableDataFormatNodePair(fanout_transpose, fanin_transpose);
}

Status EraseCancellableNodes(TransposeContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_7(mht_7_v, 426, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "EraseCancellableNodes");

  const int original_num_nodes = context->num_nodes;
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();

  for (int i = original_num_nodes; i < num_nodes; ++i) {
    auto* node = graph_view->GetNode(i);
    if (node->NumRegularFanins() < 1) {
      continue;
    }
    const auto& regular_fanin_0 = node->GetRegularFanin(0);
    auto* fanin_node = regular_fanin_0.node_view();
    // TODO(lyandy): Lift restriction once original nodes in the graph can be
    // pruned away.
    if (fanin_node->node_index() < original_num_nodes) {
      continue;
    }
    if (!IsCancellableNodePair(*node, *fanin_node)) {
      continue;
    }
    const auto& fanin_to_forward = fanin_node->GetRegularFanin(0);
    TensorId fanin_id_to_forward(fanin_to_forward.node_view()->GetName(),
                                 fanin_to_forward.index());
    for (const auto& regular_fanout : node->GetRegularFanout(0)) {
      mutation->AddOrUpdateRegularFanin(regular_fanout.node_view(),
                                        regular_fanout.index(),
                                        fanin_id_to_forward);
    }
    mutation->RemoveNode(node);
    if (node->NumRegularFanins() > 1) {
      mutation->RemoveNode(node->GetRegularFanin(1).node_view());
    }
    mutation->RemoveNode(fanin_node);
    if (fanin_node->NumRegularFanins() > 1) {
      mutation->RemoveNode(fanin_node->GetRegularFanin(1).node_view());
    }
  }
  return mutation->Apply();
}

// TODO(ezhulenev): This is a temporary workaround for a graph pattern
// in Resnet models. We should be able to push down transpose nodes across Pad
// and many other ops, and then rely on cancellation to remove them.
//
// From: Transpose[NHWC->NCHW] -> Pad[paddings] -> Transpose[NCHW->NHWC]
// To:   Pad[Permute(paddings)]
Status EraseCancellableNodesAroundPad(TransposeContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_8(mht_8_v, 476, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "EraseCancellableNodesAroundPad");

  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  absl::flat_hash_set<utils::MutableNodeView*> cancelled_transposes;

  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    // Transpose node after Pad.
    auto* transpose_after = graph_view->GetNode(i);
    if (!IsTranspose(*transpose_after->node())) continue;

    // This transpose was already cancelled in previous loop iteration.
    if (cancelled_transposes.contains(transpose_after)) continue;

    // Pad node.
    const auto& transpose_after_fanin = transpose_after->GetRegularFanin(0);
    auto* pad = transpose_after_fanin.node_view();
    if (!IsPad(*pad->node())) continue;

    // Transpose node before Pad.
    const auto& pad_fanin_0 = pad->GetRegularFanin(0);
    auto* transpose_before = pad_fanin_0.node_view();
    if (!IsTranspose(*transpose_before->node())) continue;

    // Transpose before output used once by the Pad node.
    if (transpose_before->NumRegularFanouts() != 1) continue;

    // Transposes are cancellable.
    if (!IsCancellableConstPermTransposeNodePair(*transpose_after,
                                                 *transpose_before))
      continue;

    // Paddings are known constant values.
    Tensor paddings_t;
    if (!GetValueAttrFromConstInputNode(*pad, IsPad, 1, &paddings_t)) continue;

    // Paddings value used once by the pad node only.
    const auto& pad_fanin_1 = pad->GetRegularFanin(1);
    auto* paddings = pad_fanin_1.node_view();
    if (paddings->NumRegularFanouts() != 1) continue;

    // Get permutation after the padding.
    Tensor permute_t;
    if (!GetValueAttrFromConstInputNode(*transpose_after, IsTranspose, 1,
                                        &permute_t))
      continue;

    // Pad output might be used multiple times by different Transpose nodes. If
    // they all have identical permutation, we can cancel all of them.
    std::vector<utils::MutableNodeView*> pad_fanout_transposes;
    pad_fanout_transposes.emplace_back(transpose_after);

    bool pad_has_unsupported_fanout = false;
    for (auto& fanout : pad->GetRegularFanout(0)) {
      auto* extra_transpose = fanout.node_view();
      if (extra_transpose == transpose_after) continue;

      // Check that fanout is a Transpose identical to the transpose_after.
      Tensor extra_permute_t;
      if (!GetValueAttrFromConstInputNode(*extra_transpose, IsTranspose, 1,
                                          &extra_permute_t) ||
          extra_permute_t.tensor_data() != permute_t.tensor_data()) {
        pad_has_unsupported_fanout = true;
        break;
      }

      pad_fanout_transposes.emplace_back(extra_transpose);
    }
    if (pad_has_unsupported_fanout) continue;

    VLOG(0) << "Cancel Transpose nodes around Pad:"
            << " transpose_before=" << transpose_before->node()->name()
            << " pad=" << pad->node()->name() << " transpose_after="
            << absl::StrJoin(pad_fanout_transposes, ",",
                             MutableNodeViewFormatter());

    // Permute paddings in place according to permutation in second transpose.
    auto permutation_s = absl::Span<int32>(permute_t.flat<int32>().data(),
                                           permute_t.NumElements());
    auto paddings_s = absl::Span<int32>(paddings_t.flat<int32>().data(),
                                        paddings_t.NumElements());
    TF_RETURN_IF_ERROR(
        PermuteDouble(absl::StrCat("paddings in ", pad->GetName()),
                      permutation_s, &paddings_s));

    // Update paddings constant value with a permuted tensor.
    AttrValue permuted_paddings_tensor;
    paddings_t.AsProtoTensorContent(permuted_paddings_tensor.mutable_tensor());
    mutation->AddOrUpdateNodeAttr(paddings, "value", permuted_paddings_tensor);

    // Transform Transpose nodes into Identity nodes.
    const auto transpose_to_identity =
        [&cancelled_transposes,
         &mutation](utils::MutableNodeView* transpose) -> void {
      mutation->UpdateNodeOp(transpose, "Identity");
      mutation->RemoveNodeAttr(transpose, "Tperm");
      mutation->RemoveRegularFanin(transpose, 1);
      cancelled_transposes.insert(transpose);
    };

    transpose_to_identity(transpose_before);
    absl::c_for_each(pad_fanout_transposes, transpose_to_identity);
  }

  return mutation->Apply();
}

Status EraseOutputShapeAttrs(TransposeContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_9(mht_9_v, 587, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "EraseOutputShapeAttrs");

  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    auto* node = graph_view->GetNode(i);
    if (IsArg(*node->node())) {
      continue;
    }
    mutation->RemoveNodeAttr(node, kAttrOutputShape);
    TF_RETURN_IF_ERROR(mutation->Apply());
  }
  return Status::OK();
}

}  // namespace

// When there is a GPU, the computation graph is converted to NCHW format.
// When there is only CPU, there will be no conversion by default, unless user
// chose to convert the graph to a desired format. Currently, NCHW -> NHWC
// format conversion is available on CPU.
Status GenericLayoutOptimizer::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizerDTcc mht_10(mht_10_v, 613, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc", "GenericLayoutOptimizer::Optimize");

  if (cluster == nullptr) {
    LOG(WARNING)
        << "generic layout optimizer was called with cluster == nullptr";
    return errors::Aborted("cluster == nullptr.");
  }
  if (!enforced_layout_.empty() && enforced_layout_ != "NHWC" &&
      enforced_layout_ != "NCHW") {
    return Status(
        tensorflow::error::Code::INVALID_ARGUMENT,
        absl::StrCat("Invalid value for enforced_layout: ", enforced_layout_,
                     ". Supported layouts: 'NHWC', 'NCHW'."));
  }
  const auto num_gpus_and_num_volta = GetNumGPUs(*cluster);
  const int num_gpus = num_gpus_and_num_volta.first;

  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;

  TransposeContext context;
  context.enforced_layout = enforced_layout_;

  if (num_gpus > 0) {
    TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
        /*assume_valid_feeds=*/is_aggressive, item, cluster, &context));

    const auto src_dst_formats = GetSrcAndDstDataFormats(
        context, num_gpus, num_gpus_and_num_volta.second);
    context.AssignDeviceAndDataFormats(kGPU, src_dst_formats.first,
                                       src_dst_formats.second);
  } else {
    TF_RETURN_IF_ERROR(TransposeContext::InitializeTransposeContext(
        /*assume_valid_feeds=*/is_aggressive, item, cluster, &context));
    switch (cpu_layout_conversion_) {
      case RewriterConfig::NCHW_TO_NHWC:
        context.AssignDeviceAndDataFormats(kCPU, kNCHW, kNHWC);
        break;
      // TODO(intel-tf): Add functionality for NHWC_TO_NCHW layout conversion on
      // CPU.
      case RewriterConfig::NHWC_TO_NCHW:
        return errors::Aborted(
            "Conversion from NHWC to NCHW is currently not  available for "
            "CPU.");
      default:
        *output = item.graph;
        VLOG(2) << "No layout conversion will take place for CPU.";
        return Status::OK();
    }
  }

  TransposerFactory transposer_factory;
  TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
  if (context.graph.node_size() > context.num_nodes || is_aggressive) {
    TF_RETURN_IF_ERROR(ExpandLayoutAgnosticOp(&context, &transposer_factory));
    TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
    TF_RETURN_IF_ERROR(EraseCancellableNodesAroundPad(&context));
    // TODO(lyandy): Remove sorting once other optimizers are migrated to using
    // `utils::GraphView`.
    TF_RETURN_IF_ERROR(
        context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));
  }
  TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

  *output = context.graph;
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
