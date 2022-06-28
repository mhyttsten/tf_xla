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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc() {
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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kOptimizedSuffix[] = "LayoutOptimizer";
constexpr char kAttrKSize[] = "ksize";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrDilations[] = "dilations";
constexpr char kAttrExplicitPaddings[] = "explicit_paddings";
constexpr char kAttrDataFormat[] = "data_format";
constexpr char kAttrIsTraining[] = "is_training";
constexpr char kAttrValue[] = "value";
constexpr char kAttrN[] = "N";
constexpr char kAttrT[] = "T";
constexpr char kAttrNumSplit[] = "num_split";
constexpr char kAttrNumOuts[] = "num_outs";
constexpr char kAttrKeepDims[] = "keep_dims";
constexpr char kAttrSqueezeDims[] = "squeeze_dims";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpDataFormatVecPermute[] = "DataFormatVecPermute";
constexpr char kOpDataFormatDimMap[] = "DataFormatDimMap";
constexpr char kOpConst[] = "Const";
constexpr char kReshape[] = "Reshape";
constexpr char kReshapeConst[] = "ReshapeConst";
constexpr int kRank = 4;
constexpr int kUnknownRank = -1;
constexpr int kInvalidRank = -2;

inline bool AttrDataFormatMatch(const utils::MutableNodeView& node,
                                absl::string_view src_data_format,
                                bool* missing) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("src_data_format: \"" + std::string(src_data_format.data(), src_data_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_0(mht_0_v, 245, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "AttrDataFormatMatch");

  const auto* attr = node.GetAttr(kAttrDataFormat);
  if (attr != nullptr) {
    return attr->s() == src_data_format;
  }
  *missing = true;
  return false;
}

inline bool AttrDataFormatMatch(const utils::MutableNodeView& node,
                                absl::string_view src_data_format) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("src_data_format: \"" + std::string(src_data_format.data(), src_data_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_1(mht_1_v, 259, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "AttrDataFormatMatch");

  bool missing = false;
  return AttrDataFormatMatch(node, src_data_format, &missing);
}

bool IsNonFloatingConv2D(const utils::MutableNodeView& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_2(mht_2_v, 267, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsNonFloatingConv2D");

  if (IsConv2D(*node.node()) || IsConv2DBackpropInput(*node.node())) {
    const auto* attr = node.GetAttr(kAttrT);
    if (attr != nullptr) {
      return !kDataTypeIsFloating.Contains(attr->type());
    }
  }
  return false;
}

// Utils for layout agnostic transposer.

bool IsComparisonOp(const NodeDef& node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_3(mht_3_v, 282, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsComparisonOp");

  bool is_compare = IsApproximateEqual(node) || IsEqual(node) ||
                    IsGreater(node) || IsGreaterEqual(node) || IsLess(node) ||
                    IsLessEqual(node) || IsNotEqual(node);
  return is_compare;
}

std::vector<int> GetRegularFaninPorts(const utils::MutableNodeView& node) {
  const int num_regular_fanins = node.NumRegularFanins();
  std::vector<int> values(num_regular_fanins);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

std::vector<int> GetConcatDataFaninPorts(const utils::MutableNodeView& node) {
  const auto* n_attr = node.GetAttr(kAttrN);
  const int n = n_attr != nullptr ? n_attr->i() : 0;
  const int start = (node.GetOp() == "Concat") ? 1 : 0;
  const int end = start + n;
  std::vector<int> values(end - start);
  std::iota(values.begin(), values.end(), start);
  return values;
}

struct ComparatorByNodeNameAndIndex {
  bool operator()(const utils::MutableFaninView& node1,
                  const utils::MutableFaninView& node2) const {
    auto* node1_view = node1.node_view();
    auto* node2_view = node2.node_view();
    auto name_compare = node1_view->GetName().compare(node2_view->GetName());
    if (name_compare == 0) {
      return node1.index() < node2.index();
    }
    return name_compare < 0;
  }
};

bool IsHostMemory(const NodeDef& node, int output_port) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_4(mht_4_v, 322, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsHostMemory");

  DeviceNameUtils::ParsedName parsed_name;
  if (DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
    DeviceType device_type(parsed_name.type);
    Status s = FindKernelDef(device_type, node, nullptr, nullptr);
    if (s.ok()) {
      tensorflow::MemoryTypeVector in_mtypes;
      tensorflow::MemoryTypeVector out_mtypes;
      s = tensorflow::MemoryTypesForNode(OpRegistry::Global(), device_type,
                                         node, &in_mtypes, &out_mtypes);
      if (s.ok()) {
        if (out_mtypes[output_port] == HOST_MEMORY) {
          return true;
        }
      }
    } else {
      return true;
    }
  }
  return false;
}

std::vector<int> GetDimensionIndicesFromLabel(
    const absl::flat_hash_map<char, int>& dim_indices,
    absl::Span<const char> labels) {
  std::vector<int> indices;
  indices.reserve(labels.size());
  for (const auto& label : labels) {
    indices.push_back(dim_indices.at(label));
  }
  return indices;
}

// RAII-styled object for keeping track of 4D to 5D data format
// upgrade/conversion. Currently only NHWC -> NDHWC and NCHW -> NCDHW are
// supported.
class ScopedDataFormatUpgrader {
 public:
  ScopedDataFormatUpgrader(TransposeContext* context, int rank)
      : context_(context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_5(mht_5_v, 364, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ScopedDataFormatUpgrader");

    if (rank == 5 && IsSupportedDataFormat(context_->src_format) &&
        IsSupportedDataFormat(context_->dst_format)) {
      old_src_format_ = context_->src_format;
      old_dst_format_ = context_->dst_format;
      std::string new_src_format = GetUpgradedDataFormat(context_->src_format);
      std::string new_dst_format = GetUpgradedDataFormat(context_->dst_format);
      context_->AssignDeviceAndDataFormats(context_->target_device,
                                           new_src_format, new_dst_format);
      upgraded_ = true;
    }
  }

  ScopedDataFormatUpgrader(const ScopedDataFormatUpgrader&) = delete;
  ScopedDataFormatUpgrader& operator=(const ScopedDataFormatUpgrader&) = delete;

  ~ScopedDataFormatUpgrader() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_6(mht_6_v, 383, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "~ScopedDataFormatUpgrader");

    if (upgraded_) {
      context_->AssignDeviceAndDataFormats(context_->target_device,
                                           old_src_format_, old_dst_format_);
    }
  }

 private:
  bool IsSupportedDataFormat(absl::string_view data_format) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("data_format: \"" + std::string(data_format.data(), data_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_7(mht_7_v, 395, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsSupportedDataFormat");

    return data_format == "NHWC" || data_format == "NCHW";
  }

  std::string GetUpgradedDataFormat(absl::string_view data_format) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("data_format: \"" + std::string(data_format.data(), data_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_8(mht_8_v, 403, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "GetUpgradedDataFormat");

    if (data_format == "NHWC") {
      return "NDHWC";
    }

    DCHECK_EQ(data_format, "NCHW");
    return "NCDHW";
  }

  TransposeContext* context_ = nullptr;
  bool upgraded_ = false;
  std::string old_src_format_;
  std::string old_dst_format_;
};

}  // namespace

// TransposeContext.

Status TransposeContext::InitializeTransposeContext(bool assume_valid_feeds,
                                                    const GrapplerItem& item,
                                                    const Cluster* cluster,
                                                    TransposeContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_9(mht_9_v, 428, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "TransposeContext::InitializeTransposeContext");

  DCHECK(context != nullptr);
  context->graph_properties = absl::make_unique<GraphProperties>(item);
  TF_RETURN_IF_ERROR(
      context->graph_properties->InferStatically(assume_valid_feeds));
  TF_RETURN_IF_ERROR(
      context->graph_properties->AnnotateOutputShapes(&context->graph));
  Status status;
  context->graph_view =
      absl::make_unique<utils::MutableGraphView>(&context->graph, &status);
  TF_RETURN_IF_ERROR(status);
  context->num_nodes = context->graph.node_size();
  const auto& nodes_to_preserve = item.NodesToPreserve();
  context->nodes_to_preserve = absl::flat_hash_set<string>(
      nodes_to_preserve.begin(), nodes_to_preserve.end());
  TF_RETURN_IF_ERROR(context->frames.InferFromGraph(context->graph));
  if (cluster != nullptr) {
    context->virtual_placer =
        absl::make_unique<const VirtualPlacer>(cluster->GetDevices());
  }
  return Status::OK();
}

// Sets data formats to convert from and to for specified device type.
void TransposeContext::AssignDeviceAndDataFormats(
    absl::string_view target_device, absl::string_view src_format,
    absl::string_view dst_format) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("target_device: \"" + std::string(target_device.data(), target_device.size()) + "\"");
   mht_10_v.push_back("src_format: \"" + std::string(src_format.data(), src_format.size()) + "\"");
   mht_10_v.push_back("dst_format: \"" + std::string(dst_format.data(), dst_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_10(mht_10_v, 460, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "TransposeContext::AssignDeviceAndDataFormats");

  this->target_device = string(target_device);
  this->src_format = string(src_format);
  this->dst_format = string(dst_format);
  this->src_dim_indices = GetDimensionIndices(src_format);
  this->dst_dim_indices = GetDimensionIndices(dst_format);
  this->src_to_dst = GetPermutation(this->src_dim_indices, dst_format);
  this->dst_to_src = GetPermutation(this->dst_dim_indices, src_format);
}

// Transposer.

bool Transposer::ShouldProcess(const TransposeContext& context,
                               const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_11(mht_11_v, 476, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::ShouldProcess");

  const auto* node_def = node.node();
  const string& device_name =
      GetDeviceName(context.virtual_placer.get(), *node_def);
  string device;
  string task;
  const bool is_on_target_device =
      DeviceNameUtils::SplitDeviceName(device_name, &task, &device) &&
      absl::StrContains(absl::AsciiStrToLower(device),
                        absl::AsciiStrToLower(context.target_device));

  // Only checks data format for layout sensitive op.
  const bool data_format_match = !IsLayoutSensitiveOp(*node_def) ||
                                 AttrDataFormatMatch(node, context.src_format);

  // Only transposes floating point nodes.
  const bool is_integer_conv2d = IsNonFloatingConv2D(node);

  return is_on_target_device && data_format_match && !is_integer_conv2d &&
         !context.nodes_to_preserve.contains(node_def->name()) &&
         !(node.NumRegularFanouts() == 0 && node.NumControlledFanouts() == 0);
}

Status Transposer::CreateConstPermNode(TransposeContext* context,
                                       absl::string_view node_name,
                                       absl::string_view device,
                                       absl::Span<const int> permutation,
                                       absl::string_view control_node_name,
                                       utils::MutationNewNode* added_node) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_12_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   mht_12_v.push_back("control_node_name: \"" + std::string(control_node_name.data(), control_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_12(mht_12_v, 510, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::CreateConstPermNode");

  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));

  NodeDef node;
  node.set_name(string(node_name));
  node.set_op(kOpConst);
  node.set_device(string(device));

  if (!control_node_name.empty()) {
    node.add_input(string(control_node_name));
  }

  AttrValue attr_data_type;
  attr_data_type.set_type(DT_INT32);
  node.mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  Tensor tensor(DT_INT32, TensorShape({(long long)permutation.size()}));
  for (int i = 0, end = permutation.size(); i < end; i++) {
    tensor.flat<int>()(i) = permutation[i];
  }
  tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
  node.mutable_attr()->insert({"value", attr_tensor});

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::CreateTransposeNode(
    TransposeContext* context, absl::string_view name_format,
    const DataType& data_type, absl::string_view device,
    TensorShapeProto fanin_shape, absl::Span<const int> permutation,
    absl::string_view control_node_name, utils::MutationNewNode* added_node,
    string* transpose_node_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name_format: \"" + std::string(name_format.data(), name_format.size()) + "\"");
   mht_13_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   mht_13_v.push_back("control_node_name: \"" + std::string(control_node_name.data(), control_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_13(mht_13_v, 552, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::CreateTransposeNode");

  const string node_name = absl::Substitute(name_format, kOpTranspose);
  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));
  *transpose_node_name = node_name;

  NodeDef node;
  node.set_name(node_name);
  node.set_op(kOpTranspose);
  node.set_device(string(device));

  AttrValue attr_data_type;
  attr_data_type.set_type(data_type);
  node.mutable_attr()->insert({"T", attr_data_type});

  AttrValue attr_data_type_perm;
  attr_data_type_perm.set_type(DT_INT32);
  node.mutable_attr()->insert({"Tperm", attr_data_type_perm});

  if (!fanin_shape.unknown_rank()) {
    TF_RETURN_IF_ERROR(
        PermuteSingle(absl::StrCat("fanin shape in", node.name()), permutation,
                      fanin_shape.mutable_dim()));
    AttrValue attr_output_shape;
    *attr_output_shape.mutable_list()->add_shape() = fanin_shape;
    node.mutable_attr()->insert({kAttrOutputShape, attr_output_shape});
  }

  // Create Const Node
  utils::MutationNewNode const_perm_added_node;
  const string const_perm_node_name =
      absl::Substitute(name_format, "PermConst");
  TF_RETURN_IF_ERROR(CreateConstPermNode(context, const_perm_node_name, device,
                                         permutation, control_node_name,
                                         &const_perm_added_node));
  // Add place holder for 1st input.
  node.add_input("");
  // Connect const_perm_node to 2nd input of transpose_node.
  node.add_input(const_perm_node_name);

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::UpdateFaninEdgesWithOp(TransposeContext* context,
                                          absl::Span<const int> dst_ports,
                                          utils::MutableNodeView* dst_node,
                                          absl::string_view op) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_14(mht_14_v, 605, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::UpdateFaninEdgesWithOp");

  const bool is_in_frame = context->frames.IsInFrame(*dst_node->node());
  for (int dst_port : dst_ports) {
    auto& fanin_port = dst_node->GetRegularFanin(dst_port);
    auto* fanin_node_view = fanin_port.node_view();

    TF_RETURN_IF_ERROR(
        UpdateEdge(context,
                   GetFaninNameFormat(dst_node->GetName(), dst_port,
                                      context->src_format, context->dst_format),
                   op, /*input_shape=*/nullptr, /*is_in_frame=*/is_in_frame,
                   /*is_src_format_to_dst_format=*/true, fanin_port.index(),
                   dst_port, fanin_node_view, dst_node));
  }
  return Status::OK();
}

Status Transposer::UpdateFanoutEdgesWithOp(TransposeContext* context,
                                           absl::Span<const int> src_ports,
                                           utils::MutableNodeView* src_node,
                                           absl::string_view op) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_15(mht_15_v, 629, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::UpdateFanoutEdgesWithOp");

  // Update attr _output_shapes for output ports.
  const auto* output_shape_attr = src_node->GetAttr(kAttrOutputShape);
  AttrValue shape_attr_copy;
  if (op == kOpTranspose && output_shape_attr != nullptr) {
    shape_attr_copy = *output_shape_attr;
    for (int port : src_ports) {
      auto* shape = shape_attr_copy.mutable_list()->mutable_shape(port);
      if (shape->unknown_rank()) continue;
      TF_RETURN_IF_ERROR(
          PermuteSingle(absl::StrCat("output shape attribute at port ", port,
                                     " in", src_node->GetName()),
                        context->src_to_dst, shape->mutable_dim()));
    }
    context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
        src_node, kAttrOutputShape, shape_attr_copy);
  }

  const bool is_in_frame = context->frames.IsInFrame(*src_node->node());
  // We might modify the output set in the loop. Make a copy first.
  // Use a set with custom comparator to order output nodes by node name,
  // so that we can keep transposer name deterministic.
  for (int src_port : src_ports) {
    const auto& fanouts_src_port = src_node->GetRegularFanout(src_port);
    std::vector<utils::MutableFaninView> sorted_fanouts(
        fanouts_src_port.begin(), fanouts_src_port.end());
    std::sort(sorted_fanouts.begin(), sorted_fanouts.end(),
              ComparatorByNodeNameAndIndex());
    int num_downstream_transposers = 0;
    for (const auto& fanout : sorted_fanouts) {
      TF_RETURN_IF_ERROR(UpdateEdge(
          context,
          GetFanoutNameFormat(src_node->GetName(), src_port,
                              num_downstream_transposers++, context->src_format,
                              context->dst_format),
          op, &shape_attr_copy, /*is_in_frame=*/is_in_frame,
          /*is_src_format_to_dst_format=*/false, src_port, fanout.index(),
          src_node, fanout.node_view()));
    }
  }
  return Status::OK();
}

Status Transposer::CreateDataFormatNode(
    TransposeContext* context, absl::string_view node_name,
    absl::string_view op, absl::string_view device, const DataType& data_type,
    bool is_fanin_on_host, bool is_src_format_to_dst_format,
    utils::MutationNewNode* added_node) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_16_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   mht_16_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_16(mht_16_v, 682, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::CreateDataFormatNode");

  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));

  // Create the node
  NodeDef node;
  node.set_name(string(node_name));

  // Set up parameters of node.
  node.set_op(string(op));
  node.set_device(string(device));
  AttrValue attr_data_type;
  attr_data_type.set_type(data_type);
  node.mutable_attr()->insert({"T", attr_data_type});

  // The inputs of a DataFormat op could be in host memory for ops such as
  // Reshape. In such cases, run the kernel on the host too.
  if (is_fanin_on_host) {
    AttrValue attr_kernel;
    attr_kernel.set_s("host");
    node.mutable_attr()->insert({"_kernel", attr_kernel});
  }

  AttrValue src_format;
  src_format.set_s(is_src_format_to_dst_format ? context->src_format
                                               : context->dst_format);
  node.mutable_attr()->insert({kAttrSrcFormat, src_format});
  AttrValue dst_format;
  dst_format.set_s(is_src_format_to_dst_format ? context->dst_format
                                               : context->src_format);
  node.mutable_attr()->insert({kAttrDstFormat, dst_format});

  // Add place holder for 1st input field.
  node.add_input("");

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::UpdateEdge(
    TransposeContext* context, absl::string_view name_format,
    absl::string_view op, const AttrValue* input_shape, bool is_in_frame,
    bool is_src_format_to_dst_format, const int src_port, const int dst_port,
    utils::MutableNodeView* src_node, utils::MutableNodeView* dst_node) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name_format: \"" + std::string(name_format.data(), name_format.size()) + "\"");
   mht_17_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_17(mht_17_v, 732, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::UpdateEdge");

  DCHECK(src_node != nullptr);
  DCHECK(dst_node != nullptr);
  auto* src_node_def = src_node->node();
  auto* dst_node_def = dst_node->node();

  // TODO(lyandy): Minimize device parsing/fetching.
  const string device = GetDeviceName(
      context->virtual_placer.get(),
      is_src_format_to_dst_format ? *dst_node_def : *src_node_def);
  DataType data_type =
      is_src_format_to_dst_format
          ? context->graph_properties
                ->GetInputProperties(dst_node->GetName())[dst_port]
                .dtype()
          : context->graph_properties
                ->GetOutputProperties(src_node->GetName())[src_port]
                .dtype();

  utils::MutationNewNode added_node;
  string added_node_name;
  if (op == kOpTranspose) {
    TensorShapeProto input_shape_proto;
    input_shape_proto.set_unknown_rank(true);
    if (input_shape != nullptr) {
      input_shape_proto = input_shape->list().shape(src_port);
    } else {
      const auto* src_node_shape_attr = src_node->GetAttr(kAttrOutputShape);
      if (src_node_shape_attr != nullptr) {
        input_shape_proto = src_node_shape_attr->list().shape(src_port);
      }
    }
    const string control_node_name =
        is_in_frame ? AsControlDependency(src_node_def->name()) : "";
    const std::vector<int>& permutation =
        is_src_format_to_dst_format ? context->src_to_dst : context->dst_to_src;
    TF_RETURN_IF_ERROR(CreateTransposeNode(
        context, name_format, data_type, device, input_shape_proto, permutation,
        control_node_name, &added_node, &added_node_name));
  } else if (op == kOpDataFormatVecPermute || op == kOpDataFormatDimMap) {
    DeviceNameUtils::ParsedName parsed_name;
    bool is_fanin_on_host =
        DeviceNameUtils::ParseFullName(
            GetDeviceName(context->virtual_placer.get(), *src_node_def),
            &parsed_name) &&
        parsed_name.type != "CPU" && IsHostMemory(*src_node_def, src_port);
    const string node_name = absl::Substitute(name_format, op);
    TF_RETURN_IF_ERROR(CreateDataFormatNode(
        context, node_name, op, device, data_type, is_fanin_on_host,
        is_src_format_to_dst_format, &added_node));
    added_node_name = node_name;
  } else {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Unsupported op \"", op,
                               "\". Supported ops are Transpose, "
                               "DataFormatVecPerm, DataFormatDimMap."));
  }

  // Connect src_node to 1st input of added_node.
  utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
  mutation->AddOrUpdateRegularFanin(added_node, 0,
                                    {src_node->GetName(), src_port});

  // Connect output of added_node to dst_node:dst_port.
  mutation->AddOrUpdateRegularFanin(dst_node, dst_port, {added_node_name, 0});

  return Status::OK();
}

int Transposer::GetFanoutPortRank(const utils::MutableNodeView& node,
                                  int port) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_18(mht_18_v, 805, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetFanoutPortRank");

  const auto* output_shape_attr = node.GetAttr(kAttrOutputShape);
  if (output_shape_attr == nullptr ||
      output_shape_attr->list().shape_size() <= port) {
    return kInvalidRank;
  }
  const auto& shape = output_shape_attr->list().shape(port);
  if (shape.unknown_rank()) {
    return kUnknownRank;
  }
  return shape.dim_size();
}

bool Transposer::IsFanoutPortRankN(const utils::MutableNodeView& node, int port,
                                   int n) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_19(mht_19_v, 822, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::IsFanoutPortRankN");

  return GetFanoutPortRank(node, port) == n;
}

bool Transposer::IsFanoutPortsRankN(const utils::MutableNodeView& node,
                                    absl::Span<const int> ports, int n) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_20(mht_20_v, 830, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::IsFanoutPortsRankN");

  for (const auto& port : ports) {
    if (!IsFanoutPortRankN(node, port, n)) {
      return false;
    }
  }
  return true;
}

int Transposer::GetFaninPortRank(const utils::MutableNodeView& node,
                                 int port) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_21(mht_21_v, 843, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetFaninPortRank");

  if (port < node.NumRegularFanins() && port >= 0) {
    const auto& regular_fanin = node.GetRegularFanin(port);
    return GetFanoutPortRank(*regular_fanin.node_view(), regular_fanin.index());
  }
  return kInvalidRank;
}

bool Transposer::IsFaninPortRankN(const utils::MutableNodeView& node, int port,
                                  int n) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_22(mht_22_v, 855, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::IsFaninPortRankN");

  return GetFaninPortRank(node, port) == n;
}

bool Transposer::IsFaninPortDimsNIfConst(const utils::MutableNodeView& node,
                                         int port,
                                         absl::Span<const int> dims) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_23(mht_23_v, 864, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::IsFaninPortDimsNIfConst");

  if (port < node.NumRegularFanins() && port >= 0) {
    const auto& regular_fanin = node.GetRegularFanin(port);
    const auto* fanin_node_view = regular_fanin.node_view();
    if (!IsConstant(*fanin_node_view->node())) {
      return true;
    }
    // If fanin is a Const, check tensor to see if dimensions match.
    const auto* value_attr = fanin_node_view->GetAttr(kAttrValue);
    if (value_attr == nullptr) {
      return false;
    }
    Tensor tensor;
    if (!tensor.FromProto(value_attr->tensor())) {
      return false;
    }
    const int dims_size = dims.size();
    if (tensor.dims() != dims_size) {
      return false;
    }
    for (int i = 0; i < dims_size; ++i) {
      if (tensor.dim_size(i) != dims[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool Transposer::IsFaninPortsDimsNIfConst(const utils::MutableNodeView& node,
                                          absl::Span<const int> ports,
                                          absl::Span<const int> dims) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_24(mht_24_v, 899, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::IsFaninPortsDimsNIfConst");

  for (const auto& port : ports) {
    if (!IsFaninPortDimsNIfConst(node, port, dims)) {
      return false;
    }
  }
  return true;
}

bool Transposer::CanProcessNode(const TransposeContext& context,
                                const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_25(mht_25_v, 912, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::CanProcessNode");

  return !context.nodes_to_preserve.contains(node.GetName()) &&
         !(node.NumRegularFanouts() == 0 && node.NumControlledFanouts() == 0);
}

string Transposer::GetFaninNameFormat(absl::string_view node_name, int port,
                                      absl::string_view src_format,
                                      absl::string_view dst_format) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_26_v.push_back("src_format: \"" + std::string(src_format.data(), src_format.size()) + "\"");
   mht_26_v.push_back("dst_format: \"" + std::string(dst_format.data(), dst_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_26(mht_26_v, 925, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetFaninNameFormat");

  return absl::StrCat(node_name, "-", port, "-$0", src_format, "To", dst_format,
                      "-", kOptimizedSuffix);
}

string Transposer::GetFanoutNameFormat(absl::string_view node_name, int port,
                                       int index, absl::string_view src_format,
                                       absl::string_view dst_format) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_27_v.push_back("src_format: \"" + std::string(src_format.data(), src_format.size()) + "\"");
   mht_27_v.push_back("dst_format: \"" + std::string(dst_format.data(), dst_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_27(mht_27_v, 938, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetFanoutNameFormat");

  return absl::StrCat(node_name, "-", port, "-", index, "-$0", dst_format, "To",
                      src_format, "-", kOptimizedSuffix);
}

string Transposer::LayoutOptimizerNode(absl::string_view node_name) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_28(mht_28_v, 947, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::LayoutOptimizerNode");

  return absl::StrCat(node_name, "-", kOptimizedSuffix);
}

string Transposer::GetReshapeNodeNameFormat(absl::string_view node_name,
                                            int index,
                                            absl::string_view src_format,
                                            absl::string_view dst_format) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_29_v.push_back("src_format: \"" + std::string(src_format.data(), src_format.size()) + "\"");
   mht_29_v.push_back("dst_format: \"" + std::string(dst_format.data(), dst_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_29(mht_29_v, 960, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetReshapeNodeNameFormat");

  return absl::StrCat(node_name, "-", index, "-", kReshape, src_format, "To",
                      dst_format);
}

string Transposer::GetShapeConstNodeNameFormat(absl::string_view node_name,
                                               int index) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_30(mht_30_v, 970, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Transposer::GetShapeConstNodeNameFormat");

  return absl::StrCat(node_name, "-", index, "-", kReshapeConst);
}

// Layout sensitive transposer.

inline string GetLayoutSensitiveNodeDataFormat(
    const utils::MutableNodeView& node) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_31(mht_31_v, 980, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "GetLayoutSensitiveNodeDataFormat");

  const auto* attr = node.GetAttr(kAttrDataFormat);
  if (attr != nullptr) {
    return attr->s();
  }
  return "";
}

Status LayoutSensitiveOpTransposer::UpdateNode(TransposeContext* context,
                                               utils::MutableNodeView* node) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_32(mht_32_v, 992, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "LayoutSensitiveOpTransposer::UpdateNode");

  utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
  AttrValue data_format_attr;
  data_format_attr.set_s(context->dst_format);
  mutation->AddOrUpdateNodeAttr(node, kAttrDataFormat, data_format_attr);

  auto permute_attr = [&context, &node,
                       &mutation](absl::string_view attr_name) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_33(mht_33_v, 1003, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "lambda");

    const auto* attr = node->GetAttr(attr_name);
    if (attr != nullptr) {
      AttrValue attr_copy(*attr);
      TF_RETURN_IF_ERROR(PermuteSingle(
          absl::StrCat(attr_name, " attribute in", node->GetName()),
          context->src_to_dst, attr_copy.mutable_list()->mutable_i()));
      mutation->AddOrUpdateNodeAttr(node, attr_name, attr_copy);
    }
    return Status::OK();
  };

  // Update attrs.
  TF_RETURN_IF_ERROR(permute_attr(kAttrStrides));
  TF_RETURN_IF_ERROR(permute_attr(kAttrKSize));
  TF_RETURN_IF_ERROR(permute_attr(kAttrDilations));

  const auto* explicit_paddings_attr = node->GetAttr(kAttrExplicitPaddings);
  if (explicit_paddings_attr != nullptr && explicit_paddings_attr->has_list() &&
      explicit_paddings_attr->list().i_size() > 0) {
    AttrValue explicit_paddings_attr_copy(*explicit_paddings_attr);
    TF_RETURN_IF_ERROR(PermuteDouble(
        absl::StrCat("explicit_paddings attribute in", node->GetName()),
        context->src_to_dst,
        explicit_paddings_attr_copy.mutable_list()->mutable_i()));
    mutation->AddOrUpdateNodeAttr(node, kAttrExplicitPaddings,
                                  explicit_paddings_attr_copy);
  }

  return Status::OK();
}

Status DefaultLayoutSensitiveOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_34(mht_34_v, 1039, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "DefaultLayoutSensitiveOpTransposer::TransposeNode");

  DCHECK(IsDefaultLayoutSensitiveOp(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status AvgPoolGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_35(mht_35_v, 1062, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "AvgPoolGradTransposer::TransposeNode");

  DCHECK(IsAvgPoolGrad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 1, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status BiasAddTransposer::TransposeNode(TransposeContext* context,
                                        utils::MutableNodeView* node) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_36(mht_36_v, 1082, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BiasAddTransposer::TransposeNode");

  // This TransposeNode allows for BiasAdd but not BiasAddV1, since BiasAdd
  // supports different data format.
  DCHECK(IsBiasAddV2(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, rank)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  // BiasAdd itself only needs NCHW/NHWC to determine whether C dim is the
  // second or the last dim. Therefore, we use the original 4D data format in
  // the context to update the node. For the input/output tensor, the
  // corresponding 4D or 5D data format is needed.
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status BiasAddGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_37(mht_37_v, 1111, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BiasAddGradTransposer::TransposeNode");

  DCHECK(IsBiasAddGrad(*node->node()));
  const int rank = GetFaninPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  // BiasAddGrad itself only needs NCHW/NHWC to determine whether C dim is the
  // second or the last dim. Therefore, we use the original 4D data format in
  // the context to update the node. For the input tensor, the corresponding 4D
  // or 5D data format is needed.
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape 1-D with size the
  // feature dimension of `out_backprop`, regardless of whether NCHW or NHWC is
  // used.
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_38(mht_38_v, 1140, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Conv2DBackpropFilterTransposer::TransposeNode");

  DCHECK(IsConv2DBackpropFilter(*node->node()) ||
         IsDepthwiseConv2dNativeBackpropFilter(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 2}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropInputTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_39(mht_39_v, 1162, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Conv2DBackpropInputTransposer::TransposeNode");

  DCHECK(IsConv2DBackpropInput(*node->node()) ||
         IsDepthwiseConv2dNativeBackpropInput(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }

  const auto& fanin = node->GetRegularFanin(0);
  auto* fanin_node = fanin.node_view();
  const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
  if (output_shape_attr == nullptr) {
    VLOG(3) << "Cannot compute the shape of " << fanin_node->GetName()
            << " because it is missing attribute " << kAttrOutputShape;
    return Status::OK();
  }
  TensorShapeProto fanin_shape = output_shape_attr->list().shape(fanin.index());
  if (fanin_shape.dim_size() != 1) {
    VLOG(3) << fanin_node->GetName() << " is not a vector.";
    return Status::OK();
  }

  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_40(mht_40_v, 1198, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Conv3DTransposer::TransposeNode");

  DCHECK(IsConv3D(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_41(mht_41_v, 1221, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Conv3DBackpropFilterTransposer::TransposeNode");

  DCHECK(IsConv3DBackpropFilterV2(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 2}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv3DBackpropInputTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_42(mht_42_v, 1247, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "Conv3DBackpropInputTransposer::TransposeNode");

  DCHECK(IsConv3DBackpropInputV2(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status FusedBatchNormExTransposer::TransposeNode(TransposeContext* context,
                                                 utils::MutableNodeView* node) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_43(mht_43_v, 1272, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "FusedBatchNormExTransposer::TransposeNode");

  DCHECK(IsFusedBatchNormEx(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  if (node->NumRegularFanins() == 6) {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 5}, node, kOpTranspose));
  } else {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  }
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool FusedBatchNormGradTransposer::IsTraining(
    const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_44(mht_44_v, 1296, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "FusedBatchNormGradTransposer::IsTraining");

  const auto* is_training_attr = node.GetAttr(kAttrIsTraining);
  if (is_training_attr != nullptr) {
    return is_training_attr->b();
  }
  return false;
}

Status FusedBatchNormGradTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_45(mht_45_v, 1308, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "FusedBatchNormGradTransposer::TransposeNode");

  DCHECK(IsFusedBatchNormGrad(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) || !IsTraining(*node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_46(mht_46_v, 1332, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "MaxPoolV2Transposer::TransposeNode");

  DCHECK(IsMaxPoolV2(*node->node()));
  // We check data_input's shape instead, because the shape inference of
  // MaxPoolV2 is not able to infer the shape when ksize or strides is not
  // constant.
  const auto& data_fanin = node->GetRegularFanin(0);
  auto* data_fanin_node = data_fanin.node_view();
  if (!ShouldProcess(*context, *node) ||
      !IsFanoutPortRankN(*data_fanin_node, data_fanin.index(), 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_47(mht_47_v, 1358, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "MaxPoolGradTransposer::TransposeNode");

  DCHECK(IsMaxPoolGrad(*node->node()) || IsMaxPoolGradGradV1(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradV2Transposer::TransposeNode(TransposeContext* context,
                                              utils::MutableNodeView* node) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_48(mht_48_v, 1377, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "MaxPoolGradV2Transposer::TransposeNode");

  DCHECK(IsMaxPoolGradV2(*node->node()) || IsMaxPoolGradGradV2(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {3, 4}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

// Layout agnostic transposer.

inline bool IsValidConstPermTransposeNode(const utils::MutableNodeView& node,
                                          absl::Span<const int> permutation) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_49(mht_49_v, 1400, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsValidConstPermTransposeNode");

  Tensor tensor;
  if (!GetValueAttrFromConstInputNode(node, IsTranspose, 1, &tensor)) {
    return false;
  }
  const int permutation_size = permutation.size();
  if (tensor.NumElements() != permutation_size) {
    return false;
  }

  const auto& tensor_data = tensor.unaligned_flat<int32>();
  for (int i = 0; i < permutation_size; i++) {
    if (permutation[i] != tensor_data(i)) {
      return false;
    }
  }
  return true;
}

inline bool IsValidDataFormatNode(const utils::MutableNodeView& node,
                                  absl::string_view src_format,
                                  absl::string_view dst_format) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("src_format: \"" + std::string(src_format.data(), src_format.size()) + "\"");
   mht_50_v.push_back("dst_format: \"" + std::string(dst_format.data(), dst_format.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_50(mht_50_v, 1426, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsValidDataFormatNode");

  if (!IsDataFormatOp(node)) {
    return false;
  }
  const auto* src_format_attr = node.GetAttr(kAttrSrcFormat);
  if (src_format_attr == nullptr || src_format_attr->s() != src_format) {
    return false;
  }
  const auto* dst_format_attr = node.GetAttr(kAttrDstFormat);
  if (dst_format_attr == nullptr || dst_format_attr->s() != dst_format) {
    return false;
  }
  return true;
}

inline bool IsLayoutOptimizerAddedDstToSrcTranspose(
    const TransposeContext& context, const utils::MutableNodeView& node) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_51(mht_51_v, 1445, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsLayoutOptimizerAddedDstToSrcTranspose");

  return node.node_index() >= context.num_nodes &&
         IsValidConstPermTransposeNode(node, context.dst_to_src);
}

inline bool IsLayoutOptimizerAddedDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_52(mht_52_v, 1454, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsLayoutOptimizerAddedDstToSrcTransform");

  return node.node_index() >= context.num_nodes &&
         (IsValidConstPermTransposeNode(node, context.dst_to_src) ||
          IsValidDataFormatNode(node, context.dst_format, context.src_format));
}

bool LayoutAgnosticOpTransposer::IsAfterDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_53(mht_53_v, 1464, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "LayoutAgnosticOpTransposer::IsAfterDstToSrcTransform");

  std::deque<utils::MutableNodeView*> queue;
  absl::flat_hash_set<utils::MutableNodeView*> visited_nodes;
  auto data_node_pos = GetDataFaninPorts(node);
  for (const int pos : data_node_pos) {
    const auto& fanin = node.GetRegularFanin(pos);
    auto* fanin_node = fanin.node_view();
    queue.push_back(fanin_node);
    visited_nodes.insert(fanin_node);
  }
  // The code will exit this while loop in one iteration in most cases, as the
  // graph is already topologically sorted.
  while (!queue.empty()) {
    utils::MutableNodeView* current_node = queue.front();
    queue.pop_front();
    if (IsLayoutOptimizerAddedDstToSrcTransform(context, *current_node)) {
      return true;
    }
    // We only continue searching if the path is connected through
    // format-agnostic nodes.
    if (IsLayoutAgnosticOp(*current_node->node())) {
      auto current_node_pos = GetDataFaninPorts(*current_node);
      for (const auto& pos : current_node_pos) {
        const auto& fanin = current_node->GetRegularFanin(pos);
        auto* fanin_node = fanin.node_view();
        if (visited_nodes.insert(fanin_node).second) {
          queue.push_back(fanin_node);
        }
      }
    }
  }
  return false;
}

std::vector<int> LayoutAgnosticOpTransposer::GetVariadicNDFaninPorts(
    const TransposeContext& context, const utils::MutableNodeView& node,
    int rank) const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_54(mht_54_v, 1503, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "LayoutAgnosticOpTransposer::GetVariadicNDFaninPorts");

  std::vector<int> ports;
  const int num_regular_fanins = node.NumRegularFanins();
  ports.reserve(num_regular_fanins);
  for (int i = 0; i < num_regular_fanins; ++i) {
    const auto& regular_fanin = node.GetRegularFanin(i);
    auto* regular_fanin_node = regular_fanin.node_view();
    int regular_fanin_port = regular_fanin.index();
    if ((IsFanoutPortRankN(*regular_fanin_node, regular_fanin_port, rank)) &&
        ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
          IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
         IsLayoutOptimizerAddedDstToSrcTranspose(context,
                                                 *regular_fanin_node))) {
      ports.push_back(i);
    }
  }
  return ports;
}

Status DefaultLayoutAgnosticOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_55(mht_55_v, 1526, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "DefaultLayoutAgnosticOpTransposer::TransposeNode");

  DCHECK(IsDefaultLayoutAgnosticOp(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status AddNTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_56(mht_56_v, 1549, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "AddNTransposer::TransposeNode");

  DCHECK(IsAddN(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, GetDataFaninPorts(*node),
                                            node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool BinaryOpTransposer::IsNDOperateWithMD(const utils::MutableNodeView& node,
                                           int n, int m) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_57(mht_57_v, 1573, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::IsNDOperateWithMD");

  return IsFaninPortRankN(node, 0, n) && IsFaninPortRankN(node, 1, m);
}

bool BinaryOpTransposer::IsFaninShapeSupported(
    const utils::MutableNodeView& node, int rank) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_58(mht_58_v, 1581, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::IsFaninShapeSupported");

  return (IsNDOperateWithMD(node, rank, 0) ||
          IsNDOperateWithMD(node, rank, 1) ||
          IsNDOperateWithMD(node, rank, rank) ||
          IsNDOperateWithMD(node, 0, rank) || IsNDOperateWithMD(node, 1, rank));
}

std::vector<int> BinaryOpTransposer::GetNDDataFaninPorts(
    const utils::MutableNodeView& node, int rank) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_59(mht_59_v, 1592, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::GetNDDataFaninPorts");

  std::vector<int> values;
  if (IsFaninPortRankN(node, 0, rank)) {
    values.push_back(0);
  }
  if (IsFaninPortRankN(node, 1, rank)) {
    values.push_back(1);
  }
  return values;
}

Status BinaryOpTransposer::AddNodeReshape(
    utils::Mutation* mutation, absl::string_view node_name,
    absl::string_view node_device, absl::string_view input_name,
    absl::string_view shape_const_node_name, const DataType& data_type) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_60_v.push_back("node_device: \"" + std::string(node_device.data(), node_device.size()) + "\"");
   mht_60_v.push_back("input_name: \"" + std::string(input_name.data(), input_name.size()) + "\"");
   mht_60_v.push_back("shape_const_node_name: \"" + std::string(shape_const_node_name.data(), shape_const_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_60(mht_60_v, 1613, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::AddNodeReshape");

  NodeDef new_node;
  new_node.set_name(string(node_name));
  new_node.add_input(string(input_name));
  new_node.add_input(string(shape_const_node_name));
  new_node.set_op(kReshape);
  new_node.set_device(string(node_device));

  AttrValue attr_type_indices;
  attr_type_indices.set_type(DT_INT32);
  new_node.mutable_attr()->insert({"Tshape", attr_type_indices});

  AttrValue attr_type_params;
  attr_type_params.set_type(data_type);
  new_node.mutable_attr()->insert({"T", attr_type_params});

  Status status;
  mutation->AddNode(std::move(new_node), &status);
  return status;
}

Status BinaryOpTransposer::AddNodeShapeConst(
    utils::Mutation* mutation, absl::string_view node_name,
    absl::string_view node_device, bool node_in_frame, int num_channels,
    absl::string_view depended_node, int rank) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   mht_61_v.push_back("node_device: \"" + std::string(node_device.data(), node_device.size()) + "\"");
   mht_61_v.push_back("depended_node: \"" + std::string(depended_node.data(), depended_node.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_61(mht_61_v, 1643, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::AddNodeShapeConst");

  NodeDef new_node;
  new_node.set_name(string(node_name));
  new_node.set_op(kOpConst);
  new_node.set_device(string(node_device));
  AttrValue attr_data_type;
  attr_data_type.set_type(DT_INT32);
  new_node.mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  Tensor tensor(DT_INT32, TensorShape({rank}));
  std::vector<int> shape(rank, 1);
  shape[1] = num_channels;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    tensor.flat<int>()(i) = shape[i];
  }
  tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
  new_node.mutable_attr()->insert({"value", attr_tensor});
  if (node_in_frame) {
    // This is to ensure the transpose node and the const node are in the same
    // frame.
    // TODO(halehri): Add Test that exercises this condition.
    new_node.add_input(AsControlDependency(string(depended_node)));
  }

  Status status;
  mutation->AddNode(std::move(new_node), &status);
  return status;
}

Status BinaryOpTransposer::MaybeReshapeVectorFanin(TransposeContext* context,
                                                   utils::MutableNodeView* node,
                                                   int rank) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_62(mht_62_v, 1678, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::MaybeReshapeVectorFanin");

  int vector_index = -1;
  if (IsNDOperateWithMD(*node, rank, 1)) {
    vector_index = 1;
  } else if (IsNDOperateWithMD(*node, 1, rank)) {
    vector_index = 0;
  }
  if (vector_index != -1) {
    const string& node_name = node->GetName();
    const string& node_device = node->GetDevice();
    string reshape_node_name = LayoutOptimizerNode(GetReshapeNodeNameFormat(
        node_name, vector_index, context->src_format, context->dst_format));
    string shape_const_node_name = LayoutOptimizerNode(
        GetShapeConstNodeNameFormat(node_name, vector_index));
    const auto& fanin = node->GetRegularFanin(vector_index);
    auto* fanin_node = fanin.node_view();
    const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
    if (output_shape_attr == nullptr) {
      return errors::InvalidArgument("Missing attribute ", kAttrOutputShape);
    }
    int vector_size =
        output_shape_attr->list().shape(fanin.index()).dim(0).size();
    utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
    TF_RETURN_IF_ERROR(
        AddNodeShapeConst(mutation, shape_const_node_name, node_device,
                          context->frames.IsInFrame(*node->node()), vector_size,
                          fanin_node->GetName(), rank));
    const auto* t_attr = node->GetAttr(kAttrT);
    if (t_attr == nullptr) {
      return errors::InvalidArgument("Missing attribute ", kAttrT);
    }
    TF_RETURN_IF_ERROR(
        AddNodeReshape(mutation, reshape_node_name, node_device,
                       TensorIdToString({fanin_node->GetName(), fanin.index()}),
                       shape_const_node_name, t_attr->type()));
    mutation->AddOrUpdateRegularFanin(node, vector_index,
                                      {reshape_node_name, 0});
  }
  return Status::OK();
}

Status BinaryOpTransposer::TransposeNode(TransposeContext* context,
                                         utils::MutableNodeView* node) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_63(mht_63_v, 1723, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "BinaryOpTransposer::TransposeNode");

  DCHECK(IsBinaryOp(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) || !IsFaninShapeSupported(*node, rank) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, GetNDDataFaninPorts(*node, rank), node, kOpTranspose));
  TF_RETURN_IF_ERROR(MaybeReshapeVectorFanin(context, node, rank));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ConcatOpTransposer::TransposeNode(TransposeContext* context,
                                         utils::MutableNodeView* node) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_64(mht_64_v, 1748, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ConcatOpTransposer::TransposeNode");

  DCHECK(IsConcat(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, GetConcatDataFaninPorts(*node), node, kOpTranspose));
  int axis_node = 0;
  if (node->GetOp() == "ConcatV2") {
    const auto* n_attr = node->GetAttr(kAttrN);
    if (n_attr != nullptr) {
      axis_node = n_attr->i();
    }
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {axis_node}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status FillOpTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_65(mht_65_v, 1781, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "FillOpTransposer::TransposeNode");

  DCHECK(IsFill(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 0, {4}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status IdentityNTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_66(mht_66_v, 1798, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IdentityNTransposer::TransposeNode");

  DCHECK(IsIdentityN(*node->node()));
  const auto ports_4d = GetVariadicNDFaninPorts(*context, *node, 4);

  // Temporarily upgrade the context to obtain the number of 5D fanin ports.
  std::vector<int> ports_5d;
  {
    ScopedDataFormatUpgrader data_format_upgrader(context, 5);
    ports_5d = GetVariadicNDFaninPorts(*context, *node, 5);
  }

  if (!ShouldProcess(*context, *node)) {
    return Status::OK();
  }

  if (!ports_4d.empty()) {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, ports_4d, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, ports_4d, node, kOpTranspose));
  }

  if (!ports_5d.empty()) {
    ScopedDataFormatUpgrader data_format_upgrader(context, 5);
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, ports_5d, node, kOpTranspose));
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, ports_5d, node, kOpTranspose));
  }
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool MergeTransposer::IsEveryFaninAfterDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_67(mht_67_v, 1834, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "MergeTransposer::IsEveryFaninAfterDstToSrcTransform");

  for (const auto& regular_fanin : node.GetRegularFanins()) {
    auto* regular_fanin_node = regular_fanin.node_view();
    if (IsFanoutPortRankN(*regular_fanin_node, regular_fanin.index(), 4) &&
        ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
          IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
         IsLayoutOptimizerAddedDstToSrcTranspose(context,
                                                 *regular_fanin_node))) {
      continue;
    }
    return false;
  }
  return true;
}

Status MergeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_68(mht_68_v, 1853, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "MergeTransposer::TransposeNode");

  DCHECK(IsMerge(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsEveryFaninAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, GetDataFaninPorts(*node),
                                            node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status PadTransposer::TransposeNode(TransposeContext* context,
                                    utils::MutableNodeView* node) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_69(mht_69_v, 1869, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "PadTransposer::TransposeNode");

  DCHECK(IsMirrorPad(*node->node()) || IsMirrorPadGrad(*node->node()) ||
         IsPad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 1, {4, 2}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool ReduceTransposer::KeepDims(const utils::MutableNodeView& node) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_70(mht_70_v, 1887, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ReduceTransposer::KeepDims");

  const auto* keep_dims_attr = node.GetAttr(kAttrKeepDims);
  if (keep_dims_attr != nullptr) {
    return keep_dims_attr->b();
  }
  return false;
}

bool ReduceTransposer::IsAlongAxis(const Tensor& tensor,
                                   absl::Span<const int> axis, int rank) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_71(mht_71_v, 1899, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ReduceTransposer::IsAlongAxis");

  const int axis_size = axis.size();
  if (tensor.dims() != 1 || tensor.dim_size(0) != axis_size) {
    return false;
  }
  for (int i = 0; i < axis_size; ++i) {
    int local_axis = 0;
    if (tensor.dtype() == DT_INT32) {
      local_axis = tensor.flat<int32>()(i);
    } else {
      local_axis = tensor.flat<int64_t>()(i);
    }
    if (local_axis < 0) {
      local_axis += rank;
    }
    bool along_axis = false;
    for (int dim : axis) {
      if (local_axis == dim) {
        along_axis = true;
        break;
      }
    }
    if (!along_axis) {
      return false;
    }
  }
  return true;
}

bool ReduceTransposer::IsReduceAxisSupported(const TransposeContext& context,
                                             const utils::MutableNodeView& node,
                                             int rank) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_72(mht_72_v, 1933, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ReduceTransposer::IsReduceAxisSupported");

  if (KeepDims(node)) {
    return true;
  }
  const auto& regular_fanin_1 = node.GetRegularFanin(1);
  auto* axis_node = regular_fanin_1.node_view();
  if (!IsConstant(*axis_node->node())) {
    return false;
  }
  const auto* value_attr = axis_node->GetAttr(kAttrValue);
  if (value_attr == nullptr) {
    return false;
  }
  Tensor tensor;
  if (!tensor.FromProto(value_attr->tensor())) {
    LOG(ERROR) << "Failed to parse TensorProto.";
    return false;
  }
  auto indices = [&context](absl::Span<const char> labels) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_73(mht_73_v, 1954, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "lambda");

    return GetDimensionIndicesFromLabel(context.src_dim_indices, labels);
  };
  if (rank == 5) {
    return IsAlongAxis(tensor, indices({'N', 'D', 'H', 'W', 'C'}), 5) ||
           IsAlongAxis(tensor, indices({'D', 'H', 'W', 'C'}), 5) ||
           IsAlongAxis(tensor, indices({'N', 'D', 'H', 'W'}), 5) ||
           IsAlongAxis(tensor, indices({'D', 'H', 'W'}), 5) ||
           IsAlongAxis(tensor, indices({'C'}), 5);
  }
  DCHECK_EQ(rank, 4);
  return IsAlongAxis(tensor, indices({'N', 'H', 'W', 'C'}), 4) ||
         IsAlongAxis(tensor, indices({'H', 'W', 'C'}), 4) ||
         IsAlongAxis(tensor, indices({'N', 'H', 'W'}), 4) ||
         IsAlongAxis(tensor, indices({'H', 'W'}), 4) ||
         IsAlongAxis(tensor, indices({'C'}), 4);
}

Status ReduceTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_74(mht_74_v, 1976, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ReduceTransposer::TransposeNode");

  DCHECK(IsReduceOp(*node->node()));
  const int rank = GetFaninPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsReduceAxisSupported(*context, *node, rank) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
  if (KeepDims(*node)) {
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  }
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ReverseV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_75(mht_75_v, 2005, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ReverseV2Transposer::TransposeNode");

  DCHECK(IsReverseV2(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SelectTransposer::IsFaninScalarVector4D(
    const utils::MutableNodeView& fanin, int port) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_76(mht_76_v, 2022, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SelectTransposer::IsFaninScalarVector4D");

  return IsFanoutPortRankN(fanin, port, 0) ||
         IsFanoutPortRankN(fanin, port, 1) || IsFanoutPortRankN(fanin, port, 4);
}

std::vector<int> SelectTransposer::GetFaninPorts(
    const utils::MutableNodeView& fanin, int port) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_77(mht_77_v, 2031, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SelectTransposer::GetFaninPorts");

  // Input 0 could be a scalar, a vector with size matching the first dimension
  // of input 1 and 2, or must have the same shape as input 1 and 2.
  if (IsFanoutPortRankN(fanin, port, 4)) {
    return {0, 1, 2};
  }
  return {1, 2};
}

Status SelectTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_78(mht_78_v, 2044, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SelectTransposer::TransposeNode");

  DCHECK(IsSelect(*node->node()));
  const auto& regular_fanin_0 = node->GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninScalarVector4D(*regular_fanin_0_node, regular_fanin_0.index()) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, GetFaninPorts(*regular_fanin_0_node, regular_fanin_0.index()),
      node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_79(mht_79_v, 2064, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ShapeTransposer::TransposeNode");

  DCHECK(IsShape(*node->node()));
  const int rank = GetFaninPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeNTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_80(mht_80_v, 2088, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "ShapeNTransposer::TransposeNode");

  DCHECK(IsShapeN(*node->node()));
  // ShapeN requires all input tensors to have the same dimensions. Therefore,
  // we simply use the 0th fanin port.
  const int rank = GetFaninPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  const auto ports = GetVariadicNDFaninPorts(*context, *node, rank);
  if (!ShouldProcess(*context, *node) || ports.empty()) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, ports, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpDataFormatVecPermute));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SliceTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_81(mht_81_v, 2115, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SliceTransposer::TransposeNode");

  DCHECK(IsSlice(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsFaninPortsDimsNIfConst(*node, {1, 2}, {rank}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_82(mht_82_v, 2141, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SplitTransposer::TransposeNode");

  DCHECK(IsSplit(*node->node()));
  const auto ports = GetDataFanoutPorts(*node);
  if (!ShouldProcess(*context, *node) || !IsFanoutPortsRankN(*node, ports, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitVTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_83(mht_83_v, 2160, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SplitVTransposer::TransposeNode");

  DCHECK(IsSplitV(*node->node()));
  const auto ports = GetDataFanoutPorts(*node);
  if (!ShouldProcess(*context, *node) || !IsFanoutPortsRankN(*node, ports, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {2}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SqueezeTransposer::IsInputConvertible(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_84(mht_84_v, 2179, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SqueezeTransposer::IsInputConvertible");

  const auto& regular_fanin_0 = node.GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  const auto* output_shape_attr =
      regular_fanin_0_node->GetAttr(kAttrOutputShape);
  if (output_shape_attr != nullptr) {
    auto& shape = output_shape_attr->list().shape(regular_fanin_0.index());
    if (shape.dim_size() != kRank) {
      return false;
    }
    const int height_dim = context.src_dim_indices.at('H');
    const int width_dim = context.src_dim_indices.at('W');
    if (shape.dim(height_dim).size() == 1 && shape.dim(width_dim).size() == 1) {
      return true;
    }
  }
  return false;
}

bool SqueezeTransposer::IsAlongAxis(const AttrValue& attr,
                                    absl::Span<const int> axis,
                                    int rank) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_85(mht_85_v, 2203, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SqueezeTransposer::IsAlongAxis");

  const auto& list = attr.list();
  // If list is empty, Squeeze op will squeeze all dimensions of size 1.
  int axis_size = axis.size();
  if (list.i_size() == 0) {
    return true;
  } else if (list.i_size() != axis_size) {
    return false;
  }
  for (int i = 0; i < axis_size; ++i) {
    int local_axis = list.i(i);
    if (local_axis < 0) {
      local_axis += rank;
    }
    bool along_axis = false;
    for (int dim : axis) {
      if (local_axis == dim) {
        along_axis = true;
        break;
      }
    }
    if (!along_axis) {
      return false;
    }
  }
  return true;
}

bool SqueezeTransposer::IsDimsSupported(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_86(mht_86_v, 2235, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SqueezeTransposer::IsDimsSupported");

  auto indices = [&context](absl::Span<const char> labels) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_87(mht_87_v, 2239, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "lambda");

    return GetDimensionIndicesFromLabel(context.src_dim_indices, labels);
  };
  const auto* squeeze_dims_attr = node.GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr == nullptr) {
    return false;
  }
  return (IsFanoutPortRankN(node, 0, 2) &&
          IsAlongAxis(*squeeze_dims_attr, indices({'H', 'W'}), kRank)) ||
         (IsFanoutPortRankN(node, 0, 1) &&
          IsAlongAxis(*squeeze_dims_attr, indices({'N', 'H', 'W'}), kRank));
}

Status SqueezeTransposer::UpdateSqueezeDims(TransposeContext* context,
                                            utils::MutableNodeView* node) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_88(mht_88_v, 2256, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SqueezeTransposer::UpdateSqueezeDims");

  const auto* squeeze_dims_attr = node->GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr == nullptr) {
    return errors::InvalidArgument("Missing attribute ", kAttrSqueezeDims);
  }
  const int num_input_dims = context->src_format.length();
  const int min_squeeze_dim = -num_input_dims;
  std::vector<int> squeeze_dims_mapped;
  const int squeeze_dims_size = squeeze_dims_attr->list().i_size();
  squeeze_dims_mapped.reserve(squeeze_dims_size);
  for (int i = 0; i < squeeze_dims_size; ++i) {
    int dim = squeeze_dims_attr->list().i(i);
    if (dim < min_squeeze_dim || dim >= num_input_dims) {
      return errors::InvalidArgument(
          "Attribute '", kAttrSqueezeDims, "' contains out of range index '",
          dim, "', index must be between [", min_squeeze_dim, ", ",
          num_input_dims, ")");
    }
    if (dim < 0) {
      dim += num_input_dims;
    }
    squeeze_dims_mapped.push_back(context->dst_to_src[dim]);
  }
  std::sort(squeeze_dims_mapped.begin(), squeeze_dims_mapped.end());
  AttrValue squeeze_dims;
  squeeze_dims.mutable_list()->mutable_i()->Reserve(squeeze_dims_size);
  for (const auto& dim : squeeze_dims_mapped) {
    squeeze_dims.mutable_list()->mutable_i()->Add(dim);
  }
  context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
      node, kAttrSqueezeDims, squeeze_dims);
  return Status::OK();
}

Status SqueezeTransposer::TransposeNode(TransposeContext* context,
                                        utils::MutableNodeView* node) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_89(mht_89_v, 2294, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SqueezeTransposer::TransposeNode");

  DCHECK(IsSqueeze(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsDimsSupported(*context, *node) ||
      !IsInputConvertible(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateSqueezeDims(context, node));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool StridedSliceTransposer::IsMaskZero(const utils::MutableNodeView& node,
                                        absl::string_view mask) {
   std::vector<std::string> mht_90_v;
   mht_90_v.push_back("mask: \"" + std::string(mask.data(), mask.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_90(mht_90_v, 2311, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "StridedSliceTransposer::IsMaskZero");

  const auto* mask_attr = node.GetAttr(mask);
  if (mask_attr != nullptr) {
    return mask_attr->i() == 0;
  }
  return true;
}

bool StridedSliceTransposer::HasOnlyBeginEndMask(
    const utils::MutableNodeView& node) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_91(mht_91_v, 2323, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "StridedSliceTransposer::HasOnlyBeginEndMask");

  return IsMaskZero(node, "ellipsis_mask") &&
         IsMaskZero(node, "new_axis_mask") &&
         IsMaskZero(node, "shrink_axis_mask");
}

Status StridedSliceTransposer::PermuteMask(TransposeContext* context,
                                           utils::MutableNodeView* node,
                                           absl::string_view mask) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("mask: \"" + std::string(mask.data(), mask.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_92(mht_92_v, 2335, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "StridedSliceTransposer::PermuteMask");

  // Computers the permutation of the masks based on the src and dst format.
  // For example:
  // src_format = NHWC
  // dst_format = NCHW
  // src_to_dst permutation = [0, 3, 1, 2].
  // mask : 0010 [Note the bit positions correspond to indexes i.e this is in
  // reverse order of the src format (CWHN)] result : 0100 (WHCN)
  const auto* mask_attr = node->GetAttr(mask);
  const int mask_i = mask_attr != nullptr ? mask_attr->i() : 0;
  if (mask_i < 0 || mask_i > 15) {
    return errors::InvalidArgument("invalid mask value: ", mask_i);
  }
  int result = 0;
  for (int i = 0, end = context->src_to_dst.size(); i < end; i++) {
    const int final_pos = context->src_to_dst[i];
    const int position_mask = 1 << final_pos;
    const int bit_i = (mask_i & position_mask) >> final_pos;
    result |= bit_i << i;
  }
  AttrValue new_mask_attr;
  new_mask_attr.set_i(result);
  context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(node, mask,
                                                                 new_mask_attr);
  return Status::OK();
}

Status StridedSliceTransposer::TransposeNode(TransposeContext* context,
                                             utils::MutableNodeView* node) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_93(mht_93_v, 2366, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "StridedSliceTransposer::TransposeNode");

  DCHECK(IsStridedSlice(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortsDimsNIfConst(*node, {1, 2, 3}, {4}) ||
      !HasOnlyBeginEndMask(*node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(PermuteMask(context, node, "begin_mask"));
  TF_RETURN_IF_ERROR(PermuteMask(context, node, "end_mask"));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1, 2, 3}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SwitchTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_94(mht_94_v, 2387, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "SwitchTransposer::TransposeNode");

  DCHECK(IsSwitch(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, GetDataFanoutPorts(*node),
                                             node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TernaryOpTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_95(mht_95_v, 2403, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "TernaryOpTransposer::TransposeNode");

  DCHECK(IsTernaryOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TileTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_96(mht_96_v, 2419, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "TileTransposer::TransposeNode");

  DCHECK(IsTile(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 1, {4}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status UnaryGradTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_97(mht_97_v, 2437, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "UnaryGradTransposer::TransposeNode");

  DCHECK(IsUnaryGrad(*node->node()));
  const int rank = GetFanoutPortRank(*node, 0);
  if (rank != 4 && rank != 5) {
    return Status::OK();
  }
  ScopedDataFormatUpgrader data_format_upgrader(context, rank);
  if (!ShouldProcess(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

// Utils.

string GetDeviceName(const VirtualPlacer* virtual_placer, const NodeDef& node) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_98(mht_98_v, 2462, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "GetDeviceName");

  return (node.device().empty() && virtual_placer != nullptr)
             ? virtual_placer->get_canonical_device_name(node)
             : node.device();
}

bool IsDefaultLayoutSensitiveOp(const NodeDef& node) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_99(mht_99_v, 2471, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsDefaultLayoutSensitiveOp");

  static absl::flat_hash_set<string>* default_layout_sensitive_ops =
      new absl::flat_hash_set<std::string>(
          {"AvgPool", "Conv2D", "DepthwiseConv2dNative", "DepthToSpace",
           "FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
           "FusedConv2DBiasActivation", "MaxPool", "SpaceToDepth"});
  return default_layout_sensitive_ops->find(node.op()) !=
         default_layout_sensitive_ops->end();
}

bool IsLayoutSensitiveOp(const NodeDef& node) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_100(mht_100_v, 2484, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsLayoutSensitiveOp");

  return IsDefaultLayoutSensitiveOp(node) || IsAvgPoolGrad(node) ||
         IsBiasAddV2(node) || IsBiasAddGrad(node) ||
         IsConv2DBackpropFilter(node) || IsConv2DBackpropInput(node) ||
         IsDepthwiseConv2dNativeBackpropFilter(node) ||
         IsDepthwiseConv2dNativeBackpropInput(node) ||
         IsFusedBatchNormEx(node) || IsFusedBatchNormGrad(node) ||
         IsMaxPoolV2(node) || IsMaxPoolGrad(node) || IsMaxPoolGradV2(node) ||
         IsMaxPoolGradGradV1(node) || IsMaxPoolGradGradV2(node) ||
         IsConv3D(node) || IsConv3DBackpropInputV2(node) ||
         IsConv3DBackpropFilterV2(node);
}

bool IsDefaultLayoutAgnosticOp(const NodeDef& node) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_101(mht_101_v, 2500, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsDefaultLayoutAgnosticOp");

  static absl::flat_hash_set<string>* agnostic_nodes =
      new absl::flat_hash_set<std::string>({"Abs",
                                            "Acos",
                                            "Acosh",
                                            "Angle",
                                            "Asin",
                                            "Asinh",
                                            "Atan",
                                            "Atanh",
                                            "Bitcast",
                                            "Cast",
                                            "Ceil",
                                            "CheckNumerics",
                                            "ComplexAbs",
                                            "Conj",
                                            "Cos",
                                            "Cosh",
                                            "Digamma",
                                            "Elu",
                                            "Enter",
                                            "Erf",
                                            "Erfc",
                                            "Exit",
                                            "Exp",
                                            "Expm1",
                                            "FakeQuantWithMinMaxVars",
                                            "FakeQuantWithMinMaxArgs",
                                            "Floor",
                                            "GuaranteeConst",
                                            "Identity",
                                            "Imag",
                                            "Inv",
                                            "IsFinite",
                                            "IsInf",
                                            "IsNan",
                                            "LeakyRelu",
                                            "Lgamma",
                                            "Log",
                                            "LogicalNot",
                                            "Log1p",
                                            "Neg",
                                            "NextIteration",
                                            "OnesLike",
                                            "PreventGradient",
                                            "QuantizeAndDequantizeV2",
                                            "QuantizeAndDequantizeV3",
                                            "QuantizeAndDequantizeV4",
                                            "Real",
                                            "Reciprocal",
                                            "Relu",
                                            "Relu6",
                                            "Rint",
                                            "Selu",
                                            "Sigmoid",
                                            "Sign",
                                            "Sin",
                                            "Sinh",
                                            "Snapshot",
                                            "Softplus",
                                            "Round",
                                            "Rsqrt",
                                            "Sqrt",
                                            "Square",
                                            "StopGradient",
                                            "Tan",
                                            "Tanh",
                                            "ZerosLike"});
  return agnostic_nodes->find(node.op()) != agnostic_nodes->end();
}

bool IsLayoutAgnosticOp(const NodeDef& node) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_102(mht_102_v, 2574, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsLayoutAgnosticOp");

  return IsDefaultLayoutAgnosticOp(node) || IsAddN(node) || IsBinaryOp(node) ||
         IsIdentityN(node) || IsMerge(node) || IsMirrorPad(node) ||
         IsMirrorPadGrad(node) || IsPad(node) || IsSelect(node) ||
         IsSwitch(node) || IsTernaryOp(node) || IsUnaryGrad(node) ||
         IsConcat(node) || IsReverseV2(node) || IsTile(node) || IsShape(node) ||
         IsShapeN(node) || IsFill(node) || IsSlice(node) || IsSplit(node) ||
         IsSqueeze(node) || IsSplitV(node) || IsStridedSlice(node) ||
         IsReduceOp(node);
}

bool IsTernaryOp(const NodeDef& node) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_103(mht_103_v, 2588, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsTernaryOp");
 return IsBetainc(node); }

bool IsUnaryGrad(const NodeDef& node) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_104(mht_104_v, 2593, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsUnaryGrad");

  bool is_unary_grad =
      IsEluGrad(node) || IsInvGrad(node) || IsLeakyReluGrad(node) ||
      IsReciprocalGrad(node) || IsRelu6Grad(node) || IsReluGrad(node) ||
      IsRsqrtGrad(node) || IsSeluGrad(node) || IsSigmoidGrad(node) ||
      IsSoftplusGrad(node) || IsSoftsignGrad(node) || IsSqrtGrad(node) ||
      IsTanhGrad(node);
  return is_unary_grad;
}

bool IsMaxPoolV2(const NodeDef& node) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_105(mht_105_v, 2606, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsMaxPoolV2");
 return node.op() == "MaxPoolV2"; }

bool IsMaxPoolGradV2(const NodeDef& node) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_106(mht_106_v, 2611, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsMaxPoolGradV2");

  return node.op() == "MaxPoolGradV2";
}

bool IsMaxPoolGradGradV1(const NodeDef& node) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_107(mht_107_v, 2618, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsMaxPoolGradGradV1");

  return node.op() == "MaxPoolGradGrad";
}

bool IsMaxPoolGradGradV2(const NodeDef& node) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_108(mht_108_v, 2625, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsMaxPoolGradGradV2");

  return node.op() == "MaxPoolGradGradV2";
}

bool IsBinaryOp(const NodeDef& node) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_109(mht_109_v, 2632, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsBinaryOp");

  bool is_binary =
      IsAdd(node) || IsAtan2(node) || IsComparisonOp(node) || IsComplex(node) ||
      IsDiv(node) || IsFloorDiv(node) || IsIgamma(node) || IsIgammac(node) ||
      IsLogicalAnd(node) || IsLogicalOr(node) || IsMaximum(node) ||
      IsMinimum(node) || IsMod(node) || IsMul(node) || IsPolygamma(node) ||
      IsPow(node) || IsRealDiv(node) || IsSquaredDifference(node) ||
      IsSub(node) || IsTruncateDiv(node) || IsTruncateMod(node) || IsZeta(node);
  return is_binary;
}

bool IsReduceOp(const NodeDef& node) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_110(mht_110_v, 2646, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsReduceOp");

  return IsSum(node) || IsMean(node) || IsProd(node) || IsMax(node) ||
         IsMin(node) || IsAll(node) || IsAny(node);
}

std::vector<int> GetDataFaninPorts(const utils::MutableNodeView& node) {
  const auto* node_def = node.node();
  if (IsAvgPoolGrad(*node_def) || IsSplit(*node_def)) {
    return {1};
  }
  if (IsStridedSliceGrad(*node_def)) {
    return {4};
  }
  if (IsBinaryOp(*node_def) || IsUnaryGrad(*node_def)) {
    return {0, 1};
  }
  if (IsTernaryOp(*node_def) || IsSelect(*node_def) ||
      IsMaxPoolGrad(*node_def) || IsMaxPoolGradV2(*node_def) ||
      IsMaxPoolGradGradV1(*node_def) || IsMaxPoolGradGradV2(*node_def)) {
    return {0, 1, 2};
  }
  if (IsShapeN(*node_def) || IsIdentityN(*node_def) || IsAddN(*node_def) ||
      IsMerge(*node_def)) {
    return GetRegularFaninPorts(node);
  }
  if (IsConcat(*node_def)) {
    return GetConcatDataFaninPorts(node);
  }
  if (node.NumRegularFanins() > 0) {
    return {0};
  }
  return {};
}

std::vector<int> GetDataFanoutPorts(const utils::MutableNodeView& node) {
  const auto* node_def = node.node();
  if (IsIdentityN(*node_def) || IsShape(*node_def) || IsShapeN(*node_def)) {
    return GetDataFaninPorts(node);
  }
  if (IsSplit(*node_def) || IsSplitV(*node_def)) {
    const auto* num_split_attr = node.GetAttr(kAttrNumSplit);
    if (num_split_attr == nullptr) {
      return {0};
    }
    std::vector<int> values(num_split_attr->i());
    std::iota(values.begin(), values.end(), 0);
    return values;
  }
  if (IsSwitch(*node_def)) {
    const auto* num_outs_attr = node.GetAttr(kAttrNumOuts);
    const int num_outs = num_outs_attr != nullptr ? num_outs_attr->i() : 2;
    std::vector<int> values(num_outs);
    std::iota(values.begin(), values.end(), 0);
    return values;
  }
  return {0};
}

bool GetValueAttrFromConstInputNode(
    const utils::MutableNodeView& node,
    const std::function<bool(const NodeDef&)>& predicate, int index,
    Tensor* tensor) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_111(mht_111_v, 2710, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "GetValueAttrFromConstInputNode");

  if (!predicate(*node.node())) {
    return false;
  }
  const auto& regular_fanin = node.GetRegularFanin(index);
  auto* regular_fanin_node = regular_fanin.node_view();
  if (!IsConstant(*regular_fanin_node->node())) {
    return false;
  }
  const auto* value_attr = regular_fanin_node->GetAttr(kAttrValue);
  if (value_attr == nullptr || value_attr->tensor().dtype() != DT_INT32) {
    return false;
  }
  if (!tensor->FromProto(value_attr->tensor())) {
    return false;
  }

  return true;
}

bool IsDataFormatOp(const utils::MutableNodeView& node) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSgeneric_layout_optimizer_transposerDTcc mht_112(mht_112_v, 2733, "", "./tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.cc", "IsDataFormatOp");

  const string& op = node.GetOp();
  return op == kOpDataFormatDimMap || op == kOpDataFormatVecPermute;
}

absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format) {
  const int size = data_format.size();
  absl::flat_hash_map<char, int> index;
  index.reserve(size);
  for (int i = 0; i < size; i++) {
    index[data_format[i]] = i;
  }
  return index;
}

std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format) {
  // Generate permutation for transformation between src and dst format.
  // Example:
  // src = NWHC, dst = NCWH
  // index = { N:0 W:1 H:2 C:3 }
  // permutation = [0, 3, 1, 2]
  DCHECK(src_dim_indices.size() == dst_format.size());
  std::vector<int> permutation;
  const int size = dst_format.size();
  permutation.reserve(size);
  for (int i = 0; i < size; i++) {
    permutation.push_back(src_dim_indices.at(dst_format[i]));
  }
  return permutation;
}

}  // namespace grappler
}  // namespace tensorflow
