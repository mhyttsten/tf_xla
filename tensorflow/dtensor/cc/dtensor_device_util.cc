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
class MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_device_util.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/small_constant_optimization.h"

namespace tensorflow {
namespace dtensor {
namespace {
// Represents an input node during graph construction.
// When executing a Function, `output` is used to align graph inputs
// with the inputs to the function call.
struct FunctionArgument {
  Node* node;
  NodeDefBuilder::NodeOut output;
};

bool LayoutsAreCompatible(absl::optional<Layout> first_layout,
                          absl::optional<Layout> second_layout) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "LayoutsAreCompatible");

  if (!first_layout.has_value() && !second_layout.has_value()) {
    return true;
  }
  if (!first_layout.has_value() || !second_layout.has_value()) {
    return false;
  }
  return first_layout.value() == second_layout.value();
}

// Parse a pair of attribute of (indices, layouts) into a map.
Status ParseAttrMap(const Node& node, absl::string_view indices_attr,
                    absl::string_view layout_attr,
                    std::map<int, Layout>* indices_layout_map) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("indices_attr: \"" + std::string(indices_attr.data(), indices_attr.size()) + "\"");
   mht_1_v.push_back("layout_attr: \"" + std::string(layout_attr.data(), layout_attr.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_1(mht_1_v, 233, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ParseAttrMap");

  std::vector<std::string> layouts;
  if (!TryGetNodeAttr(node.attrs(), layout_attr, &layouts)) {
    return Status::OK();
  }
  const TensorProto* indices;
  if (!TryGetNodeAttr(node.attrs(), indices_attr, &indices)) {
    return errors::Internal(
        "Arg indices must be set when setting inferred resource layouts.");
  }
  if (indices->int_val_size() != layouts.size()) {
    return errors::Internal(
        "Arg indices for inferred resource argument must match the "
        "size of inferred resource layout.");
  }
  for (int i = 0; i < indices->int_val_size(); ++i) {
    const auto arg_index = indices->int_val(i);
    const auto& arg_layout = layouts[i];
    indices_layout_map->emplace(
        arg_index,
        tensorflow::dtensor::Layout::FromString(arg_layout).ValueOrDie());
  }
  return Status::OK();
}

Status ParseResourceArgumentLayouts(
    const Node& node, std::map<int, Layout>* inferred_resource_input_layouts) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_2(mht_2_v, 262, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ParseResourceArgumentLayouts");

  return ParseAttrMap(node, kNewResourceLayoutIndices, kNewResourceArgLayouts,
                      inferred_resource_input_layouts);
}

Status ParseShapeInputLayouts(const Node& node,
                              std::map<int, Layout>* shape_output_metadata) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_3(mht_3_v, 271, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ParseShapeInputLayouts");

  return ParseAttrMap(node, kShapeOpInputLayoutIndices, kShapeOpInputLayout,
                      shape_output_metadata);
}

// Gets the layout attached to a specific node at a given index, ignoring any
// Identity ops.
StatusOr<Layout> GetLayoutThroughIdentityOps(Node* op, int output_index) {
  while (op->op_def().name() == "Identity" ||
         op->op_def().name() == "IdentityN") {
    const Edge* edge;
    TF_RETURN_IF_ERROR(op->input_edge(output_index, &edge));
    op = edge->src();
    output_index = edge->src_output();
  }
  const auto serialized_layouts = op->attrs().Find(kLayoutAttr);

  if (!serialized_layouts) {
    return errors::InvalidArgument(
        op->op_def().name(), " doesn't contain attribute : ", kLayoutAttr);
  }

  // We assume that there is one layout for each output.
  if (serialized_layouts->list().s_size() != op->num_outputs()) {
    return errors::InvalidArgument(
        "Number of outputs to ", op->op_def().name(),
        " does not match number of layouts attached");
  }

  return Layout::FromString(serialized_layouts->list().s(output_index));
}

}  // namespace

tensorflow::Fprint128 TensorWithLayout::CacheKey() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_4(mht_4_v, 308, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::CacheKey");

  tensorflow::Fprint128 f = tensorflow::Fingerprint128(layout_.ToString());
  // Use exact shape to compute the key.
  for (const int64_t dim : local_shape()) {
    f = FingerprintCat128(f, dim);
  }
  if (const_value_.has_value()) {
    std::string serialized;
    SerializeToStringDeterministic(const_value_.value(), &serialized);
    f = FingerprintCat128(f, tensorflow::Fingerprint128(serialized));
  }
  return f;
}

std::unique_ptr<TensorWithLayout> TensorWithLayout::Broadcast(
    TFE_Context* context, TFE_TensorHandle* tensor,
    const MeshWithParallelDevice& mesh, const std::string& dtensor_device_name,
    TF_Status* status) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("dtensor_device_name: \"" + dtensor_device_name + "\"");
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_5(mht_5_v, 329, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::Broadcast");

  const char* input_device = TFE_TensorHandleDeviceName(tensor, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  if (dtensor_device_name == input_device) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Input to Broadcast must be eager tensor.");
    return nullptr;
  }

  if (TFE_TensorHandleDataType(tensor) == TF_RESOURCE) {
    std::string error_message =
        "Using a non-DTensor variable with DTensor is not supported. If you "
        "are using a scope-based API, create variables inside the DTensor "
        "scope.\n";

    // Resolve the Tensor as resource handle and try to get the stack_trace and
    // Summaries out of it.
    std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tf_tensor(
        TFE_TensorHandleResolve(tensor, status), TF_DeleteTensor);
    Tensor t;
    Status convert_status = TF_TensorToTensor(tf_tensor.get(), &t);
    if (convert_status.ok() && t.dtype() == DataType::DT_RESOURCE) {
      ResourceHandle r = t.flat<ResourceHandle>()(0);
      absl::StrAppend(
          &error_message, "Offending variable summary: ", r.SummarizeValue(),
          "\nStack trace: ", DefinitionLocationMsg(r.definition_stack_trace()));
    }
    TF_SetStatus(status, TF_INVALID_ARGUMENT, error_message.c_str());
    return nullptr;
  }

  if (mesh.mesh_config().is_remote()) {
    TF_DataType dtype = TFE_TensorHandleDataType(tensor);
    std::vector<int64_t> shape(TensorShapeAsVector(tensor, status));
    if (TF_GetCode(status) != TF_OK) return nullptr;
    auto layout = Layout::ReplicatedOnMesh(mesh.mesh_config(), shape.size());

    auto ret = TensorWithLayout::Dummy(shape, dtype, mesh, layout);
    absl::optional<NodeDef> const_value =
        ExtractSmallTensorValue(context, tensor, layout, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    if (const_value) {
      ret->set_const_value(const_value.value());
    }
    return ret;
  }

  // Broadcast tensor value to local devices.
  const Mesh& target_mesh = mesh.mesh_config();
  absl::Span<const std::string> local_devices = target_mesh.local_devices();
  const int num_local_devices = local_devices.size();

  std::vector<parallel_device::TensorHandlePtr> components;
  components.reserve(num_local_devices);
  for (int i = 0; i < num_local_devices; ++i) {
    // Create tensor copies to each local devices specifie by `target_mesh`.
    components.emplace_back(TFE_TensorHandleCopyToDevice(
        tensor, context, local_devices[i].c_str(), status));
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(
          status, TF_INTERNAL,
          absl::StrCat(
              "Unable to copy tensor value for broadcast. Original message: ",
              TF_Message(status))
              .c_str());
      return nullptr;
    }
  }

  std::unique_ptr<parallel_device::ParallelTensor> parallel_tensor =
      parallel_device::ParallelTensor::FromTensorHandles(
          mesh.parallel_device(), std::move(components), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  const std::vector<int64_t>* shape;
  Status s = parallel_tensor->Shape(&shape);
  if (!s.ok()) {
    TF_SetStatus(status, static_cast<TF_Code>(s.code()),
                 s.error_message().c_str());
    return nullptr;
  }
  size_t num_dims = shape->size();

  const Layout layout = Layout::ReplicatedOnMesh(mesh.mesh_config(), num_dims);
  absl::optional<NodeDef> const_value =
      ExtractSmallTensorValue(context, tensor, layout, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  std::unique_ptr<TensorWithLayout> result(new TensorWithLayout(
      std::move(parallel_tensor), mesh, std::move(layout), *shape,
      /*dtype=*/absl::nullopt, std::move(const_value)));
  return result;
}

StatusOr<std::unique_ptr<TensorWithLayout>> TensorWithLayout::Wrap(
    std::unique_ptr<parallel_device::ParallelTensor> tensor,
    const MeshWithParallelDevice& mesh, const Layout& layout) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_6(mht_6_v, 429, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::Wrap");

  const std::vector<int64_t>* shape;
  TF_RETURN_IF_ERROR(tensor->Shape(&shape));

  if (tensor->dtype() != TF_RESOURCE) {
    return std::unique_ptr<TensorWithLayout>(
        new TensorWithLayout(std::move(tensor), mesh, layout, *shape));
  } else {
    return std::unique_ptr<TensorWithLayout>(
        new ResourceHandleWithLayout(std::move(tensor), mesh, layout, *shape));
  }
}

std::unique_ptr<TensorWithLayout> TensorWithLayout::Dummy(
    const std::vector<int64_t>& local_shape, const TF_DataType dtype,
    const MeshWithParallelDevice& mesh, const Layout& layout) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_7(mht_7_v, 447, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::Dummy");

  if (dtype != TF_RESOURCE) {
    return std::unique_ptr<TensorWithLayout>(new TensorWithLayout(
        /*tensor=*/nullptr, mesh, layout, local_shape, dtype));
  } else {
    return std::unique_ptr<TensorWithLayout>(new ResourceHandleWithLayout(
        /*tensor=*/nullptr, mesh, layout, local_shape));
  }
}

std::string TensorWithLayout::SummarizeValue() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_8(mht_8_v, 460, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::SummarizeValue");

  std::string value_summary;
  Status status;
  if (layout().IsFullyReplicated()) {
    status =
        tensorflow::unwrap(tensor()->tensor(0))->SummarizeValue(value_summary);
  } else {
    // Note that this just prints the local values for sharded tensors. We could
    // instead run a collective here to relayout to replicated.
    status = tensor()->SummarizeValue(value_summary);
  }
  if (!status.ok()) {
    value_summary = "<error computing value>";
  }
  return absl::StrCat(value_summary, ", layout=\"", layout().ToString(), "\"");
}

std::string TensorWithLayout::DebugString() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_9(mht_9_v, 480, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "TensorWithLayout::DebugString");

  auto dtype = static_cast<DataType>(tensor()->dtype());

  const auto& shape_vector = global_shape();
  return absl::StrCat("DTensor(", SummarizeValue(),
                      ", shape=", ShapeToDebugString(shape_vector),
                      ", type=", DataTypeString(dtype), ")");
}

void ResourceHandleWithLayout::EncodeAttributes(
    tensorflow::NodeDefBuilder& builder) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_10(mht_10_v, 493, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ResourceHandleWithLayout::EncodeAttributes");

  // If set, attach shape and dtype to the given node def.
  if (dereferenced_shape().has_value()) {
    builder.Attr("_handle_shapes", {*dereferenced_shape()});
  }
  if (dereferenced_dtype().has_value()) {
    builder.Attr("_handle_dtypes", {*dereferenced_dtype()});
  }
}

tensorflow::Fprint128 ResourceHandleWithLayout::CacheKey() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_11(mht_11_v, 506, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ResourceHandleWithLayout::CacheKey");

  tensorflow::Fprint128 f = tensorflow::Fingerprint128(layout().ToString());
  if (dereferenced_shape().has_value()) {
    std::string serialized;
    SerializeToStringDeterministic(dereferenced_shape().value(), &serialized);
    f = FingerprintCat128(f, tensorflow::Fingerprint128(serialized));
  }
  if (dereferenced_dtype().has_value()) {
    f = FingerprintCat128(f, dereferenced_dtype().value());
  }
  return f;
}

void ResourceHandleWithLayout::UpdateLayout(const Layout& new_layout,
                                            TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_12(mht_12_v, 523, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "ResourceHandleWithLayout::UpdateLayout");

  // Only set the value for deferenced layout if the incoming layout is not
  // empty. This is still hacky as we use empty layout as placeholder for
  // eagerly placed VarHandleOp.
  if (!dereferenced_layout_.has_value() && new_layout.IsEmpty()) return;
  if (dereferenced_layout_.has_value() &&
      !LayoutsAreCompatible(dereferenced_layout_, new_layout)) {
    // TODO(xiejw, allenl): Consider allowing variables to switch layouts.
    RETURN_STATUS(status, TF_INVALID_ARGUMENT,
                  "Attempted to overwrite an existing Layout.");
  }
  dereferenced_layout_.emplace(new_layout);
}

StatusOr<std::unique_ptr<TensorWithLayout>> SparseTensorWithLayout::Wrap(
    std::unique_ptr<parallel_device::ParallelTensor> indices_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> values_tensor,
    std::unique_ptr<parallel_device::ParallelTensor> shapes_tensor,
    const MeshWithParallelDevice& mesh, const Layout& layout,
    std::vector<int64_t> local_shape) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_13(mht_13_v, 545, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "SparseTensorWithLayout::Wrap");

  return std::unique_ptr<TensorWithLayout>(new SparseTensorWithLayout(
      std::move(indices_tensor), std::move(values_tensor),
      std::move(shapes_tensor), mesh, layout, local_shape));
}

std::string SparseTensorWithLayout::SummarizeValue() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_14(mht_14_v, 554, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "SparseTensorWithLayout::SummarizeValue");

  std::string indices_summary;
  std::string values_summary;
  std::string dense_shapes_summary;

  Status indices_status;
  Status values_status;
  Status dense_shapes_status;

  if (layout().IsFullyReplicated()) {
    indices_status = tensorflow::unwrap(indices_->tensor(0))
                         ->SummarizeValue(indices_summary);
    values_status =
        tensorflow::unwrap(values_->tensor(0))->SummarizeValue(values_summary);
    dense_shapes_status = tensorflow::unwrap(dense_shapes_->tensor(0))
                              ->SummarizeValue(dense_shapes_summary);
  } else {
    indices_status = indices_->SummarizeValue(indices_summary);
    values_status = values_->SummarizeValue(values_summary);
    dense_shapes_status = dense_shapes_->SummarizeValue(dense_shapes_summary);
  }

  if (!indices_status.ok())
    values_summary = "<error computing summary for indices>";
  if (!values_status.ok())
    indices_summary = "<error computing summary for values>";
  if (!dense_shapes_status.ok())
    indices_summary = "<error computing summary for dense_shapes>";

  return absl::StrCat("indices: ", indices_summary, ", ",
                      "values: ", values_summary, ", ",
                      "dense_shapes: ", dense_shapes_summary, ", layout=\"",
                      layout().ToString(), "\"");
}

std::string SparseTensorWithLayout::DebugString() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_15(mht_15_v, 592, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "SparseTensorWithLayout::DebugString");

  auto dtype = static_cast<DataType>(values_->dtype());

  const auto& shape_vector = global_shape();
  return absl::StrCat("DTensor(", SummarizeValue(),
                      ", shape=", ShapeToDebugString(shape_vector),
                      ", type=", DataTypeString(dtype), ")");
}

TF_DataType SparseTensorWithLayout::dtype() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_16(mht_16_v, 604, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "SparseTensorWithLayout::dtype");

  if (dtype_.has_value()) {
    return dtype_.value();
  } else {
    return values_->dtype();
  }
}

TFE_TensorHandle* SparseTensorWithLayout::get_tensor(size_t index) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_17(mht_17_v, 615, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "SparseTensorWithLayout::get_tensor");

  int num_sparse_tensors = num_tensors() / 3;
  if (index < num_sparse_tensors) {
    return indices()->tensor(index);
  } else if (index < 2 * num_sparse_tensors) {
    return values()->tensor(index % num_sparse_tensors);
  } else {
    return dense_shapes()->tensor(index % num_sparse_tensors);
  }
}

std::vector<int64_t> TensorShapeAsVector(TFE_TensorHandle* tensor,
                                         TF_Status* status) {
  std::vector<int64_t> shape(TFE_TensorHandleNumDims(tensor, status));
  if (TF_GetCode(status) != TF_OK) return {};
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(tensor, i, status);
    if (TF_GetCode(status) != TF_OK) return {};
  }
  return shape;
}

Status PrepareGraphForMlir(
    const std::vector<TensorWithLayout*>& inputs,
    const DTensorOperation& doperation,
    const tensorflow::FunctionLibraryDefinition& flib_def,
    const NameAttrList& attributes,
    const absl::optional<Layout>& default_layout, tensorflow::Graph* graph,
    std::vector<PartialTensorShape>* global_output_shapes,
    std::vector<const Layout*>* output_layouts) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_18(mht_18_v, 647, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "PrepareGraphForMlir");

  // We run shape inference on the graph to find output shapes, which may
  // determine default layouts.
  ShapeRefiner shape_refiner(TF_GRAPH_DEF_VERSION, &flib_def);
  shape_refiner.set_function_library_for_shape_inference(&flib_def);
  tensorflow::Status status;
  {
    // We include an _Arg node for the device ID, but this isn't used by the
    // initial function. It will be provided a value, though, so it's available
    // for use in rewrites.
    tensorflow::NodeDefBuilder builder("device_id", "_Arg");
    tensorflow::PartialTensorShape partial_shape;
    TF_RETURN_IF_ERROR(tensorflow::PartialTensorShape::MakePartialShape(
        static_cast<int*>(nullptr), 0, &partial_shape));
    tensorflow::NodeDef arg_node_def;
    TF_RETURN_IF_ERROR(builder.Attr("shape", partial_shape)
                           .Attr("T", tensorflow::DT_INT32)
                           .Attr("index", 0)
                           .Finalize(&arg_node_def, /*consume=*/true));
    tensorflow::Node* arg_node = graph->AddNode(arg_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    graph->AddControlEdge(graph->source_node(), arg_node);
    TF_RETURN_IF_ERROR(shape_refiner.AddNode(arg_node));
  }
  std::vector<FunctionArgument> graph_op_inputs;
  graph_op_inputs.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    const TensorWithLayout* input = inputs[i];
    // TODO(allenl): This will block until async execution is complete, which
    // will be slow. We should find a non-blocking way of fetching the shape,
    // at least pre-cache.
    // The shape passed into MLIR transformation represents the global shape of
    // the tensor. Ideally, the local shape on each parallel device should not
    // be consulted at all and we should use the shape on our input tensor
    // directly.
    const auto& shape = input->global_shape();
    std::vector<tensorflow::int64> cast_shape(shape.begin(), shape.end());
    tensorflow::PartialTensorShape partial_shape;
    // For resource tensors, `shape` attribute should not be specified as shape
    // of resource tensors is specified by resource shape subtype -- not the
    // shape attribute.
    auto* resource = dynamic_cast<const ResourceHandleWithLayout*>(input);
    if (!resource) {
      TF_RETURN_IF_ERROR(tensorflow::PartialTensorShape::MakePartialShape(
          cast_shape.data(), cast_shape.size(), &partial_shape));
    }

    tensorflow::NodeDef arg_node_def;
    auto dtype = static_cast<tensorflow::DataType>(input->dtype());
    tensorflow::NodeDefBuilder builder(absl::StrCat("op_input_", i), "_Arg");

    // Delegate TensorWithLayout to encode attributes if applicable.
    input->EncodeAttributes(builder);

    TF_RETURN_IF_ERROR(
        builder.Attr("shape", partial_shape)
            .Attr("T", dtype)
            .Attr("index", i + 1)  // Indices are offset by 1 for device_id
            .Attr(kLayoutAttr, input->layout().ToString())
            .Attr(kMeshAttr, input->mesh().mesh_config().ToString())
            .Finalize(&arg_node_def, /*consume=*/true));
    Node* arg_node = graph->AddNode(arg_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(shape_refiner.AddNode(arg_node));

    shape_inference::InferenceContext* inference_context =
        shape_refiner.GetContext(arg_node);
    shape_inference::ShapeHandle shape_handle;
    TF_RETURN_IF_ERROR(inference_context->MakeShapeFromPartialTensorShape(
        partial_shape, &shape_handle));
    TF_RETURN_IF_ERROR(shape_refiner.SetShape(arg_node, 0, shape_handle));

    // Small constants are converted into constant graph nodes, instead of being
    // passed in as input arguments. This provides more information to the SPMD
    // and layout propagation passes.
    if (!input->const_value().has_value()) {
      graph_op_inputs.push_back(FunctionArgument{
          arg_node, NodeDefBuilder::NodeOut{arg_node->name(), i, dtype}});
      graph->AddControlEdge(graph->source_node(), arg_node);
    } else {
      // TODO(xiejw): Refactor the TensorWithLayout representation to avoid
      // special code here.
      NodeDef const_node = input->const_value().value();
      const_node.set_name(absl::StrCat("input_", i, "_const_value"));
      Node* const_value_n = graph->AddNode(const_node, &status);
      TF_RETURN_IF_ERROR(status);
      TF_RETURN_IF_ERROR(shape_refiner.AddNode(const_value_n));
      graph_op_inputs.push_back(FunctionArgument{
          const_value_n, tensorflow::NodeDefBuilder::NodeOut{
                             const_value_n->name(), i, dtype}});
    }
  }

  tensorflow::NodeDef op_node_def;
  const FunctionDef* function_def = doperation.function_def;
  if (function_def) {
    AttrValue func_attr;
    func_attr.mutable_func()->set_name(doperation.name);
    std::vector<tensorflow::NodeDefBuilder::NodeOut> func_inputs;
    std::vector<tensorflow::DataType> inputs_types;
    for (const auto& in : graph_op_inputs) {
      func_inputs.emplace_back(in.output);
      inputs_types.emplace_back(in.output.data_type);
    }

    std::vector<tensorflow::DataType> output_types;
    for (const auto& out : function_def->signature().output_arg())
      output_types.emplace_back(out.type());

    TF_RETURN_IF_ERROR(
        NodeDefBuilder("eager_operation", "StatefulPartitionedCall")
            .Attr("Tin", inputs_types)
            .Attr("Tout", output_types)
            .Attr("f", func_attr)
            .Input(func_inputs)
            .Finalize(&op_node_def, true));
  } else {
    op_node_def.set_op(doperation.name);
    op_node_def.set_name("eager_operation");
  }

  op_node_def.mutable_attr()->insert(attributes.attr().begin(),
                                     attributes.attr().end());

  tensorflow::Node* op_node = graph->AddNode(op_node_def, &status);
  TF_RETURN_IF_ERROR(status);

  for (int i = 0; i < graph_op_inputs.size(); ++i) {
    graph->AddEdge(graph_op_inputs[i].node, 0, op_node, i);
  }
  TF_RETURN_IF_ERROR(shape_refiner.AddNode(op_node));

  output_layouts->clear();
  output_layouts->reserve(op_node->num_outputs());
  global_output_shapes->reserve(op_node->num_outputs());
  for (int output_index = 0; output_index < op_node->num_outputs();
       ++output_index) {
    tensorflow::NodeDefBuilder builder(absl::StrCat("op_output_", output_index),
                                       "_Retval");
    tensorflow::NodeDef ret_node_def;
    tensorflow::DataType output_type = op_node->output_type(output_index);

    TF_RETURN_IF_ERROR(builder.Attr("T", output_type)
                           .Attr("index", output_index)
                           .Input("eager_operation", output_index, output_type)
                           .Finalize(&ret_node_def, /*consume=*/true));
    tensorflow::Node* ret_node = graph->AddNode(ret_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(op_node, output_index, ret_node, 0);
    graph->AddControlEdge(ret_node, graph->sink_node());

    shape_inference::InferenceContext* inference_context =
        shape_refiner.GetContext(op_node);
    shape_inference::ShapeHandle output_shape_handle =
        inference_context->output(output_index);
    TensorShapeProto output_shape_proto;
    inference_context->ShapeHandleToProto(output_shape_handle,
                                          &output_shape_proto);
    PartialTensorShape global_output_shape(output_shape_proto);
    VLOG(3) << "Inferred shape for operation '" << doperation.name
            << "':" << global_output_shape.DebugString();
    global_output_shapes->push_back(global_output_shape);

    const Layout* layout = nullptr;
    if (default_layout.has_value() && output_index == 0) {
      // Record the user's requested output layout. The scope currently only
      // covers the first output of an op.
      layout = &default_layout.value();
      ret_node->AddAttr(kDefaultLayoutAttr, layout->ToString());
    }
    output_layouts->push_back(layout);
  }
  return Status::OK();
}

// Returns set of functions to run to execute DTensor computation.
StatusOr<ExecutionFunctions> IdentifyAllFunctionsToExecute(
    const tensorflow::Graph& graph,
    const std::vector<PartialTensorShape>& global_output_shapes) {
  ExecutionFunctions execution_functions;
  execution_functions.function_list = std::vector<TranslatedFunction>();
  for (Node* node : graph.nodes()) {
    if (node->op_def().name() != "StatefulPartitionedCall") continue;
    // Extract mesh to execute the function.
    std::string serialized_mesh;
    TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), kMeshAttr, &serialized_mesh));
    Mesh mesh;
    TF_ASSIGN_OR_RETURN(mesh, Mesh::FromString(serialized_mesh));

    TranslatedFunction function;
    function.function_mesh = std::move(mesh);
    function.node_to_execute = node;

    // Identify input arg information.
    TF_RETURN_IF_ERROR(
        ParseResourceArgumentLayouts(*node, &function.resource_input_layouts));

    TF_RETURN_IF_ERROR(
        ParseShapeInputLayouts(*node, &function.shape_output_metadata));

    function.input_index_map.resize(node->num_inputs());
    // Identity mapping between local mesh function input index and global
    // input index.
    for (int in_index = 0; in_index < node->num_inputs(); ++in_index) {
      Node* input_node;

      TF_RETURN_IF_ERROR(node->input_node(in_index, &input_node));
      if (!input_node->IsArg())
        return errors::InvalidArgument(
            "Input node to mesh computation must be arg node.");

      int global_index;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(input_node->attrs(), "index", &global_index));
      function.input_index_map[in_index] = global_index;
    }

    // Identify output mappings and layouts for each outputs.
    std::map<int, const Edge*> output_edges;
    for (const Edge* out_edge : node->out_edges()) {
      if (out_edge->IsControlEdge()) continue;

      const Node* retval_or_identity_node = out_edge->dst();
      while (retval_or_identity_node->IsIdentity()) {
        retval_or_identity_node =
            *(retval_or_identity_node->out_nodes().begin());
      }

      TF_RET_CHECK(retval_or_identity_node->IsRetval());
      int global_index;
      TF_RETURN_IF_ERROR(GetNodeAttr(retval_or_identity_node->attrs(), "index",
                                     &global_index));
      output_edges[global_index] = out_edge;
    }

    for (auto it = output_edges.begin(); it != output_edges.end(); it++) {
      const int global_index = it->first;
      function.output_index_map.emplace_back(global_index);

      const Edge* retval_edge = it->second;
      const int output_index = retval_edge->src_output();

      // Add output layout and shape information.
      TF_ASSIGN_OR_RETURN(
          const Layout output_layout,
          GetLayoutThroughIdentityOps(retval_edge->src(), output_index));

      function.output_layouts.emplace_back(output_layout);
      function.local_output_shapes.emplace_back(
          output_layout.LocalShapeFromGlobalShape(
              global_output_shapes[global_index]));
    }

    execution_functions.function_list.emplace_back(std::move(function));
  }

  if (execution_functions.function_list.empty()) {
    return errors::InvalidArgument(
        "MLIR transformed graph does not have any functions to execute for "
        "mesh.");
  }

  return execution_functions;
}

// For functions with control outputs, add identity nodes between
// StatefulPartitionedCall and _Retvals, in order to preserve control output
// dependencies after StatefulPartitionedCall is inlined at runtime.
// Consider calling this in PrepareGraphForMlir, once the identity nodes won't
// be dropped during MLIR lowering.
// TODO(b/171265131): fix the underlying issue to avoid inserting identity
// nodes.
Status MaybeInsertIdentityNodes(const FunctionDef* function_def, Graph* graph) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_19(mht_19_v, 922, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "MaybeInsertIdentityNodes");

  if (function_def == nullptr || function_def->control_ret().empty()) {
    return Status::OK();
  }
  tensorflow::Status status;
  for (Node* n : graph->nodes()) {
    if (!n->IsRetval()) {
      continue;
    }
    const Edge* edge;
    TF_RETURN_IF_ERROR(n->input_edge(0, &edge));
    int ret_index;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &ret_index));
    tensorflow::NodeDefBuilder identity_builder(
        absl::StrCat("op_output_identity_", ret_index), "Identity");
    tensorflow::NodeDef ret_identity_node_def;
    tensorflow::DataType output_type = n->input_type(0);
    TF_RETURN_IF_ERROR(
        identity_builder.Attr("T", output_type)
            .Input(edge->src()->name(), edge->src_output(), output_type)
            .Finalize(&ret_identity_node_def, /*consume=*/true));
    Node* ret_identity_node = graph->AddNode(ret_identity_node_def, &status);
    TF_RETURN_IF_ERROR(status);
    // Delete the edge between StatefulPartitionedCall and _Retval.
    graph->RemoveEdge(edge);
    // Add an edge between StatefulPartitionedCall and Identity.
    graph->AddEdge(edge->src(), edge->src_output(), ret_identity_node, 0);
    graph->AddControlEdge(edge->src(), ret_identity_node);
    // Add an edge between Identity and _Retval.
    graph->AddEdge(ret_identity_node, 0, n, 0);
  }
  return Status::OK();
}

void AddDTensorFunctionAttr(FunctionDef& function_def) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_device_utilDTcc mht_20(mht_20_v, 959, "", "./tensorflow/dtensor/cc/dtensor_device_util.cc", "AddDTensorFunctionAttr");

  // Do not xla compile function returned by DTensor MLIR graph transformation
  // as it already returns compiled graph.
  AttrValue xla_must_compile_val;
  xla_must_compile_val.set_b(false);
  function_def.mutable_attr()->insert(
      {"_XlaMustCompile", xla_must_compile_val});

  // Explicitly place function outputs on the default function device to avoid
  // redundant host <-> device copies (Placer may place outputs on the host
  // CPU).
  AttrValue outputs_on_op_device;
  outputs_on_op_device.set_b(true);
  function_def.mutable_attr()->insert(
      {"_OutputsOnOpDevice", outputs_on_op_device});
}

}  // namespace dtensor
}  // namespace tensorflow
