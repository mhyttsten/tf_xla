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
class MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/export.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/functiondef_export.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#define DEBUG_TYPE "graphdef-to-mlir"

using tensorflow::DataType;
using tensorflow::FunctionDef;
using tensorflow::GetValueNameFn;
using tensorflow::GradientDef;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::OpDef;
using tensorflow::Status;
using tensorflow::VersionDef;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {
namespace {

constexpr StringRef kNameAttr = TFGraphDialect::getNameAttrKey();
constexpr StringRef kDeviceAttr = TFGraphDialect::getDeviceAttrKey();
constexpr StringRef kFullTypeAttr = TFGraphDialect::getFullTypeAttrKey();
constexpr char kAliasingAttr[] = "tf.aliasing_output";

// Compute the name to use in GraphDef for a given Value (either the result of
// an operation or a block operand if a function argument) and store the result
// in the provided name string. The `control_ty` is the instance of the
// `ControlType` to compare against and detect a control dependency case.
static Status GetValueName(Value operand, std::string &name, Type control_ty) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_0(mht_0_v, 266, "", "./tensorflow/core/ir/importexport/export.cc", "GetValueName");

  OpResult op_result = operand.dyn_cast<OpResult>();
  if (!op_result) {
    BlockArgument block_operand = operand.dyn_cast<BlockArgument>();
    bool is_control = (block_operand.getType() == control_ty);
    int arg_num = block_operand.getArgNumber();
    name.clear();
    // Function arguments are coming as pair: the even are the actual tensors
    // while the odd position are the associated control input.
    if (is_control) name = "^";
    DictionaryAttr arg_attrs = function_interface_impl::getArgAttrDict(
        block_operand.getParentBlock()->getParentOp(), arg_num - is_control);
    if (!arg_attrs)
      return InvalidArgument("Missing attribute for argument #", arg_num);
    StringAttr arg_name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!arg_name)
      return InvalidArgument(
          "Can't export graph with missing op-name for function parameter #",
          arg_num);
    absl::StrAppend(&name, arg_name.getValue().str());
    return {};
  }
  Operation *producer = op_result.getDefiningOp();
  auto nameAttr = producer->getAttrOfType<StringAttr>(kNameAttr);
  if (!nameAttr)
    return InvalidArgument("Can't export graph with missing op-name");

  name.clear();
  if (op_result.getType() == control_ty) name = "^";
  absl::StrAppend(&name, nameAttr.getValue().str());
  if (op_result.getType() != control_ty && op_result.getResultNumber())
    absl::StrAppend(&name, ":", op_result.getResultNumber());
  return {};
}

Status GetArgumentNode(GraphFuncOp func, NodeDef *node_def, unsigned index,
                       StringRef name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_1(mht_1_v, 305, "", "./tensorflow/core/ir/importexport/export.cc", "GetArgumentNode");

  node_def->set_name(name.str());
  node_def->set_op(tensorflow::FunctionLibraryDefinition::kArgOp);
  TensorType arg_type = func.getArgument(index).getType().cast<TensorType>();

  if (auto resource_type = arg_type.getElementType().dyn_cast<ResourceType>()) {
    ArrayRef<TensorType> subtypes = resource_type.getSubtypes();
    if (!subtypes.empty()) {
      tensorflow::AttrValue handle_dtypes_attr;
      tensorflow::AttrValue handle_shapes_attr;
      for (TensorType subtype : subtypes) {
        DataType dtype;
        TF_RETURN_IF_ERROR(ConvertToDataType(subtype.getElementType(), &dtype));
        handle_dtypes_attr.mutable_list()->add_type(dtype);

        SetTensorShapeProto(subtype,
                            handle_shapes_attr.mutable_list()->add_shape());
      }

      (*node_def->mutable_attr())["_handle_dtypes"] = handle_dtypes_attr;
      (*node_def->mutable_attr())["_handle_shapes"] = handle_shapes_attr;
    }
  }

  if (arg_type.isa<RankedTensorType>())
    TF_RETURN_IF_ERROR(SetShapeAttribute("_output_shapes", arg_type,
                                         node_def->mutable_attr()));

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(arg_type.getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;

  tensorflow::AttrValue index_attr;
  index_attr.set_i(index / 2);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr = func.getArgAttrOfType<StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();
  if (tf_type::FullTypeAttr fulltype_attr =
          func.getArgAttrOfType<tf_type::FullTypeAttr>(
              index, "tfg.experimental_full_type")) {
    TF_ASSIGN_OR_RETURN(*node_def->mutable_experimental_type(),
                        ConvertAttribute(fulltype_attr));
  }

  ArrayRef<NamedAttribute> func_arg_i_attrs = func.getArgAttrs(index);
  TF_RETURN_IF_ERROR(
      ConvertAttributes(func_arg_i_attrs,
                        {kDeviceAttr, kAliasingAttr, "tfg.name", "tfg.dtype",
                         "tfg.experimental_full_type", "tfg.handle_data"},
                        /*remove_ref_type=*/false, node_def->mutable_attr()));

  return Status::OK();
}

Status GetReturnNode(GraphFuncOp function, Value operand, unsigned index,
                     StringRef name, NodeDef *node_def,
                     ControlType control_ty) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_2(mht_2_v, 367, "", "./tensorflow/core/ir/importexport/export.cc", "GetReturnNode");

  node_def->set_name(name.str() + "_mlir_ret");
  node_def->set_op(tensorflow::FunctionLibraryDefinition::kRetOp);

  std::string input_name;
  TF_RETURN_IF_ERROR(GetValueName(operand, input_name, control_ty));
  node_def->add_input(input_name);

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(
      operand.getType().cast<TensorType>().getElementType(), &dtype));
  tensorflow::AttrValue type_attr;
  type_attr.set_type(dtype);
  (*node_def->mutable_attr())["T"] = type_attr;
  tensorflow::AttrValue index_attr;
  index_attr.set_i(index);
  (*node_def->mutable_attr())["index"] = index_attr;

  if (auto device_attr =
          function.getResultAttrOfType<StringAttr>(index, kDeviceAttr))
    *node_def->mutable_device() = device_attr.getValue().str();
  if (auto fulltype_attr = function.getResultAttrOfType<tf_type::FullTypeAttr>(
          index, "tfg.experimental_full_type")) {
    TF_ASSIGN_OR_RETURN(*node_def->mutable_experimental_type(),
                        ConvertAttribute(fulltype_attr));
  }

  ArrayRef<NamedAttribute> func_res_i_attrs = function.getResultAttrs(index);
  TF_RETURN_IF_ERROR(
      ConvertAttributes(func_res_i_attrs,
                        {kDeviceAttr, kAliasingAttr, "tfg.name", "tfg.dtype",
                         "tfg.experimental_full_type", "tfg.handle_data"},
                        /*remove_ref_type=*/false, node_def->mutable_attr()));

  return Status::OK();
}

// Converts a location to the debug information for the node def, if we find
// supported location, that is a top-level NameLoc or any NameLoc nested inside
// a FusedLoc. Other kind of location are ignored. If a NameLoc is of the form
// "name@func" we parse it and import the two appropriately.
void ExtractExperimentalDebugInfoFromLocation(
    mlir::Location inst_loc, NodeDef::ExperimentalDebugInfo *debug_info) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_3(mht_3_v, 412, "", "./tensorflow/core/ir/importexport/export.cc", "ExtractExperimentalDebugInfoFromLocation");

  auto add_name_loc = [&](mlir::NameLoc name_loc) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_4(mht_4_v, 416, "", "./tensorflow/core/ir/importexport/export.cc", "lambda");

    StringRef node, func;
    std::tie(node, func) = name_loc.getName().strref().split('@');
    debug_info->add_original_node_names(node.str());
    if (!func.empty()) debug_info->add_original_func_names(func.str());
  };
  if (auto fused = inst_loc.dyn_cast<mlir::FusedLoc>()) {
    for (Location loc : fused.getLocations())
      if (auto name_loc = loc.dyn_cast<mlir::NameLoc>()) add_name_loc(name_loc);
    return;
  }
  if (auto name_loc = inst_loc.dyn_cast<mlir::NameLoc>())
    add_name_loc(name_loc);
}

// Convert an MLIR operation to a NodeDef. The `control_ty` is the instance of
// the `ControlType` to compare against and detect a control dependency case.
Status ConvertOperationToNodeImpl(Operation &op, NodeDef *node,
                                  GetValueNameFn get_value_name) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_5(mht_5_v, 437, "", "./tensorflow/core/ir/importexport/export.cc", "ConvertOperationToNodeImpl");

  auto nameAttr = op.getAttrOfType<StringAttr>(kNameAttr);
  if (nameAttr) node->set_name(nameAttr.getValue().str());
  auto deviceAttr = op.getAttrOfType<StringAttr>(kDeviceAttr);
  if (deviceAttr) node->set_device(deviceAttr.getValue().str());
  if (auto fulltype_attr =
          op.getAttrOfType<tf_type::FullTypeAttr>(kFullTypeAttr)) {
    TF_ASSIGN_OR_RETURN(*node->mutable_experimental_type(),
                        ConvertAttribute(fulltype_attr));
  }
  std::string name;
  for (Value operand : op.getOperands()) {
    TF_RETURN_IF_ERROR(get_value_name(operand, name));
    node->add_input(name);
  }
  StringRef op_name = op.getName().stripDialect();
  if (op_name == "LegacyCall") {
    auto callee = op.getAttrOfType<FuncAttr>("callee");
    if (!callee)
      return InvalidArgument("Missing callee attribute on LegacyCall");
    StringRef callee_name = callee.getName().getRootReference().getValue();
    node->set_op({callee_name.data(), callee_name.size()});
    TF_RETURN_IF_ERROR(ConvertAttributes(
        callee.getAttrs().getValue(), {kNameAttr, kDeviceAttr, kFullTypeAttr},
        /*remove_ref_type=*/false, node->mutable_attr()));
    auto optional_device =
        op.getAttrDictionary().getNamed("_mlir_assigned_device");
    if (optional_device.hasValue()) {
      NamedAttrList assigned_device;
      assigned_device.push_back(*optional_device);
      TF_RETURN_IF_ERROR(ConvertAttributes(assigned_device, {},
                                           /*remove_ref_type=*/false,
                                           node->mutable_attr()));
    }
  } else {
    node->set_op({op_name.data(), op_name.size()});
    TF_RETURN_IF_ERROR(ConvertAttributes(
        op.getAttrs(), {kNameAttr, kDeviceAttr, kFullTypeAttr},
        /*remove_ref_type=*/false, node->mutable_attr()));
  }
  // Eliminate empty "_mlir_assigned_device" from the export. This is just
  // more friendly to the serialization.
  {
    auto it = node->mutable_attr()->find("_mlir_assigned_device");
    if (it != node->mutable_attr()->end() && it->second.s().empty())
      node->mutable_attr()->erase("_mlir_assigned_device");
  }

  // Export the location as debug info on the nodes.
  ExtractExperimentalDebugInfoFromLocation(
      op.getLoc(), node->mutable_experimental_debug_info());
  if (node->experimental_debug_info().original_node_names().empty())
    node->clear_experimental_debug_info();

  return Status::OK();
}

// Convert the handle_data_arr to the `handle_data` field of the provided arg.
// Each entry of the array is itself an array with two entries: a Type and a
// ShapeAttr.
static Status ConvertHandleDataImpl(ArrayAttr handle_data_arr,
                                    OpDef::ArgDef *arg) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_6(mht_6_v, 501, "", "./tensorflow/core/ir/importexport/export.cc", "ConvertHandleDataImpl");

  if (!handle_data_arr) return {};
  for (auto handle_data_attr : handle_data_arr.getAsRange<TypeAttr>()) {
    TensorType handle_type = handle_data_attr.getValue().dyn_cast<TensorType>();
    if (!handle_type) {
      return InvalidArgument("Expected an array of tensor types, but got ",
                             debugString(handle_data_arr));
    }
    auto *handle_data = arg->add_handle_data();
    if (handle_type.hasRank()) {
      ConvertToTensorShapeProto(handle_type.getShape(),
                                handle_data->mutable_shape());
    } else {
      handle_data->mutable_shape()->set_unknown_rank(true);
    }
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(handle_type.getElementType(), &dtype));
    handle_data->set_dtype(dtype);
  }
  return {};
}

Status BuildFunctionSignature(GraphFuncOp func_op, FunctionDef &fdef) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_7(mht_7_v, 526, "", "./tensorflow/core/ir/importexport/export.cc", "BuildFunctionSignature");

  const std::string func_name = func_op.getName().str();
  OpDef *signature = fdef.mutable_signature();
  signature->set_name(func_name);
  if (func_op->getAttr("is_stateful")) signature->set_is_stateful(true);
  if (auto description = func_op->getAttrOfType<StringAttr>("description"))
    signature->set_description(description.getValue().str());

  // Handle the results now.
  // An ArgDef entry needs to be constructed for all non-control returned value.
  auto return_op = cast<ReturnOp>(func_op.getBody()->getTerminator());
  if (!return_op.control_ret_attrs()) {
    return InvalidArgument(
        "Can't export function ", func_name,
        " because return op is missing \"control_ret_attrs\"");
  }
  StringAttr tfg_name_key =
      cast<TFGraphDialect>(func_op->getDialect())->getTfgNameAttrIdentifier();

  // Export the data operands.
  for (auto it :
       llvm::zip(llvm::enumerate(TFOp(return_op).getNonControlOperands()),
                 func_op.getAllResultAttrs().getAsRange<DictionaryAttr>())) {
    DictionaryAttr attrs = std::get<1>(it);
    auto name = attrs.getAs<StringAttr>(tfg_name_key);
    if (!name) {
      return InvalidArgument(
          "Can't export function ", func_name,
          " because missing \"tfg.name\" attribute for result #",
          std::get<0>(it).index());
    }
    OpDef::ArgDef *arg = signature->add_output_arg();
    arg->set_name(name.getValue().str());
    StringAttr description = attrs.getAs<StringAttr>("tfg.description");
    if (description) arg->set_description(description.getValue().str());
    TF_RETURN_IF_ERROR(
        ConvertHandleData(attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
    if (tf_type::FullTypeAttr full_type =
            attrs.getAs<tf_type::FullTypeAttr>("tfg.experimental_full_type")) {
      TF_ASSIGN_OR_RETURN(*arg->mutable_experimental_full_type(),
                          ConvertAttribute(full_type));
    }
  }

  // Export the control operands.
  for (auto it : llvm::zip(
           llvm::enumerate(TFOp(return_op).getControlOperands()),
           return_op.control_ret_attrsAttr().getAsRange<DictionaryAttr>())) {
    auto name = std::get<1>(it).getAs<StringAttr>(tfg_name_key);
    if (!name) {
      return InvalidArgument(
          "Can't export function ", func_name,
          " because missing \"tfg.name\" attribute for control result #",
          std::get<0>(it).index());
    }
    signature->add_control_output(name.getValue().str());
  }

  return Status::OK();
}

// Given an MLIR module, returns a GraphDef.
Status ExportMlirToGraphdefImpl(ModuleOp module, GraphDef *graphdef) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_8(mht_8_v, 591, "", "./tensorflow/core/ir/importexport/export.cc", "ExportMlirToGraphdefImpl");

  // Check that this module is valid for export: it should only contains at most
  // a single Graph operation and one or more GraphFunc operations.
  GraphOp graph_op;
  for (Operation &op : *module.getBody()) {
    if (isa<GraphFuncOp>(op)) continue;
    if (auto new_graph_op = dyn_cast<GraphOp>(op)) {
      if (graph_op) {
        return InvalidArgument(
            "Can't export module with two different tfg.graph");
      }
      graph_op = new_graph_op;
      continue;
    }
    return InvalidArgument(
        absl::StrCat("Can't export module with other ops than tfg.graph or "
                     "tfg.func, has: ",
                     op.getName().getStringRef().data()));
  }
  if (graph_op) {
    // A graph is mostly a flat "sea of nodes" to export.
    auto control_ty = tfg::ControlType::get(graph_op.getContext());
    VersionDef *version = graphdef->mutable_versions();
    tfg::VersionAttr versionAttr = graph_op.version();
    version->set_producer(versionAttr.getProducer());
    version->set_min_consumer(versionAttr.getMinConsumer());
    for (int32_t bad_consumer : versionAttr.getBadConsumers())
      version->add_bad_consumers(bad_consumer);
    for (Operation &op : *graph_op.getBody()) {
      NodeDef *node = graphdef->add_node();
      TF_RETURN_IF_ERROR(ConvertOperationToNode(
          op, node, [&](Value operand, std::string &output_name) {
            return GetValueName(operand, output_name, control_ty);
          }));
    }
  }

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             *graphdef->mutable_library());
  // Export the functions, if any.
  for (GraphFuncOp func_op :
       llvm::reverse(module.getBody()->getOps<GraphFuncOp>())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Exporting function @" << func_op.getName() << "\n");
    if (flib.Find(func_op.getName().str())) continue;
    if (!func_op.generic()) {
      // Export only the signature here, we'll export these below.
      FunctionDef fdef;
      TF_RETURN_IF_ERROR(BuildFunctionSignature(func_op, fdef));
      TF_RETURN_IF_ERROR(flib.AddFunctionDef(fdef));
      continue;
    }
    // We can immediately export generic functions, because they don't need to
    // go through a "Graph" and aren't sensitive to importing called function
    // first.
    TF_ASSIGN_OR_RETURN(FunctionDef fdef,
                        ConvertGenericFunctionToFunctionDef(func_op));
    if (flib.Find(fdef.signature().name()))
      TF_RETURN_IF_ERROR(flib.ReplaceFunction(fdef.signature().name(), fdef));
    else
      TF_RETURN_IF_ERROR(flib.AddFunctionDef(fdef));
  }
  for (GraphFuncOp func_op :
       llvm::reverse(module.getBody()->getOps<GraphFuncOp>())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Exporting function @" << func_op.getName() << "\n");
    if (func_op.generic()) continue;
    TF_RETURN_IF_ERROR(ExportFunction(func_op, flib));
  }
  *graphdef->mutable_library() = flib.ToProto();

  return Status::OK();
}

}  // namespace

Status ExportFunction(GraphFuncOp func_op,
                      tensorflow::FunctionLibraryDefinition &flib) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_9(mht_9_v, 671, "", "./tensorflow/core/ir/importexport/export.cc", "ExportFunction");

  const std::string func_name = func_op.getName().str();
  // The function->gradient mapping is stored separately in the library.
  if (auto gradient_attr =
          func_op->getAttrOfType<FlatSymbolRefAttr>("gradient")) {
    GradientDef gradient;
    gradient.set_gradient_func(gradient_attr.getValue().str());
    gradient.set_function_name(func_name);
    TF_RETURN_IF_ERROR(flib.AddGradientDef(gradient));
  }

  auto control_ty = tfg::ControlType::get(func_op.getContext());
  GraphDef graph_def;
  ArrayAttr args_attr = func_op.getAllArgAttrs();
  for (int arg_num : llvm::seq<int>(0, func_op.getNumArguments())) {
    // Odd position are just for control dependencies.
    if (arg_num % 2) continue;
    DictionaryAttr arg_attrs = args_attr[arg_num].dyn_cast<DictionaryAttr>();
    auto name = arg_attrs.getAs<StringAttr>("tfg.name");
    if (!name || name.getValue().empty())
      return tensorflow::errors::InvalidArgument(
          "Missing tfg.name on argument ", arg_num);
    NodeDef *node_def = graph_def.add_node();
    TF_RETURN_IF_ERROR(
        GetArgumentNode(func_op, node_def, arg_num, name.getValue()));
  }
  // Convert the invidual nodes in the function body, since the function is
  // terminated by a return operation we skip it in this loop and handled it
  // separately later.
  for (Operation &op : func_op.getBody()->without_terminator())
    TF_RETURN_IF_ERROR(ConvertOperationToNode(
        op, graph_def.add_node(), [&](Value operand, std::string &output_name) {
          return GetValueName(operand, output_name, control_ty);
        }));

  auto return_op = cast<ReturnOp>(func_op.getBody()->getTerminator());
  if (!return_op.control_ret_attrs()) {
    return InvalidArgument(
        "Can't export function ", func_name,
        " because return op is missing \"control_ret_attrs\"");
  }
  ArrayAttr results_attr = func_op.getAllResultAttrs();
  StringAttr tfg_name_key =
      cast<TFGraphDialect>(func_op->getDialect())->getTfgNameAttrIdentifier();

  for (auto it :
       llvm::zip(llvm::enumerate(TFOp(return_op).getNonControlOperands()),
                 results_attr.getAsRange<DictionaryAttr>())) {
    unsigned res_num = std::get<0>(it).index();
    auto name = std::get<1>(it).getAs<StringAttr>(tfg_name_key);
    if (!name) {
      return InvalidArgument(
          "Can't export function ", func_name,
          " because missing \"tfg.name\" attribute for result #", res_num);
    }
    NodeDef *node_def = graph_def.add_node();
    TF_RETURN_IF_ERROR(GetReturnNode(func_op, std::get<0>(it).value(), res_num,
                                     name.getValue(), node_def, control_ty));
  }

  tensorflow::GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  // TODO(aminim): remove dependency on the global registry and allow for
  // injection.
  tensorflow::Graph graph(&flib);

  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToGraph(
      options, std::move(graph_def), &graph));

  FunctionDef func_def;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(graph, func_name, &func_def));

  OpDef *signature = func_def.mutable_signature();
  if (func_op->getAttr("is_stateful")) signature->set_is_stateful(true);
  if (auto description = func_op->getAttrOfType<StringAttr>("description"))
    signature->set_description(description.getValue().str());

  // Some ArgDef updates couldn't be carried through the graph nodes, like
  // "handle_data".
  for (int arg_num : llvm::seq<int>(0, func_op.getNumArguments())) {
    // Odd position are just for control dependencies.
    if (arg_num % 2) continue;
    DictionaryAttr arg_attrs =
        function_interface_impl::getArgAttrDict(func_op, arg_num);
    OpDef::ArgDef *arg = signature->mutable_input_arg(arg_num / 2);
    StringAttr description = arg_attrs.getAs<StringAttr>("tfg.description");
    if (description) arg->set_description(description.getValue().str());
    TF_RETURN_IF_ERROR(
        ConvertHandleData(arg_attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
    if (auto full_type = arg_attrs.getAs<tf_type::FullTypeAttr>(
            "tfg.experimental_full_type")) {
      TF_ASSIGN_OR_RETURN(*arg->mutable_experimental_full_type(),
                          ConvertAttribute(full_type));
    }
  }
  // Handle the results now.
  // An ArgDef entry needs to be constructed for all non-control returned value,
  // and a mapping from the output name to the signature is also recorded in the
  // FunctionDef.
  for (auto it :
       llvm::zip(llvm::enumerate(TFOp(return_op).getNonControlOperands()),
                 results_attr.getAsRange<DictionaryAttr>())) {
    unsigned res_num = std::get<0>(it).index();
    DictionaryAttr attrs = std::get<1>(it);
    auto name = std::get<1>(it).getAs<StringAttr>(tfg_name_key);
    if (!name) {
      return InvalidArgument(
          "Can't export function ", func_name,
          " because missing \"tfg.name\" attribute for result #", res_num);
    }
    OpDef::ArgDef *arg = signature->mutable_output_arg(res_num);
    auto ret_it = func_def.mutable_ret()->find(arg->name());
    if (ret_it == func_def.mutable_ret()->end()) {
      return tensorflow::errors::Internal(
          "Mismatch in name mapping for returned value");
    }
    func_def.mutable_ret()->insert({name.getValue().str(), ret_it->second});
    func_def.mutable_ret()->erase(ret_it);
    arg->set_name(name.getValue().str());
    StringAttr description = attrs.getAs<StringAttr>("tfg.description");
    if (description) arg->set_description(description.getValue().str());
    TF_RETURN_IF_ERROR(
        ConvertHandleData(attrs.getAs<ArrayAttr>("tfg.handle_data"), arg));
    if (auto full_type =
            attrs.getAs<tf_type::FullTypeAttr>("tfg.experimental_full_type")) {
      TF_ASSIGN_OR_RETURN(*arg->mutable_experimental_full_type(),
                          ConvertAttribute(full_type));
    }
  }

  std::string ret_name;
  for (auto it : llvm::zip(
           llvm::enumerate(TFOp(return_op).getControlOperands()),
           return_op.control_ret_attrsAttr().getAsRange<DictionaryAttr>())) {
    auto name = std::get<1>(it).getAs<StringAttr>(tfg_name_key);
    if (!name) {
      return InvalidArgument("Can't export function ", func_name,
                             " because missing \"tfg.name\" "
                             "attribute for control result #",
                             std::get<0>(it).index());
    }
    // When we return a control dependency, it is not really a returned value
    // but it is added to the `control_ret` field of the FunctionDef.
    TF_RETURN_IF_ERROR(
        GetValueName(std::get<0>(it).value(), ret_name, control_ty));
    func_def.mutable_control_ret()->insert(
        {name.getValue().str(), ret_name.substr(1)});
    signature->add_control_output(name.getValue().str());
  }

  // Handled the `resource_arg_unique_id` entries. At the moment it is
  // represented as two vectors of integers which are expected of the same
  // length.
  auto unique_ids_keys = func_op->getAttrOfType<DenseIntElementsAttr>(
      "resource_arg_unique_ids_keys");
  if (unique_ids_keys) {
    auto unique_ids_values = func_op->getAttrOfType<DenseIntElementsAttr>(
        "resource_arg_unique_ids_values");
    if (!unique_ids_values)
      return InvalidArgument(
          "Can't export function ", func_name,
          " because \"resource_arg_unique_ids_keys\" attribute is present "
          "but "
          "\"resource_arg_unique_ids_values\" is missing");
    if (unique_ids_keys.size() != unique_ids_values.size())
      return InvalidArgument(
          "Can't export function ", func_name,
          " because \"resource_arg_unique_ids_keys\" array does not have the "
          "same size as \"resource_arg_unique_ids_values\"");

    auto *unique_ids_map = func_def.mutable_resource_arg_unique_id();
    for (auto key_value : llvm::zip(unique_ids_keys.getValues<int32_t>(),
                                    unique_ids_values.getValues<int32_t>()))
      (*unique_ids_map)[std::get<0>(key_value)] = std::get<1>(key_value);
  }

  // Finally the dialect attributes (prefixed by `tf.` in general) are converted
  // as-is and stored on the `attr` field of the FunctionDef.
  SmallVector<NamedAttribute> funcAttrs(func_op->getDialectAttrs());
  TF_RETURN_IF_ERROR(ConvertAttributes(funcAttrs, {},
                                       /*remove_ref_type=*/false,
                                       func_def.mutable_attr()));
  if (flib.Find(func_def.signature().name()))
    TF_RETURN_IF_ERROR(
        flib.ReplaceFunction(func_def.signature().name(), func_def));
  else
    TF_RETURN_IF_ERROR(flib.AddFunctionDef(func_def));
  return {};
}

}  // namespace tfg
}  // namespace mlir

namespace tensorflow {

Status ConvertHandleData(mlir::ArrayAttr handle_data_arr, OpDef::ArgDef *arg) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_10(mht_10_v, 870, "", "./tensorflow/core/ir/importexport/export.cc", "ConvertHandleData");

  return mlir::tfg::ConvertHandleDataImpl(handle_data_arr, arg);
}

Status ExportMlirToGraphdef(mlir::ModuleOp module, GraphDef *output_graph) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_11(mht_11_v, 877, "", "./tensorflow/core/ir/importexport/export.cc", "ExportMlirToGraphdef");

  return mlir::tfg::ExportMlirToGraphdefImpl(module, output_graph);
}

Status ExportMlirToSavedModel(mlir::ModuleOp module,
                              const SavedModel &original_saved_model,
                              SavedModel *output_saved_model) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_12(mht_12_v, 886, "", "./tensorflow/core/ir/importexport/export.cc", "ExportMlirToSavedModel");

  if (original_saved_model.meta_graphs_size() == 0) {
    return tensorflow::errors::InvalidArgument(
        "Original saved model has no meta graphs");
  }

  tensorflow::GraphDef new_graphdef;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(ExportMlirToGraphdef(module, &new_graphdef),
                                  "while converting TFG to GraphDef");

  // Overwrite the graph def portion of the saved model with the new one.
  tensorflow::MetaGraphDef meta_graph_def = original_saved_model.meta_graphs(0);
  *(meta_graph_def.mutable_graph_def()) = std::move(new_graphdef);
  *output_saved_model = original_saved_model;
  *(output_saved_model->mutable_meta_graphs(0)) = std::move(meta_graph_def);

  return Status::OK();
}

Status ConvertOperationToNode(mlir::Operation &op, NodeDef *node,
                              GetValueNameFn get_value_name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_13(mht_13_v, 909, "", "./tensorflow/core/ir/importexport/export.cc", "ConvertOperationToNode");

  return mlir::tfg::ConvertOperationToNodeImpl(op, node, get_value_name);
}
Status ConvertOperationToNode(mlir::Operation &op, NodeDef *node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSexportDTcc mht_14(mht_14_v, 915, "", "./tensorflow/core/ir/importexport/export.cc", "ConvertOperationToNode");

  auto control_ty = mlir::tfg::ControlType::get(op.getContext());
  return mlir::tfg::ConvertOperationToNodeImpl(
      op, node, [&](mlir::Value operand, std::string &output_name) {
        return mlir::tfg::GetValueName(operand, output_name, control_ty);
      });
}

}  //  namespace tensorflow
