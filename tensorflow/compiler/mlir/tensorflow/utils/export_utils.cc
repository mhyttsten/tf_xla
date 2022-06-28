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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/location_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {
// static TensorFlow op prefix set.
std::set<std::string>* GlobalOpPrefixes() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_0(mht_0_v, 231, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "GlobalOpPrefixes");

  static std::set<std::string>* global_op_prefixes = [] {
    std::set<std::string>* result = new std::set<std::string>;
    result->insert("tf.");
    result->insert("tf_executor.");
    return result;
  }();
  return global_op_prefixes;
}

// Converts a location to the debug information for the node def.
Status ConvertLocation(mlir::Location inst_loc, llvm::StringRef node_name,
                       NodeDef::ExperimentalDebugInfo* debug_info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertLocation");

  mlir::Location unwrapped_inst_loc = GetLocationWithoutOpType(inst_loc);

  if (auto call_site = unwrapped_inst_loc.dyn_cast<mlir::CallSiteLoc>()) {
    if (auto name_loc = GetLocationWithoutOpType(call_site.getCallee())
                            .dyn_cast<mlir::NameLoc>()) {
      llvm::StringRef original_node_name, original_func_name;
      std::tie(original_node_name, original_func_name) =
          name_loc.getName().strref().split('@');
      // The location points to the current node def.
      if (node_name == original_node_name && original_func_name.empty()) {
        return Status::OK();
      }
      debug_info->add_original_node_names(original_node_name.str());
      if (!original_func_name.empty()) {
        debug_info->add_original_func_names(original_func_name.str());
      }
    }
  } else if (auto fused = unwrapped_inst_loc.dyn_cast<mlir::FusedLoc>()) {
    auto locations = fused.getLocations();
    if (locations.size() <= 1)
      return errors::InvalidArgument("expected experimental debuf info.");
    // skip the first one, which is the name of the node_def.
    for (int i = 0, end = locations.size() - 1; i < end; ++i) {
      TF_RETURN_IF_ERROR(ConvertLocation(locations[i], node_name, debug_info));
    }
  }
  return Status::OK();
}

Status ConvertAttribute(const mlir::BoolAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_2(mht_2_v, 279, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->set_b(attr.getValue());
  return Status::OK();
}

Status ConvertAttribute(const mlir::IntegerAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->set_i(attr.getInt());
  return Status::OK();
}

Status ConvertAttribute(const mlir::FloatAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_4(mht_4_v, 295, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->set_f(attr.getValueAsDouble());
  return Status::OK();
}

Status ConvertAttribute(const mlir::ElementsAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_5(mht_5_v, 303, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  return ConvertToTensorProto(attr, value->mutable_tensor());
}

Status ConvertAttribute(const mlir::TF::PlaceholderAttr& attr,
                        AttrValue* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_6(mht_6_v, 311, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->set_placeholder(attr.getValue().str());
  return Status::OK();
}

Status ConvertAttribute(const mlir::TF::ShapeAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_7(mht_7_v, 319, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  SetTensorShapeProto(attr, value->mutable_shape());
  return Status::OK();
}

Status ConvertAttribute(const mlir::FlatSymbolRefAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_8(mht_8_v, 327, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->mutable_func()->set_name(attr.getValue().str());
  return Status::OK();
}

Status ConvertAttribute(const mlir::TF::FuncAttr& attr, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_9(mht_9_v, 336, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  TF_RETURN_IF_ERROR(
      ConvertAttribute(attr.getName().cast<mlir::FlatSymbolRefAttr>(), value));
  TF_RETURN_IF_ERROR(ConvertAttributes(attr.getAttrs().getValue(),
                                       /*attrs_to_ignore=*/{}, remove_ref_type,
                                       value->mutable_func()->mutable_attr()));
  return Status::OK();
}

Status ConvertAttribute(const mlir::StringAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_10(mht_10_v, 348, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  absl::string_view attr_value(attr.getValue().data(), attr.getValue().size());
  switch (mangling_util::GetMangledKind(attr_value)) {
    case mangling_util::MangledKind::kUnknown: {
      value->set_s(std::string(attr_value));
      return Status::OK();
    }
    case mangling_util::MangledKind::kDataType: {
      DataType dtype;
      TF_RETURN_IF_ERROR(mangling_util::DemangleDataType(attr_value, &dtype));
      value->set_type(dtype);
      return Status::OK();
    }
    case mangling_util::MangledKind::kTensorShape:
      TF_RETURN_IF_ERROR(
          mangling_util::DemangleShape(attr_value, value->mutable_shape()));
      return Status::OK();
    default:
      return errors::Unimplemented("Mangled string couldn't be handled!");
  }
  return Status::OK();
}

Status ConvertAttribute(mlir::Type type, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_11(mht_11_v, 375, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  if (tensorflow::IsRefType(dtype)) dtype = tensorflow::RemoveRefType(dtype);
  value->set_type(dtype);
  return Status::OK();
}

Status ConvertAttribute(const mlir::TypeAttr& type, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_12(mht_12_v, 387, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  return ConvertAttribute(type.getValue(), remove_ref_type, value);
}

Status ConvertAttribute(const mlir::UnitAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_13(mht_13_v, 394, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  value->clear_value();
  return Status::OK();
}

Status ConvertAttribute(const mlir::ArrayAttr& attr, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_14(mht_14_v, 403, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttribute");

  auto* list = value->mutable_list();
  for (mlir::Attribute a : attr.getValue()) {
    if (auto attr = a.dyn_cast<mlir::BoolAttr>()) {
      list->add_b(attr.getValue());
    } else if (auto attr = a.dyn_cast<mlir::IntegerAttr>()) {
      list->add_i(attr.getInt());
    } else if (auto attr = a.dyn_cast<mlir::FloatAttr>()) {
      list->add_f(attr.getValueAsDouble());
    } else if (auto attr = a.dyn_cast<mlir::StringAttr>()) {
      AttrValue nested_value;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &nested_value));
      switch (nested_value.value_case()) {
        case AttrValue::kS:
          list->add_s(nested_value.s());
          break;
        case AttrValue::kType:
          list->add_type(nested_value.type());
          break;
        case AttrValue::kShape:
          *list->add_shape() = nested_value.shape();
          break;
        default:
          return errors::Unimplemented("Unhandled nested attribute!");
      }
    } else if (auto attr = a.dyn_cast<mlir::ElementsAttr>()) {
      TensorProto tensor;
      TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor));
      *list->add_tensor() = tensor;
    } else if (auto attr = a.dyn_cast<mlir::FlatSymbolRefAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = a.dyn_cast<mlir::TypeAttr>()) {
      AttrValue attr_val;
      // For type attributes, we only propagate the element type.
      mlir::Type elt_type = attr.getValue();
      if (auto shaped_type = elt_type.dyn_cast<mlir::ShapedType>()) {
        elt_type = shaped_type.getElementType();
      }
      TF_RETURN_IF_ERROR(
          ConvertAttribute(elt_type, remove_ref_type, &attr_val));
      list->add_type(attr_val.type());
    } else if (auto attr = a.dyn_cast<mlir::TF::ShapeAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_shape() = attr_val.shape();
    } else if (auto attr = a.dyn_cast<mlir::ArrayAttr>()) {
      std::vector<int64_t> vals;
      for (mlir::Attribute a : attr.getValue()) {
        auto i = a.dyn_cast<mlir::IntegerAttr>();
        if (!i)
          return errors::Unimplemented(
              "Expected 64-bit integer array attributes!");
        vals.push_back(i.getInt());
      }
      mlir::OpBuilder builder(attr.getContext());
      mlir::TensorType ty =
          mlir::RankedTensorType::get(vals.size(), builder.getIntegerType(64));
      TensorProto tensor;
      TF_RETURN_IF_ERROR(ConvertToTensorProto(
          mlir::DenseIntElementsAttr::get(ty, vals), &tensor));
      *list->add_tensor() = tensor;
    } else {
      return errors::Unimplemented("Unhandled attribute!");
    }
  }
  return Status::OK();
}

// Returns true if the executor/control dialect op should map to Ref node in
// TensorFlow Graph. For control dialect NextIteration it uses the 1st operand
// type. For executor dialect NextIteration it uses the 2nd operand type. For
// all others (Enter/Exit/Merge/Switch), if the output type is ref, they
// correspond to the Ref equivalent op in TF Graph.
static bool IsRefTypeControlOp(mlir::Operation* op) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_15(mht_15_v, 481, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "IsRefTypeControlOp");

  if (auto next_iter_sink =
          llvm::dyn_cast<mlir::tf_executor::NextIterationSinkOp>(op))
    return mlir::getElementTypeOrSelf(next_iter_sink.input().getType())
        .isa<mlir::TF::TensorFlowRefType>();

  auto op_name_or_status = GetTensorFlowOpName(op->getName().getStringRef());
  if (!op_name_or_status.ok()) return false;

  auto op_name = op_name_or_status.ConsumeValueOrDie();
  if (op_name.equals("NextIteration"))
    return mlir::getElementTypeOrSelf(op->getOperand(0).getType())
        .isa<mlir::TF::TensorFlowRefType>();

  if (op_name.equals("Enter") || op_name.equals("Exit") ||
      op_name.equals("Switch") || op_name.equals("Merge")) {
    return getElementTypeOrSelf(op->getResult(0).getType())
        .isa<mlir::TF::TensorFlowRefType>();
  }
  return false;
}

}  // anonymous namespace

StatusOr<llvm::StringRef> GetTensorFlowOpName(llvm::StringRef op_name) {
  // When being converted to MLIR, some prefixes and suffixes are added to the
  // operation types, and we have to remove them when converting the
  // operations back to a graph:
  // - "tf." or "tf_executor." : every operation type has this prefix.
  // - ".sink" or ".Sink": only the NextIteration operation has this suffix. We
  // don't need to consider ".source"/".Source" because the nodes with this
  // suffix are skipped by the caller and will not be added to the graph.
  auto prefixes = GlobalOpPrefixes();
  if (std::none_of(prefixes->begin(), prefixes->end(), [&](std::string prefix) {
        return op_name.consume_front(prefix);
      })) {
    return errors::FailedPrecondition("op node '", op_name.str(),
                                      "' was not a TF op!");
  }
  // Control dialect NextIteration sink ends with ".sink" and Executor dialect
  // NextIteration sink ends with ".Sink".
  if (!op_name.consume_back(".sink")) op_name.consume_back(".Sink");
  return op_name;
}

StatusOr<std::unique_ptr<NodeDef>> GetOperationNodeDef(
    mlir::Operation* inst, llvm::StringRef name) {
  auto node_def = absl::make_unique<NodeDef>();
  // Note: we do not use NodeBuilder or NodeDefBuilder as that would require
  // mapping back from the inputs to the input arguments.

  llvm::SmallString<64> op_name;
  if (IsLegacyCallInstruction(inst)) {
    // The op_name is the name of the function.
    op_name.append(inst->getAttrOfType<mlir::SymbolRefAttr>("f")
                       .getLeafReference()
                       .getValue());
    // Remove the attribute from the instruction as it is already converted to
    // op_name.
    auto attr_id = mlir::StringAttr::get(inst->getContext(), "f");
    inst->removeAttr(attr_id);
  } else {
    // Some control flow ops in TensorFlow Graph have their respective "Ref" ops
    // as well. For example there is Enter and RefEnter op. RefEnter forwards
    // the input ref buffer to output. However both Enter and RefEnter are
    // mapped to tf_executor::EnterOp during import. Check if it is a Ref op to
    // correctly map to the TensorFlow Graph op.
    if (IsRefTypeControlOp(inst)) op_name = "Ref";
    TF_ASSIGN_OR_RETURN(auto tf_name,
                        GetTensorFlowOpName(inst->getName().getStringRef()));
    op_name.append(tf_name);
  }

  node_def->set_name(name.str());
  node_def->set_op(std::string(op_name.str()));

  // Update NodeDef constructed out of an MLIR Case/If/While op to map it to
  // either TensorFlow StatelessX or X op depending on the additional attribute.
  if (llvm::isa<mlir::TF::CaseOp, mlir::TF::IfOp, mlir::TF::WhileOp>(inst)) {
    auto stateless = inst->getAttrOfType<mlir::BoolAttr>("is_stateless");
    if (stateless && stateless.getValue())
      *node_def->mutable_op() = "Stateless" + node_def->op();
  }

  // Add inputs to the NodeDef based on the number of operands. This is required
  // as later when edges are added to the Node using Graph::AddEdge the
  // associated NodeDef is not updated.
  for (int i = 0, e = inst->getNumOperands(); i < e; ++i) {
    node_def->add_input();
  }
  if (auto attr = inst->getAttrOfType<mlir::StringAttr>("device")) {
    node_def->set_device(std::string(attr.getValue()));
  }

  // Add the node debug info.
  TF_RETURN_IF_ERROR(ConvertLocation(
      inst->getLoc(), name, node_def->mutable_experimental_debug_info()));

  return node_def;
}

Status ConvertAttributes(
    const llvm::ArrayRef<mlir::NamedAttribute> attrs,
    const absl::flat_hash_set<absl::string_view>& attrs_to_ignore,
    bool remove_ref_type, AttrValueMap* values) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_16(mht_16_v, 588, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "ConvertAttributes");

  AttrValueMap func_call_attrs;
  for (const mlir::NamedAttribute& named_attr : attrs) {
    auto name_strref = named_attr.getName().str();
    auto attr = named_attr.getValue();
    absl::string_view name(name_strref.data(), name_strref.size());
    if (name == "name" || name == "device" || attrs_to_ignore.contains(name)) {
      // The name, device spec of a TF op or function are not stored as
      // AttrValue inside NodeDef, but we model them using attribute inside
      // MLIR. So we need to ignore them when going back to AttrValue here.
      continue;
    }
    if (mangling_util::IsMangledAttributeName(name)) {
      // In MLIR, attributes for functions requires dialect prefix. We need to
      // remove TF dialect prefix before converting to AttrValue.
      name = mangling_util::DemangleAttributeName(name);
    }
    AttrValue value;
    if (auto symbol_ref = attr.dyn_cast<mlir::SymbolRefAttr>()) {
      TF_RETURN_IF_ERROR(
          ConvertAttribute(symbol_ref.cast<mlir::FlatSymbolRefAttr>(), &value));
      func_call_attrs[string(name)] = value;
      continue;
    }
    if (auto func_attr = attr.dyn_cast<mlir::TF::FuncAttr>()) {
      TF_RETURN_IF_ERROR(ConvertAttribute(func_attr, remove_ref_type, &value));
      func_call_attrs[string(name)] = value;
      continue;
    }
    if (attr.isa<mlir::AffineMapAttr>()) {
      // AffineMapAttr is not implemented.
      return errors::Unimplemented("AffineMap attribute (needed for '",
                                   name_strref, "') unimplemented");
    }
    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, Status>(attr)
            .Case<mlir::BoolAttr, mlir::IntegerAttr, mlir::FloatAttr,
                  mlir::StringAttr, mlir::ElementsAttr, mlir::UnitAttr,
                  mlir::TF::ShapeAttr, mlir::TF::PlaceholderAttr>(
                [&](auto derived_attr) {
                  return ConvertAttribute(derived_attr, &value);
                })
            .Case<mlir::ArrayAttr, mlir::TypeAttr>([&](auto derived_attr) {
              return ConvertAttribute(derived_attr, remove_ref_type, &value);
            })
            .Default([&](mlir::Attribute) {
              return errors::Unimplemented(
                  "Unhandled attribute kind for attribute '", name_strref,
                  '\'');
            }));

    // According to the NodeDef proto definition, an attribute name from the
    // input TensorFlow GraphDef shouldn't contain '.'. If it does appear in
    // the attribute from MLIR, it is treated as an attribute from function
    // calls.
    std::vector<string> name_tokens =
        absl::StrSplit(name, '.', absl::SkipEmpty());
    TF_RET_CHECK(name_tokens.size() <= 2);
    auto it = func_call_attrs.find(name_tokens[0]);
    if (it == func_call_attrs.end()) {
      (*values)[string(name)] = value;
    } else {
      (*it->second.mutable_func()->mutable_attr())[name_tokens[1]] = value;
    }
  }
  for (const auto& it : func_call_attrs) {
    (*values)[it.first] = it.second;
  }
  return Status::OK();
}

Status SetShapeAttribute(absl::string_view name, mlir::ShapedType shaped_type,
                         AttrValueMap* values) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_17(mht_17_v, 664, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "SetShapeAttribute");

  AttrValue value;
  SetTensorShapeProto(shaped_type, value.mutable_list()->add_shape());

  auto result = values->insert({string(name), value});
  if (!result.second) {
    // This should be extremely rare as it means we are adding the same
    // attribute multiple times/have some redundancy in representing this
    // attribute.
    TensorShapeProto actual_shape = result.first->second.shape();
    // Just check via string output as we shouldn't get here and if we do they
    // should be trivially the same, else fail.
    std::string new_shape_string = value.list().shape(0).ShortDebugString();
    if (actual_shape.ShortDebugString() != new_shape_string) {
      return errors::InvalidArgument("Expected ", new_shape_string, " '", name,
                                     "' attribute but found ",
                                     actual_shape.ShortDebugString());
    }
  }
  return Status::OK();
}

bool IsLegacyCallInstruction(mlir::Operation* inst) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_18(mht_18_v, 689, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "IsLegacyCallInstruction");

  return llvm::dyn_cast<mlir::TF::LegacyCallOp>(inst);
}

Status AddTensorFlowOpPrefix(std::string prefix) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSexport_utilsDTcc mht_19(mht_19_v, 697, "", "./tensorflow/compiler/mlir/tensorflow/utils/export_utils.cc", "AddTensorFlowOpPrefix");

  GlobalOpPrefixes()->insert(prefix);
  return Status::OK();
}

}  // namespace tensorflow
