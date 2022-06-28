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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

// Sets type list attribute with the given `name` to the given `types`. If the
// attribute already exists with a different value, returns an error.
template <typename ContainerT,
          typename = typename std::enable_if<
              std::is_same<mlir::Type, decltype(*std::declval<ContainerT>()
                                                     .begin())>::value>::type>
Status SetTypeAttribute(absl::string_view name, ContainerT types,
                        AttrValueMap* values) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.cc", "SetTypeAttribute");

  AttrValue value;
  auto& type_list = *value.mutable_list();
  for (auto type : types) {
    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, &dtype));
    type_list.add_type(dtype);
  }

  auto result = values->insert({string(name), value});
  assert(result.second && "cannot have multiple attributes with the same name");
  (void)result;

  return Status::OK();
}

// Sets shape list attribute with the given `name` to the given `shapes`. If the
// attribute already exists then this will just retain the set value.
template <typename ContainerT,
          typename = typename std::enable_if<std::is_same<
              llvm::Optional<llvm::ArrayRef<int64_t>>,
              decltype(*std::declval<ContainerT>().begin())>::value>::type>
void SetShapeAttribute(absl::string_view name, ContainerT shapes,
                       AttrValueMap* values) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.cc", "SetShapeAttribute");

  AttrValue value;
  auto& shape_list = *value.mutable_list();
  for (const llvm::Optional<llvm::ArrayRef<int64_t>>& shape : shapes) {
    TensorShapeProto& tshape = *shape_list.add_shape();
    if (shape.hasValue()) {
      for (int64_t dim : *shape) tshape.add_dim()->set_size(dim);
    } else {
      tshape.set_unknown_rank(true);
    }
  }

  // If shape is already set, override it. This can happen if we import
  // without shape inference enabled and so couldn't be removed on import and
  // are not explicitly dropped later.
  (*values)[string(name)] = value;
}

// Collects all the unregistered attributes for an TF dialect operation.
// Attributes "name" and "device" are not included because they are not part
// of an TF op attributes.
Status GetUnregisteredAttrs(
    mlir::Operation* inst, const tensorflow::OpRegistrationData* op_reg_data,
    absl::flat_hash_set<absl::string_view>* attrs_to_ignore) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc mht_2(mht_2_v, 271, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.cc", "GetUnregisteredAttrs");

  if (!op_reg_data) {
    // This is likely a function call node, so we should continue.
    return Status::OK();
  }

  // Collect all the registered attributes.
  llvm::DenseSet<llvm::StringRef> registered_attrs;
  registered_attrs.insert("name");
  registered_attrs.insert("device");
  for (const auto& attr_def : op_reg_data->op_def.attr()) {
    registered_attrs.insert(attr_def.name());
  }
  // Attributes are not in the registered attributes set will be ignored.
  for (auto& attr : inst->getAttrs()) {
    if (registered_attrs.find(attr.getName()) == registered_attrs.end()) {
      attrs_to_ignore->insert(
          absl::string_view(attr.getName().data(), attr.getName().size()));
    }
  }
  return Status::OK();
}

// Collects all attribute names to ignore in an MLIR operation when exporting to
// a TensorFlow NodeDef.
StatusOr<absl::flat_hash_set<absl::string_view>> GetAttributesToIgnore(
    mlir::Operation* inst, mlir::DictionaryAttr derived_attrs,
    const tensorflow::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs) {
  // The elements are owned by the MLIRContext.
  absl::flat_hash_set<absl::string_view> attrs_to_ignore;

  // We ignore attributes attached to the operation when there is already a
  // derived attribute defined in ODS.
  if (derived_attrs) {
    for (auto derived_attr : derived_attrs) {
      attrs_to_ignore.insert(
          mlir::StringRefToView(derived_attr.getName().strref()));
    }
  }

  if (ignore_unregistered_attrs) {
    TF_RETURN_IF_ERROR(
        GetUnregisteredAttrs(inst, op_reg_data, &attrs_to_ignore));
  }

  if (inst->hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
    // TODO(b/146937733): Don't use <void> here.
    llvm::StringRef attr_name = mlir::OpTrait::AttrSizedOperandSegments<
        void>::getOperandSegmentSizeAttr();
    attrs_to_ignore.insert(attr_name.data());
  }

  if (inst->hasTrait<mlir::OpTrait::AttrSizedResultSegments>()) {
    // TODO(b/146937733): Don't use <void> here.
    llvm::StringRef attr_name = mlir::OpTrait::AttrSizedResultSegments<
        void>::getResultSegmentSizeAttr();
    attrs_to_ignore.insert(attr_name.data());
  }

  if (llvm::isa<mlir::TF::CaseOp, mlir::TF::IfOp, mlir::TF::WhileOp>(inst))
    attrs_to_ignore.insert("is_stateless");

  if (llvm::isa<mlir::TF::WhileOp>(inst))
    attrs_to_ignore.insert("shape_invariant");

  return attrs_to_ignore;
}

// Populates all derived attributes of a MLIR operation in a proto
// map<string, AttrValue>.
Status PopulateDerivedAttributes(mlir::Operation* inst, llvm::StringRef name,
                                 mlir::DictionaryAttr derived_attrs,
                                 bool ignore_unregistered_attrs,
                                 AttrValueMap* attributes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc mht_3(mht_3_v, 348, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.cc", "PopulateDerivedAttributes");

  if (derived_attrs) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        ConvertAttributes(derived_attrs.getValue(), /*attrs_to_ignore=*/{},
                          /*remove_ref_type=*/true, attributes),
        "while converting derived attributes for node: ",
        mlir::StringRefToView(name));
  }

  // Here we only add the shapes for the leading values with ShapedType,
  // assuming values with non-ShapedType are put at the end of the result.
  if (!ignore_unregistered_attrs && inst->getNumResults() > 0) {
    auto values = inst->getResults();
    auto begin = values.begin();
    auto end = values.begin();
    while (end != values.end() && (*end).getType().isa<mlir::ShapedType>())
      end++;
    if (begin != end) {
      mlir::TF::ResultShapeRange output_shapes = {
          mlir::TF::ResultShapeIterator(begin),
          mlir::TF::ResultShapeIterator(end)};
      SetShapeAttribute("_output_shapes", output_shapes, attributes);
    }
  }

  return Status::OK();
}

}  // namespace

Status GetAttrValuesFromOperation(
    mlir::Operation* inst, llvm::StringRef name,
    const tensorflow::OpRegistrationData* op_reg_data,
    bool ignore_unregistered_attrs, AttrValueMap* attributes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSexport_tf_dialect_opDTcc mht_4(mht_4_v, 384, "", "./tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.cc", "GetAttrValuesFromOperation");

  mlir::DictionaryAttr derived_attrs = nullptr;
  if (auto interface = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(inst))
    derived_attrs = interface.materializeDerivedAttributes();
  TF_ASSIGN_OR_RETURN(auto attrs_to_ignore,
                      GetAttributesToIgnore(inst, derived_attrs, op_reg_data,
                                            ignore_unregistered_attrs));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertAttributes(inst->getAttrs(), attrs_to_ignore,
                        /*remove_ref_type=*/false, attributes),
      "while converting attributes for node: ", mlir::StringRefToView(name));
  TF_RETURN_IF_ERROR(PopulateDerivedAttributes(
      inst, name, derived_attrs, ignore_unregistered_attrs, attributes));

  //  Explicitly handle XlaHostCompute op which has required function attribute
  //  in TensorFlow op def but it could have an empty value to represent missing
  //  functions. This value can't be represented using MLIR SymbolRefAttr and
  //  instead uses optional symbol ref attribute.
  //
  // TODO(b/182315488): Remove custom handling by finding a better
  // representation in MLIR for empty function names. One option could be to use
  // TensorFlow op defs to figure out function attributes that are missing in
  // MLIR. This will also require some trait to identify optional attributes in
  // MLIR.
  constexpr char kShapeInferenceGraph[] = "shape_inference_graph";
  if (mlir::isa<mlir::TF::XlaHostComputeOp>(inst) &&
      !inst->hasAttr(kShapeInferenceGraph) &&
      !attrs_to_ignore.contains(kShapeInferenceGraph)) {
    AttrValue value;
    value.mutable_func()->set_name("");
    (*attributes)[kShapeInferenceGraph] = value;
  }
  return Status::OK();
}

StatusOr<std::unique_ptr<NodeDef>> ConvertTFDialectOpToNodeDef(
    mlir::Operation* inst, llvm::StringRef name,
    bool ignore_unregistered_attrs) {
  TF_ASSIGN_OR_RETURN(auto node_def, GetOperationNodeDef(inst, name));
  TF_ASSIGN_OR_RETURN(auto op_name,
                      GetTensorFlowOpName(inst->getName().getStringRef()));
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(op_name.str());
  TF_RETURN_IF_ERROR(GetAttrValuesFromOperation(inst, name, op_reg_data,
                                                ignore_unregistered_attrs,
                                                node_def->mutable_attr()));
  return node_def;
}

}  // namespace tensorflow
