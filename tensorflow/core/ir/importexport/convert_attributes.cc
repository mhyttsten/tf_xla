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
class MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc() {
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

#include "tensorflow/core/ir/importexport/convert_attributes.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/mangling.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"

using tensorflow::AttrValue;
using tensorflow::AttrValueMap;
using tensorflow::DataType;
using tensorflow::NodeDef;
using tensorflow::Status;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::Unimplemented;

namespace mlir {
namespace tfg {

namespace {
// Converts a location to the debug information for the node def.
Status ConvertLocation(Location inst_loc,
                       NodeDef::ExperimentalDebugInfo* debug_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertLocation");

  if (auto call_site = inst_loc.dyn_cast<CallSiteLoc>()) {
    if (auto name_loc = call_site.getCallee().dyn_cast<NameLoc>()) {
      debug_info->add_original_node_names(name_loc.getName().data());
    }
  } else if (auto fused = inst_loc.dyn_cast<FusedLoc>()) {
    auto locations = fused.getLocations();
    if (locations.size() <= 1)
      return InvalidArgument("Expected experimental debug info.");
    // skip the first one, which is the name of the node_def.
    for (int i = 0, end = locations.size() - 1; i < end; ++i) {
      TF_RETURN_IF_ERROR(ConvertLocation(locations[i], debug_info));
    }
  }
  return Status::OK();
}

Status ConvertAttribute(const BoolAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->set_b(attr.getValue());
  return Status::OK();
}

Status ConvertAttribute(const IntegerAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_2(mht_2_v, 252, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->set_i(attr.getInt());
  return Status::OK();
}

Status ConvertAttribute(const FloatAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->set_f(attr.getValueAsDouble());
  return Status::OK();
}

Status ConvertAttribute(const ElementsAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  return ConvertToTensorProto(attr, value->mutable_tensor());
}

Status ConvertAttribute(const PlaceholderAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->set_placeholder(attr.getValue().str());
  return Status::OK();
}

Status ConvertAttribute(const ShapeAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  SetTensorShapeProto(attr, value->mutable_shape());
  return Status::OK();
}

Status ConvertAttribute(const FlatSymbolRefAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_7(mht_7_v, 291, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->mutable_func()->set_name(attr.getValue().str());
  return Status::OK();
}

Status ConvertAttribute(const FuncAttr& attr, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  TF_RETURN_IF_ERROR(
      ConvertAttribute(attr.getName().cast<FlatSymbolRefAttr>(), value));
  TF_RETURN_IF_ERROR(ConvertAttributes(attr.getAttrs().getValue(),
                                       /*attrs_to_ignore=*/{}, remove_ref_type,
                                       value->mutable_func()->mutable_attr()));
  return Status::OK();
}

Status ConvertAttribute(const StringAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_9(mht_9_v, 312, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

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
      return Unimplemented("Mangled string couldn't be handled!");
  }
  return Status::OK();
}

Status ConvertAttribute(Type type, bool remove_ref_type, AttrValue* value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_10(mht_10_v, 338, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  if (tensorflow::IsRefType(dtype)) dtype = tensorflow::RemoveRefType(dtype);
  value->set_type(dtype);
  return Status::OK();
}

Status ConvertAttribute(const TypeAttr& type, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_11(mht_11_v, 350, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  return ConvertAttribute(type.getValue(), remove_ref_type, value);
}

Status ConvertAttribute(const UnitAttr& attr, AttrValue* value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_12(mht_12_v, 357, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  value->clear_value();
  return Status::OK();
}

Status ConvertAttribute(const ArrayAttr& attr, bool remove_ref_type,
                        AttrValue* value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_13(mht_13_v, 366, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttribute");

  auto* list = value->mutable_list();
  for (Attribute a : attr.getValue()) {
    if (auto attr = a.dyn_cast<BoolAttr>()) {
      list->add_b(attr.getValue());
    } else if (auto attr = a.dyn_cast<IntegerAttr>()) {
      list->add_i(attr.getInt());
    } else if (auto attr = a.dyn_cast<FloatAttr>()) {
      list->add_f(attr.getValueAsDouble());
    } else if (auto attr = a.dyn_cast<StringAttr>()) {
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
          return Unimplemented("Unhandled nested attribute!");
      }
    } else if (auto attr = a.dyn_cast<ElementsAttr>()) {
      TensorProto tensor;
      TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor));
      *list->add_tensor() = tensor;
    } else if (auto attr = a.dyn_cast<FlatSymbolRefAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = a.dyn_cast<FuncAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, remove_ref_type, &attr_val));
      *list->add_func() = attr_val.func();
    } else if (auto attr = a.dyn_cast<TypeAttr>()) {
      AttrValue attr_val;
      // For type attributes, we only propagate the element type.
      Type elt_type = attr.getValue();
      if (auto shaped_type = elt_type.dyn_cast<ShapedType>()) {
        elt_type = shaped_type.getElementType();
      }
      TF_RETURN_IF_ERROR(
          ConvertAttribute(elt_type, remove_ref_type, &attr_val));
      list->add_type(attr_val.type());
    } else if (auto attr = a.dyn_cast<ShapeAttr>()) {
      AttrValue attr_val;
      TF_RETURN_IF_ERROR(ConvertAttribute(attr, &attr_val));
      *list->add_shape() = attr_val.shape();
    } else {
      return Unimplemented("Unhandled MLIR attribute in export to graph:",
                           debugString(a));
    }
  }
  return Status::OK();
}
}  // namespace

tensorflow::StatusOr<AttrValue> ConvertAttribute(Attribute attr) {
  AttrValue value;
  if (auto symbol_ref = attr.dyn_cast<SymbolRefAttr>()) {
    TF_RETURN_IF_ERROR(
        ConvertAttribute(symbol_ref.cast<FlatSymbolRefAttr>(), &value));
    return value;
  }
  if (auto func_attr = attr.dyn_cast<FuncAttr>()) {
    TF_RETURN_IF_ERROR(
        ConvertAttribute(func_attr, /*remove_ref_type=*/false, &value));
    return value;
  }
  if (attr.isa<AffineMapAttr>())
    return Unimplemented("AffineMap attribute unimplemented");
  TF_RETURN_IF_ERROR(
      llvm::TypeSwitch<Attribute, Status>(attr)
          .Case<BoolAttr, IntegerAttr, FloatAttr, StringAttr, ElementsAttr,
                UnitAttr, ShapeAttr, PlaceholderAttr>([&](auto derived_attr) {
            return ConvertAttribute(derived_attr, &value);
          })
          .Case<ArrayAttr, TypeAttr>([&](auto derived_attr) {
            return ConvertAttribute(derived_attr,
                                    /*remove_ref_type=*/false, &value);
          })
          .Default([&](Attribute attr) {
            return Unimplemented("Unhandled attribute kind for attribute");
          }));
  return value;
}

Status ConvertAttributes(ArrayRef<NamedAttribute> attrs,
                         ArrayRef<StringRef> attrs_to_ignore,
                         bool remove_ref_type, AttrValueMap* values) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_14(mht_14_v, 461, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "ConvertAttributes");

  StringSet<> ignored_attrs;
  ignored_attrs.insert(attrs_to_ignore.begin(), attrs_to_ignore.end());
  AttrValueMap func_call_attrs;
  for (const NamedAttribute& named_attr : attrs) {
    std::string name_str =
        PrepareTFGAttributeForExport(named_attr.getName()).str();
    auto attr = named_attr.getValue();
    absl::string_view name = name_str;
    if (ignored_attrs.contains(name_str)) {
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
    TF_ASSIGN_OR_RETURN(AttrValue value, ConvertAttribute(attr));
    if (attr.isa<SymbolRefAttr>()) {
      func_call_attrs[std::string(name)] = value;
      continue;
    }
    if (attr.isa<FuncAttr>()) {
      func_call_attrs[std::string(name)] = value;
      continue;
    }
    // According to the NodeDef proto definition, an attribute name from the
    // input TensorFlow GraphDef shouldn't contain '.'. If it does appear in
    // the attribute from MLIR, it is treated as an attribute from function
    // calls.
    std::vector<std::string> name_tokens =
        absl::StrSplit(name, '.', absl::SkipEmpty());
    TF_RET_CHECK(name_tokens.size() <= 2);
    auto it = func_call_attrs.find(name_tokens[0]);
    if (it == func_call_attrs.end())
      (*values)[std::string(name)] = value;
    else
      (*it->second.mutable_func()->mutable_attr())[name_tokens[1]] = value;
  }
  for (const auto& it : func_call_attrs) {
    (*values)[it.first] = it.second;
  }
  return Status::OK();
}

Status SetShapeAttribute(absl::string_view name, ShapedType shaped_type,
                         AttrValueMap* values) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_15(mht_15_v, 514, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "SetShapeAttribute");

  AttrValue value;
  SetTensorShapeProto(shaped_type, value.mutable_shape());

  auto result = values->insert({std::string(name), value});
  if (!result.second) {
    // This should be extremely rare as it means we are adding the same
    // attribute multiple times/have some redundancy in representing this
    // attribute.
    TensorShapeProto actual_shape = result.first->second.shape();
    // Just check via string output as we shouldn't get here and if we do they
    // should be trivially the same, else fail.
    std::string new_shape_string = value.shape().ShortDebugString();
    if (actual_shape.ShortDebugString() != new_shape_string) {
      return InvalidArgument("Expected ", new_shape_string, " '", name,
                             "' attribute but found ",
                             actual_shape.ShortDebugString());
    }
  }
  return Status::OK();
}

// Converts non func AttrValue proto into an MLIR attribute. Func attribute is
// exclused in this function because the function might be renamed when the
// function definition is imported.
tensorflow::StatusOr<Attribute> ConvertNonFuncAttributeValue(
    const AttrValue& value, Builder& builder, TFGraphDialect* tfgDialect) {
  assert(tfgDialect == builder.getContext()->getLoadedDialect("tfg"));
  switch (value.value_case()) {
    case AttrValue::kI:
      return builder.getI64IntegerAttr(value.i());
    case AttrValue::kS:
      return builder.getStringAttr(value.s());
    case AttrValue::kF:
      return builder.getFloatAttr(builder.getF32Type(), value.f());
    case AttrValue::kB:
      return builder.getBoolAttr(value.b());
    case AttrValue::kType: {
      Type type;
      TF_RETURN_IF_ERROR(ConvertDataType(value.type(), builder, &type));
      return TypeAttr::get(type);
    }
    case AttrValue::kShape:
      return ConvertTensorShapeProto(value.shape(), builder.getContext());
    case AttrValue::kTensor:
      return ConvertTensorProto(value.tensor(), builder, tfgDialect);
    case AttrValue::kList: {
      absl::InlinedVector<Attribute, 8> attrs;
      for (const auto& item : value.list().i())
        attrs.push_back(builder.getI64IntegerAttr(item));
      for (const auto& item : value.list().s())
        attrs.push_back(builder.getStringAttr(item));
      for (const auto& item : value.list().f())
        attrs.push_back(builder.getFloatAttr(builder.getF32Type(), item));
      for (const auto& item : value.list().b())
        attrs.push_back(builder.getBoolAttr(item));
      for (const auto& item : value.list().type()) {
        Type type;
        TF_RETURN_IF_ERROR(ConvertDataType(DataType(item), builder, &type));
        attrs.push_back(TypeAttr::get(type));
      }
      for (const auto& item : value.list().shape()) {
        TF_ASSIGN_OR_RETURN(
            auto attr, ConvertTensorShapeProto(item, builder.getContext()));
        attrs.push_back(attr);
      }
      for (const auto& item : value.list().tensor()) {
        TF_ASSIGN_OR_RETURN(auto attr,
                            ConvertTensorProto(item, builder, tfgDialect));
        attrs.push_back(attr);
      }
      for (const auto& func_attr : value.list().func()) {
        NamedAttrList subattrs;
        for (const auto& subattr : func_attr.attr()) {
          TF_ASSIGN_OR_RETURN(
              auto attr,
              ConvertAttributeValue(subattr.second, builder, tfgDialect));
          if (subattr.first.empty())
            return InvalidArgument("empty func_attr name");
          subattrs.push_back(builder.getNamedAttr(subattr.first, attr));
        }
        attrs.push_back(FuncAttr::get(builder.getContext(), func_attr.name(),
                                      builder.getDictionaryAttr(subattrs)));
      }
      return builder.getArrayAttr(
          llvm::makeArrayRef(attrs.begin(), attrs.end()));
    }
    case AttrValue::VALUE_NOT_SET:
      return builder.getUnitAttr();
    case AttrValue::kPlaceholder:
      return PlaceholderAttr::get(builder.getContext(), value.placeholder());
    default:
      return tensorflow::errors::Unimplemented(
          absl::StrCat("Attribute ", value.DebugString()));
  }
}

tensorflow::StatusOr<Attribute> ConvertAttributeValue(
    const AttrValue& value, Builder& builder, TFGraphDialect* tfgDialect) {
  switch (value.value_case()) {
    case AttrValue::kFunc: {
      NamedAttrList attrs;
      for (const auto& func_attr : value.func().attr()) {
        if (func_attr.first.empty()) return InvalidArgument("empty attr name");
        TF_ASSIGN_OR_RETURN(
            auto attr,
            ConvertAttributeValue(func_attr.second, builder, tfgDialect));
        attrs.push_back(builder.getNamedAttr(func_attr.first, attr));
      }
      auto func_attrs = builder.getDictionaryAttr(attrs);
      return FuncAttr::get(builder.getContext(), value.func().name(),
                           func_attrs);
    }
    default:
      return ConvertNonFuncAttributeValue(value, builder, tfgDialect);
  }
}

static constexpr StringLiteral kTpuReplicate = "_tpu_replicate";

StringRef PromoteToTFGAttribute(StringRef tf_attr_name) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_16(mht_16_v, 637, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "PromoteToTFGAttribute");

  return StringSwitch<StringRef>(tf_attr_name)
      // `_tpu_replicate` -> `tfg.tpu_replicate`
      //   This attribute assigns ops to TPU clusters. When transformations
      //   create new ops, they must ensure that these new ops are assigned to
      //   the same cluster.
      .Case(kTpuReplicate, TFGraphDialect::getTfgTpuReplicateAttrKey())
      .Default(tf_attr_name);
}

StringRef PrepareTFGAttributeForExport(StringRef tfg_attr_name) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSconvert_attributesDTcc mht_17(mht_17_v, 650, "", "./tensorflow/core/ir/importexport/convert_attributes.cc", "PrepareTFGAttributeForExport");

  return StringSwitch<StringRef>(tfg_attr_name)
      .Case(TFGraphDialect::getTfgTpuReplicateAttrKey(), kTpuReplicate)
      .Default(tfg_attr_name);
}

tensorflow::StatusOr<::mlir::tf_type::FullTypeAttr> ConvertAttribute(
    const tensorflow::FullTypeDef& full_type, Builder& builder,
    TFGraphDialect* tfgDialect) {
  using FullTypeAttr = ::mlir::tf_type::FullTypeAttr;

  SmallVector<FullTypeAttr> args;
  for (const tensorflow::FullTypeDef& it : full_type.args()) {
    TF_ASSIGN_OR_RETURN(FullTypeAttr arg,
                        ConvertAttribute(it, builder, tfgDialect));
    args.push_back(arg);
  }

  Attribute attr;
  switch (full_type.attr_case()) {
    case tensorflow::FullTypeDef::AttrCase::kS:
      attr = builder.getStringAttr(full_type.s());
      break;
    case tensorflow::FullTypeDef::AttrCase::kI:
      attr = builder.getI64IntegerAttr(full_type.i());
      break;
    case tensorflow::FullTypeDef::ATTR_NOT_SET:
      break;
    default:
      return InvalidArgument("Unsupported attr kind in FullType");
  }

  return FullTypeAttr::get(builder.getContext(), full_type.type_id(), args,
                           attr);
}

tensorflow::StatusOr<tensorflow::FullTypeDef> ConvertAttribute(
    tf_type::FullTypeAttr full_type) {
  using FullTypeDef = tensorflow::FullTypeDef;

  FullTypeDef ret;
  for (tf_type::FullTypeAttr it : full_type.getArgs()) {
    TF_ASSIGN_OR_RETURN(*ret.add_args(), ConvertAttribute(it));
  }

  if (full_type.getAttr()) {
    bool converted = llvm::TypeSwitch<Attribute, bool>(full_type.getAttr())
                         .Case<StringAttr>([&](StringAttr sattr) {
                           ret.set_s(sattr.str());
                           return true;
                         })
                         .Case<IntegerAttr>([&](IntegerAttr iattr) {
                           ret.set_i(iattr.getInt());
                           return true;
                         })
                         .Default([&](Attribute attr) { return false; });
    if (!converted)
      return InvalidArgument("Unsupported attr kind in FullType:",
                             mlir::debugString(full_type.getAttr()));
  }

  ret.set_type_id(static_cast<tensorflow::FullTypeId>(full_type.getType_id()));

  return ret;
}

}  // namespace tfg
}  // namespace mlir
