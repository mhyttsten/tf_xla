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
class MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/full_type_util.h"

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace full_type {

OpTypeConstructor Nullary(FullTypeId t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/framework/full_type_util.cc", "Nullary");

  return [t](OpDef* op_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);
    return Status::OK();
  };
}

OpTypeConstructor Unary(FullTypeId t, const string& var_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("var_name: \"" + var_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/framework/full_type_util.cc", "Unary");

  return [t, var_name](OpDef* op_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_3(mht_3_v, 226, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);

    FullTypeDef* arg = tdef->add_args();
    arg->set_type_id(TFT_VAR);
    arg->set_s(var_name);

    return Status::OK();
  };
}

OpTypeConstructor UnaryGeneric(FullTypeId t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/framework/full_type_util.cc", "UnaryGeneric");

  return [t](OpDef* op_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_5(mht_5_v, 246, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);

    FullTypeDef* arg = tdef->add_args();
    arg->set_type_id(TFT_ANY);

    return Status::OK();
  };
}

OpTypeConstructor UnaryTensorContainer(FullTypeId t, FullTypeId dtype) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/framework/full_type_util.cc", "UnaryTensorContainer");

  return [t, dtype](OpDef* op_def) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_7(mht_7_v, 265, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);

    FullTypeDef* arg = tdef->add_args();
    arg->set_type_id(TFT_TENSOR);
    FullTypeDef* targ = arg->add_args();
    targ->set_type_id(dtype);

    return Status::OK();
  };
}

OpTypeConstructor UnaryTensorContainer(FullTypeId t, const string& var_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("var_name: \"" + var_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_8(mht_8_v, 283, "", "./tensorflow/core/framework/full_type_util.cc", "UnaryTensorContainer");

  return [t, var_name](OpDef* op_def) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_9(mht_9_v, 287, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);

    FullTypeDef* targ = tdef->add_args();
    targ->set_type_id(TFT_TENSOR);
    FullTypeDef* varg = targ->add_args();
    varg->set_type_id(TFT_VAR);
    varg->set_s(var_name);

    return Status::OK();
  };
}

OpTypeConstructor VariadicTensorContainer(FullTypeId t,
                                          const string& var_name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("var_name: \"" + var_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_10(mht_10_v, 307, "", "./tensorflow/core/framework/full_type_util.cc", "VariadicTensorContainer");

  return [t, var_name](OpDef* op_def) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_11(mht_11_v, 311, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* tdef =
        op_def->mutable_output_arg(0)->mutable_experimental_full_type();
    tdef->set_type_id(t);

    FullTypeDef* for_each = tdef->add_args();
    for_each->set_type_id(TFT_FOR_EACH);
    for_each->add_args()->set_type_id(TFT_PRODUCT);

    FullTypeDef* tpl = for_each->add_args();
    tpl->set_type_id(TFT_TENSOR);
    FullTypeDef* targ = tpl->add_args();
    targ->set_type_id(TFT_VAR);
    targ->set_s(var_name);

    FullTypeDef* tvar = for_each->add_args();
    tvar->set_type_id(TFT_VAR);
    tvar->set_s(var_name);

    return Status::OK();
  };
}

namespace {

typedef absl::flat_hash_map<StringPiece, const AttrValue*> AttrMap;

inline Status SubstituteFromAttrs(AttrMap& attrs, FullTypeDef& t);

Status SubstituteVar(AttrMap& attrs, FullTypeDef& t) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_12(mht_12_v, 343, "", "./tensorflow/core/framework/full_type_util.cc", "SubstituteVar");

  DCHECK_EQ(t.args_size(), 0);

  StringPiece var_name = t.s();
  if (!attrs.contains(var_name)) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("could not find an attribute for key '", var_name, "'"));
  }
  const AttrValue* attr = attrs.at(var_name);

  const auto attr_type = attr->value_case();
  if (attr_type == AttrValue::kType) {
    map_dtype_to_tensor(attr->type(), t);
  } else if (attr_type == AttrValue::kList) {
    const auto& attr_list = attr->list();
    if (attr_list.type_size() != 1) {
      return Status(error::UNIMPLEMENTED,
                    absl::StrCat("lists or other than one type element\n",
                                 attr_list.DebugString(), "\nkey=", var_name));
    }
    map_dtype_to_tensor(attr_list.type(0), t);
  } else {
    return Status(error::UNIMPLEMENTED,
                  absl::StrCat("unsupported attribute type ",
                               attr->DebugString(), " for name ", var_name));
  }
  t.clear_s();
  return Status::OK();
}

Status SubstituteForEach(AttrMap& attrs, FullTypeDef& t) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_13(mht_13_v, 377, "", "./tensorflow/core/framework/full_type_util.cc", "SubstituteForEach");

  DCHECK_EQ(t.args_size(), 3);

  const auto& cont = t.args(0);
  const auto& tmpl = t.args(1);
  const auto& t_var = t.args(2);

  StringPiece var_name = t_var.s();
  if (!attrs.contains(var_name)) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("could not find an attribute for key '", var_name, "'"));
  }
  const AttrValue* attr = attrs.at(var_name);

  FullTypeDef result;
  result.set_type_id(cont.type_id());

  const auto attr_type = attr->value_case();
  if (attr_type == AttrValue::kType) {
    FullTypeDef* target = result.add_args();
    *target = tmpl;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        SubstituteFromAttrs(attrs, *target), "while substituting '", var_name,
        "' from\n", attr->DebugString(), "\ninto ", target->DebugString());

  } else if (attr_type == AttrValue::kList) {
    const auto& attr_list = attr->list();
    int tsize = attr_list.type_size();
    if (tsize == 0) {
      return Status(error::UNIMPLEMENTED,
                    absl::StrCat("unsupported list attribute type\n",
                                 attr_list.DebugString(), "\nkey=", var_name));
    }
    AttrValue replacement;
    attrs[var_name] = &replacement;
    for (int i = 0; i < tsize; i++) {
      replacement.set_type(attr_list.type(i));
      FullTypeDef* target = result.add_args();
      *target = tmpl;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(SubstituteFromAttrs(attrs, *target),
                                      "while substituting '", var_name,
                                      "' from\n", attr->DebugString(), "\n[", i,
                                      "] into\n", target->DebugString());
    }
    // In case of error, it's ok for the attributes map to remain in an invalid
    // state.
    attrs[var_name] = attr;

  } else {
    return Status(error::UNIMPLEMENTED,
                  absl::StrCat("unsupported attribute type\n",
                               attr->DebugString(), "\nfor name ", var_name));
  }
  t = result;
  return Status::OK();
}

Status SubstituteGeneric(AttrMap& attrs, FullTypeDef& t) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_14(mht_14_v, 438, "", "./tensorflow/core/framework/full_type_util.cc", "SubstituteGeneric");

  int nargs = t.args_size();
  for (int j = 0; j < nargs; j++) {
    FullTypeDef* arg_t = t.mutable_args(j);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(SubstituteFromAttrs(attrs, *arg_t),
                                    "while substituting arg ", j, ": ",
                                    arg_t->DebugString());

    // Special case for DT_VARIANT tensors. We leave those unset to avoid even
    // more special casing downstream.
    if (arg_t->type_id() == TFT_TENSOR && arg_t->args_size() &&
        arg_t->args(0).type_id() == TFT_LEGACY_VARIANT) {
      t.clear_args();
      break;
    }
  }
  return Status::OK();
}

inline Status SubstituteFromAttrs(AttrMap& attrs, FullTypeDef& t) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_15(mht_15_v, 460, "", "./tensorflow/core/framework/full_type_util.cc", "SubstituteFromAttrs");

  // Resolve dependent types. The convention for op registrations is to use
  // attributes as type variables.
  // See https://www.tensorflow.org/guide/create_op#type_polymorphism.
  // Once the op signature can be defined entirely in FullType, this
  // convention can be deprecated.
  //
  // Note: While this code performs some basic verifications, it generally
  // assumes consistent op defs and attributes. If more complete
  // verifications are needed, they should be done by separately, and in a
  // way that can be reused for type inference.
  switch (t.type_id()) {
    case TFT_VAR:
      return SubstituteVar(attrs, t);

    case TFT_FOR_EACH:
      return SubstituteForEach(attrs, t);

    default:
      return SubstituteGeneric(attrs, t);
  }
  return Status::OK();
}

}  // namespace

Status SpecializeType(const AttrSlice& attrs, const OpDef& op_def,
                      FullTypeDef& target) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_16(mht_16_v, 490, "", "./tensorflow/core/framework/full_type_util.cc", "SpecializeType");

  target.Clear();
  target.set_type_id(TFT_PRODUCT);

  AttrMap map;
  for (const auto& attr : attrs) {
    map.emplace(attr.first, &attr.second);
  }

  int nargs = op_def.output_arg_size();
  for (int i = 0; i < nargs; i++) {
    auto& t = *(target.add_args());
    t = op_def.output_arg(i).experimental_full_type();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        SubstituteFromAttrs(map, t), "while expanding vars of\n",
        t.DebugString(), "\nfrom\n", attrs.SummarizeNode());
  }

  return Status::OK();
}

const FullTypeDef& GetArgDefaultUnset(const FullTypeDef& t, int i) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_17(mht_17_v, 514, "", "./tensorflow/core/framework/full_type_util.cc", "GetArgDefaultUnset");

  static FullTypeDef* unset_type = []() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_18(mht_18_v, 518, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* t = new FullTypeDef();
    return t;
  }();

  if (i < t.args_size()) {
    return t.args(i);
  }
  return *unset_type;
}

const FullTypeDef& GetArgDefaultAny(const FullTypeDef& t, int i) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_19(mht_19_v, 532, "", "./tensorflow/core/framework/full_type_util.cc", "GetArgDefaultAny");

  static FullTypeDef* any_type = []() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_20(mht_20_v, 536, "", "./tensorflow/core/framework/full_type_util.cc", "lambda");

    FullTypeDef* t = new FullTypeDef();
    t->set_type_id(TFT_ANY);
    return t;
  }();

  if (i < t.args_size()) {
    const FullTypeDef& f_val = t.args(i);
    if (f_val.type_id() == TFT_UNSET) {
      return *any_type;
    }
    return f_val;
  }
  return *any_type;
}

bool IsEqual(const FullTypeDef& lhs, const FullTypeDef& rhs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_21(mht_21_v, 555, "", "./tensorflow/core/framework/full_type_util.cc", "IsEqual");

  if (lhs.type_id() != rhs.type_id()) {
    return false;
  }
  const auto& lhs_s = lhs.s();
  const auto& rhs_s = rhs.s();
  if (lhs_s.empty()) {
    if (!rhs_s.empty()) {
      return false;
    }
  } else if (rhs_s != lhs_s) {
    return false;
  }
  for (int i = 0; i < std::max(lhs.args_size(), rhs.args_size()); i++) {
    const FullTypeDef& lhs_arg = GetArgDefaultAny(lhs, i);
    const FullTypeDef& rhs_arg = GetArgDefaultAny(rhs, i);

    if (!IsEqual(lhs_arg, rhs_arg)) {
      return false;
    }
  }
  return true;
}

uint64_t Hash(const FullTypeDef& arg) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_22(mht_22_v, 582, "", "./tensorflow/core/framework/full_type_util.cc", "Hash");

  // Following style of IsEqual above and walking across FullTypeDef.
  uint64_t val = Hash64Combine(arg.type_id(), 0);

  const auto& arg_s = arg.s();
  val = Hash64Combine(val, Hash64(arg_s));
  for (int i = 0, e = arg.args_size(); i < e; ++i) {
    const FullTypeDef& arg_arg = GetArgDefaultAny(arg, i);
    val = Hash64Combine(val, Hash(arg_arg));
  }

  return val;
}

bool IsSubtype(const FullTypeDef& lhs, const FullTypeDef& rhs, bool covariant) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfull_type_utilDTcc mht_23(mht_23_v, 599, "", "./tensorflow/core/framework/full_type_util.cc", "IsSubtype");

  // Rule: ANY is a supertype of all types.
  if (rhs.type_id() == TFT_ANY) {
    return true;
  }
  // Compatibility rule: UNSET is treated as ANY for the purpose of subtyping.
  if (rhs.type_id() == TFT_UNSET) {
    return true;
  }
  // Default rule: type IDs must match.
  if (lhs.type_id() != rhs.type_id()) {
    return false;
  }

  for (int i = 0; i < std::max(lhs.args_size(), rhs.args_size()); i++) {
    const FullTypeDef& lhs_arg = GetArgDefaultAny(lhs, i);
    const FullTypeDef& rhs_arg = GetArgDefaultAny(rhs, i);

    if (covariant) {
      if (!IsSubtype(lhs_arg, rhs_arg)) {
        return false;
      }
    } else {
      if (!IsSubtype(rhs_arg, lhs_arg)) {
        return false;
      }
    }
  }

  // Invariant: type IDs are eaqual, and all args are subtype of one another.
  return true;
}

}  // namespace full_type

}  // namespace tensorflow
