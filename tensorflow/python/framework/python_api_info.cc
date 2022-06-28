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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc() {
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
#include "tensorflow/python/framework/python_api_info.h"

#include <Python.h>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_STRING_CHECK(x) (PyString_Check(x) || PyUnicode_Check(x))
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_STRING_FROMSTRING(x) (PyString_FromString(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#define PY_STRING_AS_CSTR(x) (PyString_AsString(x))
#else
// Python 3.x:
#define PY_STRING_CHECK(x) (PyBytes_Check(x) || PyUnicode_Check(x))
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_FROMSTRING(x) (PyUnicode_FromString(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#define PY_STRING_AS_CSTR(x) (PyUnicode_AsUTF8AndSize((x), nullptr))
#endif

namespace {

// Converts the given object to an interned Python string, and returns its
// data pointer.  (This means we don't need to worry about ownership for
// this string.)
const char* InternPyString(const std::string& s) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_0(mht_0_v, 221, "", "./tensorflow/python/framework/python_api_info.cc", "InternPyString");

  Safe_PyObjectPtr interned(PY_STRING_INTERN_FROM_STRING(s.c_str()));
  return PY_STRING_AS_CSTR(interned.get());
}

template <typename T, typename UnaryPredicate>
void RemoveIf(UnaryPredicate p, std::vector<T>* vec) {
  vec->erase(std::remove_if(vec->begin(), vec->end(), p), vec->end());
}

struct DataTypeFormatter {
  void operator()(std::string* out, DataType dtype) const {
    out->append(DataType_Name(dtype));
  }
};

// Populates `param_names` and `defaults_tuple` based on the given OpDef.
void GetOpDefNamesAndDefaults(const tensorflow::OpDef& op_def,
                              std::vector<string>& param_names,
                              Safe_PyObjectPtr& defaults_tuple) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_1(mht_1_v, 243, "", "./tensorflow/python/framework/python_api_info.cc", "GetOpDefNamesAndDefaults");

  param_names.reserve(op_def.input_arg_size() + op_def.attr_size());
  std::set<std::string> inferred_attrs;

  // Input parameters come first, in the order they occur in the OpDef.
  for (const auto& input : op_def.input_arg()) {
    param_names.push_back(input.name());
    if (!input.type_attr().empty()) {
      inferred_attrs.insert(input.type_attr());
    }
    if (!input.type_list_attr().empty()) {
      inferred_attrs.insert(input.type_list_attr());
    }
    if (!input.number_attr().empty()) {
      inferred_attrs.insert(input.number_attr());
    }
  }

  // Next come attribute params without defaults, followed by attributes with
  // defaults (but inferred attributes are not included).
  std::vector<std::string> param_names_with_default;
  std::vector<Safe_PyObjectPtr> defaults;
  for (const auto& attr : op_def.attr()) {
    if (inferred_attrs.count(attr.name()) == 0) {
      if (attr.has_default_value()) {
        param_names_with_default.push_back(attr.name());
        defaults.push_back(AttrValueToPyObject(attr.default_value()));
      } else {
        param_names.push_back(attr.name());
      }
    }
  }
  param_names.insert(param_names.end(), param_names_with_default.begin(),
                     param_names_with_default.end());

  // Finally, the 'name' parameter comes at the end, and its default value
  // is the operation's name.
  param_names.push_back("name");
  defaults.emplace_back(PY_STRING_FROMSTRING(op_def.name().c_str()));

  defaults_tuple.reset(PyTuple_New(defaults.size()));
  for (int i = 0; i < defaults.size(); ++i) {
    PyTuple_SET_ITEM(defaults_tuple.get(), i, defaults[i].release());
  }
}

}  // namespace

PythonAPIInfo::PythonAPIInfo(const std::string& api_name)
    : api_name_(InternPyString(api_name)) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("api_name: \"" + api_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_2(mht_2_v, 296, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::PythonAPIInfo");
}

Status PythonAPIInfo::Initialize(const OpDef& op_def,
                                 const std::vector<string> param_names,
                                 PyObject* defaults_tuple) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_3(mht_3_v, 303, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::Initialize");

  // Intern the parameter names.
  param_names_.reserve(param_names.size());
  for (const auto& param_name : param_names) {
    param_names_.push_back(InternPyString(param_name));
  }

  Py_INCREF(defaults_tuple);
  defaults_tuple_.reset(defaults_tuple);

  // Build an index to look up parameter index by name.  (Does not include
  // inferred attributes.)
  std::map<std::string, int> param_name_to_index;
  for (int i = 0; i < param_names_.size(); ++i) {
    param_name_to_index[param_names_[i]] = i;
  }

  // Initialize each attribute & input parameter.
  attributes_.reserve(op_def.attr_size());
  for (const auto& attr_def : op_def.attr()) {
    TF_RETURN_IF_ERROR(InitializeAttribute(attr_def, param_name_to_index));
  }

  inputs_.reserve(op_def.input_arg_size());
  for (const auto& arg_def : op_def.input_arg()) {
    TF_RETURN_IF_ERROR(InitializeInput(arg_def, param_name_to_index));
  }

  TF_RETURN_IF_ERROR(CheckParamNames());

  // Filter out any unused entries from inputs_with_*_attrs_.
  RemoveIf(
      [](const InputsWithTypeAttr& input) {
        return input.tensor_params.empty() && input.tensor_list_params.empty();
      },
      &inputs_with_type_attrs_);
  RemoveIf(
      [](const InputsWithTypeListAttr& input) {
        return input.tensor_list_params.empty();
      },
      &inputs_with_type_list_attrs_);
  RemoveIf(
      [](const InputsWithNumberAttr& input) {
        return input.tensor_list_params.empty();
      },
      &inputs_with_number_attrs_);

  return Status::OK();
}

Status PythonAPIInfo::CheckParamNames() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_4(mht_4_v, 356, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::CheckParamNames");

  std::vector<bool> param_found(param_names_.size());
  for (const auto& attr : attributes_) {
    if (attr.index != -1) {
      param_found[attr.index] = true;
    }
  }
  for (const auto& input : inputs_) {
    param_found[input.index] = true;
  }

  for (int i = 0; i < param_names_.size(); ++i) {
    if (param_names_[i] == std::string("name")) {
      continue;
    }
    if (!param_found[i]) {
      return errors::InvalidArgument(
          api_name_, ": missing specification for parameter ", param_names_[i]);
    }
  }
  return Status::OK();
}

Status PythonAPIInfo::InitializeFromRegisteredOp(const std::string& op_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_5(mht_5_v, 383, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::InitializeFromRegisteredOp");

  const tensorflow::OpDef* op_def = nullptr;
  TF_RETURN_IF_ERROR(
      tensorflow::OpRegistry::Global()->LookUpOpDef(op_name, &op_def));
  std::vector<std::string> param_names;
  Safe_PyObjectPtr defaults_tuple;
  GetOpDefNamesAndDefaults(*op_def, param_names, defaults_tuple);
  TF_RETURN_IF_ERROR(Initialize(*op_def, param_names, defaults_tuple.get()));
  return Status::OK();
}

Status PythonAPIInfo::InitializeFromParamSpecs(
    const std::map<std::string, std::string>& input_specs,
    const std::map<std::string, std::string>& attr_specs,
    const std::vector<string> param_names, PyObject* defaults_tuple) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_6(mht_6_v, 400, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::InitializeFromParamSpecs");

  OpDefBuilder op_def_builder(api_name_);
  op_def_builder.AllowAttrTypeAny();
  for (const auto& attr_spec : attr_specs) {
    op_def_builder.Attr(absl::StrCat(attr_spec.first, ": ", attr_spec.second));
  }
  for (const auto& input_spec : input_specs) {
    op_def_builder.Input(
        absl::StrCat(input_spec.first, ": ", input_spec.second));
  }
  OpRegistrationData op_reg_data;
  TF_RETURN_IF_ERROR(op_def_builder.Finalize(&op_reg_data));

  TF_RETURN_IF_ERROR(
      Initialize(op_reg_data.op_def, param_names, defaults_tuple));

  return Status::OK();
}

Status PythonAPIInfo::InitializeAttribute(
    const OpDef::AttrDef& attr_def,
    const std::map<std::string, int>& param_name_to_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_7(mht_7_v, 424, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::InitializeAttribute");

  if (attr_def.name() == "name") {
    return errors::InvalidArgument(
        api_name_, ": Reserved parameter `name` was used as an attribute.");
  }
  const char* name = InternPyString(attr_def.name());

  const int param_index =
      gtl::FindWithDefault(param_name_to_index, attr_def.name(), -1);
  const AttributeType dtype = AttributeTypeFromName(attr_def.type());
  const int inferred_index = -1;
  attributes_.push_back({param_index, dtype, name, inferred_index});
  Attribute& attr = attributes_.back();
  if (attr.type == AttributeType::UNKNOWN) {
    return errors::InvalidArgument(api_name_, ": Bad attribute type for ",
                                   attr_def.name(), ": '", attr_def.type(),
                                   "'");
  }
  std::vector<DataType>* ok_dtypes = nullptr;

  if (attr.type == AttributeType::DTYPE) {
    DataType default_dtype = attr_def.has_default_value()
                                 ? attr_def.default_value().type()
                                 : DT_INVALID;
    inputs_with_type_attrs_.push_back({&attr, default_dtype});
    ok_dtypes = &inputs_with_type_attrs_.back().ok_dtypes;

  } else if (attr.type == AttributeType::LIST_DTYPE) {
    inputs_with_type_list_attrs_.push_back({&attr});
    for (int d : attr_def.default_value().list().type()) {
      inputs_with_type_list_attrs_.back().default_dtypes.push_back(
          static_cast<DataType>(d));
    }
    ok_dtypes = &inputs_with_type_list_attrs_.back().ok_dtypes;
  }

  if (attr_def.has_allowed_values() && ok_dtypes) {
    const auto& dtypes = attr_def.allowed_values().list();
    for (int i = 0; i < dtypes.type_size(); ++i) {
      ok_dtypes->push_back(dtypes.type(i));
    }
  }

  if (attr.type == AttributeType::INT) {
    int64_t default_len =
        attr_def.has_default_value() ? attr_def.default_value().i() : -1;
    inputs_with_number_attrs_.push_back({&attr, default_len});
  }

  // If this is an inferred attribute, then record its name and index.
  if (attr.index == -1) {
    std::vector<const char*>* inferred_attr_names =
        attr.type == AttributeType::DTYPE        ? &inferred_type_attrs_
        : attr.type == AttributeType::LIST_DTYPE ? &inferred_type_list_attrs_
        : attr.type == AttributeType::INT        ? &inferred_length_attrs_
                                                 : nullptr;
    if (inferred_attr_names == nullptr) {
      return errors::InvalidArgument(
          api_name_, ": Missing specification for parameter ", attr_def.name());
    } else {
      attr.inferred_index = inferred_attr_names->size();
      inferred_attr_names->push_back(attr.name);
    }
  }

  return Status::OK();
}

Status PythonAPIInfo::InitializeInput(
    const OpDef::ArgDef& arg_def,
    const std::map<std::string, ParamIndex>& param_name_to_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_8(mht_8_v, 497, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::InitializeInput");

  if (arg_def.name() == "name") {
    return errors::InvalidArgument(
        api_name_, ": Reserved parameter `name` was used as a tensor input.");
  }
  const ParamIndex param_index =
      gtl::FindWithDefault(param_name_to_index, arg_def.name(), -1);
  if (param_index == -1) {
    return errors::InvalidArgument(
        api_name_, ": Missing specification for parameter ", arg_def.name());
  }
  if (arg_def.is_ref()) {
    // TODO(b/164980194): Support reference parameters.
    //   - Pass as_ref to convert_to_tensor
    //   - Check that values for ref inputs have ref types.
    return errors::InvalidArgument(api_name_,
                                   ": PythonAPIInfo doesn't support reference "
                                   "parameters yet.");
  }
  bool is_list =
      !arg_def.number_attr().empty() || !arg_def.type_list_attr().empty();
  inputs_.push_back({param_index, is_list});

  if (!arg_def.type_list_attr().empty()) {
    // list(input) with dtypes specified by a `list(type)` attribute.
    InputsWithTypeListAttr* input =
        FindInputsWithTypeListAttr(arg_def.type_list_attr());
    if (!input) {
      return errors::InvalidArgument(
          api_name_, ": Type attribute ", arg_def.type_list_attr(),
          " for parameter ", arg_def.name(), " not found.");
    }
    input->tensor_list_params.push_back(param_index);
  } else if (!arg_def.type_attr().empty()) {
    InputsWithTypeAttr* input = FindInputsWithTypeAttr(arg_def.type_attr());
    // input or list(input) with dtype specified by a `type` attribute.
    if (!input) {
      return errors::InvalidArgument(api_name_, ": Type attribute ",
                                     arg_def.type_attr(), " for parameter ",
                                     arg_def.name(), " not found.");
    }
    if (arg_def.number_attr().empty()) {
      input->tensor_params.push_back(param_index);
    } else {
      input->tensor_list_params.push_back(param_index);
    }
  } else {
    // input or list(input) with fixed dtype
    inputs_with_fixed_dtype_.push_back({arg_def.type(), param_index, is_list});
  }

  if (!arg_def.number_attr().empty()) {
    InputsWithNumberAttr* input =
        FindInputsWithNumberAttr(arg_def.number_attr());
    if (!input) {
      return errors::InvalidArgument(api_name_, ": Length attribute ",
                                     arg_def.number_attr(), " for parameter ",
                                     arg_def.name(), " not found.");
    }
    input->tensor_list_params.push_back(param_index);
  }

  return Status::OK();
}

PythonAPIInfo::InputsWithTypeAttr* PythonAPIInfo::FindInputsWithTypeAttr(
    const string& name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_9(mht_9_v, 567, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::FindInputsWithTypeAttr");

  for (auto& input : inputs_with_type_attrs_) {
    if (name == input.type_attr->name) {
      return &input;
    }
  }
  return nullptr;
}

PythonAPIInfo::InputsWithTypeListAttr*
PythonAPIInfo::FindInputsWithTypeListAttr(const string& name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_10(mht_10_v, 581, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::FindInputsWithTypeListAttr");

  for (auto& input : inputs_with_type_list_attrs_) {
    if (name == input.type_list_attr->name) {
      return &input;
    }
  }
  return nullptr;
}

PythonAPIInfo::InputsWithNumberAttr* PythonAPIInfo::FindInputsWithNumberAttr(
    const string& name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_11(mht_11_v, 595, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::FindInputsWithNumberAttr");

  for (auto& input : inputs_with_number_attrs_) {
    if (name == input.number_attr->name) {
      return &input;
    }
  }
  return nullptr;
}

string PythonAPIInfo::DebugInfo() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTcc mht_12(mht_12_v, 607, "", "./tensorflow/python/framework/python_api_info.cc", "PythonAPIInfo::DebugInfo");

  string s = absl::StrCat("DebugInfo for ", api_name_, ":\n");
  absl::StrAppend(&s, "  param_names=[", absl::StrJoin(param_names_, ", "),
                  "]\n");
  Safe_PyObjectPtr defaults_repr(PyObject_Repr(defaults_tuple_.get()));
  absl::StrAppend(
      &s, "  defaults_tuple=", TFE_GetPythonString(defaults_repr.get()), "\n");
  if (!attributes_.empty()) {
    absl::StrAppend(&s, "  attributes=[");
    for (const auto& attrib : attributes_) {
      if (attrib.index != -1) {
        absl::StrAppend(&s, "\n    {index=", attrib.index);
        DCHECK_EQ(attrib.inferred_index, -1);
      } else {
        absl::StrAppend(&s, "\n    {inferred_index=", attrib.inferred_index);
      }
      absl::StrAppend(&s, ", name=", attrib.name,
                      ", type=", AttributeTypeToName(attrib.type), "},");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inputs_.empty()) {
    absl::StrAppend(&s, "  inputs=[");
    for (const auto& input : inputs_) {
      absl::StrAppend(&s, "\n    {index=", input.index,
                      ", name=", param_names_[input.index],
                      ", is_list=", input.is_list, "},");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inputs_with_fixed_dtype_.empty()) {
    absl::StrAppend(&s, "  inputs_with_fixed_dtype=[");
    for (const auto& input : inputs_with_fixed_dtype_) {
      absl::StrAppend(&s, "\n    {index=", input.index,
                      ", dtype=", DataType_Name(input.dtype),
                      ", is_list=", input.is_list, "},");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inputs_with_type_attrs_.empty()) {
    absl::StrAppend(&s, "  inputs_with_type_attr=[");
    for (const auto& input : inputs_with_type_attrs_) {
      absl::StrAppend(&s, "\n    {type_attr=", input.type_attr->name);
      if (input.default_dtype != DT_INVALID) {
        absl::StrAppend(&s,
                        ", default_dtype=", DataType_Name(input.default_dtype));
      }
      if (!input.tensor_params.empty()) {
        absl::StrAppend(&s, ", tensor_params=[",
                        absl::StrJoin(input.tensor_params, ", "), "]");
      }
      if (!input.tensor_list_params.empty()) {
        absl::StrAppend(&s, ", tensor_list_params=[",
                        absl::StrJoin(input.tensor_list_params, ", "), "]");
      }
      if (!input.ok_dtypes.empty()) {
        absl::StrAppend(
            &s, ", ok_dtypes=[",
            absl::StrJoin(input.ok_dtypes, ", ", DataTypeFormatter()), "]");
      }
      absl::StrAppend(&s, "},");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inputs_with_type_list_attrs_.empty()) {
    absl::StrAppend(&s, "  inputs_with_type_list_attrs=[");
    for (const auto& input : inputs_with_type_list_attrs_) {
      absl::StrAppend(&s, "\n    {type_list_attr=", input.type_list_attr->name);
      if (!input.default_dtypes.empty()) {
        absl::StrAppend(
            &s, ", default_dtypes=[",
            absl::StrJoin(input.default_dtypes, ", ", DataTypeFormatter()),
            "]");
      }
      if (!input.tensor_list_params.empty()) {
        absl::StrAppend(&s, ", tensor_list_params=[",
                        absl::StrJoin(input.tensor_list_params, ", "), "]");
      }
      if (!input.ok_dtypes.empty()) {
        absl::StrAppend(
            &s, ", ok_dtypes=[",
            absl::StrJoin(input.ok_dtypes, ", ", DataTypeFormatter()), "]");
      }
      absl::StrAppend(&s, "},");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inputs_with_number_attrs_.empty()) {
    absl::StrAppend(&s, "  inputs_with_number_attrs=[");
    for (const auto& input : inputs_with_number_attrs_) {
      absl::StrAppend(&s, "\n    {number_attr=", input.number_attr->name,
                      ", default_length=", input.default_length,
                      ", tensor_list_params=[",
                      absl::StrJoin(input.tensor_list_params, ", "), "],\n");
    }
    absl::StrAppend(&s, "]\n");
  }
  if (!inferred_type_attrs_.empty()) {
    absl::StrAppend(&s, "  inferred_type_attrs=[",
                    absl::StrJoin(inferred_type_attrs_, ", "), "]\n");
  }
  if (!inferred_type_list_attrs_.empty()) {
    absl::StrAppend(&s, "  inferred_type_list_attrs=[",
                    absl::StrJoin(inferred_type_list_attrs_, ", "), "]\n");
  }
  if (!inferred_length_attrs_.empty()) {
    absl::StrAppend(&s, "  inferred_length_attrs=[",
                    absl::StrJoin(inferred_length_attrs_, ", "), "]\n");
  }
  return s;
}

}  // namespace tensorflow
