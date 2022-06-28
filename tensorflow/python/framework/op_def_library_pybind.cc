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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc() {
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

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "Python.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

namespace {

using ::tensorflow::AttributeType;
using ::tensorflow::AttributeTypeFromName;
using ::tensorflow::AttrValue;
using ::tensorflow::CheckOpDeprecation;
using ::tensorflow::ConvertPyObjectToAttributeType;
using ::tensorflow::DataType;
using ::tensorflow::DataTypeToPyObject;
using ::tensorflow::MaybeRaiseFromStatus;
using ::tensorflow::OpDef;
using ::tensorflow::OpRegistry;
using ::tensorflow::protobuf::RepeatedField;
using ::tensorflow::protobuf::RepeatedPtrField;
using AttrDef = ::tensorflow::OpDef::AttrDef;
using ArgDef = ::tensorflow::OpDef::ArgDef;
// Keys: attr.name(); Values: attr_def.allowed_values().list().type()
using AllowedAttrMap =
    absl::flat_hash_map<std::string, absl::flat_hash_set<int>>;
// Keys: attr.name(); Values; attr_def.default_value().type()
using DefaultAttrMap = absl::flat_hash_map<std::string, py::object>;
// Keys: attr.name(); Values: corresponding attr serialized as an AttrValue
using AttrProtosMap = absl::flat_hash_map<std::string, AttrValue>;

constexpr char kType[] = "type";
constexpr char kTypeEnum[] = "_type_enum";
constexpr char kDType[] = "dtype";
constexpr char kBaseDType[] = "base_dtype";
constexpr char kAsProto[] = "as_proto";
constexpr char kSerialize[] = "SerializeToString";
constexpr char kListPrefix[] = "list(";
constexpr char kPop[] = "pop";

inline py::error_already_set PyTypeError(const std::string& error_msg) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("error_msg: \"" + error_msg + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_0(mht_0_v, 245, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "PyTypeError");

  PyErr_SetString(PyExc_TypeError, error_msg.c_str());
  return pybind11::error_already_set();
}

inline py::error_already_set PyValueError(const std::string& error_msg) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("error_msg: \"" + error_msg + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_1(mht_1_v, 254, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "PyValueError");

  PyErr_SetString(PyExc_ValueError, error_msg.c_str());
  return pybind11::error_already_set();
}

inline std::string PyRepr(const py::handle& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_2(mht_2_v, 262, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "PyRepr");

  return value.attr("__repr__")().cast<std::string>();
}

py::object DataTypeToPybindObject(const DataType& data_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_3(mht_3_v, 269, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "DataTypeToPybindObject");

  return py::reinterpret_borrow<py::object>(
      DataTypeToPyObject(data_type).release());
}

// Converts the py:object to the AttributeType.
// ToAttributeType corrupts the value's representation when it fails. So this
// should be stored before hand if it is needed for error msgs.
py::object ToAttributeType(const py::handle& value, const AttributeType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_4(mht_4_v, 280, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ToAttributeType");

  auto result = ConvertPyObjectToAttributeType(value.ptr(), type);
  if (result == nullptr) {
    throw std::runtime_error("Failed to perform conversion.");
  }
  return py::reinterpret_borrow<py::object>(result.release());
}

inline bool MakeBool(const py::handle& value, const std::string& arg_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("arg_name: \"" + arg_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_5(mht_5_v, 292, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "MakeBool");

  if (!py::isinstance<py::bool_>(value)) {
    throw PyTypeError(
        absl::StrCat("Expected bool for argument '", arg_name, "' not ",
                     value.attr("__repr__")().cast<std::string>(), "."));
  }
  return value.cast<py::bool_>();
}

inline int MakeInt(const py::handle& value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_6(mht_6_v, 304, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "MakeInt");

  try {
    // Needed for TF1 compatibility where a tf.Dimension may be passed in.
    return value.attr("value").cast<float>();
  } catch (...) {
    return value.cast<float>();  // Cast to float to match Python's behaviour.
  }
}

inline DataType MakeType(const py::handle& value, const std::string& arg_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("arg_name: \"" + arg_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_7(mht_7_v, 317, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "MakeType");

  std::string repr_v = PyRepr(value);
  try {
    return ToAttributeType(value, AttributeType::DTYPE)
        .attr(kBaseDType)
        .cast<DataType>();
  } catch (...) {
    throw PyTypeError(absl::StrCat("Expected DataType for argument '", arg_name,
                                   "' not ", repr_v, "."));
  }
}

inline std::string MakeShape(const py::handle& value,
                             const std::string& arg_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("arg_name: \"" + arg_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_8(mht_8_v, 334, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "MakeShape");

  std::string repr_v = PyRepr(value);
  try {
    return ToAttributeType(value, AttributeType::SHAPE)
        .attr(kAsProto)()
        .attr(kSerialize)()
        .cast<std::string>();
  } catch (...) {
    throw PyTypeError(absl::StrCat("Error converting ", repr_v, " (arg name = ",
                                   arg_name, ") to a TensorShape"));
  }
}

AttrValue ValueToAttrValue(const py::object& value,
                           const std::string& attr_type,
                           const std::string& arg_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_type: \"" + attr_type + "\"");
   mht_9_v.push_back("arg_name: \"" + arg_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_9(mht_9_v, 354, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ValueToAttrValue");

  AttrValue attr_value;
  if (absl::StartsWith(attr_type, kListPrefix)) {
    if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
      throw PyTypeError(absl::StrCat(
          "Expected list for attr ", arg_name, ", obtained ",
          py::type::handle_of(value).attr("__name__").cast<std::string>(),
          " instead."));
    }
  }

  try {
    const AttributeType type_enum = AttributeTypeFromName(attr_type);
    switch (type_enum) {
      case AttributeType::STRING:
        attr_value.set_s(value.cast<std::string>());
        break;
      case AttributeType::LIST_STRING: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_s(v.cast<std::string>());
        }
        break;
      }
      case AttributeType::INT:
        attr_value.set_i(MakeInt(value));
        break;
      case AttributeType::LIST_INT: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_i(MakeInt(v));
        }
        break;
      }
      case AttributeType::FLOAT:
        attr_value.set_f(value.cast<float>());
        break;
      case AttributeType::LIST_FLOAT: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_f(v.cast<float>());
        }
        break;
      }
      case AttributeType::BOOL:
        attr_value.set_b(MakeBool(value, arg_name));
        break;
      case AttributeType::LIST_BOOL: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_b(MakeBool(v, arg_name));
        }
        break;
      }
      case AttributeType::DTYPE: {
        attr_value.set_type(MakeType(value, arg_name));
        break;
      }
      case AttributeType::LIST_DTYPE: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_type(MakeType(v, arg_name));
        }
        break;
      }
      case AttributeType::SHAPE:
        attr_value.mutable_shape()->ParseFromString(MakeShape(value, arg_name));
        break;
      case AttributeType::LIST_SHAPE: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_shape()->ParseFromString(MakeShape(v, arg_name));
        }
        break;
      }
      case AttributeType::TENSOR:
        attr_value.mutable_tensor()->ParseFromString(
            ToAttributeType(value, type_enum)
                .attr(kSerialize)()
                .cast<std::string>());
        break;
      case AttributeType::LIST_TENSOR: {
        auto* list = attr_value.mutable_list();
        for (const auto& v : value) {
          list->add_tensor()->ParseFromString(
              ToAttributeType(v, AttributeType::TENSOR)
                  .attr(kSerialize)()
                  .cast<std::string>());
        }
        break;
      }
      default:
        throw PyTypeError(absl::StrCat("Unrecognized Attr type ", attr_type,
                                       " for ", arg_name, "."));
    }
  } catch (const py::error_already_set& e) {
    throw e;
  } catch (...) {
    throw PyTypeError(absl::StrCat(
        "Expected ", attr_type, " for argument '", arg_name, "' not ",
        value.attr("__repr__")().cast<std::string>(), "."));
  }

  return attr_value;
}

py::object AttrValueToSerializedBytesPyObject(const AttrValue& attr_value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_10(mht_10_v, 463, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AttrValueToSerializedBytesPyObject");

  std::string serialized_attr_value;
  if (!attr_value.SerializeToString(&serialized_attr_value)) {
    throw std::runtime_error("Failed to serialized AttrValue to string");
  }
  return py::reinterpret_borrow<py::object>(py::bytes(serialized_attr_value));
}

void AssertSatisfiesLengthConstraint(const py::object& attr,
                                     const AttrDef& attr_def,
                                     const std::string& attr_name,
                                     const std::string& op_type_name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_11_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_11(mht_11_v, 479, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesLengthConstraint");

  if (!absl::StartsWith(attr_def.type(), kListPrefix)) return;
  int attr_size = attr.cast<py::list>().size();
  if (attr_def.has_minimum() && attr_size < attr_def.minimum()) {
    throw PyValueError(absl::StrCat("Attr '", attr_name, "' of '", op_type_name,
                                    "' Op passed list of length ", attr_size,
                                    " less than minimum ", attr_def.minimum(),
                                    "."));
  }
}

void AssertSatisfiesAllowedStringConstraint(
    const std::string& attr,
    const RepeatedPtrField<std::string>& allowed_values,
    const std::string& attr_name, const std::string& op_type_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr: \"" + attr + "\"");
   mht_12_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_12_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_12(mht_12_v, 499, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesAllowedStringConstraint");

  if (!absl::c_linear_search(allowed_values, attr)) {
    const std::string allowed_values_str =
        absl::StrJoin(allowed_values, "\", \"");
    throw PyValueError(absl::StrCat("Attr '", attr_name, "' of '", op_type_name,
                                    "' Op passed string '", attr,
                                    "' not in: \"", allowed_values_str, "\"."));
  }
}

void AssertSatisfiesAllowedStringsConstraint(const AttrValue& attr,
                                             const AttrDef& attr_def,
                                             const std::string& attr_name,
                                             const AttributeType attr_type,
                                             const std::string& op_type_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_13_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_13(mht_13_v, 518, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesAllowedStringsConstraint");

  if (!attr_def.has_allowed_values()) return;
  const auto& allowed_values = attr_def.allowed_values().list().s();
  if (attr_type == AttributeType::STRING) {
    AssertSatisfiesAllowedStringConstraint(attr.s(), allowed_values, attr_name,
                                           op_type_name);
  } else if (attr_type == AttributeType::LIST_STRING) {
    for (const std::string& v : attr.list().s()) {
      AssertSatisfiesAllowedStringConstraint(v, allowed_values, attr_name,
                                             op_type_name);
    }
  }
}

void AssertSatisfiesIntMinimumConstraint(const AttrValue& attr,
                                         const AttrDef& attr_def,
                                         const std::string& attr_name,
                                         const AttributeType attr_type,
                                         const std::string& op_type_name) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_14_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_14(mht_14_v, 541, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesIntMinimumConstraint");

  if (attr_def.has_minimum() && attr_type == AttributeType::INT &&
      attr.i() < attr_def.minimum()) {
    throw PyValueError(absl::StrCat(
        "Attr '", attr_name, "' of '", op_type_name, "' Op passed ", attr.i(),
        " less than minimum ", attr_def.minimum(), "."));
  }
}

void AssertSatisfiesAllowedListAttrTypeConstraint(
    const std::string& type_attr, const AllowedAttrMap& allowed_list_attr_map,
    const py::object& dtype, const std::string& input_name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("type_attr: \"" + type_attr + "\"");
   mht_15_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_15(mht_15_v, 557, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesAllowedListAttrTypeConstraint");

  auto it = allowed_list_attr_map.find(type_attr);
  if (it != allowed_list_attr_map.end() &&
      !it->second.contains(dtype.cast<DataType>())) {
    std::vector<std::string> allowed_values;
    for (const auto& allowed_value : it->second) {
      allowed_values.emplace_back(
          DataTypeToPybindObject(static_cast<DataType>(allowed_value))
              .attr("name")
              .cast<std::string>());
    }
    throw PyTypeError(absl::StrCat("Value passed to parameter '", input_name,
                                   "' has DataType ",
                                   dtype.attr("name").cast<std::string>(),
                                   " not in list of allowed values: ",
                                   absl::StrJoin(allowed_values, ", ")));
  }
}

void AssertSatisfiesDTypeConstraint(const int attr,
                                    const RepeatedField<int>& allowed_values,
                                    const std::string& attr_name,
                                    const std::string& op_type_name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_16_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_16(mht_16_v, 584, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesDTypeConstraint");

  if (!absl::c_linear_search(allowed_values, attr)) {
    std::string allowed_vals_str;
    for (const auto& v : allowed_values) {
      if (!allowed_vals_str.empty()) absl::StrAppend(&allowed_vals_str, ", ");
      absl::StrAppend(&allowed_vals_str,
                      DataTypeToPybindObject(static_cast<DataType>(v))
                          .attr("name")
                          .cast<std::string>());
    }
    throw PyTypeError(absl::StrCat(
        "Value passed to parameter '", attr_name, "' has DataType ",
        DataTypeToPybindObject(static_cast<DataType>(attr))
            .attr("name")
            .cast<std::string>(),
        " not in list of allowed values: ", allowed_vals_str));
  }
}

void AssertSatisfiesTypeConstraint(const AttrValue& attr,
                                   const AttrDef& attr_def,
                                   const std::string& attr_name,
                                   const AttributeType attr_type,
                                   const std::string& op_type_name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_17_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_17(mht_17_v, 612, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "AssertSatisfiesTypeConstraint");

  if (!attr_def.has_allowed_values()) return;
  const auto& allowed_values = attr_def.allowed_values().list().type();
  if (attr_type == AttributeType::DTYPE) {
    AssertSatisfiesDTypeConstraint(attr.type(), allowed_values, attr_name,
                                   op_type_name);
  } else if (attr_type == AttributeType::LIST_DTYPE) {
    for (const auto& v : attr.list().type()) {
      AssertSatisfiesDTypeConstraint(v, allowed_values, attr_name,
                                     op_type_name);
    }
  }
}

// Returns the OpDef from the global registry. Raises runtime_error if the
// OpDef is not found.
const OpDef* GetOpDef(const std::string& op_type_name, int producer_version) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_18(mht_18_v, 632, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "GetOpDef");

  const OpDef* op_def = nullptr;
  auto status = OpRegistry::Global()->LookUpOpDef(op_type_name, &op_def);
  if (!status.ok() || op_def == nullptr) {
    throw std::runtime_error(
        absl::StrCat("Unrecognized Op name ", op_type_name));
  }
  return op_def;
}

// Extracts the default_type_attr_map and the allowed_list_attr_map from the
// OpDef.
void ExtractDefaultTypesAndAllowedTypes(const OpDef& op_def,
                                        DefaultAttrMap& default_type_attr_map,
                                        AllowedAttrMap& allowed_list_attr_map) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_19(mht_19_v, 649, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ExtractDefaultTypesAndAllowedTypes");

  for (const AttrDef& attr_def : op_def.attr()) {
    if (attr_def.type() != kType) continue;
    const std::string& attr_name = attr_def.name();
    if (attr_def.has_default_value()) {
      default_type_attr_map[attr_name] =
          DataTypeToPybindObject(attr_def.default_value().type());
    }
    if (attr_def.has_allowed_values()) {
      const auto& types = attr_def.allowed_values().list().type();
      absl::flat_hash_set<int> allowed_values(types.begin(), types.end());
      allowed_list_attr_map[attr_name] = std::move(allowed_values);
    }
  }
}

// Returns the input Tensor corresponding to `input_name` from `keywords`.
// Updates `input_name` if it is a Python keyword or built-in.
py::object GetInputTensor(std::string& input_name, const py::dict& keywords,
                          const OpDef& op_def) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_20(mht_20_v, 671, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "GetInputTensor");

  if (keywords.contains(input_name)) {
    return py::reinterpret_borrow<py::object>(
        keywords.attr(kPop)(input_name.c_str()));
  } else if (keywords.contains(absl::StrCat(input_name, "_"))) {
    absl::StrAppend(&input_name, "_");
    return py::reinterpret_borrow<py::object>(
        keywords.attr(kPop)(input_name.c_str()));
  } else {
    throw PyTypeError(absl::StrCat("No argument for input ", input_name,
                                   " found in ", op_def.DebugString()));
  }
}

// Returns the input Tensor's DType.
py::object GetInputType(
    const py::object& input_tensor, const ArgDef& input_arg,
    const AllowedAttrMap& allowed_list_attr_map,
    const std::string& op_type_name, const std::string& input_name,
    py::dict& attrs,
    absl::flat_hash_map<std::string, std::string>& inferred_from) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("op_type_name: \"" + op_type_name + "\"");
   mht_21_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_21(mht_21_v, 696, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "GetInputType");

  py::object dtype = input_tensor.attr(kDType);
  py::object base_type = dtype.attr(kBaseDType);

  // Check that the input_arg and the input are compatible.
  if (input_arg.type() != DataType::DT_INVALID &&
      input_arg.type() != dtype.cast<DataType>() &&
      input_arg.type() != base_type.cast<DataType>()) {
    throw PyTypeError(absl::StrCat("Input '", input_name, "' of '",
                                   op_type_name, "' Op has type ",
                                   base_type.attr("name").cast<std::string>(),
                                   " that does not match expected type of ",
                                   DataTypeToPybindObject(input_arg.type())
                                       .attr("name")
                                       .cast<std::string>(),
                                   "."));
  }

  const std::string& type_attr = input_arg.type_attr();
  if (!type_attr.empty()) {
    if (attrs.contains(type_attr) &&
        attrs[type_attr.c_str()].cast<py::object>() != base_type) {
      throw PyTypeError(absl::StrCat(
          "Input '", input_name, "' of '", op_type_name, "' Op has type ",
          base_type.attr("name").cast<std::string>(),
          " that does not match type ",
          attrs[type_attr.c_str()].attr("name").cast<std::string>(),
          " of argument '", inferred_from.at(type_attr), "'."));
    } else {
      AssertSatisfiesAllowedListAttrTypeConstraint(
          type_attr, allowed_list_attr_map, base_type, input_name);
      attrs[type_attr.c_str()] = base_type;
      inferred_from[input_arg.type_attr()] = input_name;
    }
  } else if (base_type.cast<DataType>() != input_arg.type()) {
    // Added to match the python behaviour.
    throw PyTypeError("Unreachable");
  }
  if (input_arg.is_ref()) return dtype;
  return base_type;
}

// Extracts `inputs`, `input_types` and `attrs`.
void ExtractInputsAndAttrs(const std::string& op_type_name, const OpDef& op_def,
                           const AllowedAttrMap& allowed_list_attr_map,
                           py::dict& keywords, py::dict& attrs,
                           py::list& inputs, py::list& input_types) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_22(mht_22_v, 746, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ExtractInputsAndAttrs");

  absl::flat_hash_map<std::string, std::string> inferred_from;
  for (const ArgDef& input_arg : op_def.input_arg()) {
    std::string input_name = input_arg.name();
    py::object input_tensor = GetInputTensor(input_name, keywords, op_def);
    inputs.append(input_tensor);
    py::object dtype =
        GetInputType(input_tensor, input_arg, allowed_list_attr_map,
                     op_type_name, input_name, attrs, inferred_from);
    input_types.append(dtype);
  }
}

// Extracts the remaining attributes from the OpDef to `attrs`.
void ExtractRemainingAttrs(const std::string& op_type_name, const OpDef& op_def,
                           const py::dict& keywords,
                           const DefaultAttrMap& default_type_attr_map,
                           py::dict& attrs) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_23(mht_23_v, 767, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ExtractRemainingAttrs");

  for (const AttrDef& attr : op_def.attr()) {
    const std::string& attr_name = attr.name();
    if (attrs.contains(attr_name)) {
      if (keywords.contains(attr_name)) {
        throw PyTypeError(
            absl::StrCat("Should not specify value for inferred attr '",
                         attr_name, "' for ", op_type_name, "."));
      }
      continue;
    }
    if (keywords.contains(attr_name)) {
      attrs[attr_name.c_str()] =
          keywords.attr(kPop)(attr_name.c_str()).cast<py::object>();
    } else if (keywords.contains(absl::StrCat(attr_name, "_"))) {
      attrs[attr_name.c_str()] =
          keywords.attr(kPop)(absl::StrCat(attr_name, "_").c_str())
              .cast<py::object>();
    } else if (default_type_attr_map.contains(attr_name)) {
      attrs[attr_name.c_str()] = default_type_attr_map.at(attr_name);
    } else {
      throw PyTypeError(absl::StrCat("No argument found for attr ", attr_name,
                                     " for ", op_type_name));
    }
  }
}

void SetAttrProto(const std::string& key, const AttrValue& value,
                  py::dict& attr_protos, AttrProtosMap& attr_protos_map) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_24(mht_24_v, 799, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "SetAttrProto");

  attr_protos_map[key] = value;
  attr_protos[key.c_str()] = AttrValueToSerializedBytesPyObject(value);
}

// Converts attr values to AttrValues.
void ExtractAttrProto(const std::string& op_type_name, const OpDef& op_def,
                      const py::dict& attrs, py::dict& attr_protos,
                      AttrProtosMap& attr_protos_map) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_25(mht_25_v, 811, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ExtractAttrProto");

  for (const AttrDef& attr_def : op_def.attr()) {
    const std::string& attr_name = attr_def.name();
    const py::object attr = attrs[attr_name.c_str()].cast<py::object>();

    if (attr_def.has_default_value() && attr.is_none()) {
      SetAttrProto(attr_name, attr_def.default_value(), attr_protos,
                   attr_protos_map);
      continue;
    }

    const AttrValue attr_value =
        ValueToAttrValue(attr, attr_def.type(), attr_name);
    const AttributeType attr_type = AttributeTypeFromName(attr_def.type());
    AssertSatisfiesLengthConstraint(attr, attr_def, attr_name, op_type_name);
    AssertSatisfiesAllowedStringsConstraint(attr_value, attr_def, attr_name,
                                            attr_type, op_type_name);
    AssertSatisfiesIntMinimumConstraint(attr_value, attr_def, attr_name,
                                        attr_type, op_type_name);
    AssertSatisfiesTypeConstraint(attr_value, attr_def, attr_name, attr_type,
                                  op_type_name);
    SetAttrProto(attr_name, attr_value, attr_protos, attr_protos_map);
  }
}

inline const AttrValue& MaybeGetAttrValue(const py::dict& attr_protos,
                                          const AttrProtosMap& attr_protos_map,
                                          const std::string& attr_name,
                                          const std::string& op_type_name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("attr_name: \"" + attr_name + "\"");
   mht_26_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_26(mht_26_v, 844, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "MaybeGetAttrValue");

  auto it = attr_protos_map.find(attr_name);
  if (it != attr_protos_map.end()) return it->second;
  throw PyTypeError(absl::StrCat(
      "Inconsistent OpDef for '", op_type_name, "', missing attr '", attr_name,
      "' from '", attr_protos.attr("__repr__")().cast<std::string>(), "'."));
}

void ExtractOutputStructure(const std::string& op_type_name,
                            const OpDef& op_def, const py::dict& attr_protos,
                            const AttrProtosMap& attr_protos_map,
                            py::list& output_structure) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_27(mht_27_v, 859, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "ExtractOutputStructure");

  for (const ArgDef& arg : op_def.output_arg()) {
    if (!arg.number_attr().empty()) {
      const auto& value = MaybeGetAttrValue(attr_protos, attr_protos_map,
                                            arg.number_attr(), op_type_name);
      output_structure.append(value.i());
    } else if (!arg.type_attr().empty()) {
      const auto& _ = MaybeGetAttrValue(attr_protos, attr_protos_map,
                                        arg.type_attr(), op_type_name);
      output_structure.append(py::none());
    } else if (!arg.type_list_attr().empty()) {
      const auto& value = MaybeGetAttrValue(attr_protos, attr_protos_map,
                                            arg.type_list_attr(), op_type_name);
      output_structure.append(value.list().type_size());
    } else {
      output_structure.append(py::none());
    }
  }
}

void CheckAllInputsUsed(const std::string& op_type_name,
                        const py::dict& keywords) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("op_type_name: \"" + op_type_name + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSop_def_library_pybindDTcc mht_28(mht_28_v, 884, "", "./tensorflow/python/framework/op_def_library_pybind.cc", "CheckAllInputsUsed");

  if (!keywords.empty()) {
    std::string all_keywords;
    for (const auto& item : keywords) {
      if (!all_keywords.empty()) absl::StrAppend(&all_keywords, ", ");
      absl::StrAppend(&all_keywords, item.first.cast<std::string>());
    }
    throw PyTypeError(absl::StrCat(
        op_type_name, " got unexpected keyword arguments: ", all_keywords));
  }
}

}  // namespace

// This module provides a subset of the functionality from op_def_library.py
// and relies on op_def_library_test.py for test coverage.
PYBIND11_MODULE(_op_def_library_pybind, m) {
  // Method assumes all inputs in `keywords` are of type tf.Tensor.
  m.def("process_inputs", [](std::string& op_type_name, int producer_version,
                             py::dict& keywords) {
    const OpDef* op_def = GetOpDef(op_type_name, producer_version);
    MaybeRaiseFromStatus(CheckOpDeprecation(*op_def, producer_version));

    DefaultAttrMap default_type_attr_map;
    AllowedAttrMap allowed_list_attr_map;
    AttrProtosMap attr_protos_map;
    py::dict attrs, attr_protos;
    py::list inputs, input_types, output_structure;

    ExtractDefaultTypesAndAllowedTypes(*op_def, default_type_attr_map,
                                       allowed_list_attr_map);
    ExtractInputsAndAttrs(op_type_name, *op_def, allowed_list_attr_map,
                          keywords, attrs, inputs, input_types);
    ExtractRemainingAttrs(op_type_name, *op_def, keywords,
                          default_type_attr_map, attrs);
    ExtractAttrProto(op_type_name, *op_def, attrs, attr_protos,
                     attr_protos_map);
    ExtractOutputStructure(op_type_name, *op_def, attr_protos, attr_protos_map,
                           output_structure);
    CheckAllInputsUsed(op_type_name, keywords);

    return py::make_tuple(attr_protos, inputs, input_types, output_structure);
  });
};
