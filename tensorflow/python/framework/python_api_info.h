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
#ifndef TENSORFLOW_PYTHON_UTIL_PYTHON_API_INFO_H_
#define TENSORFLOW_PYTHON_UTIL_PYTHON_API_INFO_H_
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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh() {
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


#include <Python.h>

#include <map>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/framework/python_tensor_converter.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Precomputed information about a TensorFlow Python API.
//
// PythonAPIInfo records information about a single TensorFlow Python API,
// in order to allow calls to the API to be executed more efficiently.  This
// information includes:
//
// * The name of the API.  (E.g. "tf.math.add")
//
// * The name of the registered op that implements the API, if applicable
//   (e.g. "AddV2").
//
// * Information about the API's parameters.  Parameters are divided into two
//   "kinds": inputs and attributes.  An *input* is a parameter that
//   expects a Tensor or list of Tensors, and it is described by an `ArgDef`.
//   An *attribute* is a parameter that expects any other value type, and it is
//   described by an `AttrDef`.
//
// * Default values for the API's attribute parameters.
//
// * Information about "inferred attributes" -- attributes whose values are
//   inferred from `input` parameters.  There are two kinds of inferred
//   attributes: Tensor dtypes, which are inferred from tensor and list(tensor)
//   parameters; and list lengths, which are inferred from list(tensor)
//   parameters.
class PythonAPIInfo {
 public:
  // The index of a parameter in the canonicalized parameter list.  The
  // canonicalized parameter list includes inputs and attributes (but does
  // not include inferred attributes).  `-1` is used for inferred attributes.
  using ParamIndex = int;

  // Information about a parameter that expects a non-Tensor value.
  struct Attribute {
    ParamIndex index;  // -1 if this is an inferred attribute
    AttributeType type;
    const char* name;    // Interned python string
    int inferred_index;  // index to store attribute in InferredAttributes
  };

  // Information about a parameter that expects a Tensor or list(Tensor).
  // Additional information about tensor parameters is stored in types
  // defined below, in order to simplify dtype/length inference:
  //   * FixedDTypeInput: inputs with fixed dtypes.
  //   * InputsWithTypeAttr: groups inputs that use a type_attr for dtype.
  //   * InputsWithTypeListAttr: groups inputs that use a type_list_attr.
  //   * InputsWithNumberAttr: groups inputs by a number_attr for length.
  struct Input {
    ParamIndex index;
    bool is_list;
  };

  // Information about a Tensor parameter w/ fixed dtype.
  struct InputWithFixedDType {
    DataType dtype;
    ParamIndex index;
    bool is_list;
  };

  // Information about Tensor parameters whose DType is specified by a single
  // `type_attr` attribute.
  struct InputsWithTypeAttr {
    Attribute* type_attr;                        // not owned.
    DataType default_dtype;                      // DT_INVALID if no default.
    std::vector<ParamIndex> tensor_params;       // single-tensor inputs.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
    std::vector<DataType> ok_dtypes;
  };

  // Information about Tensor parameters whose DType is specified by a single
  // `type_list_attr` attribute.
  struct InputsWithTypeListAttr {
    Attribute* type_list_attr;                   // not owned.
    std::vector<DataType> default_dtypes;        // empty if no default.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
    std::vector<DataType> ok_dtypes;
  };

  // Information about Tensor-list parameters whose length is specified by a
  // single `int` attribute.
  struct InputsWithNumberAttr {
    Attribute* number_attr;                      // not owned.
    int64_t default_length;                      // -1 for no default.
    std::vector<ParamIndex> tensor_list_params;  // list(tensor) inputs.
  };

  // Structure used to return inferred attribute values.
  //   * types[i] is the inferred value for inferred_type_attrs()[i]
  //   * type_lists[i] is the inferred value for inferred_type_list_attrs()[i]
  //   * lengths[i] is the inferred value for inferred_length_attrs()[i]
  struct InferredAttributes {
    std::vector<DataType> types;
    std::vector<std::vector<DataType>> type_lists;
    std::vector<int64_t> lengths;
  };

  // Constructs a new PythonAPIInfo.
  //
  // Note: One of the `Initialize()` functions must be called before the
  // `PythonAPIInfo` is used.
  //
  // Args:
  //   api_name: The fully-qualified name of the python API (e.g., tf.math.sum).
  explicit PythonAPIInfo(const std::string& api_name);

  // Initializes this PythonAPIInfo.
  //
  // Args:
  //   op_def: Contains information about the parameters.
  //   param_names: The argument names for the python API, in canonical order.
  //   defaults_tuple: Tuple containing default values for the parameters,
  //     right-aligned with `param_names` -- i.e., `defaults[-i]` is the default
  //     for `param_names[-i]`.
  Status Initialize(const OpDef& op_def, const std::vector<string> param_names,
                    PyObject* defaults_tuple);

  // Initialize this PythonAPIInfo based on the registered OpDef for the given
  // operation.
  //
  // Args:
  //   op_name: The registered name of the operation (e.g. "AddV2").
  Status InitializeFromRegisteredOp(const std::string& op_name);

  // Initializes this PythonAPIInfo based on a set of parameter specifications.
  //
  // Args:
  //   input_specs: Mapping from parameter name to specification string for
  //     each input (parameter that expects a tensor value).
  //   attr_specs: Mapping from parameter name to specification string for
  //     each attribute (parameter that expects a non-tensor value).
  //   param_names: The argument names for the python API, in canonical order.
  //   defaults_tuple: Tuple containing default values for the parameters,
  //     right-aligned with `param_names` -- i.e., `defaults[-i]` is the default
  //     for `param_names[-i]`.
  //
  // Note: the `name` parameter should not be included in `input_specs` or
  // `attr_specs`.
  Status InitializeFromParamSpecs(
      const std::map<std::string, std::string>& input_specs,
      const std::map<std::string, std::string>& attr_specs,
      const std::vector<string> param_names, PyObject* defaults_tuple);

  // The name of the API that is described by this PythonAPIInfo.
  const char* api_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_0(mht_0_v, 345, "", "./tensorflow/python/framework/python_api_info.h", "api_name");
 return api_name_; }

  // The ordered names of the canononical parameters that this API expects.
  const std::vector<const char*>& param_names() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_1(mht_1_v, 351, "", "./tensorflow/python/framework/python_api_info.h", "param_names");
 return param_names_; }

  // A Python tuple containing the default values for parameters.  This is
  // right-aligned with `param_name` -- i.e., `defaults[-i]` is the default
  // for `param_names[-i]`.
  const PyObject* defaults_tuple() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_2(mht_2_v, 359, "", "./tensorflow/python/framework/python_api_info.h", "defaults_tuple");
 return defaults_tuple_.get(); }

  // Information about the attribute (non-tensor) parameters for this API.
  const std::vector<Attribute>& attributes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_3(mht_3_v, 365, "", "./tensorflow/python/framework/python_api_info.h", "attributes");
 return attributes_; }

  // Information about the input (tensor) parameters for this API.
  const std::vector<Input>& inputs() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_4(mht_4_v, 371, "", "./tensorflow/python/framework/python_api_info.h", "inputs");
 return inputs_; }
  const std::vector<InputWithFixedDType>& inputs_with_fixed_dtype() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_5(mht_5_v, 375, "", "./tensorflow/python/framework/python_api_info.h", "inputs_with_fixed_dtype");

    return inputs_with_fixed_dtype_;
  }
  const std::vector<InputsWithTypeAttr>& inputs_with_type_attrs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_6(mht_6_v, 381, "", "./tensorflow/python/framework/python_api_info.h", "inputs_with_type_attrs");

    return inputs_with_type_attrs_;
  }
  const std::vector<InputsWithTypeListAttr>& inputs_with_type_list_attrs()
      const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_7(mht_7_v, 388, "", "./tensorflow/python/framework/python_api_info.h", "inputs_with_type_list_attrs");

    return inputs_with_type_list_attrs_;
  }
  const std::vector<InputsWithNumberAttr>& inputs_with_number_attrs() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_8(mht_8_v, 394, "", "./tensorflow/python/framework/python_api_info.h", "inputs_with_number_attrs");

    return inputs_with_number_attrs_;
  }

  // Names of inferred attributes.
  const std::vector<const char*>& inferred_type_attrs() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_9(mht_9_v, 402, "", "./tensorflow/python/framework/python_api_info.h", "inferred_type_attrs");

    return inferred_type_attrs_;
  }
  const std::vector<const char*>& inferred_type_list_attrs() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_10(mht_10_v, 408, "", "./tensorflow/python/framework/python_api_info.h", "inferred_type_list_attrs");

    return inferred_type_list_attrs_;
  }
  const std::vector<const char*>& inferred_length_attrs() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_infoDTh mht_11(mht_11_v, 414, "", "./tensorflow/python/framework/python_api_info.h", "inferred_length_attrs");

    return inferred_length_attrs_;
  }

  // Returns a string summarizing the internal state of this type converter.
  string DebugInfo() const;

 private:
  // Adds an entry to the attributes_ vector based on the given `AttrDef`.
  //
  // If `attr_def` describes a type attribute, then adds a value to
  // inputs_with_type_attrs_ or inputs_with_type_list_attrs_ (to record any
  // tensor inputs that use this dtype).
  //
  // If `attr_def` describes an int attribute, then adds a value to
  // inputs_with_number_attrs_ (to record any tensor inputs that use this
  // value as a list length).
  Status InitializeAttribute(
      const OpDef::AttrDef& attr_def,
      const std::map<std::string, ParamIndex>& param_name_to_index);

  // Adds an entry to the inputs_ vector based on the given `ArgDef`.
  //
  // If `arg_def` has a fixed dtype, then adds a value to `fixed_dtype_inputs`.
  //
  // If `arg_def`'s dtype is described by a `type` attr, then updates the
  // appropriate value in `inputs_with_type_attrs_` with information about the
  // `arg_def`.
  //
  // If `arg_def`'s dtype is described by a `list(type)` attr, then updates the
  // appropriate value in `inputs_with_type_list_attrs_` with information about
  // the `arg_def`.
  Status InitializeInput(const OpDef::ArgDef& arg_def,
                         const std::map<std::string, int>& param_name_to_index);

  // Checks that the OpDef used to initialize this PythonAPIInfo
  // had an AttrDef or ArgDef specification for each parameter.
  Status CheckParamNames() const;

  // Searches inputs_with_type_attrs_ for an input with the given name.
  InputsWithTypeAttr* FindInputsWithTypeAttr(const string& name);

  // Searches inputs_with_type_list_attrs_ for an input with the given name.
  InputsWithTypeListAttr* FindInputsWithTypeListAttr(const string& name);

  // Searches inputs_with_type_list_attrs_ for an input with the given name.
  InputsWithNumberAttr* FindInputsWithNumberAttr(const string& name);

  ABSL_MUST_USE_RESULT
  bool InferLengthAttributes(const absl::Span<PyObject*> params,
                             std::vector<int64_t>& inferred_length_attrs) const;

  // ==========================================================================
  // Member Variables
  // ==========================================================================

  // The name of the API that is described by this PythonAPIInfo.
  // (Interned python string).
  const char* api_name_;

  // The names of the parameters that this API expects.
  // (Interned python strings.)
  std::vector<const char*> param_names_;

  // Tuple containing default values for the parameters, right-aligned with
  // `param_names` -- i.e., `defaults[-i]` is the default for `param_names[-i]`.
  Safe_PyObjectPtr defaults_tuple_;

  // Information about the non-tensor-valued parameters that this API expects.
  std::vector<Attribute> attributes_;

  // Information about the tensor-valued parameters that this API expects.
  std::vector<Input> inputs_;
  std::vector<InputWithFixedDType> inputs_with_fixed_dtype_;
  std::vector<InputsWithTypeAttr> inputs_with_type_attrs_;
  std::vector<InputsWithTypeListAttr> inputs_with_type_list_attrs_;
  std::vector<InputsWithNumberAttr> inputs_with_number_attrs_;

  // Names of inferred attributes.  (Interned python strings.)
  std::vector<const char*> inferred_type_attrs_;
  std::vector<const char*> inferred_type_list_attrs_;
  std::vector<const char*> inferred_length_attrs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_UTIL_PYTHON_API_INFO_H_
