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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh() {
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


#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class Converter;

// Specifies the expected type taken by a TRT_TensorOrWeights input during op
// conversion.
enum class TrtInputArg { kTensor = 1, kWeight = 2, kBoth = 3 };

// Parameters for each op converter.
struct OpConverterParams {
  // Constructor used for validation only.
  OpConverterParams(const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store,
                    TrtPrecisionMode precision_mode, bool use_calibration,
                    bool use_implicit_batch, bool use_explicit_precision);

  // Constructor used for conversion.
  OpConverterParams(Converter* converter, const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store);

  Converter* converter = nullptr;
  const NodeDef& node_def;
  const std::vector<TRT_TensorOrWeights>& inputs;
  std::vector<TRT_TensorOrWeights>* outputs;
  const bool validation_only;
  TrtWeightStore* weight_store;
  const TrtPrecisionMode precision_mode;
  const bool use_calibration;
  const bool use_implicit_batch;
  const bool use_explicit_precision;
};

// Operation converter function specification.
using OpConverter = std::function<Status(OpConverterParams*)>;

struct InputArgSpec {
  absl::string_view name;
  TrtInputArg allowed_roles;

  static constexpr InputArgSpec Create(absl::string_view n, TrtInputArg role) {
    return InputArgSpec{n, role};
  }
};

// A Curiously recurring template pattern (CRTP) template class for operation
// converters.
template <typename Impl>
class OpConverterBase {
 public:
  explicit OpConverterBase(OpConverterParams* params)
      : params_(params), node_def_attrs_(params->node_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh mht_0(mht_0_v, 253, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter.h", "OpConverterBase");
}

  // Default NodeDef attribute name to inspect in order to determine node data
  // type. The Impl class can override this by implementing the same function.
  static constexpr const char* NodeDefDataTypeAttributeName() { return "T"; }

  // Default allowed data types for the NodeDef data type attribute. The Impl
  // class can override this by implementing the same function.
  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  // Validate data type of the given NodeDef against allowed types.
  Status ValidateNodeDefDataType() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh mht_1(mht_1_v, 269, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter.h", "ValidateNodeDefDataType");

    // If the attribute name is empty, we should skip this check.
    if (absl::string_view(Impl::NodeDefDataTypeAttributeName()).empty()) {
      return Status::OK();
    }

    // Get the NodeDef data type.
    auto dtype = GetAttrValue<DataType>(Impl::NodeDefDataTypeAttributeName());
    if (!dtype.ok()) {
      return errors::InvalidArgument("Attribute with name ",
                                     Impl::NodeDefDataTypeAttributeName(),
                                     " not found.");
    }

    // Check allowed data types.
    const auto& node_def = params_->node_def;
    const auto& allowed_dtypes = Impl::AllowedDataTypes();
    if (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), *dtype) ==
        allowed_dtypes.end()) {
      std::string allowed_types_string = absl::StrJoin(
          allowed_dtypes, ", ", [](std::string* out, const DataType& type) {
            absl::StrAppendFormat(out, "%s", DataTypeString(type));
          });
      return errors::Unimplemented("Data type ", DataTypeString(*dtype),
                                   " is not supported for ", node_def.op(),
                                   ", must be one of [", allowed_types_string,
                                   "], at ", node_def.name());
    }
    return Status::OK();
  }

  // Validates input argument roles and data types.
  Status ValidateInputs() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh mht_2(mht_2_v, 304, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter.h", "ValidateInputs");

    const NodeDef& node_def = params_->node_def;
    const auto& inputs = params_->inputs;
    TRT_ENSURE(inputs.size() == Impl::InputSpec().size());
    for (int i = 0; i < inputs.size(); i++) {
      const InputArgSpec arg_spec = Impl::InputSpec()[i];
      if (arg_spec.allowed_roles == TrtInputArg::kWeight &&
          inputs.at(i).is_tensor()) {
        return errors::Unimplemented("The input \"", arg_spec.name, "\" for ",
                                     node_def.op(), " must be a constant, at ",
                                     node_def.name());
      }
      if (arg_spec.allowed_roles == TrtInputArg::kTensor &&
          inputs.at(i).is_weights()) {
        return errors::Unimplemented("The input \"", arg_spec.name, "\" for ",
                                     node_def.op(), " must be a tensor, at ",
                                     node_def.name());
      }
    }
    return Status::OK();
  }

  Status operator()() {
    // Validate data type and inputs.
    TF_RETURN_IF_ERROR(this->ValidateNodeDefDataType());
    TF_RETURN_IF_ERROR(this->ValidateInputs());

    // Perform op-level validation.
    TF_RETURN_IF_ERROR(reinterpret_cast<Impl*>(this)->Validate());
    if (params_->validation_only) {
      return Status::OK();
    }

    // Perform conversion.
    return reinterpret_cast<Impl*>(this)->Convert();
  }

 protected:
  void AddOutput(const TRT_TensorOrWeights& out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh mht_3(mht_3_v, 345, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter.h", "AddOutput");

    params_->outputs->push_back(out);
  }

  template <typename T>
  StatusOr<T> GetAttrValue(absl::string_view key) const {
    T result;
    TF_RETURN_IF_ERROR(GetNodeAttr(node_def_attrs_, key, &result));
    return result;
  }

  OpConverterParams* const params_;
  AttrSlice node_def_attrs_;
};

// Constructs and returns a converter function for a given operation converter
// class T. This requires T to be a derived class of StructuredOpConverter.
template <typename T>
OpConverter MakeConverterFunction() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSop_converterDTh mht_4(mht_4_v, 366, "", "./tensorflow/compiler/tf2tensorrt/convert/op_converter.h", "MakeConverterFunction");

  return [](OpConverterParams* params) -> Status {
    T converter(params);
    return converter();
  };
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
