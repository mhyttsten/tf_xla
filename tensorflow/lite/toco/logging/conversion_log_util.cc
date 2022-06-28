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
class MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc() {
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
#include "tensorflow/lite/toco/logging/conversion_log_util.h"

#ifdef __linux__
#include <sys/utsname.h>
#endif

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/export.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/lite/version.h"

namespace toco {

namespace {

std::string TryGetOperatorName(const Operator& op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "TryGetOperatorName");

  std::string op_name;
  if (!op.tensorflow_node_def.empty()) {
    // Parse op name from serialized NodeDef.
    tensorflow::NodeDef node_def;
    if (!node_def.ParseFromString(op.tensorflow_node_def)) {
      LOG(ERROR) << "Failed to parse Tensorflow NodeDef";
    } else {
      op_name = node_def.op();
      if (!op_name.empty()) return op_name;
    }
  }
  if (op.type == OperatorType::kUnsupported) {
    // If we failed to get op name from serialized NodeDef (either because
    // the tensorflow_node_def is an empty string, or we failed to parse
    // from it), fall back to use 'tensorflow_op' field if this op is a
    // TensorflowUnsupportedOperator.
    const TensorFlowUnsupportedOperator& unsupported_op =
        static_cast<const TensorFlowUnsupportedOperator&>(op);
    if (!unsupported_op.tensorflow_op.empty()) {
      op_name = unsupported_op.tensorflow_op;
      return op_name;
    }
  }
  // If this is a built-in op.
  op_name = OperatorTypeName(op.type);
  return op_name;
}

std::string GetOSVersion() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_1(mht_1_v, 238, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetOSVersion");

  std::string os_info;
#ifdef __linux__
  utsname info;
  if (uname(&info)) {
    // Failed
    LOG(ERROR) << "Cannot get OS info.";
    return "";
  }
  os_info =
      std::string(info.sysname) + ";OSVer=" + std::string(info.release) + ";";
#endif
  return os_info;
}

std::string ShapeToStringNoSpace(const Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "ShapeToStringNoSpace");

  if (shape.dimensions_count() == 0) {
    return "[]";
  }

  return absl::StrCat("[", absl::StrJoin(shape.dims(), ","), "]");
}

std::string GetOperatorSignature(
    const Model& model, const Operator& op,
    const std::map<OperatorType, std::unique_ptr<tflite::BaseOperator>>&
        op_types_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_3(mht_3_v, 270, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetOperatorSignature");

  // The signature of an op has the following schema:
  // INPUT:SHAPE::TYPE::OUTPUT:SHAPE::TYPE::NAME:VERSION:
  std::string op_signature;
  constexpr char delimiter[] = "::";

  // Get input shapes and types.
  op_signature.append("INPUT:");
  for (const auto& input : op.inputs) {
    const auto& array = model.GetArray(input);
    if (array.has_shape()) {
      op_signature.append(ShapeToStringNoSpace(array.shape()));
    } else {
      op_signature.append("None");
    }
    op_signature.append(delimiter);
    op_signature.append(ArrayDataTypeName(array.data_type) + delimiter);
  }
  // Get output shapes and types.
  op_signature.append("OUTPUT:");
  for (const auto& output : op.outputs) {
    const auto& array = model.GetArray(output);
    if (array.has_shape()) {
      op_signature.append(ShapeToStringNoSpace(array.shape()));
    } else {
      op_signature.append("None");
    }
    op_signature.append(delimiter);
    op_signature.append(ArrayDataTypeName(array.data_type) + delimiter);
  }
  // Append Op name.
  op_signature.append("NAME:");
  op_signature.append(TryGetOperatorName(op) + delimiter);
  // Append Op version.
  op_signature.append("VERSION:");
  OperatorSignature toco_op_signature;
  toco_op_signature.op = &op;
  toco_op_signature.model = &model;
  if (op_types_map.find(op.type) != op_types_map.end()) {
    const int version = op_types_map.at(op.type)->GetVersion(toco_op_signature);
    op_signature.append(std::to_string(version));
  } else {
    op_signature.append("None");
  }
  return op_signature;
}

}  // namespace

std::vector<std::string> GetOperatorNames(const Model& model) {
  std::vector<std::string> op_names;
  for (const auto& op : model.operators) {
    op_names.push_back(TryGetOperatorName(*op));
  }
  return op_names;
}

void CountOperatorsByType(const Model& model,
                          std::map<std::string, int>* built_in_ops,
                          std::map<std::string, int>* custom_ops,
                          std::map<std::string, int>* select_ops) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_4(mht_4_v, 333, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "CountOperatorsByType");

  for (const auto& op : model.operators) {
    OperatorSignature op_signature = {op.get(), &model};
    const auto ops_by_type =
        tflite::BuildOperatorByTypeMap(true /*enable_select_tf_ops*/);
    tflite::details::OperatorKey op_key(op_signature, ops_by_type,
                                        true /*enable_select_tf_ops*/);

    const std::string op_name = TryGetOperatorName(*op);
    if (op_key.is_custom_op()) {
      (*custom_ops)[op_name]++;
    } else if (op_key.is_flex_op()) {
      (*select_ops)[op_name]++;
    } else {
      (*built_in_ops)[op_name]++;
    }
  }
}

void GetInputAndOutputTypes(
    const Model& model,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* input_types,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* output_types) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_5(mht_5_v, 358, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetInputAndOutputTypes");

  for (const auto& input_array : model.flags.input_arrays()) {
    const Array& array = model.GetArray(input_array.name());
    input_types->Add(ArrayDataTypeName(array.data_type));
  }
  for (const auto& output_array : model.flags.output_arrays()) {
    const Array& array = model.GetArray(output_array);
    output_types->Add(ArrayDataTypeName(array.data_type));
  }
}

std::string GetTfLiteVersion() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_6(mht_6_v, 372, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetTfLiteVersion");
 return TFLITE_VERSION_STRING; }

std::string GetCachedOSVersion() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_7(mht_7_v, 377, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetCachedOSVersion");

  static std::string* version = new std::string(GetOSVersion());
  return *version;
}

void GetOpSignatures(
    const Model& model,
    TFLITE_PROTO_NS::RepeatedPtrField<std::string>* op_signatures) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_8(mht_8_v, 387, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetOpSignatures");

  const auto& op_types_map =
      tflite::BuildOperatorByTypeMap(true /*enable_select_tf_ops*/);
  for (const auto& op : model.operators) {
    op_signatures->Add(GetOperatorSignature(model, *op, op_types_map));
  }
}

std::string GetModelHash(const Model& model) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_9(mht_9_v, 398, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "GetModelHash");

  // TODO(b/123519920): Implement the hash function for Model.
  // Need to consider different implementations for public/private models.
  return "";
}

// This function scans through the error message string, extracts the part about
// missing ops and prunes away all other information in the error info.
std::string SanitizeErrorMessage(const std::string& error_message) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_10(mht_10_v, 410, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "SanitizeErrorMessage");

  const std::string s1 = "Ops that can be supported by the flex runtime";
  const std::string s2 = "Ops that need custom implementation";
  std::string pruned_message;
  size_t pos = error_message.find(s1);
  if (pos != std::string::npos) {
    // Find the terminate point for flex op list.
    auto end = error_message.find('.', pos);
    pruned_message.append(error_message.substr(pos, end - pos + 1));
  }
  pos = error_message.find(s2);
  if (pos != std::string::npos) {
    // Find the terminate point for custom op list.
    auto end = error_message.find('.', pos);
    pruned_message.append(error_message.substr(pos, end - pos + 1));
  }
  return pruned_message;
}

void PopulateConversionLog(const Model& model, TocoConversionLog* log) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPSloggingPSconversion_log_utilDTcc mht_11(mht_11_v, 432, "", "./tensorflow/lite/toco/logging/conversion_log_util.cc", "PopulateConversionLog");

  // Get the list of ops after conversion.
  const std::vector<std::string> op_names = GetOperatorNames(model);
  for (const auto& op_name : op_names) {
    log->add_op_list(op_name);
  }

  // Get op signatures.
  TFLITE_PROTO_NS::RepeatedPtrField<std::string> op_signatures;
  GetOpSignatures(model, &op_signatures);
  log->mutable_op_signatures()->CopyFrom(op_signatures);

  // Get op counts by category: custom, built-in or select.
  std::map<std::string, int> custom_ops, select_ops, built_in_ops;
  CountOperatorsByType(model, &built_in_ops, &custom_ops, &select_ops);
  log->mutable_custom_ops()->insert(custom_ops.cbegin(), custom_ops.cend());
  log->mutable_built_in_ops()->insert(built_in_ops.cbegin(),
                                      built_in_ops.cend());
  log->mutable_select_ops()->insert(select_ops.cbegin(), select_ops.cend());

  // Get the model's input and output types.
  TFLITE_PROTO_NS::RepeatedPtrField<std::string> input_types, output_types;
  GetInputAndOutputTypes(model, &input_types, &output_types);
  log->mutable_input_tensor_types()->CopyFrom(input_types);
  log->mutable_output_tensor_types()->CopyFrom(output_types);

  log->set_log_generation_ts(absl::ToUnixMicros(absl::Now()));

  log->set_model_size(model.operators.size());
  log->set_tf_lite_version(GetTfLiteVersion());
  log->set_os_version(GetCachedOSVersion());
  log->set_model_hash(GetModelHash(model));
  // TODO(b/123519920): Populate TOCO error logs.
  // Currently we will focus on external installation of TOCO via pip, where
  // the C++ TOCO binary is invoked via subprocess command, this will make our
  // life easier collecting the error logs emitted by TOCO. However, note that
  // if a user directly invokes the C++ TOCO binary, this log might not be
  // available.
}

}  // namespace toco
