/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
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
class MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh {
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
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh() {
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


#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/util.h"

namespace toco {

namespace tflite {

enum class QuantizedBufferType { NONE, INT8, FLOAT16 };

// The parameters for exporting a TFLite model.
struct ExportParams {
  bool allow_custom_ops = false;
  bool allow_dynamic_tensors = true;
  bool enable_select_tf_ops = false;
  QuantizedBufferType quantize_weights = QuantizedBufferType::NONE;
  // Whether to use per-tensor (false) or per-channel (true) for hybrid quant.
  bool disable_per_channel = false;
};

// Transform the given tf.mini model into a TF Lite flatbuffer and deposit the
// result in the given string.
tensorflow::Status Export(const Model& model, std::string* output_file_contents,
                          const ExportParams& params);

// Export API with custom TFLite operator mapping.
tensorflow::Status Export(
    const Model& model, std::string* output_file_contents,
    const ExportParams& params,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type);

// This is for backward-compatibility.
// TODO(ycling): Remove the deprecated entry functions.
inline void Export(const Model& model, bool allow_custom_ops,
                   bool quantize_weights, std::string* output_file_contents) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/toco/tflite/export.h", "Export");

  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights =
      quantize_weights ? QuantizedBufferType::INT8 : QuantizedBufferType::NONE;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

// This is for backward-compatibility.
// TODO(ycling): Remove the deprecated entry functions.
inline void Export(
    const Model& model, bool allow_custom_ops, bool quantize_weights,
    std::string* output_file_contents,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_1(mht_1_v, 238, "", "./tensorflow/lite/toco/tflite/export.h", "Export");

  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights =
      quantize_weights ? QuantizedBufferType::INT8 : QuantizedBufferType::NONE;
  auto status = Export(model, output_file_contents, params, ops_by_type);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

// This is for backward-compatibility.
// TODO(ycling): Remove the deprecated entry functions.
inline void Export(const Model& model, std::string* output_file_contents) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_2(mht_2_v, 252, "", "./tensorflow/lite/toco/tflite/export.h", "Export");

  ExportParams params;
  params.allow_custom_ops = true;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

namespace details {

// A map from tensor name to its final position in the TF Lite buffer.
using TensorsMap = std::unordered_map<std::string, int>;

// A key to identify an operator.
// Only when `type` is `kUnsupported`, `custom_code` is filled to
// identify which operation is used.
class OperatorKey {
 public:
  OperatorKey() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_3(mht_3_v, 272, "", "./tensorflow/lite/toco/tflite/export.h", "OperatorKey");
}

  // Construct OperatorKey by Toco op.
  OperatorKey(
      const ::toco::OperatorSignature& op_signature,
      const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
      bool enable_select_tf_ops);

  // Construct OperatorKey by type, custom code and version.
  // Note that this construct doesn't set the additional information including
  // `is_custom_op`, `is_flex_op`, `is_unsupported_flex_op`.
  OperatorKey(::tflite::BuiltinOperator type, const std::string& custom_code,
              int version)
      : type_(type), custom_code_(custom_code), version_(version) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("custom_code: \"" + custom_code + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_4(mht_4_v, 289, "", "./tensorflow/lite/toco/tflite/export.h", "OperatorKey");
}

  // Only `type`, `custom_code` and `version` is used to compute hash and
  // identity.
  ::tflite::BuiltinOperator type() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_5(mht_5_v, 296, "", "./tensorflow/lite/toco/tflite/export.h", "type");
 return type_; }
  const std::string& custom_code() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_6(mht_6_v, 300, "", "./tensorflow/lite/toco/tflite/export.h", "custom_code");
 return custom_code_; }
  int version() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_7(mht_7_v, 304, "", "./tensorflow/lite/toco/tflite/export.h", "version");
 return version_; }

  // The attributes below are not used to compute hash and identity.
  //
  // Return true if the op is a custom op. Note it will return false for Flex
  // ops.
  bool is_custom_op() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_8(mht_8_v, 313, "", "./tensorflow/lite/toco/tflite/export.h", "is_custom_op");
 return is_custom_op_; }
  // Return true if the op is a Flex op.
  bool is_flex_op() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_9(mht_9_v, 318, "", "./tensorflow/lite/toco/tflite/export.h", "is_flex_op");
 return is_flex_op_; }
  // Return true if the op is a Flex op but it's knwon that the op is not
  // supported by Flex runtime.
  bool is_unsupported_flex_op() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_10(mht_10_v, 324, "", "./tensorflow/lite/toco/tflite/export.h", "is_unsupported_flex_op");
 return is_unsupported_flex_op_; }
  // Return the original TensorFlow op name for a Flex op.
  const std::string& flex_tensorflow_op() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_11(mht_11_v, 329, "", "./tensorflow/lite/toco/tflite/export.h", "flex_tensorflow_op");
 return flex_tensorflow_op_; }

  bool operator<(const OperatorKey& other) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStocoPStflitePSexportDTh mht_12(mht_12_v, 334, "", "./tensorflow/lite/toco/tflite/export.h", "operator<");

    if (type_ < other.type_)
      return true;
    else if (type_ > other.type_)
      return false;
    else if (custom_code_ < other.custom_code_)
      return true;
    else if (custom_code_ > other.custom_code_)
      return false;
    else
      return version_ < other.version_;
  }

  bool operator==(const OperatorKey& other) const {
    return type_ == other.type_ && custom_code_ == other.custom_code_ &&
           version_ == other.version_;
  }

  struct Hash {
    size_t operator()(const OperatorKey& key) const {
      return ::tflite::CombineHashes(
          {std::hash<size_t>()(static_cast<size_t>(key.type())),
           std::hash<std::string>()(key.custom_code()),
           std::hash<int>()(key.version())});
    }
  };

 private:
  ::tflite::BuiltinOperator type_ = ::tflite::BuiltinOperator_CUSTOM;
  std::string custom_code_;
  int version_ = 1;

  bool is_custom_op_ = false;
  bool is_flex_op_ = false;
  bool is_unsupported_flex_op_ = false;
  // The original TensorFlow op name for the flex op. Filled only when
  // `is_flex_op` is true.
  std::string flex_tensorflow_op_;
};

// A map from OperatorKey to its final position in the TF Lite buffer.
using OperatorsMap = std::unordered_map<OperatorKey, int, OperatorKey::Hash>;

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map);
void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    bool enable_select_tf_ops);

}  // namespace details
}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
