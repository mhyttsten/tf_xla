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

// This header file defines node specs for quantization and the methods to parse
// command line flags to these specs.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh() {
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


#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/tools/optimize/reduced_precision_support.h"

namespace mlir {
namespace quant {

// Stores information about how to quantize a user-specified custom operation.
struct CustomOpInfo {
  std::vector<std::int32_t> quantizable_input_indices;
  bool is_weight_only = false;
  bool no_side_effect = true;
};

using ::tflite::optimize::ReducedPrecisionSupport;
using StringSet = absl::flat_hash_set<std::string>;
using CustomOpMap = std::unordered_map<std::string, CustomOpInfo>;
enum CustomOpUpdateOptions { kINputIndices, kWeightOnly, kNoSideEffect };

struct QuantizationSpecs {
  // Which function this node quant specifications belong to.
  std::string target_func = "main";

  // Whether the quantization passes are triggered for post-training
  // quantization. If it is true, the model input doesn't require user specified
  // input ranges.
  bool post_training_quantization = false;

  // Whether allow dynamic range quantization. This is the easiest quantization
  // mode which doesn't require QAT or sample inputs. But it can only target
  // DT_HALF and DT_QINT8 inference type.
  bool weight_quantization = false;

  // Whether use the MLIR dynamic range quantizer instead of the old TOCO one.
  bool enable_mlir_dynamic_range_quantizer = false;

  // Whether allow weight-only quantization. This scheme quantize weights but
  // will dequantize them back at runtime which is useful to save memory when
  // the kernel support is not yet avilable in lower precisions. Used in MLIR
  // dynamic range quantizer.
  bool weight_only_quantization = false;

  // The minimum number of elements in a weights array required to apply
  // quantization. This is especially useful not to quantize small tensors as
  // it is hard to get performance benefits from them with quantization. Used
  // in MLIR dynamic range quantizer with int8 weight data type.
  int64_t minimum_elements_for_weights = 1024;

  // Calculate scales in float to keep quantized values the same with old TOCO
  // quantizer.
  bool legacy_float_scale = false;

  // When set to true, quantization will be done per-tensor. Currently, this
  // option is only valid when the quantization parameters need to be created by
  // scanning the constant content (post-training quantization or QAT without
  // weight FakeQuant).
  bool disable_per_channel = false;

  // When set to true, the fixed output ranges of the activation ops (tanh,
  // sigmoid, etc.) and the weight constants are not inferred. Then, to quantize
  // these ops, quantization emulation ops should be placed after the ops in the
  // input graph. This flag should be set to false for post-training
  // quantization.
  bool disable_infer_tensor_range = false;

  // The node type when the model is exported. Currently this is limited to
  // DT_FLOAT, DT_HALF, DT_QINT8, and DT_QUINT8. When DT_HALF is used, the
  // `weight_quantization` flag needs to set to true. When DT_QUINT8 is used,
  // the `weight_quantization` flag needs to set to false.
  tensorflow::DataType inference_type = tensorflow::DT_FLOAT;

  // The input and output data type during inference. This flag is only used
  // when `inference_type` is different from DT_FLOAT. This flag can only be set
  // to DT_FLOAT or as same as `inference_type`. If this flag is different
  // from `inference_type`, adaptor ops are inserted as heading and tailing ops
  // in the result model.
  tensorflow::DataType inference_input_type = tensorflow::DT_FLOAT;

  // Input node ranges. These ranges are stored as the same order of function
  // arguments. They are only used when `weight_quantization` is set to false,
  // and the model is required to have quantization parameters, either from
  // quantization aware training or calibration, for the remaining tensors.
  std::vector<std::pair<llvm::Optional<double>, llvm::Optional<double>>>
      input_ranges;

  // The default ranges can be used when a tensor doesn't have quantization
  // parameters and couldn't be quantized. Used only for latency tests.
  std::pair<llvm::Optional<double>, llvm::Optional<double>> default_ranges;

  // A serialized "QuantizationInfo" object to specify value ranges for some of
  // the tensors with known names.
  std::string serialized_quant_stats = "";

  // A bitmask to encode support for reduced precision inference in the model.
  ReducedPrecisionSupport support_mask = ReducedPrecisionSupport::None;

  // Whether run the passes to propagate the quantization parameters and graph
  // rewrites. Returns false if the inference_type is DT_FLOAT or
  // `weight_quantization` flag is set.
  bool RunPropagationAndRewriteQuantizationPasses() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh mht_0(mht_0_v, 298, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.h", "RunPropagationAndRewriteQuantizationPasses");

    return inference_type != tensorflow::DT_FLOAT && !weight_quantization;
  }

  // TODO(b/202075505): make implicit weight type clearer
  // Whether run the passes and graph rewrites for dynamic range quantization.
  bool RunAndRewriteDynamicRangeQuantizationPasses() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh mht_1(mht_1_v, 307, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.h", "RunAndRewriteDynamicRangeQuantizationPasses");

    // TODO(b/201389248): add condition that symmetric, signed, int8 only
    // If fail, log will appear to let user know nothing happened.
    bool dynamic_range_quantize =
        (inference_type != tensorflow::DT_FLOAT) && weight_quantization &&
        !post_training_quantization && !disable_infer_tensor_range &&
        enable_mlir_dynamic_range_quantizer;
    return dynamic_range_quantize;
  }

  // Whether this inference type represents a signed storage type.
  bool IsSignedInferenceType() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh mht_2(mht_2_v, 321, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.h", "IsSignedInferenceType");

    switch (inference_type) {
      case tensorflow::DT_QUINT8:
      case tensorflow::DT_QUINT16:
        return false;
      default:
        return true;
    }
  }

  // Gets the width of this quantization type. Returns 0 if it isn't a
  // quantization type.
  int64_t GetQuantizationTypeWidth() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTh mht_3(mht_3_v, 336, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.h", "GetQuantizationTypeWidth");

    switch (inference_type) {
      case tensorflow::DT_QINT8:
      case tensorflow::DT_QUINT8:
        return 8;
      case tensorflow::DT_QINT16:
      case tensorflow::DT_QUINT16:
        return 16;
      case tensorflow::DT_QINT32:
        return 32;
      default:
        return 0;
    }
  }

  // Whether add the NumericVerify ops to verify numbers before and after
  // quantization.
  bool verify_numeric = false;
  // Whether to add verification for layer by layer, or on whole model. When
  // disabled (per-layer) float and quantized ops will be run from same input
  // (output of previous quantized layer). When enabled, float and quantized ops
  // will run with respective float and quantized output of previous ops.
  bool whole_model_verify = false;

  // Whether to use fake quant attributes to calculate quantization parameters.
  bool use_fake_quant_num_bits = false;

  // Names of ops to block from quantization. Used in QuantizePass.
  // For dynamic range quantization, ops in blocklist are quantized in weight-
  // only manner.
  StringSet ops_blocklist;

  // Names of locations to block from quantization. Used in QuantizePass.
  StringSet nodes_blocklist;

  // Map from custom op code to custom op quantization information.
  // For dynamic range quantization, among the custom ops in the graph those
  // specified in this map are subject to quantization.
  CustomOpMap custom_map;
};

// Parses the command line flag strings to the CustomOpMap specification.
void ParseCustomOpSpecs(absl::string_view node_names,
                        const CustomOpUpdateOptions& update_option,
                        CustomOpMap& custom_op_map);

// Parses the command line flag strings to the quantization specification for
// input arrays of a graph. The array names are not stored in the spec, and will
// be matched by position. Returns true if failed.
bool ParseInputNodeQuantSpecs(absl::string_view node_names,
                              absl::string_view min_values,
                              absl::string_view max_values,
                              absl::string_view inference_type,
                              QuantizationSpecs* quant_specs);

// Gets the quantization specification for input arrays. The array names are not
// stored in the spec, and will be matched by position. The min/max will be
// ignored if the inference_type isn't a quantized type. Returns true if failed.
bool GetInputNodeQuantSpecs(
    const std::vector<std::string>& node_names,
    const std::vector<llvm::Optional<double>>& node_mins,
    const std::vector<llvm::Optional<double>>& node_maxs,
    tensorflow::DataType inference_type, QuantizationSpecs* quant_specs);

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONFIG_H_
