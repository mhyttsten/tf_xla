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
class MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc() {
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
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

namespace {

const std::string GetOpName(const OpSignature& op_sig) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "GetOpName");

  if (op_sig.op == tflite::BuiltinOperator_CUSTOM) {
    return op_sig.custom_name;
  }
  return tflite::EnumNamesBuiltinOperator()[op_sig.op];
}

int NumElements(const std::vector<int32_t>& dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_1(mht_1_v, 208, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "NumElements");

  int count = 1;
  for (int i = 0; i < dims.size(); ++i) {
    count *= dims.at(i);
  }
  return count;
}

// Helper functions from
// tensorflow/lite/delegates/gpu/common/model_builder_helper.cc

#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok()) return c; \
  }

template <typename ParamsT>
absl::Status RetrieveBuiltinData(const OpSignature& op_sig,
                                 const ParamsT** tf_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "RetrieveBuiltinData");

  *tf_options = static_cast<const ParamsT*>(op_sig.builtin_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve builtin_data.");
  }
  return absl::OkStatus();
}

template <typename ParamsT>
absl::Status RetrieveCustomInitialData(const OpSignature& op_sig,
                                       const ParamsT** tf_options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_3(mht_3_v, 243, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "RetrieveCustomInitialData");

  *tf_options = static_cast<const ParamsT*>(op_sig.custom_initial_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve custom_initial_data.");
  }
  return absl::OkStatus();
}

absl::Status IsActivationSupported(TfLiteFusedActivation fused_activation) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_4(mht_4_v, 254, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "IsActivationSupported");

  switch (fused_activation) {
    case kTfLiteActNone:
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
    case kTfLiteActSigmoid:
      return absl::OkStatus();
    case kTfLiteActSignBit:
      return absl::UnimplementedError(
          "TfLiteFusedActivation.kTfLiteActSignBit");

      // Do not add default; we want compilation error rather than run-time
      // error.
  }
}

// Returns the number of runtime inputs of the given OpSignature.
// runtime inputs are input tensors which are not constant or optional tensors.
int GetNumberOfRuntimeInputs(const OpSignature& op_sig) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_5(mht_5_v, 277, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "GetNumberOfRuntimeInputs");

  int number_of_runtime_inputs = 0;
  for (auto& input : op_sig.inputs) {
    if (!input.is_const && input.type != kTfLiteNoType) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

// Checks if the given OpSignature has required number of inputs and outputs.
// - required_runtime_inputs: number of inputs which are not constants.
// - required_outputs: number of outputs
absl::Status CheckInputsOutputs(const OpSignature& op_sig,
                                const int required_runtime_inputs,
                                const int required_outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_6(mht_6_v, 295, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckInputsOutputs");

  const int runtime_inputs_from_model = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs_from_model != required_runtime_inputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", required_runtime_inputs,
                     " runtime input tensor(s), but node has ",
                     runtime_inputs_from_model, " runtime input(s)."));
  }
  const int outputs_from_model = op_sig.outputs.size();
  if (outputs_from_model != required_outputs) {
    return absl::InternalError(absl::StrCat("Expected ", required_outputs,
                                            " output tensor(s), but node has ",
                                            outputs_from_model, " output(s)."));
  }
  return absl::OkStatus();
}

// Checks if the given OpSignature has required number of inputs and outputs.
// - required_runtime_inputs: number of inputs which are not constants.
// - required_const_inputs: number of inputs which are constants.
// - required_outputs: number of outputs
absl::Status CheckInputsConstsOutputs(const OpSignature& op_sig,
                                      int required_runtime_inputs,
                                      int required_const_inputs,
                                      int required_outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_7(mht_7_v, 322, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckInputsConstsOutputs");

  int const_inputs_from_model = 0;
  for (auto& input : op_sig.inputs) {
    if (input.is_const) {
      ++const_inputs_from_model;
    }
  }
  if (const_inputs_from_model != required_const_inputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", required_const_inputs,
                     " const input tensor(s), but node has ",
                     const_inputs_from_model, " const input(s)."));
  }
  return CheckInputsOutputs(op_sig, required_runtime_inputs, required_outputs);
}

absl::Status CheckTensorIsAvailable(const OpSignature& op_sig, int idx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_8(mht_8_v, 341, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckTensorIsAvailable");

  // If tensor id is in range, it's guaranteed that it'll be available.
  if (idx >= op_sig.inputs.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Requested index goes beyond array size: ", idx, " vs ",
                     op_sig.inputs.size()));
  }
  return absl::OkStatus();
}

// Checks if the given OpSignature has required number of inputs and outputs for
// convolution operators. The number of input should be either 2 runtime inputs
// or 1 runtime and 1 constant input. The number of output should be one.
absl::Status CheckConvoultionInputOutput(const OpSignature& op_sig) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_9(mht_9_v, 357, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckConvoultionInputOutput");

  const int runtime_inputs = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs > 2) {
    return absl::InternalError(
        absl::StrCat("Expected 1 or 2 input tensor(s), but node has ",
                     runtime_inputs, " runtime inputs."));
  }
  const int runtime_outputs = op_sig.outputs.size();
  if (runtime_outputs != 1) {
    return absl::InternalError(
        absl::StrCat("Expected 1 output tensor(s), but node has ",
                     runtime_outputs, " runtime outputs."));
  }
  if (runtime_inputs == 1) {
    RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
  }
  return absl::OkStatus();
}

absl::Status CheckStrides(int strides_h, int strides_w) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_10(mht_10_v, 379, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckStrides");

  if (strides_h <= 0 || strides_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect stride values: stride_height = ", strides_h,
                     ", stride_width = ", strides_w));
  }
  return absl::OkStatus();
}

absl::Status CheckDilation(int dilation_h, int dilation_w) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_11(mht_11_v, 391, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckDilation");

  if (dilation_h <= 0 || dilation_w <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect dilation values: dilation_height = ", dilation_h,
        ", dilation_width = ", dilation_w));
  }
  return absl::OkStatus();
}

absl::Status CheckStridesAndDilation(int strides_h, int strides_w,
                                     int dilation_h, int dilation_w) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_12(mht_12_v, 404, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckStridesAndDilation");

  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  RETURN_IF_ERROR(CheckDilation(dilation_h, dilation_w));
  return absl::OkStatus();
}

absl::Status CheckKernels(int kernel_h, int kernel_w) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_13(mht_13_v, 413, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckKernels");

  if (kernel_h <= 0 || kernel_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect kernel values: kernel_height = ", kernel_h,
                     ", kernel_width = ", kernel_w));
  }
  return absl::OkStatus();
}

absl::Status CheckKernelsAndStrides(int kernel_h, int kernel_w, int strides_h,
                                    int strides_w) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_14(mht_14_v, 426, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckKernelsAndStrides");

  RETURN_IF_ERROR(CheckKernels(kernel_h, kernel_w));
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  return absl::OkStatus();
}

// Checks if the axes tensor at the given index is a integer32 constant tensor.
absl::Status CheckAxesAreInt32Const(const OpSignature& op_sig, int idx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_15(mht_15_v, 436, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckAxesAreInt32Const");

  auto axes = op_sig.inputs.at(idx);
  if (!axes.is_const) {
    return absl::UnimplementedError(GetOpName(op_sig) +
                                    " is only supported with constant axes.");
  }
  if (axes.type != kTfLiteInt32) {
    return absl::UnimplementedError(absl::StrCat(
        GetOpName(op_sig) + " supports int32 tensor for axes. But node has ",
        TfLiteTypeGetName(axes.type)));
  }
  return absl::OkStatus();
}

absl::Status CheckPooling2DGpuDelegateCompatibility(const OpSignature& op_sig) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_16(mht_16_v, 453, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckPooling2DGpuDelegateCompatibility");

  const TfLitePoolParams* tf_options;
  if (op_sig.custom_initial_data) {  // custom case with indices as a second
                                     // output
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/1,
                                       /*required_outputs=*/2));
  } else {  // common pooling with 1 output
    RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/1,
                                       /*required_outputs=*/1));
  }
  RETURN_IF_ERROR(CheckKernelsAndStrides(
      tf_options->filter_height, tf_options->filter_width,
      tf_options->stride_height, tf_options->stride_width));
  return IsActivationSupported(tf_options->activation);
}

absl::Status CheckDepthwiseConvGpuDelegateCompatibility(
    const OpSignature& op_sig) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_17(mht_17_v, 477, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckDepthwiseConvGpuDelegateCompatibility");

  RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
  const TfLiteDepthwiseConvParams* tf_options;
  RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
  RETURN_IF_ERROR(CheckStridesAndDilation(
      tf_options->stride_height, tf_options->stride_width,
      tf_options->dilation_height_factor, tf_options->dilation_width_factor));
  RETURN_IF_ERROR(IsActivationSupported(tf_options->activation));

  const int depth_multiplier = tf_options->depth_multiplier;
  const auto* input = &op_sig.inputs[0];
  const auto* filter = &op_sig.inputs[1];
  const auto* bias = op_sig.inputs.size() > 2 ? &op_sig.inputs[2] : nullptr;
  const auto* output = &op_sig.outputs[0];
  if (input->dims.size() != 4) {
    return absl::InvalidArgumentError("input.dims.size != 4");
  }
  if (filter->dims.size() != 4) {
    return absl::InvalidArgumentError("filter.dims.size != 4");
  }
  if (output->dims.size() != 4) {
    return absl::InvalidArgumentError("output.dims.size != 4");
  }
  if (input->dims[0] != output->dims[0]) {
    return absl::InvalidArgumentError("input.b != output.b");
  }
  const int input_depth = input->dims[3];
  const int output_depth = output->dims[3];
  if (filter->dims[3] != output_depth) {
    return absl::InvalidArgumentError("filter.i != output.c");
  }
  if (output_depth != input_depth * depth_multiplier) {
    return absl::InvalidArgumentError("output.c != input.c * depth_multiplier");
  }
  if (bias && NumElements(bias->dims) != output_depth) {
    return absl::InvalidArgumentError("bias.size != output.c");
  }
  if (depth_multiplier != 1 && input_depth != 1) {
    return absl::UnimplementedError("depth_multiplier != 1 && input.c != 1");
  }
  return absl::OkStatus();
}

absl::Status CheckCustomOpsGpuDelegateCompatibility(const OpSignature& op_sig) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_18(mht_18_v, 523, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckCustomOpsGpuDelegateCompatibility");

  if (op_sig.custom_name == "Convolution2DTransposeBias") {
    RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
    const TfLiteTransposeConvParams* tf_options;
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(
        CheckStrides(tf_options->stride_height, tf_options->stride_width));
    return absl::OkStatus();
  }
  if (op_sig.custom_name == "MaxPoolingWithArgmax2D") {
    return CheckPooling2DGpuDelegateCompatibility(op_sig);
  }
  if (op_sig.custom_name == "MaxUnpooling2D") {
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/2,
                                       /*required_outputs=*/1));
    const TfLitePoolParams* tf_options;
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckKernelsAndStrides(
        tf_options->filter_height, tf_options->filter_width,
        tf_options->stride_height, tf_options->stride_width));
    return absl::OkStatus();
  }
  if (op_sig.custom_name == "Resampler") {
    return CheckInputsOutputs(op_sig,
                              /*required_runtime_inputs=*/2,
                              /*required_outputs=*/1);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Not supported custom op ", op_sig.custom_name));
}

}  // namespace

// Logics here used to be in TFLiteOperationParser:IsSupported()
// of tensorflow/lite/delegates/gpu/common/model_builder.cc but they're all
// migrated into here.
absl::Status CheckGpuDelegateCompatibility(const OpSignature& op_sig) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_19(mht_19_v, 563, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckGpuDelegateCompatibility");

  TfLiteBuiltinOperator opcode = static_cast<TfLiteBuiltinOperator>(op_sig.op);
  switch (opcode) {
    case kTfLiteBuiltinAdd: {
      if (op_sig.inputs.size() != 2) {
        return absl::UnimplementedError("ADD requires two input tensors.");
      }
      const TfLiteAddParams* tf_options;
      return RetrieveBuiltinData(op_sig, &tf_options);
    }

    case kTfLiteBuiltinAveragePool2d:
      return CheckPooling2DGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinBatchMatmul:
      return CheckInputsOutputs(op_sig, /*required_runtime_inputs=*/2,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinCast:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      if (op_sig.inputs.at(0).type == kTfLiteBool &&
          (op_sig.outputs.at(0).type == kTfLiteFloat16 ||
           op_sig.outputs.at(0).type == kTfLiteFloat32)) {
        return absl::OkStatus();
      } else {
        return absl::UnimplementedError(absl::StrCat(
            "Not supported Cast case. Input type: ",
            TfLiteTypeGetName(op_sig.inputs.at(0).type), " and output type: ",
            TfLiteTypeGetName(op_sig.outputs.at(0).type)));
      }

    case kTfLiteBuiltinConcatenation: {
      const TfLiteConcatenationParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinConv2d: {
      RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
      const TfLiteConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(CheckStridesAndDilation(
          tf_options->stride_height, tf_options->stride_width,
          tf_options->dilation_height_factor,
          tf_options->dilation_width_factor));
      return IsActivationSupported(tf_options->activation);
    }

    case kTfLiteBuiltinDensify:
      return CheckInputsOutputs(op_sig, /*required_runtime_inputs=*/0,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinDepthwiseConv2d:
      return CheckDepthwiseConvGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinDepthToSpace: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteDepthToSpaceParams* d2s_params;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &d2s_params));
      if (d2s_params->block_size == 1) {
        return absl::InvalidArgumentError(
            "DEPTH_TO_SPACE block_size = 1 is a no-op.");
      }
      if (d2s_params->block_size < 1) {
        return absl::InvalidArgumentError(
            "DEPTH_TO_SPACE block_size must be > 1.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinDequantize: {
      const int num_inputs = op_sig.inputs.size();
      const int num_outputs = op_sig.outputs.size();
      if (num_inputs != 1 || num_outputs != 1) {
        return absl::InternalError(absl::StrCat(
            "Expected 1 input & output each from Dequantize, got: %d, %d",
            num_inputs, num_outputs));
      }
      if (op_sig.inputs[0].type == kTfLiteInt16) {
        return absl::UnimplementedError("Unsupported dequantization type.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinFullyConnected: {
      const TfLiteFullyConnectedParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->weights_format !=
          kTfLiteFullyConnectedWeightsFormatDefault) {
        return absl::UnimplementedError(
            absl::StrCat("Unsupported FullyConnected weights format: ",
                         tf_options->weights_format));
      }
      if (GetNumberOfRuntimeInputs(op_sig) > 2) {
        return absl::UnimplementedError(
            "FullyConnected doesn't support more than 2 runtime inputs.");
      }
      if (tf_options->keep_num_dims == true) {
        const auto& input = op_sig.inputs.at(0);
        const auto& output = op_sig.outputs.at(0);
        if (input.dims.size() != output.dims.size()) {
          return absl::UnimplementedError(
              "Input and output dimensions different and FullyConnected "
              "doesn't "
              "support keep_num_dims.");
        }
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinHardSwish:
      return CheckInputsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinLstm: {
      const TfLiteLSTMParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      switch (tf_options->kernel_type) {
        case kTfLiteLSTMFullKernel: {
          const int inputs = op_sig.inputs.size();
          if (inputs != 20 && inputs != 24) {
            return absl::InternalError(
                absl::StrCat("Expected 20 or 24 input tensors, but node has ",
                             inputs, " input(s)."));
          }
          const int runtime_outputs = op_sig.outputs.size();
          if (runtime_outputs != 1) {
            return absl::InternalError(
                absl::StrCat("Expected 1 output tensor, but node has ",
                             runtime_outputs, " output(s)."));
          }
          if (tf_options->activation != kTfLiteActSigmoid &&
              tf_options->activation != kTfLiteActTanh) {
            return absl::UnimplementedError(absl::StrCat(
                "Only sigmoid or tanh activation is supported, but node has ",
                tf_options->activation));
          }
          return absl::OkStatus();
        }
        case kTfLiteLSTMBasicKernel:
          RETURN_IF_ERROR(
              CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/3,
                                       /*required_const_inputs=*/2,
                                       /*required_outputs=*/4));
          if (tf_options->activation != kTfLiteActTanh) {
            return absl::UnimplementedError(
                absl::StrCat("Only TANH activation is supported. but node has ",
                             tf_options->activation));
          }
          if (tf_options->cell_clip != 0.0f) {
            return absl::UnimplementedError("cell_clip is not supported.");
          }
          if (tf_options->proj_clip != 0.0f) {
            return absl::UnimplementedError("proj_clip is not supported.");
          }
          return absl::OkStatus();
      }
    }

    case kTfLiteBuiltinMaxPool2d:
      return CheckPooling2DGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinMean: {
      RETURN_IF_ERROR(CheckInputsConstsOutputs(op_sig,
                                               /*required_runtime_inputs=*/1,
                                               /*required_const_inputs=*/1,
                                               /*required_outputs=*/1));
      return CheckAxesAreInt32Const(op_sig, 1);
    }

    case kTfLiteBuiltinMul: {
      if (op_sig.inputs.size() != 2) {
        return absl::UnimplementedError("MUL requires two input tensors.");
      }
      const auto& input0 = op_sig.inputs.at(0);
      const auto& input1 = op_sig.inputs.at(1);
      if (input0.dims.size() == input1.dims.size()) {
        // this code checks that at least one input of Mul not smaller in all
        // dimensions. Sometimes Mul used for matrix-vector multiplication that
        // we currently don't support. For example input0 HWC(1, 256, 1), input1
        // HWC(1, 1, 256) -> output HWC (1, 256, 256). In this case it can be
        // replaced with Convolution operation.
        bool first_has_smaller_dim = false;
        bool second_has_smaller_dim = false;
        for (int i = 0; i < input0.dims.size(); ++i) {
          if (input0.dims[i] < input1.dims[i]) {
            first_has_smaller_dim = true;
          }
          if (input1.dims[i] < input0.dims[i]) {
            second_has_smaller_dim = true;
          }
        }
        if (first_has_smaller_dim && second_has_smaller_dim) {
          return absl::UnimplementedError(
              "MUL requires one tensor that not less than second in all "
              "dimensions.");
        }
      }
      const TfLiteMulParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return IsActivationSupported(tf_options->activation);
    }

    case kTfLiteBuiltinPack:
      return absl::OkStatus();

    case kTfLiteBuiltinQuantize:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinReluN1To1:
      return absl::OkStatus();

    case kTfLiteBuiltinPrelu:
      return absl::OkStatus();

    case kTfLiteBuiltinReshape:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinSlice: {
      if (op_sig.inputs.size() < 3) {
        return absl::UnimplementedError(
            absl::StrCat("SLICE requires 3 inputs, but node has ",
                         op_sig.inputs.size(), " inputs."));
      }
      const auto& input = op_sig.inputs.at(0);
      if (input.dims.size() != 3 && input.dims.size() != 4) {
        return absl::UnimplementedError(absl::StrCat(
            "SLICE supports for 3 or 4 dimensional tensors only, but node has ",
            input.dims.size(), " dimensional tensors."));
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSoftmax: {
      const TfLiteSoftmaxParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->beta != 1) {
        return absl::UnimplementedError("Softmax.beta != 1 is not supported.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSpaceToDepth: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteSpaceToDepthParams* s2d_params;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &s2d_params));
      if (s2d_params->block_size == 1) {
        return absl::InvalidArgumentError(
            "SPACE_TO_DEPTH block_size = 1 is a no-op.");
      }
      if (s2d_params->block_size < 1) {
        return absl::InvalidArgumentError(
            "SPACE_TO_DEPTH block_size must be > 1.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSplit:
      return absl::OkStatus();

    case kTfLiteBuiltinSplitV:
      return absl::OkStatus();

    case kTfLiteBuiltinStridedSlice: {
      const TfLiteStridedSliceParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->ellipsis_mask) {
        return absl::UnimplementedError(
            "Slice does not support ellipsis_mask.");
      }
      if (tf_options->new_axis_mask) {
        return absl::UnimplementedError(
            "Slice does not support new_axis_mask.");
      }
      if (tf_options->shrink_axis_mask) {
        return absl::UnimplementedError(
            "Slice does not support shrink_axis_mask parameter. ");
      }

      if (op_sig.inputs.size() < 4) {
        return absl::UnimplementedError("STRIDED_SLICE requires 4 inputs.");
      }
      const auto& input = op_sig.inputs.at(0);
      if (input.dims.size() != 3 && input.dims.size() != 4) {
        return absl::UnimplementedError(
            "STRIDED_SLICE supports for 3 or 4 dimensional tensors only.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinTile:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinTranspose:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinTransposeConv: {
      RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
      const TfLiteTransposeConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(
          CheckStrides(tf_options->stride_height, tf_options->stride_width));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinResizeBilinear: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteResizeBilinearParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->align_corners && tf_options->half_pixel_centers) {
        return absl::InternalError(
            "If half_pixel_centers is True, align_corners must be False.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinResizeNearestNeighbor: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteResizeNearestNeighborParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLeakyRelu:
      return absl::OkStatus();

    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceProd:
    case kTfLiteBuiltinSum: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return CheckAxesAreInt32Const(op_sig, 1);
    }

    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinMirrorPad: {
      if (opcode == kTfLiteBuiltinMirrorPad) {
        const TfLiteMirrorPaddingParams* tf_options;
        RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
        if (tf_options->mode !=
            TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect) {
          return absl::InvalidArgumentError(
              absl::StrCat("Only Reflective padding is supported for Mirror "
                           "Pad operation. But node has ",
                           tf_options->mode));
        }
      }
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
      auto& pad_tensor = op_sig.inputs.at(1);
      if (pad_tensor.dims.size() != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid paddings tensor dimension: expected 2 dim, got ",
            pad_tensor.dims.size(), " dim"));
      }
      bool supported = pad_tensor.dims[0] == 3 || pad_tensor.dims[0] == 4;
      if (!supported || pad_tensor.dims[1] != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid paddings tensor shape: expected 4x2 or 3x2, got ",
            pad_tensor.dims[0], "x", pad_tensor.dims[1]));
      }
      return absl::OkStatus();
    }

    // One argument elemenetwise operations
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinCos:
    case kTfLiteBuiltinElu:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinFloor:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinLogistic:  // Sigmoid
    case kTfLiteBuiltinNeg:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinSin:
    case kTfLiteBuiltinSqrt:
    case kTfLiteBuiltinSquare:
    case kTfLiteBuiltinTanh:
      return (CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                       /*required_const_inputs=*/0,
                                       /*required_outputs=*/1));

    // Two arguments elemenetwise operations
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinFloorDiv:
    case kTfLiteBuiltinFloorMod:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinSquaredDifference:
    case kTfLiteBuiltinSub: {
      if (!CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/2,
                                    /*required_const_inputs=*/0,
                                    /*required_outputs=*/1)
               .ok() &&
          !CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                    /*required_const_inputs=*/1,
                                    /*required_outputs=*/1)
               .ok()) {
        return absl::InvalidArgumentError(
            "Op can only handle 1 or 2 operand(s).");
      }
      TfLiteFusedActivation activation = kTfLiteActNone;
      if (opcode == kTfLiteBuiltinDiv) {
        const TfLiteDivParams* tf_options;
        auto status = RetrieveBuiltinData(op_sig, &tf_options);
        activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      } else if (opcode == kTfLiteBuiltinSub) {
        const TfLiteSubParams* tf_options;
        auto status = RetrieveBuiltinData(op_sig, &tf_options);
        activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      }
      return IsActivationSupported(activation);
    }

    case kTfLiteBuiltinCustom:
      return CheckCustomOpsGpuDelegateCompatibility(op_sig);

    default:
      break;
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Not supported op ", tflite::EnumNamesBuiltinOperator()[op_sig.op]));
}

absl::Status CheckGpuDelegateCompatibility(const OperatorCode* op_code,
                                           const Operator* op,
                                           const SubGraph* subgraph,
                                           const Model* model) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_20(mht_20_v, 1029, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckGpuDelegateCompatibility");

  OpSignature op_sig = GetOpSignature(op_code, op, subgraph, model);
  auto status = CheckGpuDelegateCompatibility(op_sig);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

absl::Status CheckGpuDelegateCompatibility(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSgpu_compatibilityDTcc mht_21(mht_21_v, 1043, "", "./tensorflow/lite/tools/versioning/gpu_compatibility.cc", "CheckGpuDelegateCompatibility");

  return CheckGpuDelegateCompatibility(
      GetOpSignature(context, node, registration));
}

}  // namespace tflite
