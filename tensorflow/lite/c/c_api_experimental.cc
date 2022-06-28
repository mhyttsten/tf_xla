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
class MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc {
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
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_experimental.h"

#include <stdint.h>

#include <memory>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"

extern "C" {

TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterResetVariableTensors");

  return interpreter->impl->ResetVariableTensors();
}

void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterOptionsAddBuiltinOp");

  options->mutable_op_resolver.AddBuiltin(
      static_cast<tflite::BuiltinOperator>(op), registration, min_version,
      max_version);
}

TfLiteInterpreter* TfLiteInterpreterCreateWithSelectedOps(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterCreateWithSelectedOps");

  tflite::MutableOpResolver resolver;
  return tflite::internal::InterpreterCreateWithOpResolver(
      model, optional_options, &resolver);
}

void TfLiteInterpreterOptionsAddCustomOp(TfLiteInterpreterOptions* options,
                                         const char* name,
                                         const TfLiteRegistration* registration,
                                         int32_t min_version,
                                         int32_t max_version) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_3(mht_3_v, 235, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterOptionsAddCustomOp");

  options->mutable_op_resolver.AddCustom(name, registration, min_version,
                                         max_version);
}

void TfLiteInterpreterOptionsSetOpResolver(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data, const char* op,
                                                int version),
    void* op_resolver_user_data) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_4(mht_4_v, 251, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterOptionsSetOpResolver");

  options->op_resolver_callbacks.find_builtin_op = find_builtin_op;
  options->op_resolver_callbacks.find_custom_op = find_custom_op;
  options->op_resolver_callbacks.user_data = op_resolver_user_data;
}

void TfLiteInterpreterOptionsSetUseNNAPI(TfLiteInterpreterOptions* options,
                                         bool enable) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_5(mht_5_v, 261, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterOptionsSetUseNNAPI");

  options->use_nnapi = enable;
}

void TfLiteInterpreterOptionsSetEnableDelegateFallback(
    TfLiteInterpreterOptions* options, bool enable) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_6(mht_6_v, 269, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterOptionsSetEnableDelegateFallback");

  options->enable_delegate_fallback = enable;
}

void TfLiteSetAllowBufferHandleOutput(const TfLiteInterpreter* interpreter,
                                      bool allow_buffer_handle_output) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_7(mht_7_v, 277, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSetAllowBufferHandleOutput");

  interpreter->impl->SetAllowBufferHandleOutput(allow_buffer_handle_output);
}

TfLiteStatus TfLiteInterpreterModifyGraphWithDelegate(
    const TfLiteInterpreter* interpreter, TfLiteDelegate* delegate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_8(mht_8_v, 285, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterModifyGraphWithDelegate");

  return interpreter->impl->ModifyGraphWithDelegate(delegate);
}

int32_t TfLiteInterpreterGetInputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_9(mht_9_v, 293, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterGetInputTensorIndex");

  return interpreter->impl->inputs()[input_index];
}

int32_t TfLiteInterpreterGetOutputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_10(mht_10_v, 301, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterGetOutputTensorIndex");

  return interpreter->impl->outputs()[output_index];
}

int32_t TfLiteInterpreterGetSignatureCount(
    const TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_11(mht_11_v, 309, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterGetSignatureCount");

  return static_cast<int32_t>(interpreter->impl->signature_keys().size());
}

const char* TfLiteInterpreterGetSignatureName(
    const TfLiteInterpreter* interpreter, int32_t signature_index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_12(mht_12_v, 317, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterGetSignatureName");

  int32_t signature_count = TfLiteInterpreterGetSignatureCount(interpreter);
  if (signature_index < 0 || signature_index >= signature_count) {
    return nullptr;
  }
  return interpreter->impl->signature_keys()[signature_index]->c_str();
}

TfLiteSignatureRunner* TfLiteInterpreterGetSignatureRunner(
    const TfLiteInterpreter* interpreter, const char* signature_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("signature_name: \"" + (signature_name == nullptr ? std::string("nullptr") : std::string((char*)signature_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_13(mht_13_v, 330, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteInterpreterGetSignatureRunner");

  tflite::SignatureRunner* signature_runner =
      interpreter->impl->GetSignatureRunner(signature_name);
  return new TfLiteSignatureRunner{signature_runner};
}

size_t TfLiteSignatureRunnerGetInputCount(
    const TfLiteSignatureRunner* signature_runner) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_14(mht_14_v, 340, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetInputCount");

  return signature_runner->impl->input_size();
}

const char* TfLiteSignatureRunnerGetInputName(
    const TfLiteSignatureRunner* signature_runner, const int32_t input_index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_15(mht_15_v, 348, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetInputName");

  int32_t input_count = TfLiteSignatureRunnerGetInputCount(signature_runner);
  if (input_index < 0 || input_index >= input_count) {
    return nullptr;
  }
  return signature_runner->impl->input_names()[input_index];
}

TfLiteStatus TfLiteSignatureRunnerResizeInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name,
    const int* input_dims, int32_t input_dims_size) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_16(mht_16_v, 362, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerResizeInputTensor");

  std::vector<int> dims{input_dims, input_dims + input_dims_size};
  return signature_runner->impl->ResizeInputTensorStrict(input_name, dims);
}

TfLiteStatus TfLiteSignatureRunnerAllocateTensors(
    TfLiteSignatureRunner* signature_runner) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_17(mht_17_v, 371, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerAllocateTensors");

  return signature_runner->impl->AllocateTensors();
}

TfLiteTensor* TfLiteSignatureRunnerGetInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_18(mht_18_v, 380, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetInputTensor");

  return signature_runner->impl->input_tensor(input_name);
}

TfLiteStatus TfLiteSignatureRunnerInvoke(
    TfLiteSignatureRunner* signature_runner) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_19(mht_19_v, 388, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerInvoke");

  return signature_runner->impl->Invoke();
}

size_t TfLiteSignatureRunnerGetOutputCount(
    const TfLiteSignatureRunner* signature_runner) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_20(mht_20_v, 396, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetOutputCount");

  return signature_runner->impl->output_size();
}

const char* TfLiteSignatureRunnerGetOutputName(
    const TfLiteSignatureRunner* signature_runner, int32_t output_index) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_21(mht_21_v, 404, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetOutputName");

  int32_t output_count = TfLiteSignatureRunnerGetOutputCount(signature_runner);
  if (output_index < 0 || output_index >= output_count) {
    return nullptr;
  }
  return signature_runner->impl->output_names()[output_index];
}

const TfLiteTensor* TfLiteSignatureRunnerGetOutputTensor(
    const TfLiteSignatureRunner* signature_runner, const char* output_name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_22(mht_22_v, 417, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerGetOutputTensor");

  return signature_runner->impl->output_tensor(output_name);
}

void TfLiteSignatureRunnerDelete(TfLiteSignatureRunner* signature_runner) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePScPSc_api_experimentalDTcc mht_23(mht_23_v, 424, "", "./tensorflow/lite/c/c_api_experimental.cc", "TfLiteSignatureRunnerDelete");

  delete signature_runner;
}

}  // extern "C"
