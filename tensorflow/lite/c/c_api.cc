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
class MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc {
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
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc() {
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
#include "tensorflow/lite/c/c_api.h"

#include <memory>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/create_op_resolver.h"
#include "tensorflow/lite/delegates/interpreter_utils.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"

namespace {
class CallbackErrorReporter : public tflite::ErrorReporter {
 public:
  explicit CallbackErrorReporter(TfLiteErrorReporterCallback callback)
      : callback_(callback) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/c/c_api.cc", "CallbackErrorReporter");
}

  int Report(const char* format, va_list args) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/c/c_api.cc", "Report");

    callback_.error_reporter(callback_.user_data, format, args);
    return 0;
  }

 private:
  TfLiteErrorReporterCallback callback_;
};

/// `CallbackOpResolver` is a (C++) `tflite::OpResolver` that forwards the
/// methods to (C ABI) callback functions from a `TfLiteOpResolverCallbacks`
/// struct.
///
/// The SetCallbacks method must be called before calling any of the FindOp
/// methods.
class CallbackOpResolver : public ::tflite::OpResolver {
 public:
  CallbackOpResolver() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_2(mht_2_v, 229, "", "./tensorflow/lite/c/c_api.cc", "CallbackOpResolver");
}
  void SetCallbacks(
      const struct TfLiteOpResolverCallbacks& op_resolver_callbacks) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_3(mht_3_v, 234, "", "./tensorflow/lite/c/c_api.cc", "SetCallbacks");

    op_resolver_callbacks_ = op_resolver_callbacks;
  }
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_4(mht_4_v, 241, "", "./tensorflow/lite/c/c_api.cc", "FindOp");

    if (op_resolver_callbacks_.find_builtin_op == nullptr) {
      return nullptr;
    }
    return op_resolver_callbacks_.find_builtin_op(
        op_resolver_callbacks_.user_data,
        static_cast<TfLiteBuiltinOperator>(op), version);
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_5(mht_5_v, 253, "", "./tensorflow/lite/c/c_api.cc", "FindOp");

    if (op_resolver_callbacks_.find_custom_op == nullptr) {
      return nullptr;
    }
    return op_resolver_callbacks_.find_custom_op(
        op_resolver_callbacks_.user_data, op, version);
  }

 private:
  CallbackOpResolver(const CallbackOpResolver&) = delete;
  CallbackOpResolver& operator=(const CallbackOpResolver&) = delete;

  struct TfLiteOpResolverCallbacks op_resolver_callbacks_ = {};
};

}  // namespace

extern "C" {

// LINT.IfChange

const char* TfLiteVersion() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_6(mht_6_v, 277, "", "./tensorflow/lite/c/c_api.cc", "TfLiteVersion");
 return TFLITE_VERSION_STRING; }

TfLiteModel* TfLiteModelCreate(const void* model_data, size_t model_size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_7(mht_7_v, 282, "", "./tensorflow/lite/c/c_api.cc", "TfLiteModelCreate");

  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      static_cast<const char*>(model_data), model_size);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("model_path: \"" + (model_path == nullptr ? std::string("nullptr") : std::string((char*)model_path)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_8(mht_8_v, 293, "", "./tensorflow/lite/c/c_api.cc", "TfLiteModelCreateFromFile");

  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(model_path);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model)} : nullptr;
}

void TfLiteModelDelete(TfLiteModel* model) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_9(mht_9_v, 302, "", "./tensorflow/lite/c/c_api.cc", "TfLiteModelDelete");
 delete model; }

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_10(mht_10_v, 307, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterOptionsCreate");

  return new TfLiteInterpreterOptions{};
}

void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_11(mht_11_v, 314, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterOptionsDelete");

  delete options;
}

void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options,
                                           int32_t num_threads) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_12(mht_12_v, 322, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterOptionsSetNumThreads");

  options->num_threads = num_threads;
}

void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* options,
                                         TfLiteDelegate* delegate) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_13(mht_13_v, 330, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterOptionsAddDelegate");

  options->delegates.push_back(delegate);
}

void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_14(mht_14_v, 341, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterOptionsSetErrorReporter");

  options->error_reporter_callback.error_reporter = reporter;
  options->error_reporter_callback.user_data = user_data;
}

TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_15(mht_15_v, 351, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterCreate");

  std::unique_ptr<tflite::MutableOpResolver> resolver =
      tflite::CreateOpResolver();
  return tflite::internal::InterpreterCreateWithOpResolver(
      model, optional_options, resolver.get());
}

void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_16(mht_16_v, 361, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterDelete");

  delete interpreter;
}

int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_17(mht_17_v, 369, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterGetInputTensorCount");

  return static_cast<int32_t>(interpreter->impl->inputs().size());
}

TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_18(mht_18_v, 377, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterGetInputTensor");

  return interpreter->impl->tensor(interpreter->impl->inputs()[input_index]);
}

TfLiteStatus TfLiteInterpreterResizeInputTensor(TfLiteInterpreter* interpreter,
                                                int32_t input_index,
                                                const int* input_dims,
                                                int32_t input_dims_size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_19(mht_19_v, 387, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterResizeInputTensor");

  std::vector<int> dims{input_dims, input_dims + input_dims_size};
  return interpreter->impl->ResizeInputTensor(
      interpreter->impl->inputs()[input_index], dims);
}

TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_20(mht_20_v, 396, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterAllocateTensors");

  return interpreter->impl->AllocateTensors();
}

TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_21(mht_21_v, 403, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterInvoke");

  if (interpreter->enable_delegate_fallback) {
    return tflite::delegates::InterpreterUtils::InvokeWithCPUFallback(
        interpreter->impl.get());
  } else {
    return interpreter->impl->Invoke();
  }
}

int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_22(mht_22_v, 416, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterGetOutputTensorCount");

  return static_cast<int32_t>(interpreter->impl->outputs().size());
}

const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_23(mht_23_v, 424, "", "./tensorflow/lite/c/c_api.cc", "TfLiteInterpreterGetOutputTensor");

  return interpreter->impl->tensor(interpreter->impl->outputs()[output_index]);
}

TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_24(mht_24_v, 431, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorType");
 return tensor->type; }

int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_25(mht_25_v, 436, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorNumDims");

  return tensor->dims->size;
}

int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_26(mht_26_v, 443, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorDim");

  return tensor->dims->data[dim_index];
}

size_t TfLiteTensorByteSize(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_27(mht_27_v, 450, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorByteSize");

  return tensor->bytes;
}

void* TfLiteTensorData(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_28(mht_28_v, 457, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorData");
 return tensor->data.raw; }

const char* TfLiteTensorName(const TfLiteTensor* tensor) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_29(mht_29_v, 462, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorName");

  return tensor->name;
}

TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_30(mht_30_v, 470, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorQuantizationParams");

  return tensor->params;
}

TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor,
                                        const void* input_data,
                                        size_t input_data_size) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_31(mht_31_v, 479, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorCopyFromBuffer");

  if (tensor->bytes != input_data_size) {
    return kTfLiteError;
  }
  memcpy(tensor->data.raw, input_data, input_data_size);
  return kTfLiteOk;
}

TfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor* tensor,
                                      void* output_data,
                                      size_t output_data_size) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_32(mht_32_v, 492, "", "./tensorflow/lite/c/c_api.cc", "TfLiteTensorCopyToBuffer");

  if (tensor->bytes != output_data_size) {
    return kTfLiteError;
  }
  memcpy(output_data, tensor->data.raw, output_data_size);
  return kTfLiteOk;
}

// LINT.ThenChange(//tensorflow/lite/experimental/examples/unity/TensorFlowLitePlugin/Assets/TensorFlowLite/SDK/Scripts/Interpreter.cs)

}  // extern "C"

namespace tflite {
namespace internal {

TfLiteInterpreter* InterpreterCreateWithOpResolver(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options,
    tflite::MutableOpResolver* mutable_resolver) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePScPSc_apiDTcc mht_33(mht_33_v, 512, "", "./tensorflow/lite/c/c_api.cc", "InterpreterCreateWithOpResolver");

  TFLITE_DCHECK_NE(mutable_resolver, nullptr);
  if (!model || !model->impl) {
    return nullptr;
  }

  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (optional_options &&
      optional_options->error_reporter_callback.error_reporter != nullptr) {
    optional_error_reporter.reset(
        new CallbackErrorReporter(optional_options->error_reporter_callback));
  }

  // By default, we use the provided mutable_op_resolver, adding any builtin or
  // custom ops registered with `TfLiteInterpreterOptionsAddBuiltinOp` and/or
  // `TfLiteInterpreterOptionsAddCustomOp`.
  tflite::OpResolver* op_resolver = mutable_resolver;
  if (optional_options) {
    mutable_resolver->AddAll(optional_options->mutable_op_resolver);
  }
  // However, if `TfLiteInterpreterOptionsSetOpResolver` has been called with
  // a non-null callback parameter, then we instead use a
  // `CallbackOpResolver` that will forward to the callbacks provided there.
  CallbackOpResolver callback_op_resolver;
  if (optional_options &&
      (optional_options->op_resolver_callbacks.find_builtin_op != nullptr ||
       optional_options->op_resolver_callbacks.find_custom_op != nullptr)) {
    callback_op_resolver.SetCallbacks(optional_options->op_resolver_callbacks);
    op_resolver = &callback_op_resolver;
  }

  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();
  tflite::InterpreterBuilder builder(model->impl->GetModel(), *op_resolver,
                                     error_reporter);

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    return nullptr;
  }

  if (optional_options) {
    if (optional_options->num_threads !=
        TfLiteInterpreterOptions::kDefaultNumThreads) {
      interpreter->SetNumThreads(optional_options->num_threads);
    }

    if (optional_options->use_nnapi) {
      if (interpreter->ModifyGraphWithDelegate(tflite::NnApiDelegate()) !=
          kTfLiteOk) {
        return nullptr;
      }
    }

    for (auto* delegate : optional_options->delegates) {
      if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
        return nullptr;
      }
    }
  }

  bool enable_delegate_fallback =
      optional_options != nullptr && optional_options->enable_delegate_fallback;

  return new TfLiteInterpreter{model->impl, std::move(optional_error_reporter),
                               std::move(interpreter),
                               enable_delegate_fallback};
}

}  // namespace internal
}  // namespace tflite
