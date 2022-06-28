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
class MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc {
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
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc() {
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

#include <stddef.h>
#include <stdlib.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ruy/denormal.h"  // from @ruy
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"

namespace tflite {

TfLiteStatus Interpreter::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation, int64_t flags) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetCustomAllocationForTensor");

  return primary_subgraph().SetCustomAllocationForTensor(tensor_index,
                                                         allocation, flags);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::ReleaseNonPersistentMemory");

  // TODO(b/138790287): We could do this for all subgraphs whose tensors have
  // been allocated. However, AllocateTensors() relies on Control Flow ops to
  // allocate tensors on 'children' subgraphs. Revisit this if required.
  return primary_subgraph().ReleaseNonPersistentMemory();
}

TfLiteStatus Interpreter::ResetVariableTensors() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::ResetVariableTensors");

  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->ResetVariableTensors());
  }
  return kTfLiteOk;
}

void Interpreter::SetAllowFp16PrecisionForFp32(bool allow) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_3(mht_3_v, 236, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetAllowFp16PrecisionForFp32");

  for (auto& subgraph : subgraphs_) {
    subgraph->context()->allow_fp32_relax_to_fp16 = allow;
  }
}

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_4(mht_4_v, 248, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetCancellationFunction");

  for (auto& subgraph : subgraphs_) {
    subgraph->SetCancellationFunction(data, check_cancelled_func);
  }
}

bool Interpreter::IsCancelled() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_5(mht_5_v, 257, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::IsCancelled");
 return primary_subgraph().IsCancelled(); }

TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_6(mht_6_v, 262, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::ModifyGraphWithDelegate");

  return ModifyGraphWithDelegateImpl(delegate);
}

bool Interpreter::HasDelegates() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_7(mht_7_v, 269, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::HasDelegates");
 return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_8(mht_8_v, 276, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetBufferHandle");

  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = primary_subgraph().tensor(tensor_index);

  TF_LITE_ENSURE(context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(context_, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(context_, tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_9(mht_9_v, 298, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::GetBufferHandle");

  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = primary_subgraph().tensor(tensor_index);

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::SetProfiler(Profiler* profiler) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_10(mht_10_v, 311, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetProfiler");

  // Release resources occupied by owned_profiler_ which is replaced by
  // caller-owned profiler.
  owned_profiler_.reset(nullptr);
  installed_profiler_ = profiler;
  SetSubgraphProfiler();
}

void Interpreter::SetProfiler(std::unique_ptr<Profiler> profiler) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_11(mht_11_v, 322, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::SetProfiler");

  SetProfilerImpl(std::move(profiler));
}

Profiler* Interpreter::GetProfiler() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_12(mht_12_v, 329, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::GetProfiler");

  return primary_subgraph().GetProfiler();
}

TfLiteStatus Interpreter::ApplyOptions(InterpreterOptions* options) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_13(mht_13_v, 336, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::ApplyOptions");

  return ApplyOptionsImpl(options);
}

SignatureRunner* Interpreter::GetSignatureRunner(const char* signature_key) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("signature_key: \"" + (signature_key == nullptr ? std::string("nullptr") : std::string((char*)signature_key)) + "\"");
   MHTracer_DTPStensorflowPSlitePSinterpreter_experimentalDTcc mht_14(mht_14_v, 344, "", "./tensorflow/lite/interpreter_experimental.cc", "Interpreter::GetSignatureRunner");

  auto iter = signature_runner_map_.find(signature_key);
  if (iter != signature_runner_map_.end()) {
    return &(iter->second);
  }

  // Default delegates are applied once for all subgraphs. Only returns error
  // when the status is kTfLiteError. For other statuses, it will fall back to
  // the default implementation.
  if (ApplyLazyDelegateProviders() == kTfLiteError) {
    return nullptr;
  }

  for (const auto& signature : signature_defs_) {
    if (signature.signature_key == signature_key) {
      auto status = signature_runner_map_.insert(
          {signature_key,
           SignatureRunner(&signature, subgraph(signature.subgraph_index))});
      return &(status.first->second);
    }
  }
  return nullptr;
}

}  // namespace tflite
