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
#ifndef TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_
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
class MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh {
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
   MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh() {
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


#include <memory>
#include <utility>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// This is the base class for TF Lite internal backend contexts (like a
// RUY-based cpu backend context class). A derived internal backend context is
// generally a collection of utilities (i.e. a thread pool etc.) for TF Lite to
// use certain kernel libraries, such as Gemmlowp, RUY, etc., to implement TF
// Lite operators.
class TfLiteInternalBackendContext {
 public:
  virtual ~TfLiteInternalBackendContext() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh mht_0(mht_0_v, 201, "", "./tensorflow/lite/external_cpu_backend_context.h", "~TfLiteInternalBackendContext");
}

  // Set the maximum number of threads that could be used for parallelizing
  // TfLite computation.
  virtual void SetMaxNumThreads(int max_num_threads) = 0;

  // A context may internally cache prepacked versions of constant tensors for
  // faster computation. This function will clear any caches on the context.
  virtual void ClearCaches() = 0;
};

// This TfLiteExternalContext-derived class is the default
// 'kTfLiteCpuBackendContext'-typed context that's used internally in TF Lite
// framework. The primary purpose of having this class is to allow the same cpu
// backend context to be sharable among a set of TF Lite interpreters so that
// certain system costs are saved, like saving the cost of having multiple
// thread pools in each separate cpu backend context etc..
//
// Note: as of 2019/07/19, such context sharing among a set of interpreters will
// break the execution if these interpreters are invoked simultaneously. It
// works only when these context-sharing interpreters are invoked in a
// serialized way. Here's an example to illustrate the context sharing among 2
// TF Lite interpreters:
//
//  TfLiteExternalContext* global_ctxt = new ExternalCpuBackendContext();
//  interpreter1 = /*...*/;
//  interpreter1->SetExternalContext(kTfLiteCpuBackendContext, global_ctxt);
//  interpreter2 = /*...*/;
//  interpreter2->SetExternalContext(kTfLiteCpuBackendContext, global_ctxt);
//
//  interpreter1->SetNumThreads(2);
//  interpreter1->Invoke();
//
//  interpreter2->SetNumThreads(4);
//  interpreter2->Invoke();
//
// After sharing the context, calling 'SetNumThreads' on any of the
// context-sharing interpreters will have the global impact as it also refreshes
// the #thread info in the global cpu backend context (i.e. 'global_ctxt' above)
// that affects how much parallelism an interpreter invocation will use.
// Therefore, if different number of threads are used among different
// interpreters, don't call 'SetNumThreads' consecutively but call it
// separately between each interpreter's invocation as illustrated above.
//
// Note: it is the responsibility of the user of this context (i.e. a
// TFLiteInterpreter) to clear any state from the internal backend
// context if/when the interpreter no longer needs the shared context.
// See, e.g., TFLiteInterpreter destructor clears caches in the case of a
// shared ExternalCpuBackendContext.
class ExternalCpuBackendContext : public TfLiteExternalContext {
 public:
  ExternalCpuBackendContext();
  ~ExternalCpuBackendContext() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh mht_1(mht_1_v, 256, "", "./tensorflow/lite/external_cpu_backend_context.h", "~ExternalCpuBackendContext");
}

  void set_internal_backend_context(
      std::unique_ptr<TfLiteInternalBackendContext> internal_backend_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh mht_2(mht_2_v, 262, "", "./tensorflow/lite/external_cpu_backend_context.h", "set_internal_backend_context");

    internal_backend_context_ = std::move(internal_backend_context);
  }

  TfLiteInternalBackendContext* internal_backend_context() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexternal_cpu_backend_contextDTh mht_3(mht_3_v, 269, "", "./tensorflow/lite/external_cpu_backend_context.h", "internal_backend_context");

    return internal_backend_context_.get();
  }

 private:
  // Note the actual internal backend context object is lazily initialized.
  std::unique_ptr<TfLiteInternalBackendContext> internal_backend_context_;

  ExternalCpuBackendContext(const ExternalCpuBackendContext&) = delete;
  ExternalCpuBackendContext& operator=(const ExternalCpuBackendContext&) =
      delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXTERNAL_CPU_BACKEND_CONTEXT_H_
