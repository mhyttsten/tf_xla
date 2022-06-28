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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh() {
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


#if (defined(__i386) || defined(_M_IX86) || defined(__x86_64__) || \
     defined(_M_X64))
#define TFLITE_X86_PLATFORM
#endif

#include <memory>

#include "public/gemmlowp.h"
#include "ruy/context.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/external_cpu_backend_context.h"

namespace tflite {

class CpuBackendContext final : public TfLiteInternalBackendContext {
 public:
  static CpuBackendContext* GetFromContext(TfLiteContext* context);

  CpuBackendContext();
  ~CpuBackendContext() override;

  ruy::Context* ruy_context() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "ruy_context");
 return ruy_context_.get(); }

  gemmlowp::GemmContext* gemmlowp_context() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_1(mht_1_v, 214, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "gemmlowp_context");

    return gemmlowp_context_.get();
  }

  // Sets the maximum-number-of-threads-to-use parameter, only as a means of
  // passing around this information.
  void SetMaxNumThreads(int max_num_threads) override;

  int max_num_threads() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_2(mht_2_v, 225, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "max_num_threads");
 return max_num_threads_; }

  void SetUseCaching(bool flag);

  bool use_caching() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_3(mht_3_v, 232, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "use_caching");
 return use_caching_; }

  void ClearCaches() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_4(mht_4_v, 237, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "ClearCaches");
 ruy_context_->ClearPrepackedCache(); }

  // Gemmlowp on x86 is a deprecated path but some clients may still use
  // this path based on link time dependencies.
  bool PreferGemmlowpOnX86();

 private:
  bool RuyHasAvxOrAbove();

  // Copy the wrapper class for cpuinfo from Ruy.
  class CpuInfo final {
   public:
    CpuInfo() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTh mht_5(mht_5_v, 252, "", "./tensorflow/lite/kernels/cpu_backend_context.h", "CpuInfo");
}
    ~CpuInfo();

    // X86 features
    bool Avx();
    bool Avx2Fma();
    bool Avx512();

   private:
    enum class InitStatus {
      kNotYetAttempted,
      kInitialized,
      kFailed,
    };

    InitStatus init_status_ = InitStatus::kNotYetAttempted;

    bool EnsureInitialized();
    InitStatus Initialize();
    CpuInfo(const CpuInfo&) = delete;
    CpuInfo& operator=(const CpuInfo&) = delete;
  };

  // To enable a smooth transition from the current direct usage
  // of the underlying gemmlowp context to going through abstractions
  // (see :cpu_backend_gemm), for now a CpuBackendContext always
  // stores both a gemmlowp context and a ruy context.
  // TODO(b/131416458): Once call sites all go through abstractions,
  // elide what can be elided based on TFLITE_WITH_RUY.
  const std::unique_ptr<ruy::Context> ruy_context_;
  const std::unique_ptr<gemmlowp::GemmContext> gemmlowp_context_;
  CpuInfo cpuinfo_;

  // The maximum of threads used for parallelizing TfLite ops. However,
  // cpu_backend_threadpool::Execute creates as many threads as it's
  // asked to, regardless of this. Typically a call site would query
  // cpu_backend_context->max_num_threads() and used that to determine
  // the number of tasks to create and to give to
  // cpu_backend_threadpool::Execute.
  //
  // This value also gets propagated to back-ends, where it plays the same
  // information-only role.
  int max_num_threads_;
  // For matrix muliplications with constants parameters (i.e. weights), we can
  // sometimes provide speedups by caching the "prepacked" data, for some
  // additional memory cost. This flag permits the user to route all
  // CpuBackendGem operations to a library that permits such an optimization
  // (currently the Ruy library only).
  bool use_caching_;

  CpuBackendContext(const CpuBackendContext&) = delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
