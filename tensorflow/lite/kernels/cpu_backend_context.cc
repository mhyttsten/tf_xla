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
class MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc() {
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

#include "tensorflow/lite/kernels/cpu_backend_context.h"

#include <memory>

#ifdef TFLITE_HAVE_CPUINFO
#include "include/cpuinfo.h"
#endif

#include "public/gemmlowp.h"
#include "ruy/context.h"  // from @ruy
#include "ruy/path.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace {
const int kDefaultNumThreadpoolThreads = 1;

}  // namespace

namespace tflite {

// Use weak symbols if possible to dispatch to deprecated paths.
#if TFLITE_HAS_ATTRIBUTE_WEAK && !defined(__APPLE__)
extern TFLITE_ATTRIBUTE_WEAK bool UseGemmlowpOnX86();
#endif  // defined(TFLITE_HAS_ATTRIBUTE_WEAK) && !(__APPLE__)

// TODO(b/138922878) Enable when Ruy builds on Apple.
#if defined(TFLITE_HAVE_CPUINFO) && !defined(__APPLE__)
CpuBackendContext::CpuInfo::~CpuInfo() {
  if (init_status_ == InitStatus::kInitialized) {
    cpuinfo_deinitialize();
  }
}

bool CpuBackendContext::CpuInfo::EnsureInitialized() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_0(mht_0_v, 222, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::EnsureInitialized");

  if (init_status_ == InitStatus::kNotYetAttempted) {
    init_status_ = Initialize();
  }
  return init_status_ == InitStatus::kInitialized;
}

CpuBackendContext::CpuInfo::InitStatus
CpuBackendContext::CpuInfo::Initialize() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Initialize");

  TFLITE_DCHECK_EQ(init_status_, InitStatus::kNotYetAttempted);
  if (!cpuinfo_initialize()) {
    return InitStatus::kFailed;
  }
  return InitStatus::kInitialized;
}

bool CpuBackendContext::CpuInfo::Avx2Fma() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_2(mht_2_v, 244, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx2Fma");

  return EnsureInitialized() && cpuinfo_has_x86_avx2() &&
         cpuinfo_has_x86_fma3();
}

bool CpuBackendContext::CpuInfo::Avx() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx");

  return EnsureInitialized() && cpuinfo_has_x86_avx();
}

bool CpuBackendContext::CpuInfo::Avx512() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_4(mht_4_v, 259, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx512");

  return EnsureInitialized() && cpuinfo_has_x86_avx512f() &&
         cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512cd() &&
         cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512vl();
}
#else

CpuBackendContext::CpuInfo::~CpuInfo() {}

bool CpuBackendContext::CpuInfo::EnsureInitialized() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_5(mht_5_v, 271, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::EnsureInitialized");

  if (init_status_ == InitStatus::kNotYetAttempted) {
    init_status_ = InitStatus::kInitialized;
  }
  TFLITE_DCHECK_EQ(init_status_, InitStatus::kInitialized);
  return true;
}

bool CpuBackendContext::CpuInfo::Avx2Fma() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_6(mht_6_v, 282, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx2Fma");
 return false; }

bool CpuBackendContext::CpuInfo::Avx() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_7(mht_7_v, 287, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx");
 return false; }

bool CpuBackendContext::CpuInfo::Avx512() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_8(mht_8_v, 292, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuInfo::Avx512");
 return false; }
#endif  // TFLITE_HAVE_CPUINFO

CpuBackendContext* CpuBackendContext::GetFromContext(TfLiteContext* context) {
  auto* external_context = static_cast<ExternalCpuBackendContext*>(
      context->GetExternalContext(context, kTfLiteCpuBackendContext));

  if (external_context == nullptr) {
    TF_LITE_FATAL(
        "ExternalCpuBackendContext isn't properly initialized during TFLite "
        "interpreter initialization.");
  }

  auto* cpu_backend_context = static_cast<CpuBackendContext*>(
      external_context->internal_backend_context());
  if (cpu_backend_context == nullptr) {
    // We do the lazy initialization here for the TfLiteInternalBackendContext
    // that's wrapped inside ExternalCpuBackendContext.
    cpu_backend_context = new CpuBackendContext();
    cpu_backend_context->SetMaxNumThreads(context->recommended_num_threads);
    external_context->set_internal_backend_context(
        std::unique_ptr<TfLiteInternalBackendContext>(cpu_backend_context));
  }

  return cpu_backend_context;
}

CpuBackendContext::CpuBackendContext()
    : TfLiteInternalBackendContext(),
      ruy_context_(new ruy::Context),
      gemmlowp_context_(new gemmlowp::GemmContext) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_9(mht_9_v, 325, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::CpuBackendContext");

  SetMaxNumThreads(kDefaultNumThreadpoolThreads);
// TODO(b/148289189) Remove when clients have transitioned to runtime flag.
#ifdef TFLITE_WITH_RUY_GEMV
  SetUseCaching(true);
#else
  SetUseCaching(false);
#endif
}

CpuBackendContext::~CpuBackendContext() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_10(mht_10_v, 338, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::~CpuBackendContext");
}

void CpuBackendContext::SetMaxNumThreads(int max_num_threads) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_11(mht_11_v, 343, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::SetMaxNumThreads");

  const int target_num_threads =
      max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
  max_num_threads_ = target_num_threads;
  ruy_context_->set_max_num_threads(target_num_threads);
  gemmlowp_context_->set_max_num_threads(target_num_threads);
}

void CpuBackendContext::SetUseCaching(bool flag) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_12(mht_12_v, 354, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::SetUseCaching");
 use_caching_ = flag; }

bool CpuBackendContext::PreferGemmlowpOnX86() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_13(mht_13_v, 359, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::PreferGemmlowpOnX86");

  bool use_gemmlowp_on_x86 = false;
#if defined(TFLITE_X86_PLATFORM) && TFLITE_HAS_ATTRIBUTE_WEAK && \
    !defined(__APPLE__)
  if (::tflite::UseGemmlowpOnX86 != nullptr) {
    use_gemmlowp_on_x86 = ::tflite::UseGemmlowpOnX86();
  }
#endif  // TFLITE_X86_PLATFORM && TFLITE_HAS_ATTRIBUTE_WEAK && !(__APPLE__)
  return use_gemmlowp_on_x86 || !RuyHasAvxOrAbove();
}

bool CpuBackendContext::RuyHasAvxOrAbove() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScpu_backend_contextDTcc mht_14(mht_14_v, 373, "", "./tensorflow/lite/kernels/cpu_backend_context.cc", "CpuBackendContext::RuyHasAvxOrAbove");

  // TODO(b/183178387): Use a proper query to detect AVX/optimized paths.
#if RUY_PLATFORM_X86_ENHANCEMENTS
  return cpuinfo_.Avx() || cpuinfo_.Avx2Fma() || cpuinfo_.Avx512();
#else
  return false;
#endif
}

}  // namespace tflite
