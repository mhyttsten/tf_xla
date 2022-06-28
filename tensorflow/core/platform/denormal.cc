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
class MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/denormal.h"

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/platform.h"

// If we're on gcc 4.8 or older, there's a known bug that prevents the use of
// intrinsics when the architecture is not defined in the flags. See
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57202
#if !defined(__SSE3__) && !defined(__clang__) && \
    (defined(__GNUC__) && (__GNUC__ < 4) ||      \
     ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9)))
#define GCC_WITHOUT_INTRINSICS
#endif
// Only try to use SSE3 instructions if we're on an x86 platform, and it's not
// mobile, and we're not on a known bad gcc version.
#if defined(PLATFORM_IS_X86) && !defined(IS_MOBILE_PLATFORM) && \
    !defined(GCC_WITHOUT_INTRINSICS)
#define X86_DENORM_USE_INTRINSICS
#endif

#ifdef X86_DENORM_USE_INTRINSICS
#include <pmmintrin.h>
#endif

// If on ARM, only access the control register if hardware floating-point
// support is available.
#if defined(PLATFORM_IS_ARM) && defined(__ARM_FP) && (__ARM_FP > 0)
#define ARM_DENORM_AVAILABLE
// Flush-to-zero bit on the ARM floating-point control register.
#define ARM_FPCR_FZ (1 << 24)
#endif

namespace tensorflow {
namespace port {

bool DenormalState::operator==(const DenormalState& other) const {
  return flush_to_zero() == other.flush_to_zero() &&
         denormals_are_zero() == other.denormals_are_zero();
}

bool DenormalState::operator!=(const DenormalState& other) const {
  return !(this->operator==(other));
}

#ifdef ARM_DENORM_AVAILABLE
// Although the ARM ACLE does have a specification for __arm_rsr/__arm_wsr
// for reading and writing to the status registers, they are not implemented
// by GCC, so we need to resort to inline assembly.
static inline void ArmSetFloatingPointControlRegister(uint32_t fpcr) {
#ifdef PLATFORM_IS_ARM64
  __asm__ __volatile__("msr fpcr, %[fpcr]"
                       :
                       : [fpcr] "r"(static_cast<uint64_t>(fpcr)));
#else
  __asm__ __volatile__("vmsr fpscr, %[fpcr]" : : [fpcr] "r"(fpcr));
#endif
}

static inline uint32_t ArmGetFloatingPointControlRegister() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_0(mht_0_v, 243, "", "./tensorflow/core/platform/denormal.cc", "ArmGetFloatingPointControlRegister");

  uint32_t fpcr;
#ifdef PLATFORM_IS_ARM64
  uint64_t fpcr64;
  __asm__ __volatile__("mrs %[fpcr], fpcr" : [fpcr] "=r"(fpcr64));
  fpcr = static_cast<uint32_t>(fpcr64);
#else
  __asm__ __volatile__("vmrs %[fpcr], fpscr" : [fpcr] "=r"(fpcr));
#endif
  return fpcr;
}
#endif  // ARM_DENORM_AVAILABLE

bool SetDenormalState(const DenormalState& state) {
  // For now, we flush denormals only on SSE 3 and ARM.  Other architectures
  // can be added as needed.

#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Restore flags
    _MM_SET_FLUSH_ZERO_MODE(state.flush_to_zero() ? _MM_FLUSH_ZERO_ON
                                                  : _MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(state.denormals_are_zero()
                                    ? _MM_DENORMALS_ZERO_ON
                                    : _MM_DENORMALS_ZERO_OFF);
    return true;
  }
#endif

#ifdef ARM_DENORM_AVAILABLE
  // ARM only has one setting controlling both denormal inputs and outputs.
  if (state.flush_to_zero() == state.denormals_are_zero()) {
    uint32_t fpcr = ArmGetFloatingPointControlRegister();
    if (state.flush_to_zero()) {
      fpcr |= ARM_FPCR_FZ;
    } else {
      fpcr &= ~ARM_FPCR_FZ;
    }
    ArmSetFloatingPointControlRegister(fpcr);
    return true;
  }
#endif

  // Setting denormal handling to the provided state is not supported.
  return false;
}

DenormalState GetDenormalState() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_1(mht_1_v, 293, "", "./tensorflow/core/platform/denormal.cc", "GetDenormalState");

#ifdef X86_DENORM_USE_INTRINSICS
  if (TestCPUFeature(SSE3)) {
    // Save existing flags
    bool flush_zero_mode = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
    bool denormals_zero_mode =
        _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;
    return DenormalState(flush_zero_mode, denormals_zero_mode);
  }
#endif

#ifdef ARM_DENORM_AVAILABLE
  uint32_t fpcr = ArmGetFloatingPointControlRegister();
  if ((fpcr & ARM_FPCR_FZ) != 0) {
    return DenormalState(true, true);
  }
#endif

  return DenormalState(false, false);
}

ScopedRestoreFlushDenormalState::ScopedRestoreFlushDenormalState()
    : denormal_state_(GetDenormalState()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_2(mht_2_v, 318, "", "./tensorflow/core/platform/denormal.cc", "ScopedRestoreFlushDenormalState::ScopedRestoreFlushDenormalState");
}

ScopedRestoreFlushDenormalState::~ScopedRestoreFlushDenormalState() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_3(mht_3_v, 323, "", "./tensorflow/core/platform/denormal.cc", "ScopedRestoreFlushDenormalState::~ScopedRestoreFlushDenormalState");

  SetDenormalState(denormal_state_);
}

ScopedFlushDenormal::ScopedFlushDenormal() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_4(mht_4_v, 330, "", "./tensorflow/core/platform/denormal.cc", "ScopedFlushDenormal::ScopedFlushDenormal");

  SetDenormalState(
      DenormalState(/*flush_to_zero=*/true, /*denormals_are_zero=*/true));
}

ScopedDontFlushDenormal::ScopedDontFlushDenormal() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdenormalDTcc mht_5(mht_5_v, 338, "", "./tensorflow/core/platform/denormal.cc", "ScopedDontFlushDenormal::ScopedDontFlushDenormal");

  SetDenormalState(
      DenormalState(/*flush_to_zero=*/false, /*denormals_are_zero=*/false));
}

}  // namespace port
}  // namespace tensorflow
