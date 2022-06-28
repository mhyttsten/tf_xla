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
class MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc() {
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

#include "tensorflow/core/platform/cpu_info.h"

#include "absl/base/call_once.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#if defined(PLATFORM_IS_X86)
#include <mutex>  // NOLINT
#endif

// SIMD extension querying is only available on x86.
#ifdef PLATFORM_IS_X86
#ifdef PLATFORM_WINDOWS
// Visual Studio defines a builtin function for CPUID, so use that if possible.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  {                                        \
    int cpu_info[4] = {-1};                \
    __cpuidex(cpu_info, a_inp, c_inp);     \
    a = cpu_info[0];                       \
    b = cpu_info[1];                       \
    c = cpu_info[2];                       \
    d = cpu_info[3];                       \
  }
#else
// Otherwise use gcc-format assembler to implement the underlying instructions.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  asm("mov %%rbx, %%rdi\n"                 \
      "cpuid\n"                            \
      "xchg %%rdi, %%rbx\n"                \
      : "=a"(a), "=D"(b), "=c"(c), "=d"(d) \
      : "a"(a_inp), "2"(c_inp))
#endif
#endif

namespace tensorflow {
namespace port {
namespace {

#ifdef PLATFORM_IS_X86
class CPUIDInfo;
void InitCPUIDInfo();

CPUIDInfo *cpuid = nullptr;

#ifdef PLATFORM_WINDOWS
// Visual Studio defines a builtin function, so use that if possible.
int GetXCR0EAX() { return _xgetbv(0); }
#else
int GetXCR0EAX() {
  int eax, edx;
  asm("XGETBV" : "=a"(eax), "=d"(edx) : "c"(0));
  return eax;
}
#endif

// Structure for basic CPUID info
class CPUIDInfo {
 public:
  CPUIDInfo()
      : have_adx_(0),
        have_aes_(0),
        have_amx_bf16_(0),
        have_amx_int8_(0),
        have_amx_tile_(0),
        have_avx_(0),
        have_avx2_(0),
        have_avx512f_(0),
        have_avx512cd_(0),
        have_avx512er_(0),
        have_avx512pf_(0),
        have_avx512vl_(0),
        have_avx512bw_(0),
        have_avx512dq_(0),
        have_avx512vbmi_(0),
        have_avx512ifma_(0),
        have_avx512_4vnniw_(0),
        have_avx512_4fmaps_(0),
        have_avx512_bf16_(0),
        have_avx512_vnni_(0),
        have_avx_vnni_(0),
        have_bmi1_(0),
        have_bmi2_(0),
        have_cmov_(0),
        have_cmpxchg16b_(0),
        have_cmpxchg8b_(0),
        have_f16c_(0),
        have_fma_(0),
        have_mmx_(0),
        have_pclmulqdq_(0),
        have_popcnt_(0),
        have_prefetchw_(0),
        have_prefetchwt1_(0),
        have_rdrand_(0),
        have_rdseed_(0),
        have_smap_(0),
        have_sse_(0),
        have_sse2_(0),
        have_sse3_(0),
        have_sse4_1_(0),
        have_sse4_2_(0),
        have_ssse3_(0),
        have_hypervisor_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_0(mht_0_v, 286, "", "./tensorflow/core/platform/cpu_info.cc", "CPUIDInfo");
}

  static void Initialize() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_1(mht_1_v, 291, "", "./tensorflow/core/platform/cpu_info.cc", "Initialize");

    // Initialize cpuid struct
    CHECK(cpuid == nullptr) << __func__ << " ran more than once";
    cpuid = new CPUIDInfo;

    uint32 eax, ebx, ecx, edx;

    // Get vendor string (issue CPUID with eax = 0)
    GETCPUID(eax, ebx, ecx, edx, 0, 0);
    cpuid->vendor_str_.append(reinterpret_cast<char *>(&ebx), 4);
    cpuid->vendor_str_.append(reinterpret_cast<char *>(&edx), 4);
    cpuid->vendor_str_.append(reinterpret_cast<char *>(&ecx), 4);

    // To get general information and extended features we send eax = 1 and
    // ecx = 0 to cpuid.  The response is returned in eax, ebx, ecx and edx.
    // (See Intel 64 and IA-32 Architectures Software Developer's Manual
    // Volume 2A: Instruction Set Reference, A-M CPUID).
    GETCPUID(eax, ebx, ecx, edx, 1, 0);

    cpuid->model_num_ = static_cast<int>((eax >> 4) & 0xf);
    cpuid->family_ = static_cast<int>((eax >> 8) & 0xf);

    cpuid->have_aes_ = (ecx >> 25) & 0x1;
    cpuid->have_cmov_ = (edx >> 15) & 0x1;
    cpuid->have_cmpxchg16b_ = (ecx >> 13) & 0x1;
    cpuid->have_cmpxchg8b_ = (edx >> 8) & 0x1;
    cpuid->have_mmx_ = (edx >> 23) & 0x1;
    cpuid->have_pclmulqdq_ = (ecx >> 1) & 0x1;
    cpuid->have_popcnt_ = (ecx >> 23) & 0x1;
    cpuid->have_rdrand_ = (ecx >> 30) & 0x1;
    cpuid->have_sse2_ = (edx >> 26) & 0x1;
    cpuid->have_sse3_ = ecx & 0x1;
    cpuid->have_sse4_1_ = (ecx >> 19) & 0x1;
    cpuid->have_sse4_2_ = (ecx >> 20) & 0x1;
    cpuid->have_sse_ = (edx >> 25) & 0x1;
    cpuid->have_ssse3_ = (ecx >> 9) & 0x1;
    cpuid->have_hypervisor_ = (ecx >> 31) & 1;

    const uint64 xcr0_xmm_mask = 0x2;
    const uint64 xcr0_ymm_mask = 0x4;
    const uint64 xcr0_maskreg_mask = 0x20;
    const uint64 xcr0_zmm0_15_mask = 0x40;
    const uint64 xcr0_zmm16_31_mask = 0x80;

    const uint64 xcr0_avx_mask = xcr0_xmm_mask | xcr0_ymm_mask;
    const uint64 xcr0_avx512_mask = xcr0_avx_mask | xcr0_maskreg_mask |
                                    xcr0_zmm0_15_mask | xcr0_zmm16_31_mask;

    const bool have_avx =
        // Does the OS support XGETBV instruction use by applications?
        ((ecx >> 27) & 0x1) &&
        // Does the OS save/restore XMM and YMM state?
        ((GetXCR0EAX() & xcr0_avx_mask) == xcr0_avx_mask) &&
        // Is AVX supported in hardware?
        ((ecx >> 28) & 0x1);

    const bool have_avx512 =
        // Does the OS support XGETBV instruction use by applications?
        ((ecx >> 27) & 0x1) &&
        // Does the OS save/restore ZMM state?
        ((GetXCR0EAX() & xcr0_avx512_mask) == xcr0_avx512_mask);

    cpuid->have_avx_ = have_avx;
    cpuid->have_fma_ = have_avx && ((ecx >> 12) & 0x1);
    cpuid->have_f16c_ = have_avx && ((ecx >> 29) & 0x1);

    // Get standard level 7 structured extension features (issue CPUID with
    // eax = 7 and ecx = 0), which is required to check for AVX2 support as
    // well as other Haswell (and beyond) features.  (See Intel 64 and IA-32
    // Architectures Software Developer's Manual Volume 2A: Instruction Set
    // Reference, A-M CPUID).
    GETCPUID(eax, ebx, ecx, edx, 7, 0);
    const uint32 kMaxNumSubLeaves = eax;

    cpuid->have_adx_ = (ebx >> 19) & 0x1;
    cpuid->have_avx2_ = have_avx && ((ebx >> 5) & 0x1);
    cpuid->have_bmi1_ = (ebx >> 3) & 0x1;
    cpuid->have_bmi2_ = (ebx >> 8) & 0x1;
    cpuid->have_prefetchwt1_ = ecx & 0x1;
    cpuid->have_rdseed_ = (ebx >> 18) & 0x1;
    cpuid->have_smap_ = (ebx >> 20) & 0x1;

    cpuid->have_avx512f_ = have_avx512 && ((ebx >> 16) & 0x1);
    cpuid->have_avx512cd_ = have_avx512 && ((ebx >> 28) & 0x1);
    cpuid->have_avx512er_ = have_avx512 && ((ebx >> 27) & 0x1);
    cpuid->have_avx512pf_ = have_avx512 && ((ebx >> 26) & 0x1);
    cpuid->have_avx512vl_ = have_avx512 && ((ebx >> 31) & 0x1);
    cpuid->have_avx512bw_ = have_avx512 && ((ebx >> 30) & 0x1);
    cpuid->have_avx512dq_ = have_avx512 && ((ebx >> 17) & 0x1);
    cpuid->have_avx512vbmi_ = have_avx512 && ((ecx >> 1) & 0x1);
    cpuid->have_avx512ifma_ = have_avx512 && ((ebx >> 21) & 0x1);
    cpuid->have_avx512_4vnniw_ = have_avx512 && ((edx >> 2) & 0x1);
    cpuid->have_avx512_4fmaps_ = have_avx512 && ((edx >> 3) & 0x1);
    cpuid->have_avx512_vnni_ = have_avx512 && ((ecx >> 11) & 0x1);

    // The latest Intel 64 and IA-32 Architectures Software Developer's Manual
    // Volume 2A (December 2021) does not have information on AMX yet. We use
    // the information from Xbyak in oneDNN.
    // https://github.com/oneapi-src/oneDNN/blob/acf8d214cedfe7e24c9446bacc1f9f648c9273f8/src/cpu/x64/xbyak/xbyak_util.h#L536-L538
    cpuid->have_amx_tile_ = (edx >> 24) & 0x1;
    cpuid->have_amx_int8_ = (edx >> 25) & 0x1;
    cpuid->have_amx_bf16_ = (edx >> 22) & 0x1;

    // Get more Structured Extended Feature info by issuing CPUID with
    // sub-leaf = 1 (eax = 7, ecx = 1)
    if (kMaxNumSubLeaves >= 1) {
      GETCPUID(eax, ebx, ecx, edx, 7, 1);
      cpuid->have_avx_vnni_ = (eax >> 4) & 0x1;
      cpuid->have_avx512_bf16_ = have_avx512 && ((eax >> 5) & 0x1);
    }
  }

  static bool TestFeature(CPUFeature feature) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_2(mht_2_v, 406, "", "./tensorflow/core/platform/cpu_info.cc", "TestFeature");

    InitCPUIDInfo();
    // clang-format off
    switch (feature) {
      case ADX:           return cpuid->have_adx_;
      case AES:           return cpuid->have_aes_;
      case AMX_BF16:      return cpuid->have_amx_bf16_;
      case AMX_INT8:      return cpuid->have_amx_int8_;
      case AMX_TILE:      return cpuid->have_amx_tile_;
      case AVX2:          return cpuid->have_avx2_;
      case AVX:           return cpuid->have_avx_;
      case AVX512F:       return cpuid->have_avx512f_;
      case AVX512CD:      return cpuid->have_avx512cd_;
      case AVX512PF:      return cpuid->have_avx512pf_;
      case AVX512ER:      return cpuid->have_avx512er_;
      case AVX512VL:      return cpuid->have_avx512vl_;
      case AVX512BW:      return cpuid->have_avx512bw_;
      case AVX512DQ:      return cpuid->have_avx512dq_;
      case AVX512VBMI:    return cpuid->have_avx512vbmi_;
      case AVX512IFMA:    return cpuid->have_avx512ifma_;
      case AVX512_4VNNIW: return cpuid->have_avx512_4vnniw_;
      case AVX512_4FMAPS: return cpuid->have_avx512_4fmaps_;
      case AVX512_BF16:   return cpuid->have_avx512_bf16_;
      case AVX512_VNNI:   return cpuid->have_avx512_vnni_;
      case AVX_VNNI:      return cpuid->have_avx_vnni_;
      case BMI1:          return cpuid->have_bmi1_;
      case BMI2:          return cpuid->have_bmi2_;
      case CMOV:          return cpuid->have_cmov_;
      case CMPXCHG16B:    return cpuid->have_cmpxchg16b_;
      case CMPXCHG8B:     return cpuid->have_cmpxchg8b_;
      case F16C:          return cpuid->have_f16c_;
      case FMA:           return cpuid->have_fma_;
      case MMX:           return cpuid->have_mmx_;
      case PCLMULQDQ:     return cpuid->have_pclmulqdq_;
      case POPCNT:        return cpuid->have_popcnt_;
      case PREFETCHW:     return cpuid->have_prefetchw_;
      case PREFETCHWT1:   return cpuid->have_prefetchwt1_;
      case RDRAND:        return cpuid->have_rdrand_;
      case RDSEED:        return cpuid->have_rdseed_;
      case SMAP:          return cpuid->have_smap_;
      case SSE2:          return cpuid->have_sse2_;
      case SSE3:          return cpuid->have_sse3_;
      case SSE4_1:        return cpuid->have_sse4_1_;
      case SSE4_2:        return cpuid->have_sse4_2_;
      case SSE:           return cpuid->have_sse_;
      case SSSE3:         return cpuid->have_ssse3_;
      case HYPERVISOR:    return cpuid->have_hypervisor_;
      default:
        break;
    }
    // clang-format on
    return false;
  }

  string vendor_str() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_3(mht_3_v, 463, "", "./tensorflow/core/platform/cpu_info.cc", "vendor_str");
 return vendor_str_; }
  int family() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_4(mht_4_v, 467, "", "./tensorflow/core/platform/cpu_info.cc", "family");
 return family_; }
  int model_num() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_5(mht_5_v, 471, "", "./tensorflow/core/platform/cpu_info.cc", "model_num");
 return model_num_; }

 private:
  int have_adx_ : 1;
  int have_aes_ : 1;
  int have_amx_bf16_ : 1;
  int have_amx_int8_ : 1;
  int have_amx_tile_ : 1;
  int have_avx_ : 1;
  int have_avx2_ : 1;
  int have_avx512f_ : 1;
  int have_avx512cd_ : 1;
  int have_avx512er_ : 1;
  int have_avx512pf_ : 1;
  int have_avx512vl_ : 1;
  int have_avx512bw_ : 1;
  int have_avx512dq_ : 1;
  int have_avx512vbmi_ : 1;
  int have_avx512ifma_ : 1;
  int have_avx512_4vnniw_ : 1;
  int have_avx512_4fmaps_ : 1;
  int have_avx512_bf16_ : 1;
  int have_avx512_vnni_ : 1;
  int have_avx_vnni_ : 1;
  int have_bmi1_ : 1;
  int have_bmi2_ : 1;
  int have_cmov_ : 1;
  int have_cmpxchg16b_ : 1;
  int have_cmpxchg8b_ : 1;
  int have_f16c_ : 1;
  int have_fma_ : 1;
  int have_mmx_ : 1;
  int have_pclmulqdq_ : 1;
  int have_popcnt_ : 1;
  int have_prefetchw_ : 1;
  int have_prefetchwt1_ : 1;
  int have_rdrand_ : 1;
  int have_rdseed_ : 1;
  int have_smap_ : 1;
  int have_sse_ : 1;
  int have_sse2_ : 1;
  int have_sse3_ : 1;
  int have_sse4_1_ : 1;
  int have_sse4_2_ : 1;
  int have_ssse3_ : 1;
  int have_hypervisor_ : 1;
  string vendor_str_;
  int family_;
  int model_num_;
};

absl::once_flag cpuid_once_flag;

void InitCPUIDInfo() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_6(mht_6_v, 527, "", "./tensorflow/core/platform/cpu_info.cc", "InitCPUIDInfo");

  // This ensures that CPUIDInfo::Initialize() is called exactly
  // once regardless of how many threads concurrently call us
  absl::call_once(cpuid_once_flag, CPUIDInfo::Initialize);
}

#endif  // PLATFORM_IS_X86

}  // namespace

bool TestCPUFeature(CPUFeature feature) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_7(mht_7_v, 540, "", "./tensorflow/core/platform/cpu_info.cc", "TestCPUFeature");

#ifdef PLATFORM_IS_X86
  return CPUIDInfo::TestFeature(feature);
#else
  return false;
#endif
}

std::string CPUVendorIDString() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_8(mht_8_v, 551, "", "./tensorflow/core/platform/cpu_info.cc", "CPUVendorIDString");

#ifdef PLATFORM_IS_X86
  InitCPUIDInfo();
  return cpuid->vendor_str();
#else
  return "";
#endif
}

int CPUFamily() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_9(mht_9_v, 563, "", "./tensorflow/core/platform/cpu_info.cc", "CPUFamily");

#ifdef PLATFORM_IS_X86
  InitCPUIDInfo();
  return cpuid->family();
#else
  return 0;
#endif
}

int CPUModelNum() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_10(mht_10_v, 575, "", "./tensorflow/core/platform/cpu_info.cc", "CPUModelNum");

#ifdef PLATFORM_IS_X86
  InitCPUIDInfo();
  return cpuid->model_num();
#else
  return 0;
#endif
}

int CPUIDNumSMT() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPScpu_infoDTcc mht_11(mht_11_v, 587, "", "./tensorflow/core/platform/cpu_info.cc", "CPUIDNumSMT");

#ifdef PLATFORM_IS_X86
  // https://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration
  // https://software.intel.com/en-us/articles/intel-sdm (Vol 3A)
  // Section: Detecting Hardware Multi-threads Support and Topology
  // Uses CPUID Leaf 11 to enumerate system topology on Intel x86 architectures
  // Other cases not supported
  uint32 eax, ebx, ecx, edx;
  // Check if system supports Leaf 11
  GETCPUID(eax, ebx, ecx, edx, 0, 0);
  if (eax >= 11) {
    // 1) Leaf 11 available? CPUID.(EAX=11, ECX=0):EBX != 0
    // 2) SMT_Mask_Width = CPUID.(EAX=11, ECX=0):EAX[4:0] if CPUID.(EAX=11,
    // ECX=0):ECX[15:8] is 1
    GETCPUID(eax, ebx, ecx, edx, 11, 0);
    if (ebx != 0 && ((ecx & 0xff00) >> 8) == 1) {
      return 1 << (eax & 0x1f);  // 2 ^ SMT_Mask_Width
    }
  }
#endif  // PLATFORM_IS_X86
  return 0;
}

}  // namespace port
}  // namespace tensorflow
