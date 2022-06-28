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
class MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif

#include <Windows.h>
#include <processthreadsapi.h>
#include <shlwapi.h>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/platform/windows/port.cc", "InitMain");
}

string Hostname() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/platform/windows/port.cc", "Hostname");

  char name[1024];
  DWORD name_size = sizeof(name);
  name[0] = 0;
  if (::GetComputerNameA(name, &name_size)) {
    name[name_size] = 0;
  }
  return name;
}

string JobName() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/platform/windows/port.cc", "JobName");

  const char* job_name_cs = std::getenv("TF_JOB_NAME");
  if (job_name_cs != nullptr) {
    return string(job_name_cs);
  }
  return "";
}

int64_t JobUid() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/platform/windows/port.cc", "JobUid");
 return -1; }

int NumSchedulableCPUs() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/platform/windows/port.cc", "NumSchedulableCPUs");

  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwNumberOfProcessors;
}

int MaxParallelism() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/platform/windows/port.cc", "MaxParallelism");
 return NumSchedulableCPUs(); }

int MaxParallelism(int numa_node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_6(mht_6_v, 258, "", "./tensorflow/core/platform/windows/port.cc", "MaxParallelism");

  if (numa_node != port::kNUMANoAffinity) {
    // Assume that CPUs are equally distributed over available NUMA nodes.
    // This may not be true, but there isn't currently a better way of
    // determining the number of CPUs specific to the requested node.
    return NumSchedulableCPUs() / port::NUMANumNodes();
  }
  return NumSchedulableCPUs();
}

int NumTotalCPUs() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_7(mht_7_v, 271, "", "./tensorflow/core/platform/windows/port.cc", "NumTotalCPUs");

  // TODO(ebrevdo): Make this more accurate.
  //
  // This only returns the number of processors in the current
  // processor group; which may be undercounting if you have more than 64 cores.
  // For that case, one needs to call
  // GetLogicalProcessorInformationEx(RelationProcessorCore, ...) and accumulate
  // the Size fields by iterating over the written-to buffer.  Since I can't
  // easily test this on Windows, I'm deferring this to someone who can!
  //
  // If you fix this, also consider updating GetCurrentCPU below.
  return NumSchedulableCPUs();
}

int GetCurrentCPU() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_8(mht_8_v, 288, "", "./tensorflow/core/platform/windows/port.cc", "GetCurrentCPU");

  // NOTE(ebrevdo): This returns the processor number within the processor
  // group on systems with >64 processors.  Therefore it doesn't necessarily map
  // naturally to an index in NumSchedulableCPUs().
  //
  // On the plus side, this number is probably guaranteed to be within
  // [0, NumTotalCPUs()) due to its incomplete implementation.
  return GetCurrentProcessorNumber();
}

bool NUMAEnabled() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_9(mht_9_v, 301, "", "./tensorflow/core/platform/windows/port.cc", "NUMAEnabled");

  // Not yet implemented: coming soon.
  return false;
}

int NUMANumNodes() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_10(mht_10_v, 309, "", "./tensorflow/core/platform/windows/port.cc", "NUMANumNodes");
 return 1; }

void NUMASetThreadNodeAffinity(int node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_11(mht_11_v, 314, "", "./tensorflow/core/platform/windows/port.cc", "NUMASetThreadNodeAffinity");
}

int NUMAGetThreadNodeAffinity() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_12(mht_12_v, 319, "", "./tensorflow/core/platform/windows/port.cc", "NUMAGetThreadNodeAffinity");
 return kNUMANoAffinity; }

void* AlignedMalloc(size_t size, int minimum_alignment) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_13(mht_13_v, 324, "", "./tensorflow/core/platform/windows/port.cc", "AlignedMalloc");

  return _aligned_malloc(size, minimum_alignment);
}

void AlignedFree(void* aligned_memory) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_14(mht_14_v, 331, "", "./tensorflow/core/platform/windows/port.cc", "AlignedFree");
 _aligned_free(aligned_memory); }

void* Malloc(size_t size) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_15(mht_15_v, 336, "", "./tensorflow/core/platform/windows/port.cc", "Malloc");
 return malloc(size); }

void* Realloc(void* ptr, size_t size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_16(mht_16_v, 341, "", "./tensorflow/core/platform/windows/port.cc", "Realloc");
 return realloc(ptr, size); }

void Free(void* ptr) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_17(mht_17_v, 346, "", "./tensorflow/core/platform/windows/port.cc", "Free");
 free(ptr); }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_18(mht_18_v, 351, "", "./tensorflow/core/platform/windows/port.cc", "NUMAMalloc");

  return AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_19(mht_19_v, 358, "", "./tensorflow/core/platform/windows/port.cc", "NUMAFree");
 Free(ptr); }

int NUMAGetMemAffinity(const void* addr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_20(mht_20_v, 363, "", "./tensorflow/core/platform/windows/port.cc", "NUMAGetMemAffinity");
 return kNUMANoAffinity; }

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_21(mht_21_v, 368, "", "./tensorflow/core/platform/windows/port.cc", "MallocExtension_ReleaseToSystem");

  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_22(mht_22_v, 375, "", "./tensorflow/core/platform/windows/port.cc", "MallocExtension_GetAllocatedSize");
 return 0; }

bool Snappy_Compress(const char* input, size_t length, string* output) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_23(mht_23_v, 381, "", "./tensorflow/core/platform/windows/port.cc", "Snappy_Compress");

#ifdef TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_24(mht_24_v, 398, "", "./tensorflow/core/platform/windows/port.cc", "Snappy_GetUncompressedLength");

#ifdef TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   mht_25_v.push_back("output: \"" + (output == nullptr ? std::string("nullptr") : std::string((char*)output)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_25(mht_25_v, 411, "", "./tensorflow/core/platform/windows/port.cc", "Snappy_Uncompress");

#ifdef TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("compressed: \"" + (compressed == nullptr ? std::string("nullptr") : std::string((char*)compressed)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_26(mht_26_v, 424, "", "./tensorflow/core/platform/windows/port.cc", "Snappy_UncompressToIOVec");

#ifdef TF_USE_SNAPPY
  const snappy::iovec* snappy_iov = reinterpret_cast<const snappy::iovec*>(iov);
  return snappy::RawUncompressToIOVec(compressed, compressed_length, snappy_iov,
                                      iov_cnt);
#else
  return false;
#endif
}

string Demangle(const char* mangled) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("mangled: \"" + (mangled == nullptr ? std::string("nullptr") : std::string((char*)mangled)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_27(mht_27_v, 438, "", "./tensorflow/core/platform/windows/port.cc", "Demangle");
 return mangled; }

double NominalCPUFrequency() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_28(mht_28_v, 443, "", "./tensorflow/core/platform/windows/port.cc", "NominalCPUFrequency");

  DWORD data;
  DWORD data_size = sizeof(data);
  #pragma comment(lib, "shlwapi.lib")  // For SHGetValue().
  if (SUCCEEDED(
          SHGetValueA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "~MHz", nullptr, &data, &data_size))) {
    return data * 1e6;  // Value is MHz.
  }
  return 1.0;
}

MemoryInfo GetMemoryInfo() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_29(mht_29_v, 459, "", "./tensorflow/core/platform/windows/port.cc", "GetMemoryInfo");

  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    mem_info.free = statex.ullAvailPhys;
    mem_info.total = statex.ullTotalPhys;
  }
  return mem_info;
}

MemoryBandwidthInfo GetMemoryBandwidthInfo() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_30(mht_30_v, 473, "", "./tensorflow/core/platform/windows/port.cc", "GetMemoryBandwidthInfo");

  MemoryBandwidthInfo membw_info = {INT64_MAX};
  return membw_info;
}

int NumHyperthreadsPerCore() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSportDTcc mht_31(mht_31_v, 481, "", "./tensorflow/core/platform/windows/port.cc", "NumHyperthreadsPerCore");

  static const int ht_per_core = tensorflow::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

}  // namespace port
}  // namespace tensorflow
