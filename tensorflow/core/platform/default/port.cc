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
class MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc() {
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

#include "absl/base/internal/sysinfo.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || \
    defined(__HAIKU__)
#include <thread>
#endif

#if TENSORFLOW_USE_NUMA
#include "hwloc.h"  // from @hwloc
#endif

#if defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define TENSORFLOW_HAS_CXA_DEMANGLE 0
#elif (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && \
    !defined(__mips__)
#define TENSORFLOW_HAS_CXA_DEMANGLE 1
#elif defined(__clang__) && !defined(_MSC_VER)
#define TENSORFLOW_HAS_CXA_DEMANGLE 1
#else
#define TENSORFLOW_HAS_CXA_DEMANGLE 0
#endif

#if TENSORFLOW_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace tensorflow {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("usage: \"" + (usage == nullptr ? std::string("nullptr") : std::string((char*)usage)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_0(mht_0_v, 240, "", "./tensorflow/core/platform/default/port.cc", "InitMain");
}

string Hostname() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/platform/default/port.cc", "Hostname");

  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return string(hostname);
}

string JobName() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/platform/default/port.cc", "JobName");

  const char* job_name_cs = std::getenv("TF_JOB_NAME");
  if (job_name_cs != nullptr) {
    return string(job_name_cs);
  }
  return "";
}

int64_t JobUid() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/platform/default/port.cc", "JobUid");
 return -1; }

int NumSchedulableCPUs() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_4(mht_4_v, 271, "", "./tensorflow/core/platform/default/port.cc", "NumSchedulableCPUs");

#if defined(__linux__) && !defined(__ANDROID__)
  cpu_set_t cpuset;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
    return CPU_COUNT(&cpuset);
  }
  perror("sched_getaffinity");
#endif
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || \
    defined(__HAIKU__)
  unsigned int count = std::thread::hardware_concurrency();
  if (count > 0) return static_cast<int>(count);
#endif
  const int kDefaultCores = 4;  // Semi-conservative guess
  fprintf(stderr, "can't determine number of CPU cores: assuming %d\n",
          kDefaultCores);
  return kDefaultCores;
}

int MaxParallelism() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/platform/default/port.cc", "MaxParallelism");
 return NumSchedulableCPUs(); }

int MaxParallelism(int numa_node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/platform/default/port.cc", "MaxParallelism");

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
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_7(mht_7_v, 311, "", "./tensorflow/core/platform/default/port.cc", "NumTotalCPUs");

  int count = absl::base_internal::NumCPUs();
  return (count <= 0) ? kUnknownCPU : count;
}

int GetCurrentCPU() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_8(mht_8_v, 319, "", "./tensorflow/core/platform/default/port.cc", "GetCurrentCPU");

#if defined(__EMSCRIPTEN__)
  return sched_getcpu();
#elif defined(__linux__) && !defined(__ANDROID__)
  return sched_getcpu();
  // Attempt to use cpuid on all other platforms.  If that fails, perform a
  // syscall.
#elif defined(__cpuid) && !defined(__APPLE__)
  // TODO(b/120919972): __cpuid returns invalid APIC ids on OS X.
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  __cpuid(/*level=*/1, eax, ebx, ecx, edx);
  if ((edx & /*bit_APIC=*/(1 << 9)) != 0) {
    // EBX bits 24-31 are APIC ID
    return (ebx & 0xFF) >> 24;
  }
#elif defined(__NR_getcpu)
  unsigned int cpu;
  if (syscall(__NR_getcpu, &cpu, NULL, NULL) < 0) {
    return kUnknownCPU;
  } else {
    return static_cast<int>(cpu);
  }
#endif
  return kUnknownCPU;
}

int NumHyperthreadsPerCore() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_9(mht_9_v, 351, "", "./tensorflow/core/platform/default/port.cc", "NumHyperthreadsPerCore");

  static const int ht_per_core = tensorflow::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

#ifdef TENSORFLOW_USE_NUMA
namespace {
static hwloc_topology_t hwloc_topology_handle;

bool HaveHWLocTopology() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_10(mht_10_v, 363, "", "./tensorflow/core/platform/default/port.cc", "HaveHWLocTopology");

  // One time initialization
  static bool init = []() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_11(mht_11_v, 368, "", "./tensorflow/core/platform/default/port.cc", "lambda");

    if (hwloc_topology_init(&hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_init() failed";
      return false;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_load() failed";
      return false;
    }
    return true;
  }();
  return init;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_12(mht_12_v, 387, "", "./tensorflow/core/platform/default/port.cc", "GetHWLocTypeIndex");

  hwloc_obj_t obj = nullptr;
  if (index >= 0) {
    while ((obj = hwloc_get_next_obj_by_type(hwloc_topology_handle, tp, obj)) !=
           nullptr) {
      if (obj->os_index == index) break;
    }
  }
  return obj;
}
}  // namespace
#endif  // TENSORFLOW_USE_NUMA

bool NUMAEnabled() { return (NUMANumNodes() > 1); }

int NUMANumNodes() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_13(mht_13_v, 405, "", "./tensorflow/core/platform/default/port.cc", "NUMANumNodes");

#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    int num_numanodes =
        hwloc_get_nbobjs_by_type(hwloc_topology_handle, HWLOC_OBJ_NUMANODE);
    return std::max(1, num_numanodes);
  } else {
    return 1;
  }
#else
  return 1;
#endif  // TENSORFLOW_USE_NUMA
}

void NUMASetThreadNodeAffinity(int node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_14(mht_14_v, 422, "", "./tensorflow/core/platform/default/port.cc", "NUMASetThreadNodeAffinity");

#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    // Find the corresponding NUMA node topology object.
    hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (obj) {
      hwloc_set_cpubind(hwloc_topology_handle, obj->cpuset,
                        HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    } else {
      LOG(ERROR) << "Could not find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
}

int NUMAGetThreadNodeAffinity() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_15(mht_15_v, 440, "", "./tensorflow/core/platform/default/port.cc", "NUMAGetThreadNodeAffinity");

  int node_index = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_cpuset_t thread_cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(hwloc_topology_handle, thread_cpuset,
                      HWLOC_CPUBIND_THREAD);
    hwloc_obj_t obj = nullptr;
    // Return the first NUMA node whose cpuset is a (non-proper) superset of
    // that of the current thread.
    while ((obj = hwloc_get_next_obj_by_type(
                hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
      if (hwloc_bitmap_isincluded(thread_cpuset, obj->cpuset)) {
        node_index = obj->os_index;
        break;
      }
    }
    hwloc_bitmap_free(thread_cpuset);
  }
#endif  // TENSORFLOW_USE_NUMA
  return node_index;
}

void* AlignedMalloc(size_t size, int minimum_alignment) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_16(mht_16_v, 466, "", "./tensorflow/core/platform/default/port.cc", "AlignedMalloc");

#if defined(__ANDROID__)
  return memalign(minimum_alignment, size);
#else  // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
  int err = posix_memalign(&ptr, minimum_alignment, size);
  if (err != 0) {
    return nullptr;
  } else {
    return ptr;
  }
#endif
}

void AlignedFree(void* aligned_memory) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_17(mht_17_v, 488, "", "./tensorflow/core/platform/default/port.cc", "AlignedFree");
 Free(aligned_memory); }

void* Malloc(size_t size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_18(mht_18_v, 493, "", "./tensorflow/core/platform/default/port.cc", "Malloc");
 return malloc(size); }

void* Realloc(void* ptr, size_t size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_19(mht_19_v, 498, "", "./tensorflow/core/platform/default/port.cc", "Realloc");
 return realloc(ptr, size); }

void Free(void* ptr) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_20(mht_20_v, 503, "", "./tensorflow/core/platform/default/port.cc", "Free");
 free(ptr); }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_21(mht_21_v, 508, "", "./tensorflow/core/platform/default/port.cc", "NUMAMalloc");

#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_obj_t numa_node = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (numa_node) {
      return hwloc_alloc_membind(hwloc_topology_handle, size,
                                 numa_node->nodeset, HWLOC_MEMBIND_BIND,
                                 HWLOC_MEMBIND_BYNODESET);
    } else {
      LOG(ERROR) << "Failed to find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
  return AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_22(mht_22_v, 527, "", "./tensorflow/core/platform/default/port.cc", "NUMAFree");

#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_free(hwloc_topology_handle, ptr, size);
    return;
  }
#endif  // TENSORFLOW_USE_NUMA
  Free(ptr);
}

int NUMAGetMemAffinity(const void* addr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_23(mht_23_v, 540, "", "./tensorflow/core/platform/default/port.cc", "NUMAGetMemAffinity");

  int node = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology() && addr) {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (!hwloc_get_area_memlocation(hwloc_topology_handle, addr, 4, nodeset,
                                    HWLOC_MEMBIND_BYNODESET)) {
      hwloc_obj_t obj = nullptr;
      while ((obj = hwloc_get_next_obj_by_type(
                  hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
        if (hwloc_bitmap_isincluded(nodeset, obj->nodeset)) {
          node = obj->os_index;
          break;
        }
      }
      hwloc_bitmap_free(nodeset);
    } else {
      LOG(ERROR) << "Failed call to hwloc_get_area_memlocation.";
    }
  }
#endif  // TENSORFLOW_USE_NUMA
  return node;
}

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_24(mht_24_v, 567, "", "./tensorflow/core/platform/default/port.cc", "MallocExtension_ReleaseToSystem");

  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_25(mht_25_v, 574, "", "./tensorflow/core/platform/default/port.cc", "MallocExtension_GetAllocatedSize");

#if !defined(__ANDROID__)
  return 0;
#else
  return malloc_usable_size(p);
#endif
}

bool Snappy_Compress(const char* input, size_t length, string* output) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_26(mht_26_v, 586, "", "./tensorflow/core/platform/default/port.cc", "Snappy_Compress");

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
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_27(mht_27_v, 603, "", "./tensorflow/core/platform/default/port.cc", "Snappy_GetUncompressedLength");

#ifdef TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("input: \"" + (input == nullptr ? std::string("nullptr") : std::string((char*)input)) + "\"");
   mht_28_v.push_back("output: \"" + (output == nullptr ? std::string("nullptr") : std::string((char*)output)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_28(mht_28_v, 616, "", "./tensorflow/core/platform/default/port.cc", "Snappy_Uncompress");

#ifdef TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("compressed: \"" + (compressed == nullptr ? std::string("nullptr") : std::string((char*)compressed)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_29(mht_29_v, 629, "", "./tensorflow/core/platform/default/port.cc", "Snappy_UncompressToIOVec");

#ifdef TF_USE_SNAPPY
  return snappy::RawUncompressToIOVec(compressed, compressed_length, iov,
                                      iov_cnt);
#else
  return false;
#endif
}

static void DemangleToString(const char* mangled, string* out) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("mangled: \"" + (mangled == nullptr ? std::string("nullptr") : std::string((char*)mangled)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_30(mht_30_v, 642, "", "./tensorflow/core/platform/default/port.cc", "DemangleToString");

  int status = 0;
  char* demangled = nullptr;
#if TENSORFLOW_HAS_CXA_DEMANGLE
  demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
#endif
  if (status == 0 && demangled != nullptr) {  // Demangling succeeded.
    out->append(demangled);
    free(demangled);
  } else {
    out->append(mangled);
  }
}

string Demangle(const char* mangled) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("mangled: \"" + (mangled == nullptr ? std::string("nullptr") : std::string((char*)mangled)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_31(mht_31_v, 660, "", "./tensorflow/core/platform/default/port.cc", "Demangle");

  string demangled;
  DemangleToString(mangled, &demangled);
  return demangled;
}

double NominalCPUFrequency() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_32(mht_32_v, 669, "", "./tensorflow/core/platform/default/port.cc", "NominalCPUFrequency");

  return tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency();
}

MemoryInfo GetMemoryInfo() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_33(mht_33_v, 676, "", "./tensorflow/core/platform/default/port.cc", "GetMemoryInfo");

  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
#if defined(__linux__) && !defined(__ANDROID__)
  struct sysinfo info;
  int err = sysinfo(&info);
  if (err == 0) {
    mem_info.free = info.freeram;
    mem_info.total = info.totalram;
  }
#endif
  return mem_info;
}

MemoryBandwidthInfo GetMemoryBandwidthInfo() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSplatformPSdefaultPSportDTcc mht_34(mht_34_v, 692, "", "./tensorflow/core/platform/default/port.cc", "GetMemoryBandwidthInfo");

  MemoryBandwidthInfo membw_info = {INT64_MAX};
  return membw_info;
}

}  // namespace port
}  // namespace tensorflow
