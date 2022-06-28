/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh() {
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

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32 destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
  // This contains CUpti_ActivityMemcpyKind for activity event (on device).
  // For events from other CuptiTracerEventSource, it is always 0.
  int8 copy_kind;
  // CUpti_ActivityMemoryKind of source.
  int8 src_mem_kind;
  // CUpti_ActivityMemoryKind of destination.
  int8 dst_mem_kind;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

struct MemAllocDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8 mem_kind;
  // The virtual address of allocation. 0 if it is a free operation.
  uint64 address;
};

using MemFreeDetails = MemAllocDetails;

// Memory residency contains details read from CUpti_ActivityMemory type. This
// is populated in the CUPTI tracer encounters a CUPTI_ACTIVITY_KIND_MEMORY
// event. The start of this even corresponse to a cudaMalloc, and the end
// corresponds to a cudaFree.
using MemoryResidencyDetails = MemAllocDetails;

struct MemsetDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8 mem_kind;
  // Whether or not the memset is asynchronous.
  bool async;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32 registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32 static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32 dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32 block_x;
  // Y-dimension of a thread block.
  uint32 block_y;
  // Z-dimension of a thread block.
  uint32 block_z;
  // X-dimension of a grid.
  uint32 grid_x;
  // Y-dimension of a grid.
  uint32 grid_y;
  // Z-dimension of a grid.
  uint32 grid_z;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_0(mht_0_v, 282, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "ToXStat");

  return absl::StrCat(
      "regs:", kernel_info.registers_per_thread,
      " static_shared:", kernel_info.static_shared_memory_usage,
      " dynamic_shared:", kernel_info.dynamic_shared_memory_usage,
      " grid:", kernel_info.grid_x, ",", kernel_info.grid_y, ",",
      kernel_info.grid_z, " block:", kernel_info.block_x, ",",
      kernel_info.block_y, ",", kernel_info.block_z,
      " occ_pct:", occupancy_pct);
}

// Gets the name of the CUpti_ActivityMemoryKind value.
absl::string_view GetMemoryKindName(int8_t memory_kind);

enum class CuptiTracerEventType {
  Unsupported = 0,
  Kernel = 1,
  MemcpyH2D = 2,
  MemcpyD2H = 3,
  MemcpyD2D = 4,
  MemcpyP2P = 5,
  MemcpyOther = 6,
  MemoryAlloc = 7,
  Overhead = 8,
  UnifiedMemory = 9,
  MemoryFree = 10,
  Memset = 11,
  MemoryResidency = 12,
  Generic = 100,
};

const char* GetTraceEventTypeName(const CuptiTracerEventType& type);

enum class CuptiTracerEventSource {
  Invalid = 0,
  DriverCallback = 1,
  Activity = 2,
  // Maybe consider adding runtime callback and metric api in the future.
};

struct CuptiTracerEvent {
  static constexpr uint32 kInvalidThreadId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32 kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64 kInvalidContextId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64 kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  CuptiTracerEventType type = CuptiTracerEventType::Unsupported;
  CuptiTracerEventSource source = CuptiTracerEventSource::Invalid;
  // Although CUpti_CallbackData::functionName is persistent, however
  // CUpti_ActivityKernel4::name is not persistent, therefore we need a copy of
  // it.
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view nvtx_range;
  uint64 start_time_ns = 0;
  uint64 end_time_ns = 0;
  uint32 device_id = 0;
  uint32 correlation_id = kInvalidCorrelationId;
  uint32 thread_id = kInvalidThreadId;
  int64_t context_id = kInvalidContextId;
  int64_t stream_id = kInvalidStreamId;
  union {
    // For Memcpy API and activities. `type` must be Memcpy*.
    MemcpyDetails memcpy_info;
    // Used for MemAlloc API. `type` must be MemoryAlloc.
    MemAllocDetails memalloc_info;
    // Used for kernel activities. `type` must be Kernel.
    KernelDetails kernel_info;
    // Used for MemFree activities. `type` must be MemoryFree.
    MemFreeDetails memfree_info;
    // Used for Memset API and activities. `type` must be Memset.
    MemsetDetails memset_info;
    // Used for Memory residency activities. `type` must be MemoryResidency.
    MemoryResidencyDetails memory_residency_info;
  };
};

struct CuptiTracerCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64 max_callback_api_events = 2 * 1024 * 1024;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64 max_activity_api_events = 2 * 1024 * 1024;
  // Maximum number of annotation strings that we can accommodate.
  uint64 max_annotation_strings = 1024 * 1024;
  // Number of GPUs involved.
  uint32 num_gpus;
};

class AnnotationMap {
 public:
  struct AnnotationInfo {
    absl::string_view annotation;
    absl::string_view nvtx_range;
  };

  explicit AnnotationMap(uint64 max_size, uint32 num_gpus)
      : max_size_(max_size), per_device_map_(num_gpus) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_1(mht_1_v, 388, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "AnnotationMap");
}
  void Add(uint32 device_id, uint32 correlation_id,
           const absl::string_view annotation,
           const absl::string_view nvtx_range);
  AnnotationInfo LookUp(uint32 device_id, uint32 correlation_id);

 private:
  struct PerDeviceAnnotationMap {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::node_hash_set<std::string> nvtx_ranges;
    absl::flat_hash_map<uint32, AnnotationInfo> correlation_map;
  };
  const uint64 max_size_;
  absl::FixedArray<PerDeviceAnnotationMap> per_device_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(AnnotationMap);
};

class CuptiTraceCollector {
 public:
  explicit CuptiTraceCollector(const CuptiTracerCollectorOptions& options)
      : options_(options),
        annotation_map_(options.max_annotation_strings, options.num_gpus) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_2(mht_2_v, 418, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "CuptiTraceCollector");
}
  virtual ~CuptiTraceCollector() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_3(mht_3_v, 422, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "~CuptiTraceCollector");
}

  // Producer side functions (i.e. called by CuptiTracer).
  virtual void AddEvent(CuptiTracerEvent&& event) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32 num_events) = 0;
  virtual void Flush() = 0;

  // Consumer side functions (i.e. called by GPU tracer);
  virtual bool Export(XSpace* space, uint64 end_gpu_ns) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_4(mht_4_v, 434, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "Export");
 return true; }
  virtual std::string ReportNumEventsIfDropped() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_5(mht_5_v, 438, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "ReportNumEventsIfDropped");
 return ""; }

  AnnotationMap* annotation_map() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTh mht_6(mht_6_v, 443, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.h", "annotation_map");
 return &annotation_map_; }

 protected:
  CuptiTracerCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTraceCollector);
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options, const uint64 start_walltime_ns,
    const uint64 start_gputime_ns);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_
