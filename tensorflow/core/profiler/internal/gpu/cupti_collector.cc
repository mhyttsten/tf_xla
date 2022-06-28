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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc() {
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

#include "tensorflow/core/profiler/internal/gpu/cupti_collector.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_occupancy.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/utils/parse_annotation.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

bool IsHostEvent(const CuptiTracerEvent& event, int64_t* line_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "IsHostEvent");

  // DriverCallback(i.e. kernel launching) events are host events.
  if (event.source == CuptiTracerEventSource::DriverCallback) {
    *line_id = event.thread_id;
    return true;
  }
  // Non-overhead activity events are device events.
  if (event.type != CuptiTracerEventType::Overhead) {
    *line_id = event.stream_id;
    return false;
  }
  // Overhead events can be associated with a thread or a stream, etc.
  // If a valid thread id is specified, we consider it as a host event.
  //
  if (event.stream_id != CuptiTracerEvent::kInvalidStreamId) {
    *line_id = event.stream_id;
    return false;
  } else if (event.thread_id != CuptiTracerEvent::kInvalidThreadId &&
             event.thread_id != 0) {
    *line_id = event.thread_id;
    return true;
  } else {
    *line_id = kThreadIdOverhead;
    return false;
  }
}

struct DeviceOccupancyParams {
  cudaOccFuncAttributes attributes = {};
  int block_size = 0;
  size_t dynamic_smem_size = 0;

  friend bool operator==(const DeviceOccupancyParams& lhs,
                         const DeviceOccupancyParams& rhs) {
    return 0 == memcmp(&lhs, &rhs, sizeof(lhs));
  }

  template <typename H>
  friend H AbslHashValue(H hash_state, const DeviceOccupancyParams& params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "AbslHashValue");

    return H::combine(
        std::move(hash_state), params.attributes.maxThreadsPerBlock,
        params.attributes.numRegs, params.attributes.sharedSizeBytes,
        static_cast<uint32_t>(params.attributes.partitionedGCConfig),
        static_cast<uint32_t>(params.attributes.shmemLimitConfig),
        params.attributes.maxDynamicSharedSizeBytes, params.block_size,
        params.dynamic_smem_size);
  }
};

struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

class PerDeviceCollector {
 private:
  OccupancyStats GetOccupancy(const DeviceOccupancyParams& params) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "GetOccupancy");

    OccupancyStats stats;
    if (device_properties_.computeMajor == 0) {
      return {};
    }

    const cudaOccDeviceState state = {};
    cudaOccResult occ_result;
    cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
        &occ_result, &device_properties_, &params.attributes, &state,
        params.block_size, params.dynamic_smem_size);
    if (status != CUDA_OCC_SUCCESS) {
      return {};
    }

    stats.occupancy_pct =
        occ_result.activeBlocksPerMultiprocessor * params.block_size * 100;
    stats.occupancy_pct /= device_properties_.maxThreadsPerMultiprocessor;

    status = cudaOccMaxPotentialOccupancyBlockSize(
        &stats.min_grid_size, &stats.suggested_block_size, &device_properties_,
        &params.attributes, &state, nullptr, params.dynamic_smem_size);
    if (status != CUDA_OCC_SUCCESS) {
      return {};
    }

    return stats;
  }

  void CreateXEvent(const CuptiTracerEvent& event, XPlaneBuilder* plane,
                    uint64 start_gpu_ns, uint64 end_gpu_ns,
                    XLineBuilder* line) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_3(mht_3_v, 305, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "CreateXEvent");

    if (event.start_time_ns < start_gpu_ns || event.end_time_ns > end_gpu_ns ||
        event.start_time_ns > event.end_time_ns) {
      VLOG(2) << "events have abnormal timestamps:" << event.name
              << " start time(ns): " << event.start_time_ns
              << " end time(ns): " << event.end_time_ns;
      return;
    }
    std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
    if (kernel_name.empty()) {
      kernel_name = GetTraceEventTypeName(event.type);
    }
    XEventMetadata* event_metadata =
        plane->GetOrCreateEventMetadata(std::move(kernel_name));
    XEventBuilder xevent = line->AddEvent(*event_metadata);
    VLOG(7) << "Adding event to line=" << line->Id();
    xevent.SetTimestampNs(event.start_time_ns);
    xevent.SetEndTimestampNs(event.end_time_ns);
    if (event.source == CuptiTracerEventSource::DriverCallback) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)),
          event.device_id);
    }
    if (event.correlation_id != CuptiTracerEvent::kInvalidCorrelationId) {
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kCorrelationId)),
                          event.correlation_id);
    }
    if (!event.nvtx_range.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kNVTXRange)),
          *plane->GetOrCreateStatMetadata(event.nvtx_range));
    }
    if (event.context_id != CuptiTracerEvent::kInvalidContextId) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kContextId)),
          absl::StrCat("$$", static_cast<uint64>(event.context_id)));
    }

    if (event.type == CuptiTracerEventType::Kernel &&
        event.source == CuptiTracerEventSource::Activity) {
      DeviceOccupancyParams params{};
      params.attributes.maxThreadsPerBlock = INT_MAX;
      params.attributes.numRegs =
          static_cast<int>(event.kernel_info.registers_per_thread);
      params.attributes.sharedSizeBytes =
          event.kernel_info.static_shared_memory_usage;
      params.attributes.partitionedGCConfig = PARTITIONED_GC_OFF;
      params.attributes.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
      params.attributes.maxDynamicSharedSizeBytes = 0;
      params.block_size = static_cast<int>(event.kernel_info.block_x *
                                           event.kernel_info.block_y *
                                           event.kernel_info.block_z);

      params.dynamic_smem_size = event.kernel_info.dynamic_shared_memory_usage;

      OccupancyStats& occ_stats = occupancy_cache_[params];
      if (occ_stats.occupancy_pct == 0.0) {
        occ_stats = GetOccupancy(params);
      }
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kTheoreticalOccupancyPct)),
                          occ_stats.occupancy_pct);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kOccupancyMinGridSize)),
                          static_cast<int32>(occ_stats.min_grid_size));
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kOccupancySuggestedBlockSize)),
                          static_cast<int32>(occ_stats.suggested_block_size));
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kKernelDetails)),
                          *plane->GetOrCreateStatMetadata(ToXStat(
                              event.kernel_info, occ_stats.occupancy_pct)));
    } else if (event.type == CuptiTracerEventType::MemcpyH2D ||
               event.type == CuptiTracerEventType::MemcpyD2H ||
               event.type == CuptiTracerEventType::MemcpyD2D ||
               event.type == CuptiTracerEventType::MemcpyP2P ||
               event.type == CuptiTracerEventType::MemcpyOther) {
      const auto& memcpy_info = event.memcpy_info;
      std::string value = absl::StrCat(
          "kind_src:", GetMemoryKindName(event.memcpy_info.src_mem_kind),
          " kind_dst:", GetMemoryKindName(event.memcpy_info.dst_mem_kind),
          " size:", memcpy_info.num_bytes, " dest:", memcpy_info.destination,
          " async:", memcpy_info.async);
      VLOG(7) << "Add Memcpy stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemcpyDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryAlloc) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memalloc_info.mem_kind),
                       " num_bytes:", event.memalloc_info.num_bytes);
      VLOG(7) << "Add MemAlloc stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemallocDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryFree) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memfree_info.mem_kind),
                       " num_bytes:", event.memfree_info.num_bytes);
      VLOG(7) << "Add MemFree stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemFreeDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::Memset) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memset_info.mem_kind),
                       " num_bytes:", event.memset_info.num_bytes,
                       " async:", event.memset_info.async);
      VLOG(7) << "Add Memset stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemsetDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryResidency) {
      std::string value = absl::StrCat(
          "kind:", GetMemoryKindName(event.memory_residency_info.mem_kind),
          " num_bytes:", event.memory_residency_info.num_bytes, " addr:0x",
          absl::Hex(event.memory_residency_info.address, absl::kZeroPad16));
      VLOG(7) << "Add MemoryResidency stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kMemoryResidencyDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    }

    std::vector<Annotation> annotation_stack =
        ParseAnnotationStack(event.annotation);
    if (!annotation_stack.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
          *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
    }
    // If multiple metadata have the same key name, show the values from the top
    // of the stack (innermost annotation). Concatenate the values from
    // "hlo_op".
    absl::flat_hash_set<absl::string_view> key_set;

    for (auto annotation = annotation_stack.rbegin();
         annotation != annotation_stack.rend(); ++annotation) {
      for (const Annotation::Metadata& metadata : annotation->metadata) {
        if (key_set.insert(metadata.key).second) {
          xevent.ParseAndAddStatValue(
              *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
        }
      }
    }
  }

  absl::optional<int> GetDeviceAttribute(CUdevice device,
                                         CUdevice_attribute attrib) {
    int ret_val;
    CUresult err = cuDeviceGetAttribute(&ret_val, attrib, device);
    if (err != CUDA_SUCCESS) return absl::nullopt;
    return ret_val;
  }

  std::string GetDeviceXLineName(
      int64_t stream_id,
      absl::flat_hash_set<CuptiTracerEventType>& event_types) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_4(mht_4_v, 465, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "GetDeviceXLineName");

    std::string line_name = absl::StrCat("Stream #", stream_id);
    event_types.erase(CuptiTracerEventType::Unsupported);
    if (event_types.empty()) return line_name;
    if (event_types.count(CuptiTracerEventType::Overhead))
      return "CUPTI overhead";
    std::vector<const char*> type_names;
    for (const auto event_type : event_types) {
      type_names.emplace_back(GetTraceEventTypeName(event_type));
    }
    return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
  }

 public:
  PerDeviceCollector() = default;

  void AddEvent(CuptiTracerEvent&& event) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_5(mht_5_v, 484, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "AddEvent");

    mutex_lock l(m_);
    events_.emplace_back(std::move(event));
  }

  size_t Flush(uint64 start_gpu_ns, uint64 end_gpu_ns,
               XPlaneBuilder* device_plane, XPlaneBuilder* host_plane) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_6(mht_6_v, 493, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "Flush");

    mutex_lock l(m_);
    // Tracking event types per line.
    absl::flat_hash_map<int64_t, absl::flat_hash_set<CuptiTracerEventType>>
        events_types_per_line;
    for (auto& event : events_) {
      int64_t line_id = CuptiTracerEvent::kInvalidThreadId;
      bool is_host_event = IsHostEvent(event, &line_id);
      if (line_id == CuptiTracerEvent::kInvalidThreadId ||
          line_id == CuptiTracerEvent::kInvalidStreamId) {
        VLOG(9) << "Ignoring event, type=" << static_cast<int>(event.type);
        continue;
      }
      auto* plane = is_host_event ? host_plane : device_plane;
      VLOG(9) << "Event"
              << " type=" << static_cast<int>(event.type)
              << " line_id=" << line_id
              << (is_host_event ? " host plane=" : " device plane=")
              << plane->Name();
      XLineBuilder line = plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_gpu_ns);
      CreateXEvent(event, plane, start_gpu_ns, end_gpu_ns, &line);
      events_types_per_line[line_id].emplace(event.type);
    }
    device_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(
          GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
    });
    host_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(absl::StrCat("Host Threads/", line.Id()));
    });
    size_t num_events = events_.size();
    events_.clear();
    return num_events;
  }

  void GetDeviceCapabilities(int32_t device_ordinal,
                             XPlaneBuilder* device_plane) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_7(mht_7_v, 533, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "GetDeviceCapabilities");

    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kDevVendor)),
                               kDeviceVendorNvidia);

    CUdevice device;
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) return;

    auto clock_rate_in_khz =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    if (clock_rate_in_khz) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapClockRateKHz)),
          *clock_rate_in_khz);
    }

    auto core_count =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    if (core_count) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapCoreCount)),
          *core_count);
    }

    auto mem_clock_khz =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
    auto mem_bus_width_bits =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
    if (mem_clock_khz && mem_bus_width_bits) {
      // Times 2 because HBM is DDR memory; it gets two data bits per each
      // data lane.
      auto memory_bandwidth =
          uint64{2} * (*mem_clock_khz) * 1000 * (*mem_bus_width_bits) / 8;
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
          memory_bandwidth);
    }

    size_t total_memory = 0;
    if (cuDeviceTotalMem(&total_memory, device) == CUDA_SUCCESS) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemorySize)),
          static_cast<uint64>(total_memory));
    }

    auto compute_capability_major = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    if (compute_capability_major) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
          *compute_capability_major);
    }
    auto compute_capability_minor = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    if (compute_capability_minor) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
          *compute_capability_minor);
    }

    auto max_threads_per_block =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    auto max_threads_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    auto regs_per_block =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK);
    auto regs_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    auto warp_size = GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    auto shared_mem_per_block = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    auto shared_mem_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    auto shared_mem_per_block_optin = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);

    // Precondition for calculating GPU occupancy is to have all of these
    // inputs. Otherwise, GPU occupancy will be left unset as 0%.
    if (core_count && compute_capability_major && compute_capability_minor &&
        max_threads_per_block && max_threads_per_sm && regs_per_block &&
        regs_per_sm && warp_size && shared_mem_per_block && shared_mem_per_sm &&
        shared_mem_per_block_optin) {
      device_properties_.computeMajor = *compute_capability_major;
      device_properties_.computeMinor = *compute_capability_minor;
      device_properties_.numSms = *core_count;
      device_properties_.maxThreadsPerBlock = *max_threads_per_block;
      device_properties_.maxThreadsPerMultiprocessor = *max_threads_per_sm;
      device_properties_.regsPerBlock = *regs_per_block;
      device_properties_.regsPerMultiprocessor = *regs_per_sm;
      device_properties_.warpSize = *warp_size;
      device_properties_.sharedMemPerBlock = *shared_mem_per_block;
      device_properties_.sharedMemPerMultiprocessor = *shared_mem_per_sm;
      device_properties_.sharedMemPerBlockOptin = *shared_mem_per_block_optin;
    }
  }

 private:
  mutex m_;
  std::vector<CuptiTracerEvent> events_ TF_GUARDED_BY(m_);
  cudaOccDeviceProp device_properties_;
  absl::flat_hash_map<DeviceOccupancyParams, OccupancyStats> occupancy_cache_;
};

}  // namespace

void AnnotationMap::Add(uint32 device_id, uint32 correlation_id,
                        const absl::string_view annotation,
                        const absl::string_view nvtx_range) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("annotation: \"" + std::string(annotation.data(), annotation.size()) + "\"");
   mht_8_v.push_back("nvtx_range: \"" + std::string(nvtx_range.data(), nvtx_range.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_8(mht_8_v, 651, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "AnnotationMap::Add");

  if (annotation.empty() && nvtx_range.empty()) return;
  VLOG(3) << "Add annotation: device_id: " << device_id
          << " correlation_id: " << correlation_id
          << " annotation: " << annotation;
  if (device_id >= per_device_map_.size()) return;
  auto& per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  if (per_device_map.annotations.size() < max_size_) {
    AnnotationInfo info;
    info.annotation = *per_device_map.annotations.emplace(annotation).first;
    if (!nvtx_range.empty())
      info.nvtx_range = *per_device_map.nvtx_ranges.emplace(nvtx_range).first;
    per_device_map.correlation_map.emplace(correlation_id, info);
  }
}

AnnotationMap::AnnotationInfo AnnotationMap::LookUp(uint32 device_id,
                                                    uint32 correlation_id) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_9(mht_9_v, 672, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "AnnotationMap::LookUp");

  if (device_id >= per_device_map_.size()) return AnnotationInfo();
  auto& per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  auto it = per_device_map.correlation_map.find(correlation_id);
  return it != per_device_map.correlation_map.end() ? it->second
                                                    : AnnotationInfo();
}

// CuptiTraceCollectorImpl store the CuptiTracerEvents from CuptiTracer and
// eventually convert and filter them to XSpace.
class CuptiTraceCollectorImpl : public CuptiTraceCollector {
 public:
  CuptiTraceCollectorImpl(const CuptiTracerCollectorOptions& option,
                          uint64 start_walltime_ns, uint64 start_gpu_ns)
      : CuptiTraceCollector(option),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gpu_ns_(start_gpu_ns),
        num_gpus_(option.num_gpus),
        per_device_collector_(option.num_gpus) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_10(mht_10_v, 696, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "CuptiTraceCollectorImpl");
}

  void AddEvent(CuptiTracerEvent&& event) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_11(mht_11_v, 701, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "AddEvent");

    if (event.device_id >= num_gpus_) return;
    if (event.source == CuptiTracerEventSource::DriverCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("total driver(callback) events reaches max", 1);
        return;
      }
      num_callback_events_++;
    } else {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("total device(activity) events reaches max", 1);
        return;
      }
      num_activity_events_++;
    }
    per_device_collector_[event.device_id].AddEvent(std::move(event));
  }
  void OnEventsDropped(const std::string& reason, uint32 num_events) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("reason: \"" + reason + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_12(mht_12_v, 722, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "OnEventsDropped");

    absl::MutexLock lock(&mutex_);
    dropped_events_[reason] += num_events;
  }
  void Flush() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_13(mht_13_v, 729, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "Flush");
}
  // Returns true if some GPU events are captured.
  bool Export(XSpace* space, uint64 end_gpu_ns) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_14(mht_14_v, 734, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "Export");

    LOG(INFO) << " GpuTracer has collected " << num_callback_events_
              << " callback api events and " << num_activity_events_
              << " activity events. " << ReportDroppedEvents();
    size_t num_events = 0;
    XPlaneBuilder host_plane(
        FindOrAddMutablePlaneWithName(space, kCuptiDriverApiPlaneName));
    for (int device_ordinal = 0; device_ordinal < num_gpus_; ++device_ordinal) {
      std::string name = GpuPlaneName(device_ordinal);
      XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
      device_plane.SetId(device_ordinal);
      VLOG(4) << "Creating plane for"
              << " name=" << name << " ordinal=" << device_ordinal;

      // Calculate device capabilities before flushing, so that device
      // properties are available to the occupancy calculator in Flush().
      per_device_collector_[device_ordinal].GetDeviceCapabilities(
          device_ordinal, &device_plane);
      num_events += per_device_collector_[device_ordinal].Flush(
          start_gpu_ns_, end_gpu_ns, &device_plane, &host_plane);
      NormalizeTimeStamps(&device_plane, start_walltime_ns_);
    }
    NormalizeTimeStamps(&host_plane, start_walltime_ns_);
    return num_events > 0;
  }

  std::string ReportDroppedEvents() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_15(mht_15_v, 763, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "ReportDroppedEvents");

    absl::MutexLock lock(&mutex_);
    string result;
    for (const auto& dropped : dropped_events_) {
      absl::StrAppend(&result, " ", dropped.second, " events dropped because ",
                      dropped.first, ";");
    }
    if (!result.empty()) result.back() = '.';
    return result;
  }
  std::string ReportNumEventsIfDropped() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_16(mht_16_v, 776, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "ReportNumEventsIfDropped");

    std::string events_dropped = ReportDroppedEvents();
    if (events_dropped.empty()) return "";
    return absl::StrCat("Detected GPU events dropped on ", port::Hostname(),
                        ": Profiler has collected ",
                        num_callback_events_.load(), " driver events and ",
                        num_activity_events_.load(), " device events.",
                        events_dropped);
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, uint64> dropped_events_
      ABSL_GUARDED_BY(mutex_);
  uint64 start_walltime_ns_;
  uint64 start_gpu_ns_;
  int num_gpus_;

  // Set the all XLines of specified XPlane to starting walltime.
  // Events time in both host and device planes are CUTPI timestamps.
  // We set initial CUPTI timestamp as start time for all lines to reflect
  // this fact. Eventually we change line start time to corresponding
  // start_walltime_ns to normalize with CPU wall time.
  static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                  uint64 start_walltime_ns) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_17(mht_17_v, 805, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "NormalizeTimeStamps");

    plane->ForEachLine(
        [&](XLineBuilder line) { line.SetTimestampNs(start_walltime_ns); });
  }

  absl::FixedArray<PerDeviceCollector> per_device_collector_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTraceCollectorImpl);
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options, const uint64 start_walltime_ns,
    const uint64 start_gputime_ns) {
  return absl::make_unique<CuptiTraceCollectorImpl>(options, start_walltime_ns,
                                                    start_gputime_ns);
}

// The strings are parser friendly and have no whitespaces in them.
absl::string_view GetMemoryKindName(int8_t memory_kind) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_collectorDTcc mht_18(mht_18_v, 826, "", "./tensorflow/core/profiler/internal/gpu/cupti_collector.cc", "GetMemoryKindName");

  switch (memory_kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "device_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "managed_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

}  // namespace profiler
}  // namespace tensorflow
