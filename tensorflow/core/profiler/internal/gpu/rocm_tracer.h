/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh() {
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


#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/rocm/roctracer_wrapper.h"

namespace tensorflow {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: its the current device.
  uint32_t destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
};

struct MemsetDetails {
  // The number of memory elements getting set
  size_t num_bytes;
  // Whether or not the memset is asynchronous.
  bool async;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64_t num_bytes;
};

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32_t registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32_t static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32_t dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32_t block_x;
  // Y-dimension of a thread block.
  uint32_t block_y;
  // Z-dimension of a thread block.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // kernel address. Used for calculating core occupancy
  void* func_ptr;
};

// RocmTracerSyncTypes forward decleration
enum class RocmTracerSyncTypes;
struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type;
};

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel,
  MemcpyH2D,
  MemcpyD2H,
  MemcpyD2D,
  MemcpyP2P,
  MemcpyOther,
  MemoryAlloc,
  MemoryFree,
  Memset,
  Synchronization,
  Generic,
};

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type);

enum class RocmTracerEventSource {
  Invalid = 0,
  ApiCallback,
  Activity,
};

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source);

enum class RocmTracerEventDomain {
  InvalidDomain = 0,
  HIP_API,
  HCC_OPS,  // TODO(rocm-profiler): renme this to HIP_OPS
};
enum class RocmTracerSyncTypes {
  InvalidSync = 0,
  StreamSynchronize,  // caller thread wait stream to become empty
  EventSynchronize,   // caller thread will block until event happens
  StreamWait          // compute stream will wait for event to happen
};

const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain);

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidThreadId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint32_t thread_id = kInvalidThreadId;
  int64_t stream_id = kInvalidStreamId;
  union {
    MemcpyDetails memcpy_info;                    // If type == Memcpy*
    MemsetDetails memset_info;                    // If type == Memset*
    MemAllocDetails memalloc_info;                // If type == MemoryAlloc
    KernelDetails kernel_info;                    // If type == Kernel
    SynchronizationDetails synchronization_info;  // If type == Synchronization
  };
};

void DumpRocmTracerEvent(const RocmTracerEvent& event,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns,
                         const string& message);

struct RocmTracerOptions {
  std::set<uint32_t> api_tracking_set;  // actual api set we want to profile

  // map of domain --> ops for which we need to enable the API callbacks
  // If the ops vector is empty, then enable API callbacks for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> > api_callbacks;

  // map of domain --> ops for which we need to enable the Activity records
  // If the ops vector is empty, then enable Activity records for entire domain
  absl::flat_hash_map<activity_domain_t, std::vector<uint32_t> >
      activity_tracing;
};

struct RocmTraceCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings;
  // Number of GPUs involved.
  uint32_t num_gpus;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64_t max_size) : max_size_(max_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_0(mht_0_v, 357, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "AnnotationMap");
}
  void Add(uint32_t correlation_id, const std::string& annotation);
  absl::string_view LookUp(uint32_t correlation_id);

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::flat_hash_map<uint32_t, absl::string_view> correlation_map;
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options), annotation_map_(options.max_annotation_strings) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_1(mht_1_v, 386, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "RocmTraceCollector");
}
  virtual ~RocmTraceCollector() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_2(mht_2_v, 390, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "~RocmTraceCollector");
}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;

  AnnotationMap* annotation_map() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_3(mht_3_v, 400, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "annotation_map");
 return &annotation_map_; }

 protected:
  RocmTraceCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

class RocmTracer;

class RocmApiCallbackImpl {
 public:
  RocmApiCallbackImpl(const RocmTracerOptions& options, RocmTracer* tracer,
                      RocmTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_4(mht_4_v, 423, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "RocmApiCallbackImpl");
}

  Status operator()(uint32_t domain, uint32_t cbid, const void* cbdata);

 private:
  void AddKernelEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddNormalMemcpyEventUponApiExit(uint32_t cbid,
                                       const hip_api_data_t* data,
                                       uint64_t enter_time, uint64_t exit_time);
  void AddMemcpyPeerEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                     uint64_t enter_time, uint64_t exit_time);
  void AddMemsetEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                 uint64_t enter_time, uint64_t exit_time);
  void AddMallocFreeEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                     uint32_t device_id, uint64_t enter_time,
                                     uint64_t exit_time);
  void AddStreamSynchronizeEventUponApiExit(uint32_t cbid,
                                            const hip_api_data_t* data,
                                            uint64_t enter_time,
                                            uint64_t exit_time);
  void AddSynchronizeEventUponApiExit(uint32_t cbid, const hip_api_data_t* data,
                                      uint64_t enter_time, uint64_t exit_time);

  RocmTracerOptions options_;
  RocmTracer* tracer_ = nullptr;
  RocmTraceCollector* collector_ = nullptr;
  mutex api_call_start_mutex_;
  // TODO(rocm-profiler): replace this with absl hashmap
  // keep a map from the corr. id to enter time for API callbacks.
  std::map<uint32_t, uint64_t> api_call_start_time_
      TF_GUARDED_BY(api_call_start_mutex_);
};

class RocmActivityCallbackImpl {
 public:
  RocmActivityCallbackImpl(const RocmTracerOptions& options, RocmTracer* tracer,
                           RocmTraceCollector* collector)
      : options_(options), tracer_(tracer), collector_(collector) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_5(mht_5_v, 464, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "RocmActivityCallbackImpl");
}

  Status operator()(const char* begin, const char* end);

 private:
  void AddHipKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipMemsetActivityEvent(const roctracer_record_t* record);
  void AddHipMallocActivityEvent(const roctracer_record_t* record);
  void AddHipStreamSynchronizeActivityEvent(const roctracer_record_t* record);
  void AddHccKernelActivityEvent(const roctracer_record_t* record);
  void AddNormalHipOpsMemcpyActivityEvent(const roctracer_record_t* record);
  void AddHipOpsMemsetActivityEvent(const roctracer_record_t* record);
  RocmTracerOptions options_;
  RocmTracer* tracer_ = nullptr;
  RocmTraceCollector* collector_ = nullptr;
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to RocmTraceCollector. There should be only one RocmTracer
// per process.
class RocmTracer {
 public:
  // Returns a pointer to singleton RocmTracer.
  static RocmTracer* GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector);
  void Disable();

  void ApiCallbackHandler(uint32_t domain, uint32_t cbid, const void* cbdata);
  void ActivityCallbackHandler(const char* begin, const char* end);

  static uint64_t GetTimestamp();
  static int NumGpus();

  void AddToPendingActivityRecords(uint32_t correlation_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_6(mht_6_v, 505, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "AddToPendingActivityRecords");

    pending_activity_records_.Add(correlation_id);
  }

  void RemoveFromPendingActivityRecords(uint32_t correlation_id) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_7(mht_7_v, 512, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "RemoveFromPendingActivityRecords");

    pending_activity_records_.Remove(correlation_id);
  }

  void ClearPendingActivityRecordsCount() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_8(mht_8_v, 519, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "ClearPendingActivityRecordsCount");
 pending_activity_records_.Clear(); }

  size_t GetPendingActivityRecordsCount() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_9(mht_9_v, 524, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "GetPendingActivityRecordsCount");

    return pending_activity_records_.Count();
  }

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit RocmTracer() : num_gpus_(NumGpus()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_10(mht_10_v, 533, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "RocmTracer");
}

 private:
  Status EnableApiTracing();
  Status DisableApiTracing();

  Status EnableActivityTracing();
  Status DisableActivityTracing();

  int num_gpus_;
  absl::optional<RocmTracerOptions> options_;
  RocmTraceCollector* collector_ = nullptr;

  bool api_tracing_enabled_ = false;
  bool activity_tracing_enabled_ = false;

  RocmApiCallbackImpl* api_cb_impl_;
  RocmActivityCallbackImpl* activity_cb_impl_;

  class PendingActivityRecords {
   public:
    // add a correlation id to the pending set
    void Add(uint32_t correlation_id) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_11(mht_11_v, 558, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "Add");

      absl::MutexLock lock(&mutex);
      pending_set.insert(correlation_id);
    }
    // remove a correlation id from the pending set
    void Remove(uint32_t correlation_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_12(mht_12_v, 566, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "Remove");

      absl::MutexLock lock(&mutex);
      pending_set.erase(correlation_id);
    }
    // clear the pending set
    void Clear() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_13(mht_13_v, 574, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "Clear");

      absl::MutexLock lock(&mutex);
      pending_set.clear();
    }
    // count the number of correlation ids in the pending set
    size_t Count() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTh mht_14(mht_14_v, 582, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.h", "Count");

      absl::MutexLock lock(&mutex);
      return pending_set.size();
    }

   private:
    // set of co-relation ids for which the hcc activity record is pending
    absl::flat_hash_set<uint32_t> pending_set;
    // the callback which processes the activity records (and consequently
    // removes items from the pending set) is called in a separate thread
    // from the one that adds item to the list.
    absl::Mutex mutex;
  };
  PendingActivityRecords pending_activity_records_;

 public:
  // Disable copy and move.
  RocmTracer(const RocmTracer&) = delete;
  RocmTracer& operator=(const RocmTracer&) = delete;
};

}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_
