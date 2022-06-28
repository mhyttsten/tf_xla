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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc() {
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

#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "rocm/rocm_config.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/profiler/internal/cpu/annotation_stack.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

#define RETURN_IF_ROCTRACER_ERROR(expr)                                      \
  do {                                                                       \
    roctracer_status_t status = expr;                                        \
    if (status != ROCTRACER_STATUS_SUCCESS) {                                \
      const char* errstr = wrap::roctracer_error_string();                   \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr;  \
      return errors::Internal(absl::StrCat("roctracer call error", errstr)); \
    }                                                                        \
  } while (false)

namespace {

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
int32_t GetCachedTID() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetCachedTID");

  static thread_local int32_t current_thread_id =
      Env::Default()->GetCurrentThreadId();
  return current_thread_id;
}

const char* GetActivityDomainName(uint32_t domain) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetActivityDomainName");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return "HSA API";
    case ACTIVITY_DOMAIN_HSA_OPS:
      return "HSA OPS";
    case ACTIVITY_DOMAIN_HIP_OPS:
      return "HIP OPS/HCC/VDI";
    case ACTIVITY_DOMAIN_HIP_API:
      return "HIP API";
    case ACTIVITY_DOMAIN_KFD_API:
      return "KFD API";
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT API";
    case ACTIVITY_DOMAIN_ROCTX:
      return "ROCTX";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

string GetActivityDomainOpName(uint32_t domain, uint32_t op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetActivityDomainOpName");

  std::ostringstream oss;
  oss << GetActivityDomainName(domain) << " - ";
  switch (domain) {
    case ACTIVITY_DOMAIN_HIP_API:
      oss << hip_api_name(op);
      break;
    default:
      oss << op;
      break;
  }
  return oss.str();
}

const char* GetActivityPhaseName(uint32_t phase) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetActivityPhaseName");

  switch (phase) {
    case ACTIVITY_API_PHASE_ENTER:
      return "ENTER";
    case ACTIVITY_API_PHASE_EXIT:
      return "EXIT";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

inline void DumpApiCallbackData(uint32_t domain, uint32_t cbid,
                                const void* cbdata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_4(mht_4_v, 289, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "DumpApiCallbackData");

  std::ostringstream oss;
  oss << "API callback for " << GetActivityDomainName(domain);
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t* data =
        reinterpret_cast<const hip_api_data_t*>(cbdata);
    oss << " - " << hip_api_name(cbid);
    oss << ", correlation_id=" << data->correlation_id;
    oss << ", phase=" << GetActivityPhaseName(data->phase);
    switch (cbid) {
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
      case HIP_API_ID_hipLaunchKernel:
        break;
      case HIP_API_ID_hipMemcpyDtoH:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemsetD32:
        oss << ", value=" << data->args.hipMemsetD32.value;
        oss << ", count=" << data->args.hipMemsetD32.count;
        break;
      case HIP_API_ID_hipMemsetD32Async:
        oss << ", value=" << data->args.hipMemsetD32Async.value;
        oss << ", count=" << data->args.hipMemsetD32Async.count;
        break;
      case HIP_API_ID_hipMemsetD8:
        oss << ", value=" << data->args.hipMemsetD8.value;
        oss << ", count=" << data->args.hipMemsetD8.count;
        break;
      case HIP_API_ID_hipMemsetD8Async:
        oss << ", value=" << data->args.hipMemsetD8Async.value;
        oss << ", count=" << data->args.hipMemsetD8Async.count;
        break;
      case HIP_API_ID_hipMalloc:
        oss << ", size=" << data->args.hipMalloc.size;
        break;
      case HIP_API_ID_hipFree:
        oss << ", ptr=" << data->args.hipFree.ptr;
        break;
      case HIP_API_ID_hipStreamSynchronize:
        break;
      default:
        DCHECK(false);
        break;
    }
  } else {
    oss << ": " << cbid;
  }
  VLOG(3) << oss.str();
}

void DumpActivityRecord(const roctracer_record_t* record,
                        std::string extra_info) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("extra_info: \"" + extra_info + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_5(mht_5_v, 364, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "DumpActivityRecord");

  std::ostringstream oss;
  oss << "Activity callback for " << GetActivityDomainName(record->domain);
  oss << ", op name= "
      << wrap::roctracer_op_string(record->domain, record->op, record->kind);
  oss << ", correlation_id=" << record->correlation_id;
  oss << ", begin_ns=" << record->begin_ns;
  oss << ", end_ns=" << record->end_ns;
  oss << ", duration=" << record->end_ns - record->begin_ns;
  oss << ", device_id=" << record->device_id;
  oss << ", queue_id=" << record->queue_id;
  oss << ", process_id=" << record->process_id;
  oss << ", thread_id=" << record->thread_id;
  oss << ", external_id=" << record->external_id;
  oss << ", bytes=" << record->bytes;
  oss << ", domain=" << record->domain;
  oss << ", op=" << record->op;
  oss << ", kind=" << record->kind;
  oss << ", extra_info=" << extra_info;
  VLOG(3) << oss.str();
}

}  // namespace

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_6(mht_6_v, 391, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetRocmTracerEventTypeName");

  switch (type) {
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::Kernel:
      return "Kernel";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::Generic:
      return "Generic";
    case RocmTracerEventType::Synchronization:
      return "Synchronization";
    case RocmTracerEventType::Memset:
      return "Memset";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_7(mht_7_v, 423, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetRocmTracerEventSourceName");

  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    case RocmTracerEventSource::Invalid:
      return "Invalid";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

// FIXME(rocm-profiler): These domain names are not consistent with the
// GetActivityDomainName function
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_8(mht_8_v, 446, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "GetRocmTracerEventDomainName");

  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HCC_OPS:
      return "HCC_OPS";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

void DumpRocmTracerEvent(const RocmTracerEvent& event,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns,
                         const string& message) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_9(mht_9_v, 467, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "DumpRocmTracerEvent");

  std::ostringstream oss;
  oss << "correlation_id=" << event.correlation_id;
  oss << ",type=" << GetRocmTracerEventTypeName(event.type);
  oss << ",source=" << GetRocmTracerEventSourceName(event.source);
  oss << ",domain=" << GetRocmTracerEventDomainName(event.domain);
  oss << ",name=" << event.name;
  oss << ",annotation=" << event.annotation;
  oss << ",start_time_us="
      << (start_walltime_ns + (start_gputime_ns - event.start_time_ns)) / 1000;
  oss << ",duration=" << (event.end_time_ns - event.start_time_ns) / 1000;
  oss << ",device_id=" << event.device_id;
  oss << ",thread_id=" << event.thread_id;
  oss << ",stream_id=" << event.stream_id;

  switch (event.type) {
    case RocmTracerEventType::Kernel:
      break;
    case RocmTracerEventType::MemcpyD2H:
    case RocmTracerEventType::MemcpyH2D:
    case RocmTracerEventType::MemcpyD2D:
    case RocmTracerEventType::MemcpyP2P:
      oss << ",num_bytes=" << event.memcpy_info.num_bytes;
      oss << ",destination=" << event.memcpy_info.destination;
      oss << ",async=" << event.memcpy_info.async;
      break;
    case RocmTracerEventType::MemoryAlloc:
      oss << ",num_bytes=" << event.memalloc_info.num_bytes;
      break;
    case RocmTracerEventType::Synchronization:
      break;
    case RocmTracerEventType::Generic:
      break;
    default:
      DCHECK(false);
      break;
  }
  oss << message;
  VLOG(3) << oss.str();
}

Status RocmApiCallbackImpl::operator()(uint32_t domain, uint32_t cbid,
                                       const void* cbdata) {
  /* Some APIs such as hipMalloc, implicitly work on th devices set by the
    user using APIs such as hipSetDevice. API callbacks and activity records
    for functions like hipMalloc does not return the device id (CUDA does). To
    solve this we need to track the APIs that select the device (such as
    hipSetDevice) for each thread.
    */

  thread_local uint32_t default_device = 0;

  // DumpApiCallbackData(domain, cbid, cbdata);

  if (domain != ACTIVITY_DOMAIN_HIP_API) return Status::OK();

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(cbdata);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    if (options_.api_tracking_set.find(cbid) !=
        options_.api_tracking_set.end()) {
      mutex_lock lock(api_call_start_mutex_);
      api_call_start_time_.emplace(data->correlation_id,
                                   RocmTracer::GetTimestamp());
    }

    if (cbid == HIP_API_ID_hipSetDevice) {
      default_device = data->args.hipSetDevice.deviceId;
    }
  } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    uint64_t enter_time = 0, exit_time = 0;

    if (options_.api_tracking_set.find(cbid) !=
        options_.api_tracking_set.end()) {
      mutex_lock lock(api_call_start_mutex_);
      if (api_call_start_time_.find(data->correlation_id) !=
          api_call_start_time_.end()) {
        enter_time = api_call_start_time_.at(data->correlation_id);
        api_call_start_time_.erase(data->correlation_id);
      } else {
        LOG(WARNING) << "An API exit callback received without API enter "
                        "with same correlation id. Event droped!";
        return Status::OK();  // This API does not belong to us.
      }
      exit_time = RocmTracer::GetTimestamp();
    }
    // Set up the map from correlation id to annotation string.
    const std::string& annotation = AnnotationStack::Get();
    if (!annotation.empty()) {
      collector_->annotation_map()->Add(data->correlation_id, annotation);
    }

    if (options_.api_tracking_set.find(cbid) ==
        options_.api_tracking_set.end()) {
      VLOG(3) << "API callback is from the auxilarity list. Corr. id="
              << data->correlation_id;
    }
    DumpApiCallbackData(domain, cbid, cbdata);

    switch (cbid) {
      // star in comments means it does not exist in the driver wrapper
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:  // *
      case HIP_API_ID_hipHccModuleLaunchKernel:  // *
      case HIP_API_ID_hipLaunchKernel:           // *

        this->AddKernelEventUponApiExit(cbid, data, enter_time, exit_time);

        // Add the correlation_ids for these events to the pending set
        // so that we can explicitly wait for their corresponding
        // HIP runtime activity records, before exporting the trace data
        tracer_->AddToPendingActivityRecords(data->correlation_id);
        break;
      case HIP_API_ID_hipMemcpy:
      case HIP_API_ID_hipMemcpyDtoH:
      case HIP_API_ID_hipMemcpyDtoHAsync:
      case HIP_API_ID_hipMemcpyHtoD:
      case HIP_API_ID_hipMemcpyHtoDAsync:
      case HIP_API_ID_hipMemcpyDtoD:
      case HIP_API_ID_hipMemcpyDtoDAsync:
      case HIP_API_ID_hipMemcpyAsync:
        this->AddNormalMemcpyEventUponApiExit(cbid, data, enter_time,
                                              exit_time);
        tracer_->AddToPendingActivityRecords(data->correlation_id);
        break;
      case HIP_API_ID_hipMemset:
      case HIP_API_ID_hipMemsetAsync:
      case HIP_API_ID_hipMemsetD32:
      case HIP_API_ID_hipMemsetD32Async:
      case HIP_API_ID_hipMemsetD16:
      case HIP_API_ID_hipMemsetD16Async:
      case HIP_API_ID_hipMemsetD8:
      case HIP_API_ID_hipMemsetD8Async:
        this->AddMemsetEventUponApiExit(cbid, data, enter_time, exit_time);
        break;
      case HIP_API_ID_hipMalloc:
      case HIP_API_ID_hipMallocPitch:
      case HIP_API_ID_hipHostMalloc:
      case HIP_API_ID_hipFree:
      case HIP_API_ID_hipHostFree:
        this->AddMallocFreeEventUponApiExit(cbid, data, default_device,
                                            enter_time, exit_time);
        break;
      case HIP_API_ID_hipStreamSynchronize:
      case HIP_API_ID_hipStreamWaitEvent:
        // case HIP_API_ID_hipEventSynchronize:
        this->AddSynchronizeEventUponApiExit(cbid, data, enter_time, exit_time);
        break;
      case HIP_API_ID_hipSetDevice:
        // we track this ID only to find the device ID
        //  for the current thread.
        break;
      default:
        //
        LOG(WARNING) << "API call "
                     << wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid,
                                                  0)
                     << ", corr. id=" << data->correlation_id
                     << " dropped. No capturing function was found!";
        // AddGenericEventUponApiExit(cbid, data);
        break;
    }
  }
  return Status::OK();
}

void RocmApiCallbackImpl::AddKernelEventUponApiExit(uint32_t cbid,
                                                    const hip_api_data_t* data,
                                                    const uint64_t enter_time,
                                                    const uint64_t exit_time) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_10(mht_10_v, 639, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddKernelEventUponApiExit");

  /*
  extra fields:
    kernel_info, domain

  missing fields:
    context_id
  */
  RocmTracerEvent event;

  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x = data->args.hipModuleLaunchKernel.gridDimX;
      event.kernel_info.grid_y = data->args.hipModuleLaunchKernel.gridDimY;
      event.kernel_info.grid_z = data->args.hipModuleLaunchKernel.gridDimZ;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipModuleLaunchKernel.stream;
      // TODO(rocm-profiler): wrap this API if possible.
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipExtModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipExtModuleLaunchKernel.sharedMemBytes;
      unsigned int blockDimX =
          data->args.hipExtModuleLaunchKernel.localWorkSizeX;
      unsigned int blockDimY =
          data->args.hipExtModuleLaunchKernel.localWorkSizeY;
      unsigned int blockDimZ =
          data->args.hipExtModuleLaunchKernel.localWorkSizeZ;

      event.kernel_info.block_x = blockDimX;
      event.kernel_info.block_y = blockDimY;
      event.kernel_info.block_z = blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeX / blockDimX;
      event.kernel_info.grid_y =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeY / blockDimY;
      event.kernel_info.grid_z =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeZ / blockDimZ;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipExtModuleLaunchKernel.hStream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipHccModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipHccModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipHccModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipHccModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeX /
          event.kernel_info.block_x;
      event.kernel_info.grid_y =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeY /
          event.kernel_info.block_y;
      event.kernel_info.grid_z =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeZ /
          event.kernel_info.block_z;
      event.kernel_info.func_ptr = kernelFunc;
      const hipStream_t& stream = data->args.hipHccModuleLaunchKernel.hStream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
    } break;
    case HIP_API_ID_hipLaunchKernel: {
      const void* func_addr = data->args.hipLaunchKernel.function_address;
      hipStream_t stream = data->args.hipLaunchKernel.stream;
      if (func_addr != nullptr)
        event.name = hipKernelNameRefByPtr(func_addr, stream);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipLaunchKernel.dimBlocks.x;
      event.kernel_info.block_y = data->args.hipLaunchKernel.dimBlocks.y;
      event.kernel_info.block_z = data->args.hipLaunchKernel.dimBlocks.z;
      event.kernel_info.grid_x = data->args.hipLaunchKernel.numBlocks.x;
      event.kernel_info.grid_y = data->args.hipLaunchKernel.numBlocks.y;
      event.kernel_info.grid_z = data->args.hipLaunchKernel.numBlocks.z;
      event.kernel_info.func_ptr = (void*)func_addr;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
  }
  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddNormalMemcpyEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_11(mht_11_v, 753, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddNormalMemcpyEventUponApiExit");

  /*
    missing:
      device_id(partially, have only for async), context_id,
    memcpy_info.kind(CUPTI puts CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN),
      memcpy_info.destenation(partially, only for async)( CUPTI puts device_id),

    extra:
      domain, name,
  */
  // for CUDA, it does NOT capture stream id for these types

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  /* The general hipMemcpy or hipMemcpyAsync can support any kind of memory
  copy operation, such as H2D, D2D, P2P, and D2H. Here we use MemcpyOther for
  all api calls with HipMemcpy(+Async) to carry-on this generality.
  We also assume that if we want to copy data BETWEEN devices, we do not use
  hipMemcpy(+Async) or hipMemcpyDtoD(+Async) as we explicitly always set the
  destenation as the source device id). Ultimately, to figure out the actual
  device we can use hipPointerGetAttributes but we do not do that now .In the
  other words, we assume we use hipMemcpyPeer to achieve the copy between
  devices.
  */

  switch (cbid) {
    case HIP_API_ID_hipMemcpyDtoH: {
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoH.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyDtoHAsync: {
      event.type = RocmTracerEventType::MemcpyD2H;
      const hipStream_t& stream = data->args.hipMemcpyDtoHAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoHAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpyHtoD: {
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoD.sizeBytes;
      event.memcpy_info.async = false;
      // we set the destenattion device id for it using the device id we get
      // from activities when they exchange information before flushing
    } break;
    case HIP_API_ID_hipMemcpyHtoDAsync: {
      event.type = RocmTracerEventType::MemcpyH2D;
      const hipStream_t& stream = data->args.hipMemcpyHtoDAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpyDtoD: {
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoD.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyDtoDAsync: {
      event.type = RocmTracerEventType::MemcpyD2D;
      const hipStream_t& stream = data->args.hipMemcpyDtoDAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    case HIP_API_ID_hipMemcpy: {
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.num_bytes = data->args.hipMemcpy.sizeBytes;
      event.memcpy_info.async = false;
    } break;
    case HIP_API_ID_hipMemcpyAsync: {
      event.type = RocmTracerEventType::MemcpyOther;
      const hipStream_t& stream = data->args.hipMemcpyAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
      event.memcpy_info.num_bytes = data->args.hipMemcpyAsync.sizeBytes;
      event.memcpy_info.async = true;
      event.memcpy_info.destination = event.device_id;
    } break;
    default:
      LOG(WARNING) << "Unsupported Memcpy API for profiling observed for cbid="
                   << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}
void RocmApiCallbackImpl::AddMemcpyPeerEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_12(mht_12_v, 856, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddMemcpyPeerEventUponApiExit");

  /*
    missing: context_id, memcpy_info.kind

    extra: domain, name,
  */

  RocmTracerEvent event;
  event.type = RocmTracerEventType::MemcpyP2P;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMemcpyPeer:
      event.device_id = data->args.hipMemcpyPeer.srcDeviceId;
      event.memcpy_info.destination = data->args.hipMemcpyPeer.dstDeviceId;
      event.memcpy_info.num_bytes = data->args.hipMemcpyPeer.sizeBytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyPeerAsync:
      event.device_id = data->args.hipMemcpyPeerAsync.srcDevice;
      event.memcpy_info.destination = data->args.hipMemcpyPeerAsync.dstDeviceId;
      event.memcpy_info.num_bytes = data->args.hipMemcpyPeerAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    default:
      LOG(WARNING)
          << "Unsupported MemcpyPeer API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}
void RocmApiCallbackImpl::AddMemsetEventUponApiExit(uint32_t cbid,
                                                    const hip_api_data_t* data,
                                                    uint64_t enter_time,
                                                    uint64_t exit_time) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_13(mht_13_v, 904, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddMemsetEventUponApiExit");

  /*
    misses:
      device_id(only avail. for async), context_id

    extras:
      domain, name
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMemsetD8:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetD8.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetD8Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD8Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemsetD16:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 2 * data->args.hipMemsetD16.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD16Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 2 * data->args.hipMemsetD16Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD16Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemsetD32:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 4 * data->args.hipMemsetD32.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = 4 * data->args.hipMemsetD32Async.count;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetD32Async.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipMemset:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemset.sizeBytes;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetAsync: {
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_bytes = data->args.hipMemsetAsync.sizeBytes;
      event.memset_info.async = true;
      const hipStream_t& stream = data->args.hipMemsetAsync.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    default:
      LOG(WARNING) << "Unsupported Memset API for profiling observed for cbid="
                   << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddMallocFreeEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint32_t device_id,
    uint64_t enter_time, uint64_t exit_time) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_14(mht_14_v, 988, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddMallocFreeEventUponApiExit");

  /*
    misses: context_id

    extras: domain
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = (cbid == HIP_API_ID_hipFree || cbid == HIP_API_ID_hipHostFree)
                   ? RocmTracerEventType::MemoryFree
                   : RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.device_id = device_id;
  event.thread_id = GetCachedTID();
  // We do not set stream_id (probably to zero as Malloc etc. commands seems
  // to run on  default stream). Later we use the unassigned stream_id as a
  // feature to assign events to host or device.
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipMalloc:
      event.memalloc_info.num_bytes = data->args.hipMalloc.size;
      break;
    case HIP_API_ID_hipMallocPitch:
      event.memalloc_info.num_bytes = data->args.hipMallocPitch.pitch__val *
                                      data->args.hipMallocPitch.height;
      break;
    case HIP_API_ID_hipHostMalloc:
      event.memalloc_info.num_bytes = data->args.hipHostMalloc.size;
      break;
    case HIP_API_ID_hipFree:
    case HIP_API_ID_hipHostFree:
      event.memalloc_info.num_bytes = 0;
      break;
    default:
      LOG(WARNING)
          << "Unsupported Malloc/Free API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }

  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

void RocmApiCallbackImpl::AddSynchronizeEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data, uint64_t enter_time,
    uint64_t exit_time) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_15(mht_15_v, 1044, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmApiCallbackImpl::AddSynchronizeEventUponApiExit");

  // TODO(rocm-profiler): neither CUDA and nor we capture annotaint for this
  // event
  /*
    misses: context_id

    extras: domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Synchronization;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  event.start_time_ns = enter_time;
  event.end_time_ns = exit_time;

  switch (cbid) {
    case HIP_API_ID_hipStreamSynchronize: {
      event.synchronization_info.sync_type =
          RocmTracerSyncTypes::StreamSynchronize;
      const hipStream_t& stream = data->args.hipStreamSynchronize.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    case HIP_API_ID_hipStreamWaitEvent: {
      event.synchronization_info.sync_type = RocmTracerSyncTypes::StreamWait;
      const hipStream_t& stream = data->args.hipStreamWaitEvent.stream;
      event.device_id = hipGetStreamDeviceId(stream);
    } break;
    default:
      LOG(WARNING)
          << "Unsupported Synchronization API for profiling observed for cbid="
          << cbid << ". Event dropped!";
      return;
      break;
  }
  bool is_auxiliary =
      options_.api_tracking_set.find(cbid) == options_.api_tracking_set.end();
  collector_->AddEvent(std::move(event), is_auxiliary);
}

Status RocmActivityCallbackImpl::operator()(const char* begin,
                                            const char* end) {
  // we do not dump activities in this set in logger

  static std::set<activity_op_t> dump_excluded_activities = {
      HIP_API_ID_hipGetDevice,
      HIP_API_ID_hipSetDevice,
      HIP_API_ID___hipPushCallConfiguration,
      HIP_API_ID___hipPopCallConfiguration,
      HIP_API_ID_hipEventQuery,
      HIP_API_ID_hipCtxSetCurrent,
      HIP_API_ID_hipEventRecord,
      HIP_API_ID_hipEventQuery,
      HIP_API_ID_hipGetDeviceProperties,
      HIP_API_ID_hipPeekAtLastError,
      HIP_API_ID_hipModuleGetFunction,
      HIP_API_ID_hipEventCreateWithFlags};

  const roctracer_record_t* record =
      reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record =
      reinterpret_cast<const roctracer_record_t*>(end);

  while (record < end_record) {
    // DumpActivityRecord(record);

    switch (record->domain) {
      // HIP API activities.
      case ACTIVITY_DOMAIN_HIP_API:
        switch (record->op) {
          case HIP_API_ID_hipModuleLaunchKernel:
          case HIP_API_ID_hipExtModuleLaunchKernel:
          case HIP_API_ID_hipHccModuleLaunchKernel:
          case HIP_API_ID_hipLaunchKernel:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipKernelActivityEvent(record);
            break;
          case HIP_API_ID_hipMemcpyDtoH:
          case HIP_API_ID_hipMemcpyHtoD:
          case HIP_API_ID_hipMemcpyDtoD:
          case HIP_API_ID_hipMemcpyDtoHAsync:
          case HIP_API_ID_hipMemcpyHtoDAsync:
          case HIP_API_ID_hipMemcpyDtoDAsync:
          case HIP_API_ID_hipMemcpyAsync:
          case HIP_API_ID_hipMemcpy:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddNormalHipMemcpyActivityEvent(record);
            break;
          case HIP_API_ID_hipMemset:
          case HIP_API_ID_hipMemsetAsync:
          case HIP_API_ID_hipMemsetD32:
          case HIP_API_ID_hipMemsetD32Async:
          case HIP_API_ID_hipMemsetD16:
          case HIP_API_ID_hipMemsetD16Async:
          case HIP_API_ID_hipMemsetD8:
          case HIP_API_ID_hipMemsetD8Async:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipMemsetActivityEvent(record);
            break;

          case HIP_API_ID_hipMalloc:
          case HIP_API_ID_hipMallocPitch:
          case HIP_API_ID_hipHostMalloc:
          case HIP_API_ID_hipFree:
          case HIP_API_ID_hipHostFree:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipMallocActivityEvent(record);
            break;
          case HIP_API_ID_hipStreamSynchronize:
          case HIP_API_ID_hipStreamWaitEvent:
            // case HIP_API_ID_hipStreamWaitEvent:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHipStreamSynchronizeActivityEvent(record);
            break;

          default:
            if (dump_excluded_activities.find(record->op) ==
                dump_excluded_activities.end()) {
              string drop_message(
                  "\nNot in the API tracked activities. Dropped!");
              DumpActivityRecord(record, drop_message);
            }
            break;
        }  // switch (record->op).
        break;

      // HCC ops activities.
      case ACTIVITY_DOMAIN_HIP_OPS:

        switch (record->op) {
          case HIP_OP_ID_DISPATCH:
            DumpActivityRecord(record, std::to_string(__LINE__));
            AddHccKernelActivityEvent(record);
            tracer_->RemoveFromPendingActivityRecords(record->correlation_id);
            break;
          case HIP_OP_ID_COPY:
            switch (record->kind) {
              // TODO(rocm-profiler): use enum instead.
              case 4595:   /*CopyDeviceToHost*/
              case 4596:   /*CopyDeviceToDevice*/
              case 4597: { /*CopyHostToDevice*/
                /*MEMCPY*/
                // roctracer returns CopyHostToDevice for hipMemcpyDtoD API
                //  Please look at the issue #53 in roctracer GitHub repo.
                DumpActivityRecord(record, "");
                AddNormalHipOpsMemcpyActivityEvent(record);
                tracer_->RemoveFromPendingActivityRecords(
                    record->correlation_id);
              } break;
              case 4615: /*FillBuffer*/
                /*MEMSET*/
                DumpActivityRecord(record, "");
                AddHipOpsMemsetActivityEvent(record);
                break;
              case 4606: /*MARKER*/
                // making the log shorter.
                // markers are with 0ns duration.
                break;
              default:
                string drop_message(
                    "\nNot in the HIP-OPS-COPY tracked activities. Dropeed!");
                DumpActivityRecord(record, drop_message);
                break;
            }  // switch (record->kind)
            break;
          default:
            string drop_message(
                "\nNot in the HIP-OPS tracked activities. Dropped!");
            DumpActivityRecord(record, drop_message);
            break;
        }  // switch (record->op).
        break;
      default:
        string drop_message("\nNot in the tracked domain activities. Dropped!");
        DumpActivityRecord(record, drop_message);
        break;
    }

    RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
        roctracer_next_record(record, &record)));
  }

  return Status::OK();
}

void RocmActivityCallbackImpl::AddHipKernelActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_16(mht_16_v, 1236, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHipKernelActivityEvent");

  /*
  missing:
   name, device_id(got from hcc), context_id, stream_id(got from hcc),
 nvtx_range, kernel_info

  extra:
   domain
 activity record contains process/thread ID
 */
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  // event.name =  /* we use the API name instead*/
  //    wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  // TODO(rocm-profiler): CUDA uses device id and correlation ID for finding
  // annotations.
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddNormalHipMemcpyActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_17(mht_17_v, 1267, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddNormalHipMemcpyActivityEvent");

  /*
  ---------------NormalMemcpy-------------------
    misses:context_id, memcpy_info.kind, memcpy_info.srckind,
  memcpy_info.dstkind, memcpy_info.num_bytes, memcpy_info.destenation,
  device_id, stream_id,

    extras: domain
  ---------------PeerMemcpy---------------------
    misses: device_id, context_id, stream_id, memcpy_info.kind,
      memcpy_info.num_bytes, memcpy_info.destination,
    extras:
      domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  // TODO(roc-profiler): record->bytes is not a valid value
  // event.memcpy_info.num_bytes = record->bytes;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  switch (record->op) {
    case HIP_API_ID_hipMemcpyDtoH:
    case HIP_API_ID_hipMemcpyDtoHAsync:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyDtoHAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyHtoD:
    case HIP_API_ID_hipMemcpyHtoDAsync:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyHtoDAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyDtoD:
    case HIP_API_ID_hipMemcpyDtoDAsync:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyDtoDAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpy:
    case HIP_API_ID_hipMemcpyAsync:
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyAsync) ? true : false;
      break;
    case HIP_API_ID_hipMemcpyPeer:
    case HIP_API_ID_hipMemcpyPeerAsync:
      event.type = RocmTracerEventType::MemcpyP2P;
      event.memcpy_info.async =
          (record->op == HIP_API_ID_hipMemcpyPeerAsync) ? true : false;
      break;
    default:
      LOG(WARNING) << "Unsupported Memcpy/MemcpyPeer activity for profiling "
                      "observed for cbid="
                   << record->op << ". Event dropped!";
      return;
      break;
  }

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipMemsetActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_18(mht_18_v, 1339, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHipMemsetActivityEvent");

  /*
    misses:
      device_id, context_id, stram_id, memset_info.num_bytes
      memset_info.kind

    extras:
      domain, annotation
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.type = RocmTracerEventType::Memset;

  switch (record->op) {
    case HIP_API_ID_hipMemset:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetAsync:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD8:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD16:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD16Async:
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD32:
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async:
      event.memset_info.async = true;
      break;
  }

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipMallocActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_19(mht_19_v, 1396, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHipMallocActivityEvent");

  /*
    misses: device_id, context_id, memory_residency_info (num_byts, kind,
    address)

    extras:
      annotation, domain,
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  // similar to CUDA we set this to the default stream
  event.stream_id = 0;
  event.start_time_ns = record->begin_ns;
  // making sure it does not have 0ns duration. Otherwise, it may not show up in
  // the trace view
  event.end_time_ns = std::max(record->end_ns, record->begin_ns + 1);

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipStreamSynchronizeActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_20(mht_20_v, 1427, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHipStreamSynchronizeActivityEvent");

  /*
  misses: context_id, device_id (cuda also does not provide but we can get from
  API-CB)

  extras: domain, synchronization_info.sync_type, annotation
  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Synchronization;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  event.start_time_ns = record->begin_ns;

  // making sure it does not have 0ns duration. Otherwise, it may not show up in
  // the trace view
  event.end_time_ns = std::max(record->end_ns, record->begin_ns + 1);

  switch (record->op) {
    case HIP_API_ID_hipStreamSynchronize:
      event.synchronization_info.sync_type =
          RocmTracerSyncTypes::StreamSynchronize;
      break;
    case HIP_API_ID_hipStreamWaitEvent:
      event.synchronization_info.sync_type = RocmTracerSyncTypes::StreamWait;
      break;
    default:
      event.synchronization_info.sync_type = RocmTracerSyncTypes::InvalidSync;
      break;
  }
  collector_->AddEvent(std::move(event), false);
}

// TODO(rocm-profiler): rename this function. this is HIP-OP
void RocmActivityCallbackImpl::AddHccKernelActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_21(mht_21_v, 1469, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHccKernelActivityEvent");

  /*
   missing:
     name, context_id, nvtx_range, kernel_info

   extra:
     domain (thread id from the HIP activity)

   activity record contains device/stream ID
 */
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HCC_OPS;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);
  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddNormalHipOpsMemcpyActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_22(mht_22_v, 1497, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddNormalHipOpsMemcpyActivityEvent");

  /*
    misses:
      type, name(the name set here is not clear enough but we keep it for
    debug), context_id, memcpy_info.kind, memcpy_info.num_bytes,
    memcpy_info.async, memcpy_info.src_mem_kind, memcpy_info.dst_mem_kind

    extras:
      domain,

  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HCC_OPS;
  event.source = RocmTracerEventSource::Activity;
  event.name =  // name is stored for debug
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.memcpy_info.destination = event.device_id;
  event.stream_id = record->queue_id;

  // we set the type as MemcpyOther as HIP-OPS activity record does not carry
  // this information
  event.type = RocmTracerEventType::MemcpyOther;

  collector_->AddEvent(std::move(event), false);
}

void RocmActivityCallbackImpl::AddHipOpsMemsetActivityEvent(
    const roctracer_record_t* record) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_23(mht_23_v, 1534, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmActivityCallbackImpl::AddHipOpsMemsetActivityEvent");

  /*
    misses:
      name (name recorder here is not clear enough for Memset. We only capture
    it for debug), context_id, memset_info.kind, memset_info.num_bytes,
    memset_info.async

    extras:
      dommain, annotation,

  */

  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HCC_OPS;
  event.source = RocmTracerEventSource::Activity;
  event.name =  // name is stored for debug
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  event.type = RocmTracerEventType::Memset;

  collector_->AddEvent(std::move(event), false);
}

void AnnotationMap::Add(uint32_t correlation_id,
                        const std::string& annotation) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("annotation: \"" + annotation + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_24(mht_24_v, 1569, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "AnnotationMap::Add");

  if (annotation.empty()) return;
  VLOG(3) << "Add annotation: "
          << " correlation_id=" << correlation_id
          << ", annotation: " << annotation;
  absl::MutexLock lock(&map_.mutex);
  if (map_.annotations.size() < max_size_) {
    absl::string_view annotation_str =
        *map_.annotations.insert(annotation).first;
    map_.correlation_map.emplace(correlation_id, annotation_str);
  }
}

absl::string_view AnnotationMap::LookUp(uint32_t correlation_id) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_25(mht_25_v, 1585, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "AnnotationMap::LookUp");

  absl::MutexLock lock(&map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second : absl::string_view();
}

/* static */ RocmTracer* RocmTracer::GetRocmTracerSingleton() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_26(mht_26_v, 1594, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::GetRocmTracerSingleton");

  static auto* singleton = new RocmTracer();
  return singleton;
}

// FIXME(rocm-profiler): we should also check if we have AMD GPUs
bool RocmTracer::IsAvailable() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_27(mht_27_v, 1603, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::IsAvailable");

  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

int RocmTracer::NumGpus() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_28(mht_28_v, 1610, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::NumGpus");

  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_29(mht_29_v, 1629, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::Enable");

  options_ = options;
  collector_ = collector;
  api_cb_impl_ = new RocmApiCallbackImpl(options, this, collector);
  activity_cb_impl_ = new RocmActivityCallbackImpl(options, this, collector);

  // From ROCm 3.5 onwards, the following call is required.
  // don't quite know what it does (no documentation!), only that without it
  // the call to enable api/activity tracing will run into a segfault
  wrap::roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  EnableApiTracing().IgnoreError();
  EnableActivityTracing().IgnoreError();
  LOG(INFO) << "GpuTracer started";
}

void RocmTracer::Disable() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_30(mht_30_v, 1648, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::Disable");

  // TODO(rocm-profiler): TF has a SyncAndFlush() function
  // to be called before disabling. It makes sure all the contexts
  // has finished all the tasks before shutting down the profiler
  DisableApiTracing().IgnoreError();
  DisableActivityTracing().IgnoreError();
  delete api_cb_impl_;
  delete activity_cb_impl_;
  collector_->Flush();
  collector_ = nullptr;
  options_.reset();
  LOG(INFO) << "GpuTracer stopped";
}

void ApiCallback(uint32_t domain, uint32_t cbid, const void* cbdata,
                 void* user_data) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_31(mht_31_v, 1666, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "ApiCallback");

  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ApiCallbackHandler(domain, cbid, cbdata);
}

void RocmTracer::ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                    const void* cbdata) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_32(mht_32_v, 1675, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::ApiCallbackHandler");

  if (api_tracing_enabled_) (*api_cb_impl_)(domain, cbid, cbdata);
}

Status RocmTracer::EnableApiTracing() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_33(mht_33_v, 1682, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::EnableApiTracing");

  if (api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = true;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_enable_domain_callback(domain, ApiCallback, this));
    } else {
      VLOG(3) << "Enabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_enable_op_callback(domain, op, ApiCallback, this));
      }
    }
  }
  return Status::OK();
}

Status RocmTracer::DisableApiTracing() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_34(mht_34_v, 1711, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::DisableApiTracing");

  if (!api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = false;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_disable_domain_callback(domain));
    } else {
      VLOG(3) << "Disabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_disable_op_callback(domain, op));
      }
    }
  }
  return Status::OK();
}

void ActivityCallback(const char* begin, const char* end, void* user_data) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("begin: \"" + (begin == nullptr ? std::string("nullptr") : std::string((char*)begin)) + "\"");
   mht_35_v.push_back("end: \"" + (end == nullptr ? std::string("nullptr") : std::string((char*)end)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_35(mht_35_v, 1742, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "ActivityCallback");

  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ActivityCallbackHandler(begin, end);
}

void RocmTracer::ActivityCallbackHandler(const char* begin, const char* end) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("begin: \"" + (begin == nullptr ? std::string("nullptr") : std::string((char*)begin)) + "\"");
   mht_36_v.push_back("end: \"" + (end == nullptr ? std::string("nullptr") : std::string((char*)end)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_36(mht_36_v, 1752, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::ActivityCallbackHandler");

  if (activity_tracing_enabled_) {
    (*activity_cb_impl_)(begin, end);
  } else {
    LOG(WARNING) << "ActivityCallbackHandler called when "
                    "activity_tracing_enabled_ is false";

    VLOG(3) << "Dropped Activity Records Start";
    const roctracer_record_t* record =
        reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);
    while (record < end_record) {
      DumpActivityRecord(record,
                         "activity_tracing_enabled_ is false. Dropped!");
      roctracer_next_record(record, &record);
    }
    VLOG(3) << "Dropped Activity Records End";
  }
}

Status RocmTracer::EnableActivityTracing() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_37(mht_37_v, 1776, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::EnableActivityTracing");

  if (activity_tracing_enabled_) return Status::OK();
  activity_tracing_enabled_ = true;

  if (!options_->activity_tracing.empty()) {
    // Create the memory pool to store activity records in
    if (wrap::roctracer_default_pool_expl(nullptr) == NULL) {
      roctracer_properties_t properties{};
      properties.buffer_size = 0x1000;
      properties.buffer_callback_fun = ActivityCallback;
      properties.buffer_callback_arg = this;
      VLOG(3) << "Creating roctracer activity buffer";
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_open_pool_expl(&properties, nullptr));
    }
  }

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_enable_domain_activity_expl(domain, nullptr));
    } else {
      VLOG(3) << "Enabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        // roctracer library has not exported "roctracer_enable_op_activity"
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_enable_op_activity_expl(domain, op, nullptr));
      }
    }
  }

  return Status::OK();
}

Status RocmTracer::DisableActivityTracing() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_38(mht_38_v, 1820, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::DisableActivityTracing");

  if (!activity_tracing_enabled_) return Status::OK();

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_disable_domain_activity(domain));
    } else {
      VLOG(3) << "Disabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_disable_op_activity(domain, op));
      }
    }
  }

  // TODO(rocm-profiler): this stopping mechanism needs improvement.
  // Flush the activity buffer BEFORE setting the activity_tracing_enable_
  // flag to FALSE. This is because the activity record callback routine is
  // gated by the same flag
  VLOG(3) << "Flushing roctracer activity buffer";
  RETURN_IF_ROCTRACER_ERROR(wrap::roctracer_flush_activity_expl(nullptr));
  // roctracer_flush_buf();

  // Explicitly wait for (almost) all pending activity records
  // The choice of all of the following is based what seemed to work
  // best when enabling tracing on a large testcase (BERT)
  // * 100 ms as the initial sleep duration AND
  // * 1 as the initial threshold value
  // * 6 as the maximum number of iterations
  int duration_ms = 100;
  size_t threshold = 1;
  for (int i = 0; i < 6; i++, duration_ms *= 2, threshold *= 2) {
    if (GetPendingActivityRecordsCount() < threshold) break;
    VLOG(3) << "Wait for pending activity records :"
            << " Pending count = " << GetPendingActivityRecordsCount()
            << ", Threshold = " << threshold;
    VLOG(3) << "Wait for pending activity records : sleep for " << duration_ms
            << " ms";
    tensorflow::profiler::SleepForMillis(duration_ms);
  }
  ClearPendingActivityRecordsCount();

  activity_tracing_enabled_ = false;

  return Status::OK();
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSrocm_tracerDTcc mht_39(mht_39_v, 1878, "", "./tensorflow/core/profiler/internal/gpu/rocm_tracer.cc", "RocmTracer::GetTimestamp");

  uint64_t ts;
  if (wrap::roctracer_get_timestamp(&ts) != ROCTRACER_STATUS_SUCCESS) {
    const char* errstr = wrap::roctracer_error_string();
    LOG(ERROR) << "function roctracer_get_timestamp failed with error "
               << errstr;
    // Return 0 on error.
    return 0;
  }
  return ts;
}

}  // namespace profiler
}  // namespace tensorflow
