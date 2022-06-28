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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc() {
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

#include "tensorflow/core/profiler/internal/gpu/cupti_error_manager.h"

#include <utility>

#include "absl/debugging/leak_check.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace profiler {

CuptiErrorManager::CuptiErrorManager(std::unique_ptr<CuptiInterface> interface)
    : interface_(std::move(interface)), disabled_(0), undo_disabled_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::CuptiErrorManager");
}

#define IGNORE_CALL_IF_DISABLED                                                \
  if (disabled_) {                                                             \
    LOG(ERROR) << "cupti" << __func__ << ": ignored due to a previous error."; \
    return CUPTI_ERROR_DISABLED;                                               \
  }                                                                            \
  VLOG(1) << "cupti" << __func__;

#define ALLOW_ERROR(e, ERROR)                                           \
  if (e == ERROR) {                                                     \
    VLOG(1) << "cupti" << __func__ << ": error " << static_cast<int>(e) \
            << ": " << ResultString(e) << " (allowed)";                 \
    return e;                                                           \
  }

#define LOG_AND_DISABLE_IF_ERROR(e)                                        \
  if (e != CUPTI_SUCCESS) {                                                \
    LOG(ERROR) << "cupti" << __func__ << ": error " << static_cast<int>(e) \
               << ": " << ResultString(e);                                 \
    UndoAndDisable();                                                      \
  }

void CuptiErrorManager::RegisterUndoFunction(
    const CuptiErrorManager::UndoFunction& func) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::RegisterUndoFunction");

  mutex_lock lock(undo_stack_mu_);
  undo_stack_.push_back(func);
}

CUptiResult CuptiErrorManager::ActivityDisable(CUpti_ActivityKind kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityDisable");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityDisable(kind);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityEnable(CUpti_ActivityKind kind) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityEnable");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityEnable(kind);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::ActivityDisable, this, kind);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityFlushAll(uint32_t flag) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityFlushAll");

  // There is a synchronization issue that we were assuming this will flush all
  // the activity buffers. Therefore we need to let CUPTI to flush no matter if
  // previous error is encountered or not.
  CUptiResult error = interface_->ActivityFlushAll(flag);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityGetNextRecord(
    uint8_t* buffer, size_t valid_buffer_size_bytes, CUpti_Activity** record) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityGetNextRecord");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->ActivityGetNextRecord(
      buffer, valid_buffer_size_bytes, record);
  ALLOW_ERROR(error, CUPTI_ERROR_MAX_LIMIT_REACHED);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityGetNumDroppedRecords(CUcontext context,
                                                            uint32_t stream_id,
                                                            size_t* dropped) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_6(mht_6_v, 282, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityGetNumDroppedRecords");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->ActivityGetNumDroppedRecords(context, stream_id, dropped);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_7(mht_7_v, 294, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityConfigureUnifiedMemoryCounter");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->ActivityConfigureUnifiedMemoryCounter(config, count);
  // Don't disable cupti just because the gpu don't support unified memory.
  return error;
}

CUptiResult CuptiErrorManager::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_8(mht_8_v, 307, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ActivityRegisterCallbacks");

  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI activity API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->ActivityRegisterCallbacks(
      func_buffer_requested, func_buffer_completed);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetDeviceId(CUcontext context,
                                           uint32_t* device_id) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_9(mht_9_v, 321, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::GetDeviceId");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetDeviceId(context, device_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetTimestamp(uint64_t* timestamp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_10(mht_10_v, 331, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::GetTimestamp");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetTimestamp(timestamp);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Finalize() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_11(mht_11_v, 341, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::Finalize");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->Finalize();
  ALLOW_ERROR(error, CUPTI_ERROR_API_NOT_IMPLEMENTED);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableCallback(uint32_t enable,
                                              CUpti_SubscriberHandle subscriber,
                                              CUpti_CallbackDomain domain,
                                              CUpti_CallbackId callback_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_12(mht_12_v, 355, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EnableCallback");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EnableCallback(enable, subscriber, domain, callback_id);
  if (error == CUPTI_SUCCESS) {
    if (enable == 1) {
      auto f = std::bind(&CuptiErrorManager::EnableCallback, this,
                         0 /* DISABLE */, subscriber, domain, callback_id);
      RegisterUndoFunction(f);
    }
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableDomain(uint32_t enable,
                                            CUpti_SubscriberHandle subscriber,
                                            CUpti_CallbackDomain domain) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_13(mht_13_v, 375, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EnableDomain");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EnableDomain(enable, subscriber, domain);
  if (error == CUPTI_SUCCESS) {
    if (enable == 1) {
      auto f = std::bind(&CuptiErrorManager::EnableDomain, this,
                         0 /* DISABLE */, subscriber, domain);
      RegisterUndoFunction(f);
    }
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Subscribe(CUpti_SubscriberHandle* subscriber,
                                         CUpti_CallbackFunc callback,
                                         void* userdata) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_14(mht_14_v, 394, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::Subscribe");

  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI callback API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->Subscribe(subscriber, callback, userdata);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::Unsubscribe, this, *subscriber);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::Unsubscribe(CUpti_SubscriberHandle subscriber) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_15(mht_15_v, 410, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::Unsubscribe");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->Unsubscribe(subscriber);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceEnumEventDomains(
    CUdevice device, size_t* array_size_bytes,
    CUpti_EventDomainID* domain_array) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_16(mht_16_v, 422, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DeviceEnumEventDomains");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceEnumEventDomains(
      device, array_size_bytes, domain_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceGetEventDomainAttribute(
    CUdevice device, CUpti_EventDomainID event_domain,
    CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_17(mht_17_v, 435, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DeviceGetEventDomainAttribute");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceGetEventDomainAttribute(
      device, event_domain, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DisableKernelReplayMode(CUcontext context) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_18(mht_18_v, 446, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DisableKernelReplayMode");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DisableKernelReplayMode(context);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EnableKernelReplayMode(CUcontext context) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_19(mht_19_v, 456, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EnableKernelReplayMode");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EnableKernelReplayMode(context);
  if (error == CUPTI_SUCCESS) {
    auto f =
        std::bind(&CuptiErrorManager::DisableKernelReplayMode, this, context);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceEnumMetrics(CUdevice device,
                                                 size_t* arraySizeBytes,
                                                 CUpti_MetricID* metricArray) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_20(mht_20_v, 473, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DeviceEnumMetrics");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->DeviceEnumMetrics(device, arraySizeBytes, metricArray);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::DeviceGetNumEventDomains(CUdevice device,
                                                        uint32_t* num_domains) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_21(mht_21_v, 485, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DeviceGetNumEventDomains");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->DeviceGetNumEventDomains(device, num_domains);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventDomainEnumEvents(
    CUpti_EventDomainID event_domain, size_t* array_size_bytes,
    CUpti_EventID* event_array) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_22(mht_22_v, 497, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventDomainEnumEvents");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventDomainEnumEvents(
      event_domain, array_size_bytes, event_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventDomainGetNumEvents(
    CUpti_EventDomainID event_domain, uint32_t* num_events) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_23(mht_23_v, 509, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventDomainGetNumEvents");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EventDomainGetNumEvents(event_domain, num_events);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGetAttribute(CUpti_EventID event,
                                                 CUpti_EventAttribute attrib,
                                                 size_t* value_size,
                                                 void* value) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_24(mht_24_v, 523, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGetAttribute");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->EventGetAttribute(event, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGetIdFromName(CUdevice device,
                                                  const char* event_name,
                                                  CUpti_EventID* event) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("event_name: \"" + (event_name == nullptr ? std::string("nullptr") : std::string((char*)event_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_25(mht_25_v, 537, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGetIdFromName");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGetIdFromName(device, event_name, event);
  ALLOW_ERROR(error, CUPTI_ERROR_INVALID_EVENT_NAME);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupDisable(CUpti_EventGroup event_group) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_26(mht_26_v, 548, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupDisable");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupDisable(event_group);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupEnable(CUpti_EventGroup event_group) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_27(mht_27_v, 558, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupEnable");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupEnable(event_group);
  if (error == CUPTI_SUCCESS) {
    auto f =
        std::bind(&CuptiErrorManager::EventGroupDisable, this, event_group);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupGetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t* value_size, void* value) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_28(mht_28_v, 575, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupGetAttribute");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupGetAttribute(event_group, attrib,
                                                         value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupReadEvent(
    CUpti_EventGroup event_group, CUpti_ReadEventFlags flags,
    CUpti_EventID event, size_t* event_value_buffer_size_bytes,
    uint64_t* event_value_buffer) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_29(mht_29_v, 589, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupReadEvent");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupReadEvent(
      event_group, flags, event, event_value_buffer_size_bytes,
      event_value_buffer);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t value_size, void* value) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_30(mht_30_v, 603, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupSetAttribute");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetAttribute(event_group, attrib,
                                                         value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetsCreate(
    CUcontext context, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, CUpti_EventGroupSets** event_group_passes) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_31(mht_31_v, 616, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupSetsCreate");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetsCreate(
      context, event_id_array_size_bytes, event_id_array, event_group_passes);
  if (error == CUPTI_SUCCESS) {
    auto f = std::bind(&CuptiErrorManager::EventGroupSetsDestroy, this,
                       *event_group_passes);
    RegisterUndoFunction(f);
  }
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::EventGroupSetsDestroy(
    CUpti_EventGroupSets* event_group_sets) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_32(mht_32_v, 633, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::EventGroupSetsDestroy");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->EventGroupSetsDestroy(event_group_sets);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

// CUPTI metric API
CUptiResult CuptiErrorManager::DeviceGetNumMetrics(CUdevice device,
                                                   uint32_t* num_metrics) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_33(mht_33_v, 645, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::DeviceGetNumMetrics");

  IGNORE_CALL_IF_DISABLED;
  // Disable heap checking for the first CUPTI metric API. See b/22091576.
  absl::LeakCheckDisabler disabler;
  CUptiResult error = interface_->DeviceGetNumMetrics(device, num_metrics);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetIdFromName(CUdevice device,
                                                   const char* metric_name,
                                                   CUpti_MetricID* metric) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("metric_name: \"" + (metric_name == nullptr ? std::string("nullptr") : std::string((char*)metric_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_34(mht_34_v, 660, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::MetricGetIdFromName");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->MetricGetIdFromName(device, metric_name, metric);
  ALLOW_ERROR(error, CUPTI_ERROR_INVALID_METRIC_NAME);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetNumEvents(CUpti_MetricID metric,
                                                  uint32_t* num_events) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_35(mht_35_v, 673, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::MetricGetNumEvents");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricGetNumEvents(metric, num_events);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricEnumEvents(
    CUpti_MetricID metric, size_t* event_id_array_size_bytes,
    CUpti_EventID* event_id_array) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_36(mht_36_v, 685, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::MetricEnumEvents");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricEnumEvents(
      metric, event_id_array_size_bytes, event_id_array);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetAttribute(CUpti_MetricID metric,
                                                  CUpti_MetricAttribute attrib,
                                                  size_t* value_size,
                                                  void* value) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_37(mht_37_v, 699, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::MetricGetAttribute");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->MetricGetAttribute(metric, attrib, value_size, value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::MetricGetValue(
    CUdevice device, CUpti_MetricID metric, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, size_t event_value_array_size_bytes,
    uint64_t* event_value_array, uint64_t time_duration,
    CUpti_MetricValue* metric_value) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_38(mht_38_v, 714, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::MetricGetValue");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->MetricGetValue(
      device, metric, event_id_array_size_bytes, event_id_array,
      event_value_array_size_bytes, event_value_array, time_duration,
      metric_value);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

void CuptiErrorManager::UndoAndDisable() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_39(mht_39_v, 727, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::UndoAndDisable");

  if (undo_disabled_) {  // prevent deadlock
    return;
  }
  // Iterates undo log and call undo APIs one by one.
  mutex_lock lock(undo_stack_mu_);
  undo_disabled_ = true;
  while (!undo_stack_.empty()) {
    LOG(ERROR) << "CuptiErrorManager is disabling profiling automatically.";
    undo_stack_.back()();
    undo_stack_.pop_back();
  }
  undo_disabled_ = false;
  disabled_ = 1;
}

CUptiResult CuptiErrorManager::GetResultString(CUptiResult result,
                                               const char** str) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_40(mht_40_v, 747, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::GetResultString");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetResultString(result, str);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetContextId(CUcontext context,
                                            uint32_t* context_id) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_41(mht_41_v, 758, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::GetContextId");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error = interface_->GetContextId(context, context_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

CUptiResult CuptiErrorManager::GetStreamIdEx(CUcontext context, CUstream stream,
                                             uint8_t per_thread_stream,
                                             uint32_t* stream_id) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_42(mht_42_v, 770, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::GetStreamIdEx");

  IGNORE_CALL_IF_DISABLED;
  CUptiResult error =
      interface_->GetStreamIdEx(context, stream, per_thread_stream, stream_id);
  LOG_AND_DISABLE_IF_ERROR(error);
  return error;
}

void CuptiErrorManager::CleanUp() {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_43(mht_43_v, 781, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::CleanUp");

  if (undo_disabled_) {  // prevent deadlock
    return;
  }
  mutex_lock lock(undo_stack_mu_);
  undo_disabled_ = true;
  while (!undo_stack_.empty()) {
    undo_stack_.pop_back();
  }
  undo_disabled_ = false;
}

std::string CuptiErrorManager::ResultString(CUptiResult error) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_error_managerDTcc mht_44(mht_44_v, 796, "", "./tensorflow/core/profiler/internal/gpu/cupti_error_manager.cc", "CuptiErrorManager::ResultString");

  const char* error_message = nullptr;
  if (interface_->GetResultString(error, &error_message) == CUPTI_SUCCESS &&
      error_message != nullptr) {
    return error_message;
  }
  return "";
}

}  // namespace profiler
}  // namespace tensorflow
