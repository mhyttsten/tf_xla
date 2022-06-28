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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc() {
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

#include "tensorflow/core/profiler/internal/gpu/cupti_wrapper.h"

#include <type_traits>

namespace tensorflow {
namespace profiler {

CUptiResult CuptiWrapper::ActivityDisable(CUpti_ActivityKind kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityDisable");

  return cuptiActivityDisable(kind);
}

CUptiResult CuptiWrapper::ActivityEnable(CUpti_ActivityKind kind) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityEnable");

  return cuptiActivityEnable(kind);
}

CUptiResult CuptiWrapper::ActivityFlushAll(uint32_t flag) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_2(mht_2_v, 206, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityFlushAll");

  return cuptiActivityFlushAll(flag);
}

CUptiResult CuptiWrapper::ActivityGetNextRecord(uint8_t* buffer,
                                                size_t valid_buffer_size_bytes,
                                                CUpti_Activity** record) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_3(mht_3_v, 215, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityGetNextRecord");

  return cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes, record);
}

CUptiResult CuptiWrapper::ActivityGetNumDroppedRecords(CUcontext context,
                                                       uint32_t stream_id,
                                                       size_t* dropped) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_4(mht_4_v, 224, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityGetNumDroppedRecords");

  return cuptiActivityGetNumDroppedRecords(context, stream_id, dropped);
}

CUptiResult CuptiWrapper::ActivityConfigureUnifiedMemoryCounter(
    CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_5(mht_5_v, 232, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityConfigureUnifiedMemoryCounter");

  return cuptiActivityConfigureUnifiedMemoryCounter(config, count);
}

CUptiResult CuptiWrapper::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_6(mht_6_v, 241, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::ActivityRegisterCallbacks");

  return cuptiActivityRegisterCallbacks(func_buffer_requested,
                                        func_buffer_completed);
}

CUptiResult CuptiWrapper::GetDeviceId(CUcontext context, uint32* deviceId) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_7(mht_7_v, 249, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::GetDeviceId");

  return cuptiGetDeviceId(context, deviceId);
}

CUptiResult CuptiWrapper::GetTimestamp(uint64_t* timestamp) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_8(mht_8_v, 256, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::GetTimestamp");

  return cuptiGetTimestamp(timestamp);
}

CUptiResult CuptiWrapper::Finalize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_9(mht_9_v, 263, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::Finalize");
 return cuptiFinalize(); }

CUptiResult CuptiWrapper::EnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_10(mht_10_v, 271, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EnableCallback");

  return cuptiEnableCallback(enable, subscriber, domain, cbid);
}

CUptiResult CuptiWrapper::EnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_11(mht_11_v, 280, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EnableDomain");

  return cuptiEnableDomain(enable, subscriber, domain);
}

CUptiResult CuptiWrapper::Subscribe(CUpti_SubscriberHandle* subscriber,
                                    CUpti_CallbackFunc callback,
                                    void* userdata) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_12(mht_12_v, 289, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::Subscribe");

  return cuptiSubscribe(subscriber, callback, userdata);
}

CUptiResult CuptiWrapper::Unsubscribe(CUpti_SubscriberHandle subscriber) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_13(mht_13_v, 296, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::Unsubscribe");

  return cuptiUnsubscribe(subscriber);
}

CUptiResult CuptiWrapper::DeviceEnumEventDomains(
    CUdevice device, size_t* array_size_bytes,
    CUpti_EventDomainID* domain_array) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_14(mht_14_v, 305, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DeviceEnumEventDomains");

  return cuptiDeviceEnumEventDomains(device, array_size_bytes, domain_array);
}

CUptiResult CuptiWrapper::DeviceGetEventDomainAttribute(
    CUdevice device, CUpti_EventDomainID event_domain,
    CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_15(mht_15_v, 314, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DeviceGetEventDomainAttribute");

  return cuptiDeviceGetEventDomainAttribute(device, event_domain, attrib,
                                            value_size, value);
}

CUptiResult CuptiWrapper::DisableKernelReplayMode(CUcontext context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_16(mht_16_v, 322, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DisableKernelReplayMode");

  return cuptiDisableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::EnableKernelReplayMode(CUcontext context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_17(mht_17_v, 329, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EnableKernelReplayMode");

  return cuptiEnableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::DeviceGetNumEventDomains(CUdevice device,
                                                   uint32_t* num_domains) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_18(mht_18_v, 337, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DeviceGetNumEventDomains");

  return cuptiDeviceGetNumEventDomains(device, num_domains);
}

CUptiResult CuptiWrapper::EventDomainEnumEvents(
    CUpti_EventDomainID event_domain, size_t* array_size_bytes,
    CUpti_EventID* event_array) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_19(mht_19_v, 346, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventDomainEnumEvents");

  return cuptiEventDomainEnumEvents(event_domain, array_size_bytes,
                                    event_array);
}

CUptiResult CuptiWrapper::EventDomainGetNumEvents(
    CUpti_EventDomainID event_domain, uint32_t* num_events) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_20(mht_20_v, 355, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventDomainGetNumEvents");

  return cuptiEventDomainGetNumEvents(event_domain, num_events);
}

CUptiResult CuptiWrapper::EventGetAttribute(CUpti_EventID event,
                                            CUpti_EventAttribute attrib,
                                            size_t* value_size, void* value) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_21(mht_21_v, 364, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGetAttribute");

  return cuptiEventGetAttribute(event, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGetIdFromName(CUdevice device,
                                             const char* event_name,
                                             CUpti_EventID* event) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("event_name: \"" + (event_name == nullptr ? std::string("nullptr") : std::string((char*)event_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_22(mht_22_v, 374, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGetIdFromName");

  return cuptiEventGetIdFromName(device, event_name, event);
}

CUptiResult CuptiWrapper::EventGroupDisable(CUpti_EventGroup event_group) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_23(mht_23_v, 381, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupDisable");

  return cuptiEventGroupDisable(event_group);
}

CUptiResult CuptiWrapper::EventGroupEnable(CUpti_EventGroup event_group) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_24(mht_24_v, 388, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupEnable");

  return cuptiEventGroupEnable(event_group);
}

CUptiResult CuptiWrapper::EventGroupGetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t* value_size, void* value) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_25(mht_25_v, 397, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupGetAttribute");

  return cuptiEventGroupGetAttribute(event_group, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGroupReadEvent(
    CUpti_EventGroup event_group, CUpti_ReadEventFlags flags,
    CUpti_EventID event, size_t* event_value_buffer_size_bytes,
    uint64_t* event_value_buffer) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_26(mht_26_v, 407, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupReadEvent");

  return cuptiEventGroupReadEvent(event_group, flags, event,
                                  event_value_buffer_size_bytes,
                                  event_value_buffer);
}

CUptiResult CuptiWrapper::EventGroupSetAttribute(
    CUpti_EventGroup event_group, CUpti_EventGroupAttribute attrib,
    size_t value_size, void* value) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_27(mht_27_v, 418, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupSetAttribute");

  return cuptiEventGroupSetAttribute(event_group, attrib, value_size, value);
}

CUptiResult CuptiWrapper::EventGroupSetsCreate(
    CUcontext context, size_t event_id_array_size_bytes,
    CUpti_EventID* event_id_array, CUpti_EventGroupSets** event_group_passes) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_28(mht_28_v, 427, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupSetsCreate");

  return cuptiEventGroupSetsCreate(context, event_id_array_size_bytes,
                                   event_id_array, event_group_passes);
}

CUptiResult CuptiWrapper::EventGroupSetsDestroy(
    CUpti_EventGroupSets* event_group_sets) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_29(mht_29_v, 436, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::EventGroupSetsDestroy");

  return cuptiEventGroupSetsDestroy(event_group_sets);
}

// CUPTI metric API
CUptiResult CuptiWrapper::DeviceEnumMetrics(CUdevice device,
                                            size_t* arraySizeBytes,
                                            CUpti_MetricID* metricArray) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_30(mht_30_v, 446, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DeviceEnumMetrics");

  return cuptiDeviceEnumMetrics(device, arraySizeBytes, metricArray);
}

CUptiResult CuptiWrapper::DeviceGetNumMetrics(CUdevice device,
                                              uint32_t* num_metrics) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_31(mht_31_v, 454, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::DeviceGetNumMetrics");

  return cuptiDeviceGetNumMetrics(device, num_metrics);
}

CUptiResult CuptiWrapper::MetricGetIdFromName(CUdevice device,
                                              const char* metric_name,
                                              CUpti_MetricID* metric) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("metric_name: \"" + (metric_name == nullptr ? std::string("nullptr") : std::string((char*)metric_name)) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_32(mht_32_v, 464, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::MetricGetIdFromName");

  return cuptiMetricGetIdFromName(device, metric_name, metric);
}

CUptiResult CuptiWrapper::MetricGetNumEvents(CUpti_MetricID metric,
                                             uint32_t* num_events) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_33(mht_33_v, 472, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::MetricGetNumEvents");

  return cuptiMetricGetNumEvents(metric, num_events);
}

CUptiResult CuptiWrapper::MetricEnumEvents(CUpti_MetricID metric,
                                           size_t* event_id_array_size_bytes,
                                           CUpti_EventID* event_id_array) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_34(mht_34_v, 481, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::MetricEnumEvents");

  return cuptiMetricEnumEvents(metric, event_id_array_size_bytes,
                               event_id_array);
}

CUptiResult CuptiWrapper::MetricGetAttribute(CUpti_MetricID metric,
                                             CUpti_MetricAttribute attrib,
                                             size_t* value_size, void* value) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_35(mht_35_v, 491, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::MetricGetAttribute");

  return cuptiMetricGetAttribute(metric, attrib, value_size, value);
}

CUptiResult CuptiWrapper::MetricGetValue(CUdevice device, CUpti_MetricID metric,
                                         size_t event_id_array_size_bytes,
                                         CUpti_EventID* event_id_array,
                                         size_t event_value_array_size_bytes,
                                         uint64_t* event_value_array,
                                         uint64_t time_duration,
                                         CUpti_MetricValue* metric_value) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_36(mht_36_v, 504, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::MetricGetValue");

  return cuptiMetricGetValue(device, metric, event_id_array_size_bytes,
                             event_id_array, event_value_array_size_bytes,
                             event_value_array, time_duration, metric_value);
}

CUptiResult CuptiWrapper::GetResultString(CUptiResult result,
                                          const char** str) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_37(mht_37_v, 514, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::GetResultString");

  return cuptiGetResultString(result, str);
}

CUptiResult CuptiWrapper::GetContextId(CUcontext context,
                                       uint32_t* context_id) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_38(mht_38_v, 522, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::GetContextId");

  return cuptiGetContextId(context, context_id);
}

CUptiResult CuptiWrapper::GetStreamIdEx(CUcontext context, CUstream stream,
                                        uint8_t per_thread_stream,
                                        uint32_t* stream_id) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTcc mht_39(mht_39_v, 531, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.cc", "CuptiWrapper::GetStreamIdEx");

  return cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id);
}

}  // namespace profiler
}  // namespace tensorflow
