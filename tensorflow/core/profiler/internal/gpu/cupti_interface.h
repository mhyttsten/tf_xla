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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_INTERFACE_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_interfaceDTh() {
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


#include <stddef.h>
#include <stdint.h>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

// Provides a wrapper interface to every single CUPTI API function. This class
// is needed to create an easy mock object for CUPTI API calls. All member
// functions are defined in the following order: activity related APIs, callback
// related APIs, Event APIs, and metric APIs. Within each category, we follow
// the order in the original CUPTI documentation.
class CuptiInterface {
 public:
  CuptiInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_interfaceDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/profiler/internal/gpu/cupti_interface.h", "CuptiInterface");
}

  virtual ~CuptiInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_interfaceDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/profiler/internal/gpu/cupti_interface.h", "~CuptiInterface");
}

  // CUPTI activity API
  virtual CUptiResult ActivityDisable(CUpti_ActivityKind kind) = 0;

  virtual CUptiResult ActivityEnable(CUpti_ActivityKind kind) = 0;

  virtual CUptiResult ActivityFlushAll(uint32_t flag) = 0;

  virtual CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                            size_t valid_buffer_size_bytes,
                                            CUpti_Activity** record) = 0;

  virtual CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                                   uint32_t stream_id,
                                                   size_t* dropped) = 0;

  virtual CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config, uint32_t count) = 0;

  virtual CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) = 0;

  virtual CUptiResult GetDeviceId(CUcontext context, uint32* deviceId) = 0;

  virtual CUptiResult GetTimestamp(uint64_t* timestamp) = 0;

  virtual CUptiResult Finalize() = 0;

  // CUPTI callback API
  virtual CUptiResult EnableCallback(uint32_t enable,
                                     CUpti_SubscriberHandle subscriber,
                                     CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid) = 0;

  virtual CUptiResult EnableDomain(uint32_t enable,
                                   CUpti_SubscriberHandle subscriber,
                                   CUpti_CallbackDomain domain) = 0;

  virtual CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                                CUpti_CallbackFunc callback,
                                void* userdata) = 0;

  virtual CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) = 0;

  // CUPTI event API
  virtual CUptiResult DeviceEnumEventDomains(
      CUdevice device, size_t* array_size_bytes,
      CUpti_EventDomainID* domain_array) = 0;

  virtual CUptiResult DeviceGetEventDomainAttribute(
      CUdevice device, CUpti_EventDomainID event_domain,
      CUpti_EventDomainAttribute attrib, size_t* value_size, void* value) = 0;

  virtual CUptiResult DisableKernelReplayMode(CUcontext context) = 0;

  virtual CUptiResult EnableKernelReplayMode(CUcontext context) = 0;

  virtual CUptiResult DeviceGetNumEventDomains(CUdevice device,
                                               uint32_t* num_domains) = 0;

  virtual CUptiResult EventDomainEnumEvents(CUpti_EventDomainID event_domain,
                                            size_t* array_size_bytes,
                                            CUpti_EventID* event_array) = 0;

  virtual CUptiResult EventDomainGetNumEvents(CUpti_EventDomainID event_domain,
                                              uint32_t* num_events) = 0;

  virtual CUptiResult EventGetAttribute(CUpti_EventID event,
                                        CUpti_EventAttribute attrib,
                                        size_t* value_size, void* value) = 0;

  virtual CUptiResult EventGetIdFromName(CUdevice device,
                                         const char* event_name,
                                         CUpti_EventID* event) = 0;

  virtual CUptiResult EventGroupDisable(CUpti_EventGroup event_group) = 0;

  virtual CUptiResult EventGroupEnable(CUpti_EventGroup event_group) = 0;

  virtual CUptiResult EventGroupGetAttribute(CUpti_EventGroup event_group,
                                             CUpti_EventGroupAttribute attrib,
                                             size_t* value_size,
                                             void* value) = 0;

  virtual CUptiResult EventGroupReadEvent(CUpti_EventGroup event_group,
                                          CUpti_ReadEventFlags flags,
                                          CUpti_EventID event,
                                          size_t* event_value_buffer_size_bytes,
                                          uint64_t* eventValueBuffer) = 0;

  virtual CUptiResult EventGroupSetAttribute(CUpti_EventGroup event_group,
                                             CUpti_EventGroupAttribute attrib,
                                             size_t value_size,
                                             void* value) = 0;

  virtual CUptiResult EventGroupSetsCreate(
      CUcontext context, size_t event_id_array_size_bytes,
      CUpti_EventID* event_id_array,
      CUpti_EventGroupSets** event_group_passes) = 0;

  virtual CUptiResult EventGroupSetsDestroy(
      CUpti_EventGroupSets* event_group_sets) = 0;

  // CUPTI metric API
  virtual CUptiResult DeviceEnumMetrics(CUdevice device, size_t* arraySizeBytes,
                                        CUpti_MetricID* metricArray) = 0;

  virtual CUptiResult DeviceGetNumMetrics(CUdevice device,
                                          uint32_t* num_metrics) = 0;

  virtual CUptiResult MetricGetIdFromName(CUdevice device,
                                          const char* metric_name,
                                          CUpti_MetricID* metric) = 0;

  virtual CUptiResult MetricGetNumEvents(CUpti_MetricID metric,
                                         uint32_t* num_events) = 0;

  virtual CUptiResult MetricEnumEvents(CUpti_MetricID metric,
                                       size_t* event_id_array_size_bytes,
                                       CUpti_EventID* event_id_array) = 0;

  virtual CUptiResult MetricGetAttribute(CUpti_MetricID metric,
                                         CUpti_MetricAttribute attrib,
                                         size_t* value_size, void* value) = 0;

  virtual CUptiResult MetricGetValue(CUdevice device, CUpti_MetricID metric,
                                     size_t event_id_array_size_bytes,
                                     CUpti_EventID* event_id_array,
                                     size_t event_value_array_size_bytes,
                                     uint64_t* event_value_array,
                                     uint64_t time_duration,
                                     CUpti_MetricValue* metric_value) = 0;

  virtual CUptiResult GetResultString(CUptiResult result, const char** str) = 0;

  virtual CUptiResult GetContextId(CUcontext context, uint32_t* context_id) = 0;

  virtual CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                                    uint8_t per_thread_stream,
                                    uint32_t* stream_id) = 0;

  // Interface maintenance functions. Not directly related to CUPTI, but
  // required for implementing an error resilient layer over CUPTI API.

  // Performance any clean up work that is required each time profile session
  // is done. Therefore this can be called multiple times during process life
  // time.
  virtual void CleanUp() = 0;

  // Whether CUPTI API is currently disabled due to unrecoverable errors.
  // All subsequent calls will fail immediately without forwarding calls to
  // CUPTI library.
  virtual bool Disabled() const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CuptiInterface);
};

CuptiInterface* GetCuptiInterface();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_INTERFACE_H_
