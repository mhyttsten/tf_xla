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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_WRAPPER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_WRAPPER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh() {
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
#include "tensorflow/core/profiler/internal/gpu/cupti_interface.h"

namespace tensorflow {
namespace profiler {

class CuptiWrapper : public tensorflow::profiler::CuptiInterface {
 public:
  CuptiWrapper() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh mht_0(mht_0_v, 200, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.h", "CuptiWrapper");
}

  ~CuptiWrapper() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh mht_1(mht_1_v, 205, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.h", "~CuptiWrapper");
}

  // CUPTI activity API
  CUptiResult ActivityDisable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityEnable(CUpti_ActivityKind kind) override;

  CUptiResult ActivityFlushAll(uint32_t flag) override;

  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record) override;

  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id,
                                           size_t* dropped) override;

  CUptiResult ActivityConfigureUnifiedMemoryCounter(
      CUpti_ActivityUnifiedMemoryCounterConfig* config,
      uint32_t count) override;

  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed) override;

  CUptiResult GetDeviceId(CUcontext context, uint32* deviceId) override;

  CUptiResult GetTimestamp(uint64_t* timestamp) override;

  // cuptiFinalize is only defined in CUDA8 and above.
  // To enable it in CUDA8, the environment variable CUPTI_ENABLE_FINALIZE must
  // be set to 1.
  CUptiResult Finalize() override;

  // CUPTI callback API
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid) override;

  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain) override;

  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata) override;

  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber) override;

  // CUPTI event API
  CUptiResult DeviceEnumEventDomains(
      CUdevice device, size_t* array_size_bytes,
      CUpti_EventDomainID* domain_array) override;

  CUptiResult DeviceGetEventDomainAttribute(CUdevice device,
                                            CUpti_EventDomainID event_domain,
                                            CUpti_EventDomainAttribute attrib,
                                            size_t* value_size,
                                            void* value) override;

  CUptiResult DisableKernelReplayMode(CUcontext context) override;

  CUptiResult EnableKernelReplayMode(CUcontext context) override;

  CUptiResult DeviceGetNumEventDomains(CUdevice device,
                                       uint32_t* num_domains) override;

  CUptiResult EventDomainEnumEvents(CUpti_EventDomainID event_domain,
                                    size_t* array_size_bytes,
                                    CUpti_EventID* event_array) override;

  CUptiResult EventDomainGetNumEvents(CUpti_EventDomainID event_domain,
                                      uint32_t* num_events) override;

  CUptiResult EventGetAttribute(CUpti_EventID event,
                                CUpti_EventAttribute attrib, size_t* value_size,
                                void* value) override;

  CUptiResult EventGetIdFromName(CUdevice device, const char* event_name,
                                 CUpti_EventID* event) override;

  CUptiResult EventGroupDisable(CUpti_EventGroup event_group) override;

  CUptiResult EventGroupEnable(CUpti_EventGroup event_group) override;

  CUptiResult EventGroupGetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t* value_size, void* value) override;

  CUptiResult EventGroupReadEvent(CUpti_EventGroup event_group,
                                  CUpti_ReadEventFlags flags,
                                  CUpti_EventID event,
                                  size_t* event_value_buffer_size_bytes,
                                  uint64_t* event_value_buffer) override;

  CUptiResult EventGroupSetAttribute(CUpti_EventGroup event_group,
                                     CUpti_EventGroupAttribute attrib,
                                     size_t value_size, void* value) override;

  CUptiResult EventGroupSetsCreate(
      CUcontext context, size_t event_id_array_size_bytes,
      CUpti_EventID* event_id_array,
      CUpti_EventGroupSets** event_group_passes) override;

  CUptiResult EventGroupSetsDestroy(
      CUpti_EventGroupSets* event_group_sets) override;

  // CUPTI metric API
  CUptiResult DeviceEnumMetrics(CUdevice device, size_t* arraySizeBytes,
                                CUpti_MetricID* metricArray) override;

  CUptiResult DeviceGetNumMetrics(CUdevice device,
                                  uint32_t* num_metrics) override;

  CUptiResult MetricGetIdFromName(CUdevice device, const char* metric_name,
                                  CUpti_MetricID* metric) override;

  CUptiResult MetricGetNumEvents(CUpti_MetricID metric,
                                 uint32_t* num_events) override;

  CUptiResult MetricEnumEvents(CUpti_MetricID metric,
                               size_t* event_id_array_size_bytes,
                               CUpti_EventID* event_id_array) override;

  CUptiResult MetricGetAttribute(CUpti_MetricID metric,
                                 CUpti_MetricAttribute attrib,
                                 size_t* value_size, void* value) override;

  CUptiResult MetricGetValue(CUdevice device, CUpti_MetricID metric,
                             size_t event_id_array_size_bytes,
                             CUpti_EventID* event_id_array,
                             size_t event_value_array_size_bytes,
                             uint64_t* event_value_array,
                             uint64_t time_duration,
                             CUpti_MetricValue* metric_value) override;

  CUptiResult GetResultString(CUptiResult result, const char** str) override;

  CUptiResult GetContextId(CUcontext context, uint32_t* context_id) override;

  CUptiResult GetStreamIdEx(CUcontext context, CUstream stream,
                            uint8_t per_thread_stream,
                            uint32_t* stream_id) override;

  void CleanUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh mht_2(mht_2_v, 350, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.h", "CleanUp");
}
  bool Disabled() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_wrapperDTh mht_3(mht_3_v, 354, "", "./tensorflow/core/profiler/internal/gpu/cupti_wrapper.h", "Disabled");
 return false; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CuptiWrapper);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // PERFTOOLS_ACCELERATORS_XPROF_XPROFILEZ_NVIDIA_GPU_CUPTI_WRAPPER_H_
