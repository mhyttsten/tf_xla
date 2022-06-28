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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_
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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_tracerDTh {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_tracerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_tracerDTh() {
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


#include "absl/types/optional.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_interface.h"
#include "tensorflow/core/profiler/utils/buffer_pool.h"

namespace tensorflow {
namespace profiler {

struct CuptiTracerOptions {
  bool enable_activity_api = true;

  // Use cuda events to enclose the kernel/memcpy to measure device activity.
  // enable_event_based_activity, if true, will override the enable_activity_api
  // setting.
  bool enable_event_based_activity = false;

  bool required_callback_api_events = true;
  // The callback ids that will be enabled and monitored, if empty, all
  // Callback ids to be enabled using Callback API.
  // We only care CUPTI_CB_DOMAIN_DRIVER_API domain for now. It is kind of
  // redundant to have both CUPTI_CB_DOMAIN_DRIVER_API and
  // CUPTI_CB_DOMAIN_RUNTIME_API.
  std::vector<CUpti_driver_api_trace_cbid_enum> cbids_selected;
  // Activity kinds to be collected using Activity API. If empty, the Activity
  // API is disable.
  std::vector<CUpti_ActivityKind> activities_selected;
  // Whether to call cuptiFinalize.
  bool cupti_finalize = false;
  // Whether to call cuCtxSynchronize for each device before Stop().
  bool sync_devices_before_stop = false;
  // Whether to enable NVTX tracking, we need this for TensorRT tracking.
  bool enable_nvtx_tracking = false;
};

class CuptiDriverApiHook {
 public:
  virtual ~CuptiDriverApiHook() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_tracerDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/profiler/internal/gpu/cupti_tracer.h", "~CuptiDriverApiHook");
}

  virtual Status OnDriverApiEnter(int device_id, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* callback_info) = 0;
  virtual Status OnDriverApiExit(int device_id, CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid,
                                 const CUpti_CallbackData* callback_info) = 0;
  virtual Status SyncAndFlush() = 0;

 protected:
  static Status AddDriverApiCallbackEvent(
      CuptiTraceCollector* collector, CuptiInterface* cupti_interface,
      int device_id, uint64 start_tsc, uint64 end_tsc,
      CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info);
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to CuptiTraceCollector. There should be only one CuptiTracer
// per process.
class CuptiTracer {
 public:
  // Not copyable or movable
  CuptiTracer(const CuptiTracer&) = delete;
  CuptiTracer& operator=(const CuptiTracer&) = delete;

  // Returns a pointer to singleton CuptiTracer.
  static CuptiTracer* GetCuptiTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPScupti_tracerDTh mht_1(mht_1_v, 264, "", "./tensorflow/core/profiler/internal/gpu/cupti_tracer.h", "NeedRootAccess");
 return need_root_access_; }

  void Enable(const CuptiTracerOptions& option, CuptiTraceCollector* collector);
  void Disable();

  Status HandleCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                        const CUpti_CallbackData* callback_info);

  // Returns a buffer and its size for CUPTI to store activities. This buffer
  // will be reclaimed when CUPTI makes a callback to ProcessActivityBuffer.
  void RequestActivityBuffer(uint8_t** buffer, size_t* size);

  // Parses CUPTI activity events from activity buffer, and emits events for
  // CuptiTraceCollector. This function is public because called from registered
  // callback.
  Status ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                               uint8_t* buffer, size_t size);

  static uint64 GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiTracer(CuptiInterface* cupti_interface);

 private:
  // Buffer size and alignment, 32K and 8 as in CUPTI samples.
  static constexpr size_t kBufferSizeInBytes = 32 * 1024;

  Status EnableApiTracing();
  Status EnableActivityTracing();
  Status DisableApiTracing();
  Status DisableActivityTracing();
  Status Finalize();
  void ConfigureActivityUnifiedMemoryCounter(bool enable);
  Status HandleNVTXCallback(CUpti_CallbackId cbid,
                            const CUpti_CallbackData* cbdata);

  int num_gpus_;
  absl::optional<CuptiTracerOptions> option_;
  CuptiInterface* cupti_interface_ = nullptr;
  CuptiTraceCollector* collector_ = nullptr;

  // CUPTI 10.1 and higher need root access to profile.
  bool need_root_access_ = false;

  bool api_tracing_enabled_ = false;
  // Cupti handle for driver or runtime API callbacks. Cupti permits a single
  // subscriber to be active at any time and can be used to trace Cuda runtime
  // as and driver calls for all contexts and devices.
  CUpti_SubscriberHandle subscriber_;  // valid when api_tracing_enabled_.

  bool activity_tracing_enabled_ = false;

  std::unique_ptr<CuptiDriverApiHook> cupti_driver_api_hook_;

  BufferPool buffer_pool_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_TRACER_H_
