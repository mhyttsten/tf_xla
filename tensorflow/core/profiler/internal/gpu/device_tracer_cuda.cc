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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc() {
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

#if GOOGLE_CUDA

#include <stdlib.h>

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_tracer.h"
#include "tensorflow/core/profiler/internal/gpu/cupti_wrapper.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

// GpuTracer for GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(CuptiTracer* cupti_tracer, CuptiInterface* cupti_interface)
      : cupti_tracer_(cupti_tracer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer");

    VLOG(1) << "GpuTracer created.";
  }
  ~GpuTracer() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "~GpuTracer");
}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(XSpace* space) override;

 private:
  Status DoStart();
  Status DoStop();

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  CuptiTracer* cupti_tracer_;
  CuptiTracerOptions options_;
  std::unique_ptr<CuptiTraceCollector> cupti_collector_;
};

Status GpuTracer::DoStart() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer::DoStart");

  if (!cupti_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  options_.cbids_selected = {
      // KERNEL
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      // MEMCPY
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2,
      // MemAlloc
      CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
      // MemFree
      CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
      // Memset
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async,
      // GENERIC
      CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
  };

  bool use_cupti_activity_api = true;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_USE_ACTIVITY_API", true,
                     &use_cupti_activity_api)
      .IgnoreError();
  options_.enable_event_based_activity = !use_cupti_activity_api;

  bool trace_concurrent_kernels = false;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL", false,
                     &trace_concurrent_kernels)
      .IgnoreError();
  options_.activities_selected.push_back(
      trace_concurrent_kernels ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                               : CUPTI_ACTIVITY_KIND_KERNEL);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_OVERHEAD);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMSET);

// CUDA/CUPTI 10 have issues (leaks and crashes) with CuptiFinalize.
#if CUDA_VERSION < 10000
  if (!trace_concurrent_kernels) options_.cupti_finalize = true;
#elif CUDA_VERSION >= 11000
  options_.cupti_finalize = true;
#endif

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = cupti_tracer_->NumGpus();
  uint64 start_gputime_ns = CuptiTracer::GetTimestamp();
  uint64 start_walltime_ns = GetCurrentTimeNanos();
  cupti_collector_ = CreateCuptiCollector(collector_options, start_walltime_ns,
                                          start_gputime_ns);

  cupti_tracer_->Enable(options_, cupti_collector_.get());
  return Status::OK();
}

Status GpuTracer::Start() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_3(mht_3_v, 339, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer::Start");

  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return Status::OK();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status GpuTracer::DoStop() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_4(mht_4_v, 353, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer::DoStop");

  cupti_tracer_->Disable();
  return Status::OK();
}

Status GpuTracer::Stop() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_5(mht_5_v, 361, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer::Stop");

  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return Status::OK();
}

Status GpuTracer::CollectData(XSpace* space) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPSgpuPSdevice_tracer_cudaDTcc mht_6(mht_6_v, 372, "", "./tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc", "GpuTracer::CollectData");

  VLOG(2) << "Collecting data to XSpace from GpuTracer.";
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, profiler failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      std::string cupti_error = CuptiTracer::ErrorIfAny();
      if (!cupti_error.empty()) {
        space->add_errors(std::move(cupti_error));
      }
      std::string events_dropped = cupti_collector_->ReportNumEventsIfDropped();
      if (!events_dropped.empty()) {
        space->add_warnings(std::move(events_dropped));
      }
      if (cupti_collector_) {
        uint64 end_gpu_ns = CuptiTracer::GetTimestamp();
        cupti_collector_->Export(space, end_gpu_ns);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_tracer_level() == 0) return nullptr;
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
    return nullptr;
  profiler::CuptiTracer* cupti_tracer =
      profiler::CuptiTracer::GetCuptiTracerSingleton();
  if (!cupti_tracer->IsAvailable()) {
    return nullptr;
  }
  profiler::CuptiInterface* cupti_interface = profiler::GetCuptiInterface();
  return absl::make_unique<profiler::GpuTracer>(cupti_tracer, cupti_interface);
}

auto register_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
