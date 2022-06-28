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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh() {
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


#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/profiling_info.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

// A wrapper around opencl command queue
class CLCommandQueue {
 public:
  CLCommandQueue() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.h", "CLCommandQueue");
}
  CLCommandQueue(cl_command_queue queue, bool has_ownership);

  // Move only
  CLCommandQueue(CLCommandQueue&& queue);
  CLCommandQueue& operator=(CLCommandQueue&& queue);
  CLCommandQueue(const CLCommandQueue&) = delete;
  CLCommandQueue& operator=(const CLCommandQueue&) = delete;

  virtual ~CLCommandQueue();

  cl_command_queue queue() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh mht_1(mht_1_v, 222, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.h", "queue");
 return queue_; }

  virtual absl::Status Dispatch(const CLKernel& kernel,
                                const int3& work_groups_count,
                                const int3& work_group_size);

  absl::Status Dispatch(const CLKernel& kernel, const int3& work_groups_count,
                        const int3& work_group_size, CLEvent* event);

  absl::Status EnqueueEvent(CLEvent* event);

  absl::Status EnqueueWriteImage(cl_mem memory, int3 region, const void* data,
                                 bool async = false);
  absl::Status EnqueueReadImage(cl_mem memory, int3 region, void* data,
                                bool async = false);

  absl::Status EnqueueWriteBuffer(cl_mem memory, size_t size_in_bytes,
                                  const void* data, bool async = false);
  absl::Status EnqueueReadBuffer(cl_mem memory, size_t size_in_bytes,
                                 void* data, bool async = false);

  absl::Status WaitForCompletion();

 protected:
  void Release();

  cl_command_queue queue_ = nullptr;
  bool has_ownership_ = false;
};

class ProfilingCommandQueue : public CLCommandQueue {
 public:
  ProfilingCommandQueue() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTh mht_2(mht_2_v, 257, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.h", "ProfilingCommandQueue");
}
  explicit ProfilingCommandQueue(cl_command_queue queue);

  // Move only
  ProfilingCommandQueue(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue& operator=(ProfilingCommandQueue&& queue);
  ProfilingCommandQueue(const ProfilingCommandQueue&) = delete;
  ProfilingCommandQueue& operator=(const ProfilingCommandQueue&) = delete;

  absl::Status Dispatch(const CLKernel& kernel, const int3& work_groups_count,
                        const int3& work_group_size) override;

  // for better profiling
  absl::Status DispatchNTimes(const CLKernel& kernel,
                              const int3& work_groups_count,
                              const int3& work_group_size, int n,
                              int flush_period = 0);

  // will write index for fastest work_group among work_group_sizes
  absl::Status GetBestWorkGroupIndex(const CLKernel& kernel,
                                     const GpuInfo& gpu_info,
                                     const std::vector<int3>& work_groups_count,
                                     const std::vector<int3>& work_group_sizes,
                                     int* index);

  // call ResetMeasurements() to start new seriese of measurements
  void ResetMeasurements();

  double GetQueueExecutionTimeMs() const;

  // Difference from GetQueueExecutionTimeMs is that this number doesn't include
  // time between kernels(kernels launches or preparing) on GPU. Usually, this
  // time should be 5-10% better than GetQueueExecutionTimeMs, because 5-10%
  // spend on something else(maybe kernels launches or preparing)
  double GetSumOfEventsTimeMs() const;

  // This label will be used for all subsequent dispatches.
  void SetEventsLabel(const std::string& name);

  ProfilingInfo GetProfilingInfo() const;

 private:
  std::vector<CLEvent> events_;
  std::vector<int> number_of_dispatches_;
  std::string current_label_;
};

absl::Status CreateCLCommandQueue(const CLDevice& device,
                                  const CLContext& context,
                                  CLCommandQueue* result);

absl::Status CreateProfilingCommandQueue(const CLDevice& device,
                                         const CLContext& context,
                                         ProfilingCommandQueue* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_CL_COMMAND_QUEUE_H_
