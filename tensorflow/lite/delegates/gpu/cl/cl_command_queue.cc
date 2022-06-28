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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

CLCommandQueue::CLCommandQueue(cl_command_queue queue, bool has_ownership)
    : queue_(queue), has_ownership_(has_ownership) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::CLCommandQueue");
}

CLCommandQueue::CLCommandQueue(CLCommandQueue&& queue)
    : queue_(queue.queue_), has_ownership_(queue.has_ownership_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_1(mht_1_v, 211, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::CLCommandQueue");

  queue.queue_ = nullptr;
}

CLCommandQueue& CLCommandQueue::operator=(CLCommandQueue&& queue) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "=");

  if (this != &queue) {
    Release();
    std::swap(queue_, queue.queue_);
    has_ownership_ = queue.has_ownership_;
  }
  return *this;
}

CLCommandQueue::~CLCommandQueue() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::~CLCommandQueue");
 Release(); }

void CLCommandQueue::Release() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_4(mht_4_v, 235, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::Release");

  if (has_ownership_ && queue_) {
    clReleaseCommandQueue(queue_);
    queue_ = nullptr;
  }
}

absl::Status CLCommandQueue::Dispatch(const CLKernel& kernel,
                                      const int3& work_groups_count,
                                      const int3& work_group_size,
                                      CLEvent* event) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_5(mht_5_v, 248, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::Dispatch");

  std::array<size_t, 3> local;
  std::array<size_t, 3> global;
  for (int i = 0; i < 3; ++i) {
    local[i] = work_group_size[i];
    global[i] = work_groups_count[i] * work_group_size[i];
  }
  cl_event resulting_event;
  const int error_code = clEnqueueNDRangeKernel(
      queue_, kernel.kernel(), 3, nullptr, global.data(), local.data(), 0,
      nullptr, event ? &resulting_event : nullptr);
  if (event) {
    *event = CLEvent(resulting_event);
  }
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clEnqueueNDRangeKernel - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::Dispatch(const CLKernel& kernel,
                                      const int3& work_groups_count,
                                      const int3& work_group_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_6(mht_6_v, 275, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::Dispatch");

  return Dispatch(kernel, work_groups_count, work_group_size, nullptr);
}

absl::Status CLCommandQueue::EnqueueEvent(CLEvent* event) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_7(mht_7_v, 282, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::EnqueueEvent");

  cl_event resulting_event;
  const int error_code = clEnqueueMarker(queue_, &resulting_event);
  *event = CLEvent(resulting_event);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(absl::StrCat("Failed to clEnqueueMarker - ",
                                           CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueWriteImage(cl_mem memory, int3 region,
                                               const void* data, bool async) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_8(mht_8_v, 297, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::EnqueueWriteImage");

  const size_t origin[] = {0, 0, 0};
  const size_t r[] = {static_cast<size_t>(region.x),
                      static_cast<size_t>(region.y),
                      static_cast<size_t>(region.z)};
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueWriteImage(queue_, memory, blocking, origin, r, 0,
                                        0, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to upload data to GPU (clEnqueueWriteImage) - ",
                     CLErrorCodeToString(error_code)));
  }

  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueReadImage(cl_mem memory, int3 region,
                                              void* data, bool async) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_9(mht_9_v, 318, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::EnqueueReadImage");

  const size_t origin[] = {0, 0, 0};
  const size_t r[] = {static_cast<size_t>(region.x),
                      static_cast<size_t>(region.y),
                      static_cast<size_t>(region.z)};
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueReadImage(queue_, memory, blocking, origin, r, 0,
                                       0, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to read data from GPU (clEnqueueReadImage) - ",
                     CLErrorCodeToString(error_code)));
  }

  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueWriteBuffer(cl_mem memory,
                                                size_t size_in_bytes,
                                                const void* data, bool async) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_10(mht_10_v, 340, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::EnqueueWriteBuffer");

  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueWriteBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to upload data to GPU (clEnqueueWriteBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::EnqueueReadBuffer(cl_mem memory,
                                               size_t size_in_bytes, void* data,
                                               bool async) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_11(mht_11_v, 357, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::EnqueueReadBuffer");

  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  auto error_code = clEnqueueReadBuffer(
      queue_, memory, blocking, 0, size_in_bytes, data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to read data from GPU (clEnqueueReadBuffer) - ",
                     CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

absl::Status CLCommandQueue::WaitForCompletion() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_12(mht_12_v, 372, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CLCommandQueue::WaitForCompletion");

  auto error_code = clFinish(queue_);
  if (error_code != CL_SUCCESS) {
    return absl::UnknownError(
        absl::StrCat("Failed to clFinish - ", CLErrorCodeToString(error_code)));
  }
  return absl::OkStatus();
}

ProfilingCommandQueue::ProfilingCommandQueue(cl_command_queue queue)
    : CLCommandQueue(queue, true) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_13(mht_13_v, 385, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::ProfilingCommandQueue");

  events_.reserve(128);
}

ProfilingCommandQueue::ProfilingCommandQueue(ProfilingCommandQueue&& queue)
    : CLCommandQueue(std::move(queue)),
      events_(std::move(queue.events_)),
      number_of_dispatches_(std::move(queue.number_of_dispatches_)),
      current_label_(std::move(queue.current_label_)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_14(mht_14_v, 396, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::ProfilingCommandQueue");
}

ProfilingCommandQueue& ProfilingCommandQueue::operator=(
    ProfilingCommandQueue&& queue) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_15(mht_15_v, 402, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "=");

  if (this != &queue) {
    events_ = std::move(queue.events_);
    number_of_dispatches_ = std::move(queue.number_of_dispatches_);
    current_label_ = std::move(queue.current_label_);
    CLCommandQueue::operator=(std::move(queue));
  }
  return *this;
}

void ProfilingCommandQueue::SetEventsLabel(const std::string& name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_16(mht_16_v, 416, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::SetEventsLabel");

  current_label_ = name;
}

void ProfilingCommandQueue::ResetMeasurements() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_17(mht_17_v, 423, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::ResetMeasurements");

  events_.clear();
  number_of_dispatches_.clear();
}

absl::Status ProfilingCommandQueue::Dispatch(const CLKernel& kernel,
                                             const int3& work_groups_count,
                                             const int3& work_group_size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_18(mht_18_v, 433, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::Dispatch");

  events_.push_back(CLEvent());
  number_of_dispatches_.push_back(1);
  RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                           work_group_size,
                                           &events_[events_.size() - 1]));
  events_.back().SetName(current_label_);
  return absl::OkStatus();
}

absl::Status ProfilingCommandQueue::DispatchNTimes(
    const CLKernel& kernel, const int3& work_groups_count,
    const int3& work_group_size, int n, int flush_period) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_19(mht_19_v, 448, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::DispatchNTimes");

  number_of_dispatches_.push_back(n);
  if (n == 1) {
    events_.push_back(CLEvent());
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 1]));
    events_.back().SetName(current_label_);
  } else {
    events_.push_back(CLEvent());
    events_.push_back(CLEvent());
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 2]));
    for (int i = 1; i < n - 1; ++i) {
      RETURN_IF_ERROR(
          CLCommandQueue::Dispatch(kernel, work_groups_count, work_group_size));
      if (flush_period && i % flush_period == 0) {
        clFlush(queue_);
      }
    }
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count,
                                             work_group_size,
                                             &events_[events_.size() - 1]));
    clFlush(queue_);
    events_[events_.size() - 2].SetName(current_label_);
    events_[events_.size() - 1].SetName(current_label_);
  }
  return absl::OkStatus();
}

ProfilingInfo ProfilingCommandQueue::GetProfilingInfo() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_20(mht_20_v, 482, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::GetProfilingInfo");

  ProfilingInfo result;
  result.dispatches.resize(number_of_dispatches_.size());
  int events_counter = 0;
  for (int i = 0; i < number_of_dispatches_.size(); ++i) {
    result.dispatches[i].label = events_[events_counter].GetName();
    if (number_of_dispatches_[i] == 1) {
      result.dispatches[i].duration =
          absl::Nanoseconds(events_[events_counter].GetEventTimeNs());
      events_counter += 1;
    } else {
      result.dispatches[i].duration =
          absl::Nanoseconds(events_[events_counter + 1].GetFinishedTimeNs() -
                            events_[events_counter].GetStartedTimeNs()) /
          number_of_dispatches_[i];
      events_counter += 2;
    }
  }
  return result;
}

absl::Status ProfilingCommandQueue::GetBestWorkGroupIndex(
    const CLKernel& kernel, const GpuInfo& gpu_info,
    const std::vector<int3>& work_groups_count,
    const std::vector<int3>& work_group_sizes, int* index) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_21(mht_21_v, 509, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::GetBestWorkGroupIndex");

  // Some Adreno 3xx can have wrong numbers for some events
  const bool possible_bug_with_events =
      gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx();
  events_.resize(work_group_sizes.size());
  for (int i = 0; i < work_group_sizes.size(); ++i) {
    RETURN_IF_ERROR(CLCommandQueue::Dispatch(kernel, work_groups_count[i],
                                             work_group_sizes[i], &events_[i]));

    // reducing the speed of memory leak on Mali for some kernels
    if (gpu_info.IsMali() && i % 8 == 7) {
      events_[i - 7].Wait();
    }
    if (possible_bug_with_events) {
      // We are trying to increase probability for correct result.
      RETURN_IF_ERROR(WaitForCompletion());
    }
  }

  RETURN_IF_ERROR(WaitForCompletion());

  // To release memory of some kernel pool on Mali.
  if (gpu_info.IsMali()) {
    RETURN_IF_ERROR(kernel.ReInit());
  }

  int minimum_index = 0;
  double minimum_time = std::numeric_limits<double>::max();
  if (possible_bug_with_events) {  // we will try to cut out suspicious results
    double average_time = 0.0;
    int average_samples_count = 0;
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      if (events_[i].GetEventTimeMs() < 100 * 1000) {  // 100 sec
        average_time += events_[i].GetEventTimeMs();
        average_samples_count++;
      }
    }
    average_time /= average_samples_count;
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      double time = events_[i].GetEventTimeMs();
      if (time < minimum_time && time >= 0.1 * average_time) {
        minimum_index = i;
        minimum_time = time;
      }
    }
  } else {
    for (int i = 0; i < work_group_sizes.size(); ++i) {
      double time = events_[i].GetEventTimeMs();
      if (time < minimum_time) {
        minimum_index = i;
        minimum_time = time;
      }
    }
  }

  *index = minimum_index;

  return absl::OkStatus();
}

absl::Status CreateCLCommandQueue(const CLDevice& device,
                                  const CLContext& context,
                                  CLCommandQueue* result) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_22(mht_22_v, 574, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CreateCLCommandQueue");

  int error_code;
  cl_command_queue queue =
      clCreateCommandQueue(context.context(), device.id(), 0, &error_code);
  if (!queue) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a command queue - ",
                     CLErrorCodeToString(error_code)));
  }
  *result = CLCommandQueue(queue, true);
  return absl::OkStatus();
}

double ProfilingCommandQueue::GetQueueExecutionTimeMs() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_23(mht_23_v, 590, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::GetQueueExecutionTimeMs");

  const uint64_t start = events_.front().GetStartedTimeNs();
  const uint64_t end = events_.back().GetFinishedTimeNs();
  const uint64_t time_ns = (end - start);

  return static_cast<double>(time_ns) / 1000000.0;
}

double ProfilingCommandQueue::GetSumOfEventsTimeMs() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_24(mht_24_v, 601, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "ProfilingCommandQueue::GetSumOfEventsTimeMs");

  double sum = 0.0;
  for (int i = 0; i < events_.size(); ++i) {
    sum += events_[i].GetEventTimeMs();
  }
  return sum;
}

absl::Status CreateProfilingCommandQueue(const CLDevice& device,
                                         const CLContext& context,
                                         ProfilingCommandQueue* result) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_command_queueDTcc mht_25(mht_25_v, 614, "", "./tensorflow/lite/delegates/gpu/cl/cl_command_queue.cc", "CreateProfilingCommandQueue");

  int error_code;
  cl_command_queue queue = clCreateCommandQueue(
      context.context(), device.id(), CL_QUEUE_PROFILING_ENABLE, &error_code);
  if (!queue) {
    return absl::UnknownError(
        absl::StrCat("Failed to create a command queue - ",
                     CLErrorCodeToString(error_code)));
  }

  *result = ProfilingCommandQueue(queue);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
