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

// Defines the GpuStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh() {
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


#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;

// Wraps a GpuStreamHandle in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class GpuStream : public internal::StreamInterface {
 public:
  explicit GpuStream(GpuExecutor* parent)
      : parent_(parent), gpu_stream_(nullptr), completed_event_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_0(mht_0_v, 207, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "GpuStream");
}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~GpuStream() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_1(mht_1_v, 213, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "~GpuStream");
}

  void* GpuStreamHack() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_2(mht_2_v, 218, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "GpuStreamHack");
 return gpu_stream_; }
  void** GpuStreamMemberHack() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_3(mht_3_v, 222, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "GpuStreamMemberHack");

    return reinterpret_cast<void**>(&gpu_stream_);
  }

  // Explicitly initialize the CUDA resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();
  void SetPriority(int priority) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_4(mht_4_v, 232, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "SetPriority");
 priority_ = priority; }
  int priority() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_5(mht_5_v, 236, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "priority");
 return priority_; }

  // Explicitly destroy the CUDA resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  GpuEventHandle* completed_event() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_6(mht_6_v, 251, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "completed_event");
 return &completed_event_; }

  // Returns the GpuStreamHandle value for passing to the CUDA API.
  //
  // Precond: this GpuStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  GpuStreamHandle gpu_stream() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_7(mht_7_v, 260, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "gpu_stream");

    DCHECK(gpu_stream_ != nullptr);
    return const_cast<GpuStreamHandle>(gpu_stream_);
  }

  // TODO(timshen): Migrate away and remove this function.
  GpuStreamHandle cuda_stream() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_8(mht_8_v, 269, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "cuda_stream");
 return gpu_stream(); }

  GpuExecutor* parent() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSgpu_streamDTh mht_9(mht_9_v, 274, "", "./tensorflow/stream_executor/gpu/gpu_stream.h", "parent");
 return parent_; }

 private:
  GpuExecutor* parent_;         // Executor that spawned this stream.
  GpuStreamHandle gpu_stream_;  // Wrapped CUDA stream handle.
  int priority_ = 0;

  // Event that indicates this stream has completed.
  GpuEventHandle completed_event_ = nullptr;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying GpuStream implementation.
GpuStream* AsGpuStream(Stream* stream);

// Extracts a GpuStreamHandle from a GpuStream-backed Stream object.
GpuStreamHandle AsGpuStreamValue(Stream* stream);

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
