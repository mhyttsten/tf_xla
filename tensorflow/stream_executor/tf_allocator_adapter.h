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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh() {
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


#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

// Adapter class that wraps a Tensorflow allocator.
//
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class TfAllocatorAdapter : public DeviceMemoryAllocator {
 public:
  // stream: a Stream on which the allocator can only be used. If non-null, the
  // allocator can not be used on any other stream.
  TfAllocatorAdapter(tensorflow::Allocator *wrapped, Stream *stream);

  // Constructor for the cases where `stream` can not be provided.
  TfAllocatorAdapter(tensorflow::Allocator *wrapped, Platform *platform);

  ~TfAllocatorAdapter() override;

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh mht_0(mht_0_v, 224, "", "./tensorflow/stream_executor/tf_allocator_adapter.h", "AllowsAsynchronousDeallocation");
 return true; }

  port::StatusOr<Stream *> GetStream(int device_ordinal) override;

 private:
  tensorflow::Allocator *wrapped_;
  Stream *stream_;
};

// Adapter class that wraps per-device TF allocators with corresponding streams
// as a TfAllocatorAdapter. Assumes that the Tensorflow allocator permits
// asynchronous deallocation; see comment on `AllowsAsynchronousDeallocation()`.
class MultiDeviceAdapter : public DeviceMemoryAllocator {
 public:
  using AllocatorWithStream =
      std::pair<std::unique_ptr<tensorflow::Allocator>, Stream *>;
  MultiDeviceAdapter(const Platform *platform,
                     std::vector<AllocatorWithStream> tf_allocators)
      : DeviceMemoryAllocator(platform) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh mht_1(mht_1_v, 245, "", "./tensorflow/stream_executor/tf_allocator_adapter.h", "MultiDeviceAdapter");

    tf_allocators_.reserve(tf_allocators.size());
    for (AllocatorWithStream &p : tf_allocators) {
      per_device_allocators_.emplace_back(p.first.get(), p.second);
      tf_allocators_.push_back(std::move(p.first));
    }
  }

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override {
    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal].Allocate(
        device_ordinal, size, retry_on_failure, memory_space);
  }

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh mht_2(mht_2_v, 264, "", "./tensorflow/stream_executor/tf_allocator_adapter.h", "Deallocate");

    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal].Deallocate(device_ordinal,
                                                             mem);
  }

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTh mht_3(mht_3_v, 280, "", "./tensorflow/stream_executor/tf_allocator_adapter.h", "AllowsAsynchronousDeallocation");
 return true; }

  port::StatusOr<Stream *> GetStream(int device_ordinal) override {
    return per_device_allocators_[device_ordinal].GetStream(device_ordinal);
  }

 private:
  std::vector<TfAllocatorAdapter> per_device_allocators_;
  // The wrapped TF allocators backing per_device_allocators_
  // (TfAllocatorAdapter does not take ownership of its underlying Allocator).
  std::vector<std::unique_ptr<tensorflow::Allocator>> tf_allocators_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
