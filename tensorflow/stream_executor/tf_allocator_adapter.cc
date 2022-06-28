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
class MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc() {
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

#include "tensorflow/stream_executor/tf_allocator_adapter.h"

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {

TfAllocatorAdapter::TfAllocatorAdapter(tensorflow::Allocator *wrapped,
                                       Stream *stream)
    : DeviceMemoryAllocator(stream->parent()->platform()),
      wrapped_(wrapped),
      stream_(stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::TfAllocatorAdapter");
}

TfAllocatorAdapter::TfAllocatorAdapter(tensorflow::Allocator *wrapped,
                                       Platform *platform)
    : DeviceMemoryAllocator(platform), wrapped_(wrapped), stream_(nullptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_1(mht_1_v, 206, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::TfAllocatorAdapter");
}

TfAllocatorAdapter::~TfAllocatorAdapter() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_2(mht_2_v, 211, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::~TfAllocatorAdapter");
}

port::StatusOr<OwningDeviceMemory> TfAllocatorAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_3(mht_3_v, 218, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::Allocate");

  CHECK_EQ(memory_space, 0);
  tensorflow::AllocationAttributes attrs;
  attrs.retry_on_failure = retry_on_failure;
  void *data = nullptr;
  if (size != 0) {
    data = wrapped_->AllocateRaw(tensorflow::Allocator::kAllocatorAlignment,
                                 size, attrs);
    if (data == nullptr) {
      return tensorflow::errors::ResourceExhausted(
          "Out of memory while trying to allocate ", size, " bytes.");
    }
  }
  return OwningDeviceMemory(DeviceMemoryBase(data, size), device_ordinal, this);
}

port::Status TfAllocatorAdapter::Deallocate(int device_ordinal,
                                            DeviceMemoryBase mem) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_4(mht_4_v, 238, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::Deallocate");

  wrapped_->DeallocateRaw(mem.opaque());
  return port::Status::OK();
}

port::StatusOr<Stream *> TfAllocatorAdapter::GetStream(int device_ordinal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStf_allocator_adapterDTcc mht_5(mht_5_v, 246, "", "./tensorflow/stream_executor/tf_allocator_adapter.cc", "TfAllocatorAdapter::GetStream");

  CHECK_EQ(stream_->parent()->device_ordinal(), device_ordinal);
  return stream_;
}

}  // namespace stream_executor
