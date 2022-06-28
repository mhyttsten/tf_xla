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
class MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/temporary_memory_manager.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
namespace internal {

void TemporaryMemoryManager::ForceDeallocateAll() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_0(mht_0_v, 196, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::ForceDeallocateAll");

  absl::MutexLock lock(&mutex_);
  VLOG(1) << "force-deallocating " << records_.size() << " remaining records";
  for (auto it = records_.begin(); it != records_.end(); ++it) {
    DeviceMemoryBase device_memory = it->first;
    stream_->parent()->Deallocate(&device_memory);
  }
}

void TemporaryMemoryManager::MarkFinalized(
    const DeviceMemoryBase& device_memory, uint64_t generation,
    bool must_exist) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_1(mht_1_v, 210, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::MarkFinalized");

  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    if (must_exist) {
      LOG(FATAL) << "attempted to mark finalization for temporary "
                    "memory that does not exist";
    }
    return;
  }
  it->second.finalized = true;
}

void TemporaryMemoryManager::DeallocateFinalizedTemporaries() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_2(mht_2_v, 226, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::DeallocateFinalizedTemporaries");

  absl::MutexLock lock(&mutex_);
  int deallocated_count = 0;
  for (auto it = records_.begin(); it != records_.end();) {
    if (it->second.finalized) {
      DeviceMemoryBase device_memory = it->first;
      stream_->parent()->Deallocate(&device_memory);
      ++deallocated_count;
      it = records_.erase(it);
    } else {
      ++it;
    }
  }
  VLOG(1) << "deallocated " << deallocated_count << " finalized temporaries";
}

bool TemporaryMemoryManager::IsFinalized(const DeviceMemoryBase& device_memory,
                                         uint64_t allocation_generation) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_3(mht_3_v, 246, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::IsFinalized");

  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    return true;  // If there's no record present it's vacuously finalized.
  }

  if (it->second.allocation_generation == allocation_generation) {
    return it->second.finalized;
  }

  // If the allocation generation did not match, it's vacuously true.
  return true;
}

bool TemporaryMemoryManager::HasAllocated(const DeviceMemoryBase& device_memory,
                                          uint64_t generation) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_4(mht_4_v, 265, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::HasAllocated");

  absl::MutexLock lock(&mutex_);
  auto it = records_.find(device_memory);
  if (it == records_.end()) {
    return false;
  }
  return it->second.allocation_generation == generation;
}

port::StatusOr<std::unique_ptr<TemporaryDeviceMemoryBase>>
TemporaryMemoryManager::AllocateArrayBase(uint64_t element_count,
                                          uint64_t element_size) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStemporary_memory_managerDTcc mht_5(mht_5_v, 279, "", "./tensorflow/stream_executor/temporary_memory_manager.cc", "TemporaryMemoryManager::AllocateArrayBase");

  uint64_t byte_size = element_count * element_size;
  DeviceMemoryBase device_memory =
      stream_->parent()->AllocateArray<uint8>(byte_size);
  if (device_memory == nullptr) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        absl::StrCat("could not allocate temporary memory of ",
                                     byte_size, " bytes"));
  }

  uint64_t generation;

  // Add the record before instantiating the device memory instance so we can
  // check the allocation invariant at TemporaryDeviceMemory construction time.
  {
    absl::MutexLock lock(&mutex_);
    generation = ++generation_;
    DCHECK(records_.find(device_memory) == records_.end());
    records_[device_memory] = {generation,
                               /*finalized=*/false};
  }

  VLOG(1) << absl::StreamFormat(
      "stream %p allocated temporary device memory at %p (size %u) in "
      "generation %u",
      stream_, device_memory.opaque(), byte_size, generation);
  std::unique_ptr<TemporaryDeviceMemoryBase> result(
      new TemporaryDeviceMemoryBase(stream_, device_memory, generation));
  return std::move(result);
}

}  // namespace internal
}  // namespace stream_executor
