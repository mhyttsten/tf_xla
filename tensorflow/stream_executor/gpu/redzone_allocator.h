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

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh() {
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


#include <vector>

#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"

namespace stream_executor {

// An allocator that allocates a bit of extra memory around the beginning/end of
// every allocation and can check that this memory is unmodified.
//
// This can be used to check for out-of-bounds writes, and, if the redzone is
// filled with a sufficiently "ugly" pattern, may also be able to check for
// out-of-bounds reads.  The default fill pattern of -1 is an unusual NaN
// pattern when interpreted as a floating-point number, so hopefully works for
// out-of-bounds reads and writes in those cases.
//
// This class implements ScratchAllocator, so can be used to allocate temp
// memory for cudnn convolutions.
class RedzoneAllocator : public ScratchAllocator {
 public:
  static constexpr int64_t kDefaultRedzoneSize =
      1LL << 23;  // 8MiB per side, 16MiB total.
  static constexpr uint8 kDefaultRedzonePattern = -1;
  RedzoneAllocator(Stream* stream, DeviceMemoryAllocator* memory_allocator,
                   GpuAsmOpts gpu_compilation_opts_,
                   int64_t memory_limit = (1LL << 32),  // 4GB
                   int64_t redzone_size = kDefaultRedzoneSize,
                   uint8 redzone_pattern = kDefaultRedzonePattern);

  // Redzones don't count towards the memory limit.
  int64_t GetMemoryLimitInBytes() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh mht_0(mht_0_v, 221, "", "./tensorflow/stream_executor/gpu/redzone_allocator.h", "GetMemoryLimitInBytes");
 return memory_limit_; }

  int64_t TotalAllocatedBytesExcludingRedzones() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh mht_1(mht_1_v, 226, "", "./tensorflow/stream_executor/gpu/redzone_allocator.h", "TotalAllocatedBytesExcludingRedzones");

    return allocated_bytes_excluding_redzones_;
  }

  port::StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override;

  // Non-empty redzone check status implies that there was a write into a
  // redzone, with a string communicating the location of the write.
  struct RedzoneCheckStatus {
    RedzoneCheckStatus() = default;

    RedzoneCheckStatus(absl::string_view buffer_name, void* user_buffer_address,
                       int64_t offset, uint64_t expected_value,
                       uint64_t actual_value)
        : buffer_name(buffer_name),
          user_buffer_address(user_buffer_address),
          offset(offset),
          expected_value(expected_value),
          actual_value(actual_value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buffer_name: \"" + std::string(buffer_name.data(), buffer_name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh mht_2(mht_2_v, 248, "", "./tensorflow/stream_executor/gpu/redzone_allocator.h", "RedzoneCheckStatus");
}

    static RedzoneCheckStatus OK() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh mht_3(mht_3_v, 253, "", "./tensorflow/stream_executor/gpu/redzone_allocator.h", "OK");
 return {}; }

    bool ok() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSgpuPSredzone_allocatorDTh mht_4(mht_4_v, 258, "", "./tensorflow/stream_executor/gpu/redzone_allocator.h", "ok");
 return user_buffer_address == nullptr; }

    std::string RedzoneFailureMsg() const;

    std::string buffer_name = {};
    void* user_buffer_address = nullptr;
    int64_t offset = 0;
    uint64_t expected_value = 0;
    uint64_t actual_value = 0;
  };

  // Determines whether redzones around all allocated buffers are unmodified.
  //
  // Reinitializes redzones to the expected value, so that the same buffer
  // could be reused for multiple checks.
  //
  // Returns:
  //
  //  - RedzoneCheckStatus::OK() if everything went well.
  //  - RedzoneCheckStatus with a non-empty error message iff a write into a
  //    redzone has been detected.
  //  - A stream error, if loading or launching the kernel has failed.
  port::StatusOr<RedzoneCheckStatus> CheckRedzones() const;

 private:
  const int device_ordinal_;
  Stream* stream_;

  // Memory limit of the allocator in bytes.
  const int64_t memory_limit_;

  // Redzone size on *one side* of allocation in bytes.
  //
  // Must be a multiple of kXlaAllocatedBufferAlignBytes, otherwise the buffers
  // returned to users will be misaligned.
  const int64_t redzone_size_;

  const uint8 redzone_pattern_;
  DeviceMemoryAllocator* memory_allocator_;
  GpuAsmOpts gpu_compilation_opts_;

  // The second element of the pair is the size of the user allocation.  This
  // isn't necessarily just first.size() - 2 * redzone_size_ because when the
  // user allocation size is not a multiple of 4 bytes, we round up the size of
  // the RHS redzone.
  //
  // ScratchAllocators need to free all allocated memory on destruction so we
  // use `OwningDeviceMemory` here.
  std::vector<std::pair<OwningDeviceMemory, int64_t>> allocated_buffers_;

  int64_t allocated_bytes_excluding_redzones_ = 0;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_H_
