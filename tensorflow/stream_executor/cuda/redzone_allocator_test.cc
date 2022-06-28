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
class MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc() {
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

#ifdef GOOGLE_CUDA

#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {
namespace cuda {
namespace {

using RedzoneCheckStatus = RedzoneAllocator::RedzoneCheckStatus;

static void EXPECT_REDZONE_OK(port::StatusOr<RedzoneCheckStatus> status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/stream_executor/cuda/redzone_allocator_test.cc", "EXPECT_REDZONE_OK");

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(status.ValueOrDie().ok());
}

static void EXPECT_REDZONE_VIOLATION(
    port::StatusOr<RedzoneCheckStatus> status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/stream_executor/cuda/redzone_allocator_test.cc", "EXPECT_REDZONE_VIOLATION");

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(status.ValueOrDie().ok());
}

TEST(RedzoneAllocatorTest, WriteToRedzone) {
  constexpr int64_t kRedzoneSize = 1 << 23;  // 8MiB redzone on each side
  // Redzone pattern should not be equal to zero; otherwise modify_redzone will
  // break.
  constexpr uint8 kRedzonePattern = 0x7e;

  // Allocate 32MiB + 1 byte (to make things misaligned)
  constexpr int64_t kAllocSize = (1 << 25) + 1;

  Platform* platform =
      MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();
  StreamExecutor* stream_exec = platform->ExecutorForDevice(0).ValueOrDie();
  GpuAsmOpts opts;
  StreamExecutorMemoryAllocator se_allocator(platform, {stream_exec});

  Stream stream(stream_exec);
  stream.Init();
  RedzoneAllocator allocator(&stream, &se_allocator, opts,
                             /*memory_limit=*/(1LL << 32),
                             /*redzone_size=*/kRedzoneSize,
                             /*redzone_pattern=*/kRedzonePattern);
  TF_ASSERT_OK_AND_ASSIGN(DeviceMemory<uint8> buf,
                          allocator.AllocateBytes(/*byte_size=*/kAllocSize));
  EXPECT_REDZONE_OK(allocator.CheckRedzones());

  char* buf_addr = reinterpret_cast<char*>(buf.opaque());
  DeviceMemoryBase lhs_redzone(buf_addr - kRedzoneSize, kRedzoneSize);
  DeviceMemoryBase rhs_redzone(buf_addr + kAllocSize, kRedzoneSize);

  // Check that the redzones are in fact filled with kRedzonePattern.
  auto check_redzone = [&](DeviceMemoryBase redzone, absl::string_view name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/stream_executor/cuda/redzone_allocator_test.cc", "lambda");

    std::vector<uint8> host_buf(kRedzoneSize);
    TF_ASSERT_OK(stream.ThenMemcpy(host_buf.data(), redzone, kRedzoneSize)
                     .BlockHostUntilDone());
    const int64_t kMaxMismatches = 16;
    int64_t mismatches = 0;
    for (int64_t i = 0; i < host_buf.size(); ++i) {
      if (mismatches == kMaxMismatches) {
        ADD_FAILURE() << "Hit max number of mismatches; skipping others.";
        break;
      }
      if (host_buf[i] != kRedzonePattern) {
        ++mismatches;
        EXPECT_EQ(host_buf[i], kRedzonePattern)
            << "at index " << i << " of " << name << " redzone";
      }
    }
  };
  check_redzone(lhs_redzone, "lhs");
  check_redzone(rhs_redzone, "rhs");

  // Modifies a redzone, checks that RedzonesAreUnmodified returns false, then
  // reverts it back to its original value and checks that RedzonesAreUnmodified
  // returns true.
  auto modify_redzone = [&](DeviceMemoryBase redzone, int64_t offset,
                            absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPScudaPSredzone_allocator_testDTcc mht_3(mht_3_v, 279, "", "./tensorflow/stream_executor/cuda/redzone_allocator_test.cc", "lambda");

    SCOPED_TRACE(absl::StrCat(name, ", offset=", offset));
    DeviceMemoryBase redzone_at_offset(
        reinterpret_cast<char*>(redzone.opaque()) + offset, 1);
    char old_redzone_value = 0;
    { EXPECT_REDZONE_OK(allocator.CheckRedzones()); }
    stream.ThenMemcpy(&old_redzone_value, redzone_at_offset, 1)
        .ThenMemZero(&redzone_at_offset, 1);
    EXPECT_REDZONE_VIOLATION(allocator.CheckRedzones());

    // Checking reinitializes the redzone.
    EXPECT_REDZONE_OK(allocator.CheckRedzones());
  };

  modify_redzone(lhs_redzone, /*offset=*/0, "lhs");
  modify_redzone(lhs_redzone, /*offset=*/kRedzoneSize - 1, "lhs");
  modify_redzone(rhs_redzone, /*offset=*/0, "rhs");
  modify_redzone(rhs_redzone, /*offset=*/kRedzoneSize - 1, "rhs");
}

// Older CUDA compute capabilities (<= 2.0) have a limitation that grid
// dimension X cannot be larger than 65535.
//
// Make sure we can launch kernels on sizes larger than that, given that the
// maximum number of threads per block is 1024.
TEST(RedzoneAllocatorTest, VeryLargeRedzone) {
  // Make sure the redzone size would require grid dimension > 65535.
  constexpr int64_t kRedzoneSize = 65535 * 1024 + 1;
  Platform* platform =
      MultiPlatformManager::PlatformWithName("cuda").ValueOrDie();
  StreamExecutor* stream_exec = platform->ExecutorForDevice(0).ValueOrDie();
  GpuAsmOpts opts;
  StreamExecutorMemoryAllocator se_allocator(platform, {stream_exec});
  Stream stream(stream_exec);
  stream.Init();
  RedzoneAllocator allocator(&stream, &se_allocator, opts,
                             /*memory_limit=*/(1LL << 32),
                             /*redzone_size=*/kRedzoneSize,
                             /*redzone_pattern=*/-1);
  (void)allocator.AllocateBytes(/*byte_size=*/1);
  EXPECT_REDZONE_OK(allocator.CheckRedzones());
}

}  // namespace
}  // namespace cuda
}  // namespace stream_executor

#endif  // GOOGLE_CUDA
