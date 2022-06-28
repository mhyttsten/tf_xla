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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::xla::cpu_function_runtime::BufferInfo;

TEST(XlaCompiledCpuFunctionTest, AlignmentValue) {
  // We've chosen 64 byte alignment for the tfcompile runtime to mimic the
  // regular tensorflow allocator, which was chosen to play nicely with Eigen.
  // The tfcompile runtime also has a requirement that comes from the xla
  // generated code, on the relation: buffer_size >= 16 ? 2 * sizeof(void*) : 8
  // So any value that we choose must abide by that constraint as well.
  EXPECT_EQ(xla::cpu_function_runtime::Align(), Allocator::kAllocatorAlignment);
  EXPECT_LE(xla::cpu_function_runtime::MinAlign(),
            Allocator::kAllocatorAlignment);
}

std::vector<BufferInfo> SizesToBufferInfos(const intptr_t* sizes, size_t n) {
  std::vector<BufferInfo> buffer_infos;
  std::transform(sizes, sizes + n, std::back_inserter(buffer_infos),
                 [&](intptr_t size) {
                   if (size == -1) {
                     // Use a dummy on-stack buffer allocation to indicat the
                     // the current slot does not need an allocation.
                     int64_t on_stack_buffer_size = 4;
                     return BufferInfo::MakeOnStackBuffer(on_stack_buffer_size);
                   }
                   return BufferInfo::MakeTempBuffer(size);
                 });
  return buffer_infos;
}

// Simple wrappers to make writing tests more ergonomic.

size_t AlignedBufferBytesFromSizes(const intptr_t* sizes, size_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/tf2xla/cpu_function_runtime_test.cc", "AlignedBufferBytesFromSizes");

  std::vector<BufferInfo> buffer_infos = SizesToBufferInfos(sizes, n);
  return AlignedBufferBytes(buffer_infos.data(), n,
                            /*allocate_entry_params=*/false);
}

void* MallocContiguousBuffersFromSizes(const intptr_t* sizes, size_t n,
                                       void** bufs, bool annotate_initialized) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/tf2xla/cpu_function_runtime_test.cc", "MallocContiguousBuffersFromSizes");

  std::vector<BufferInfo> buffer_infos = SizesToBufferInfos(sizes, n);
  return MallocContiguousBuffers(buffer_infos.data(), n,
                                 /*allocate_entry_params=*/false, bufs,
                                 annotate_initialized);
}

TEST(XlaCompiledCpuFunctionTest, AlignedBufferBytes) {
  EXPECT_EQ(AlignedBufferBytesFromSizes(nullptr, 0), 0);

  static constexpr intptr_t sizesA[1] = {-1};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesA, 1), 0);

  static constexpr intptr_t sizesB[1] = {3};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesB, 1), 64);

  static constexpr intptr_t sizesC[1] = {32};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesC, 1), 64);

  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  EXPECT_EQ(AlignedBufferBytesFromSizes(sizesD, 7), 320);
}

void* add_ptr(void* base, uintptr_t delta) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/tf2xla/cpu_function_runtime_test.cc", "add_ptr");

  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + delta);
}

// To test MallocContiguousBuffers and FreeContiguous, we just check for
// expected nullptrs, and write to each byte of allocated memory.  We rely on
// the leak checker to tell us if there's an inconsistency between malloc and
// free.  We also check the contiguous property.
TEST(XlaCompiledCpuFunctionTest, MallocFreeContiguousBuffers) {
  // Test empty sizes.
  void* base = MallocContiguousBuffersFromSizes(nullptr, 0, nullptr, false);
  EXPECT_EQ(base, nullptr);
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with 0 sum.
  static constexpr intptr_t sizesA[1] = {-1};
  void* bufA[1];
  base = MallocContiguousBuffersFromSizes(sizesA, 1, bufA, false);
  EXPECT_EQ(base, nullptr);
  EXPECT_EQ(bufA[0], nullptr);
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with non-0 sum.
  static constexpr intptr_t sizesB[1] = {3};
  void* bufB[1];
  base = MallocContiguousBuffersFromSizes(sizesB, 1, bufB, false);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufB[0], add_ptr(base, 0));
  char* bufB0_bytes = static_cast<char*>(bufB[0]);
  bufB0_bytes[0] = 'A';
  bufB0_bytes[1] = 'B';
  bufB0_bytes[2] = 'C';
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test non-empty sizes with non-0 sum, and annotate_initialized.
  static constexpr intptr_t sizesC[1] = {3};
  void* bufC[1];
  base = MallocContiguousBuffersFromSizes(sizesC, 1, bufC, true);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufC[0], add_ptr(base, 0));
  char* bufC0_bytes = static_cast<char*>(bufC[0]);
  bufC0_bytes[0] = 'A';
  bufC0_bytes[1] = 'B';
  bufC0_bytes[2] = 'C';
  xla::cpu_function_runtime::FreeContiguous(base);

  // Test mixed sizes.
  static constexpr intptr_t sizesD[7] = {1, -1, 32, -1, 64, 2, 3};
  void* bufD[7];
  base = MallocContiguousBuffersFromSizes(sizesD, 7, bufD, false);
  EXPECT_NE(base, nullptr);
  EXPECT_EQ(bufD[0], add_ptr(base, 0));
  EXPECT_EQ(bufD[1], nullptr);
  EXPECT_EQ(bufD[2], add_ptr(base, 64));
  EXPECT_EQ(bufD[3], nullptr);
  EXPECT_EQ(bufD[4], add_ptr(base, 128));
  EXPECT_EQ(bufD[5], add_ptr(base, 192));
  EXPECT_EQ(bufD[6], add_ptr(base, 256));
  for (int i = 0; i < 7; ++i) {
    const intptr_t size = sizesD[i];
    if (size != -1) {
      char* bufD_bytes = static_cast<char*>(bufD[i]);
      for (size_t j = 0; j < size; ++j) {
        bufD_bytes[j] = 'A' + j;
      }
    }
  }
  xla::cpu_function_runtime::FreeContiguous(base);
}

void CheckRoundTripIsOk(const BufferInfo& buffer_info) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPScpu_function_runtime_testDTcc mht_3(mht_3_v, 331, "", "./tensorflow/compiler/tf2xla/cpu_function_runtime_test.cc", "CheckRoundTripIsOk");

  BufferInfo round_trip(buffer_info.Encode());
  ASSERT_EQ(round_trip, buffer_info);
}

TEST(XlaCompiledCpuFunctionTest, BufferInfoTest) {
  CheckRoundTripIsOk(BufferInfo::MakeTempBuffer(0));
  CheckRoundTripIsOk(BufferInfo::MakeTempBuffer(4));
  CheckRoundTripIsOk(BufferInfo::MakeOnStackBuffer(0));
  CheckRoundTripIsOk(BufferInfo::MakeOnStackBuffer(4));
  CheckRoundTripIsOk(BufferInfo::MakeConstant(0));
  CheckRoundTripIsOk(BufferInfo::MakeConstant(4));
  CheckRoundTripIsOk(
      BufferInfo::MakeEntryParameter(/*size=*/0, /*param_number=*/4));
  CheckRoundTripIsOk(
      BufferInfo::MakeEntryParameter(/*size=*/4, /*param_number=*/0));
}

}  // namespace
}  // namespace tensorflow
