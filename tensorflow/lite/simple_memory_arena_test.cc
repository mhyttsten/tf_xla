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
class MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_testDTcc() {
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
#include "tensorflow/lite/simple_memory_arena.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

void ReportError(TfLiteContext* context, const char* format, ...) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arena_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/lite/simple_memory_arena_test.cc", "ReportError");
}

TEST(SimpleMemoryArenaTest, BasicArenaOperations) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[6];

  arena.Allocate(&context, 32, 2047, 0, 1, 3, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 2, 5, &allocs[1]);
  arena.Allocate(&context, 32, 2047, 2, 3, 6, &allocs[2]);
  arena.Allocate(&context, 32, 2047, 3, 5, 6, &allocs[3]);
  arena.Allocate(&context, 32, 1023, 4, 4, 6, &allocs[4]);
  arena.Allocate(&context, 32, 1023, 5, 6, 6, &allocs[5]);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);
  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 6144);
  EXPECT_EQ(allocs[5].offset, 2048);
}

TEST(SimpleMemoryArenaTest, BasicZeroAlloc) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval alloc;

  // Zero-sized allocs should have a 0 offset and size.
  ASSERT_EQ(arena.Allocate(&context, 32, 0, 0, 1, 2, &alloc), kTfLiteOk);
  EXPECT_EQ(alloc.offset, 0);
  EXPECT_EQ(alloc.size, 0);

  // The zero-sized alloc should resolve to null.
  char* resolved_ptr = nullptr;
  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);
  ASSERT_EQ(arena.ResolveAlloc(&context, alloc, &resolved_ptr), kTfLiteOk);
  EXPECT_EQ(resolved_ptr, nullptr);
}

TEST(SimpleMemoryArenaTest, InterleavedZeroAlloc) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[4];

  // Interleave some zero and non-zero-sized allocations and deallocations.
  ASSERT_EQ(arena.Allocate(&context, 32, 2047, 0, 0, 4, &allocs[0]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 0, 1, 1, 2, &allocs[1]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 1023, 2, 1, 2, &allocs[2]), kTfLiteOk);
  ASSERT_EQ(arena.Allocate(&context, 32, 2047, 3, 3, 4, &allocs[3]), kTfLiteOk);

  // Deallocation of a zero-sized alloc should not impact the allocator offsets.
  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 0);
  EXPECT_EQ(allocs[2].offset, 2048);
  EXPECT_EQ(allocs[3].offset, 2048);
}

TEST(SimpleMemoryArenaTest, TestClearPlan) {
  TfLiteContext context;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);
  arena.Allocate(&context, 32, 2047, 2, 1, 2, &allocs[2]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[0].offset, 0);
  EXPECT_EQ(allocs[1].offset, 2048);
  EXPECT_EQ(allocs[2].offset, 4096);

  arena.ClearPlan();

  // Test with smaller allocs.
  arena.Allocate(&context, 32, 1023, 3, 0, 2, &allocs[3]);
  arena.Allocate(&context, 32, 1023, 4, 1, 2, &allocs[4]);
  arena.Allocate(&context, 32, 1023, 5, 1, 2, &allocs[5]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[3].offset, 0);
  EXPECT_EQ(allocs[4].offset, 1024);
  EXPECT_EQ(allocs[5].offset, 2048);

  arena.ClearPlan();

  // Test larger allocs which should require a reallocation.
  arena.Allocate(&context, 32, 4095, 6, 0, 2, &allocs[6]);
  arena.Allocate(&context, 32, 4095, 7, 1, 2, &allocs[7]);
  arena.Allocate(&context, 32, 4095, 8, 1, 2, &allocs[8]);
  arena.Commit(&context);

  EXPECT_EQ(allocs[6].offset, 0);
  EXPECT_EQ(allocs[7].offset, 4096);
  EXPECT_EQ(allocs[8].offset, 8192);
}

TEST(SimpleMemoryArenaTest, TestClearBuffer) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);

  // Should be a no-op.
  ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);

  // Commit and ensure resolved pointers are not null.
  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);
  char* resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);

  ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
  // Base pointer should be null.
  ASSERT_EQ(arena.BasePointer(), 0);

  // Tensors cannot be resolved after ClearBuffer().
  ASSERT_NE(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);

  // Commit again and ensure resolved pointers are not null.
  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);
  ASSERT_NE(arena.BasePointer(), 0);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
}

// Test parameterized by whether ClearBuffer() is called before ClearPlan(), or
// vice versa.
class BufferAndPlanClearingTest : public ::testing::Test,
                                  public ::testing::WithParamInterface<bool> {};

TEST_P(BufferAndPlanClearingTest, TestClearBufferAndClearPlan) {
  TfLiteContext context;
  context.ReportError = ReportError;
  SimpleMemoryArena arena(64);
  ArenaAllocWithUsageInterval allocs[9];

  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);

  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);

  if (GetParam()) {
    ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
    ASSERT_EQ(arena.ClearPlan(), kTfLiteOk);
  } else {
    ASSERT_EQ(arena.ClearPlan(), kTfLiteOk);
    ASSERT_EQ(arena.ReleaseBuffer(), kTfLiteOk);
  }

  // Just committing won't work, allocations need to be made again.
  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);
  char* resolved_ptr = nullptr;
  ASSERT_NE(arena.ResolveAlloc(&context, allocs[0], &resolved_ptr), kTfLiteOk);

  // Re-allocate tensors & commit.
  arena.Allocate(&context, 32, 2047, 0, 0, 2, &allocs[0]);
  arena.Allocate(&context, 32, 2047, 1, 1, 2, &allocs[1]);
  ASSERT_EQ(arena.Commit(&context), kTfLiteOk);

  // Pointer-resolution now works.
  resolved_ptr = nullptr;
  ASSERT_EQ(arena.ResolveAlloc(&context, allocs[1], &resolved_ptr), kTfLiteOk);
  EXPECT_NE(resolved_ptr, nullptr);
}

INSTANTIATE_TEST_SUITE_P(BufferAndPlanClearingTest, BufferAndPlanClearingTest,
                         ::testing::Values(true, false));

}  // namespace
}  // namespace tflite
