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
class MHTracer_DTPStensorflowPScorePSlibPScorePSarena_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSarena_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSarena_testDTcc() {
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

#include "tensorflow/core/lib/core/arena.h"

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

// Write random data to allocated memory
static void TestMemory(void* mem, int size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarena_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/lib/core/arena_test.cc", "TestMemory");

  // Check that we can memset the entire memory
  memset(mem, 0xaa, size);

  // Do some memory allocation to check that the arena doesn't mess up
  // the internal memory allocator
  char* tmp[100];
  for (size_t i = 0; i < TF_ARRAYSIZE(tmp); i++) {
    tmp[i] = new char[i * i + 1];
  }

  memset(mem, 0xcc, size);

  // Free up the allocated memory;
  for (size_t i = 0; i < TF_ARRAYSIZE(tmp); i++) {
    delete[] tmp[i];
  }

  // Check that we can memset the entire memory
  memset(mem, 0xee, size);
}

TEST(ArenaTest, TestBasicArena) {
  Arena a(1024);
  char* memory = a.Alloc(100);
  ASSERT_NE(memory, nullptr);
  TestMemory(memory, 100);

  // Allocate again
  memory = a.Alloc(100);
  ASSERT_NE(memory, nullptr);
  TestMemory(memory, 100);
}

TEST(ArenaTest, TestAlignment) {
  Arena a(1024);
  char* byte0 = a.Alloc(1);
  char* alloc_aligned8 = a.AllocAligned(17, 8);
  EXPECT_EQ(alloc_aligned8 - byte0, 8);
  char* alloc_aligned8_b = a.AllocAligned(8, 8);
  EXPECT_EQ(alloc_aligned8_b - alloc_aligned8, 24);
  char* alloc_aligned8_c = a.AllocAligned(16, 8);
  EXPECT_EQ(alloc_aligned8_c - alloc_aligned8_b, 8);
  char* alloc_aligned8_d = a.AllocAligned(8, 1);
  EXPECT_EQ(alloc_aligned8_d - alloc_aligned8_c, 16);
}

TEST(ArenaTest, TestVariousArenaSizes) {
  {
    Arena a(1024);

    // Allocate blocksize
    char* memory = a.Alloc(1024);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 1024);

    // Allocate another blocksize
    char* memory2 = a.Alloc(1024);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 1024);
  }

  // Allocate an arena and allocate two blocks
  // that together exceed a block size
  {
    Arena a(1024);

    //
    char* memory = a.Alloc(768);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 768);

    // Allocate another blocksize
    char* memory2 = a.Alloc(768);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 768);
  }

  // Allocate larger than a blocksize
  {
    Arena a(1024);

    char* memory = a.Alloc(10240);
    ASSERT_NE(memory, nullptr);
    TestMemory(memory, 10240);

    // Allocate another blocksize
    char* memory2 = a.Alloc(1234);
    ASSERT_NE(memory2, nullptr);
    TestMemory(memory2, 1234);
  }
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
