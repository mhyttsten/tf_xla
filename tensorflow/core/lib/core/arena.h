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

// TODO(vrv): Switch this to an open-sourced version of Arena.

#ifndef TENSORFLOW_CORE_LIB_CORE_ARENA_H_
#define TENSORFLOW_CORE_LIB_CORE_ARENA_H_
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
class MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh() {
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


#include <assert.h>

#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace core {

// This class is "thread-compatible": different threads can access the
// arena at the same time without locking, as long as they use only
// const methods.
class Arena {
 public:
  // Allocates a thread-compatible arena with the specified block size.
  explicit Arena(const size_t block_size);
  ~Arena();

  char* Alloc(const size_t size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh mht_0(mht_0_v, 210, "", "./tensorflow/core/lib/core/arena.h", "Alloc");

    return reinterpret_cast<char*>(GetMemory(size, 1));
  }

  char* AllocAligned(const size_t size, const size_t alignment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/lib/core/arena.h", "AllocAligned");

    return reinterpret_cast<char*>(GetMemory(size, alignment));
  }

  void Reset();

// This should be the worst-case alignment for any type.  This is
// good for IA-32, SPARC version 7 (the last one I know), and
// supposedly Alpha.  i386 would be more time-efficient with a
// default alignment of 8, but ::operator new() uses alignment of 4,
// and an assertion will fail below after the call to MakeNewBlock()
// if you try to use a larger alignment.
#ifdef __i386__
  static const int kDefaultAlignment = 4;
#else
  static constexpr int kDefaultAlignment = 8;
#endif

 protected:
  bool SatisfyAlignment(const size_t alignment);
  void MakeNewBlock(const uint32 alignment);
  void* GetMemoryFallback(const size_t size, const int align);
  void* GetMemory(const size_t size, const int align) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/lib/core/arena.h", "GetMemory");

    assert(remaining_ <= block_size_);                  // an invariant
    if (size > 0 && size < remaining_ && align == 1) {  // common case
      void* result = freestart_;
      freestart_ += size;
      remaining_ -= size;
      return result;
    }
    return GetMemoryFallback(size, align);
  }

  size_t remaining_;

 private:
  struct AllocatedBlock {
    char* mem;
    size_t size;
  };

  // Allocate new block of at least block_size, with the specified
  // alignment.
  // The returned AllocatedBlock* is valid until the next call to AllocNewBlock
  // or Reset (i.e. anything that might affect overflow_blocks_).
  AllocatedBlock* AllocNewBlock(const size_t block_size,
                                const uint32 alignment);

  const size_t block_size_;
  char* freestart_;  // beginning of the free space in most recent block
  char* freestart_when_empty_;  // beginning of the free space when we're empty
  // STL vector isn't as efficient as it could be, so we use an array at first
  size_t blocks_alloced_;  // how many of the first_blocks_ have been alloced
  AllocatedBlock first_blocks_[16];  // the length of this array is arbitrary
  // if the first_blocks_ aren't enough, expand into overflow_blocks_.
  std::vector<AllocatedBlock>* overflow_blocks_;

  void FreeBlocks();  // Frees all except first block

  TF_DISALLOW_COPY_AND_ASSIGN(Arena);
};

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_ARENA_H_
