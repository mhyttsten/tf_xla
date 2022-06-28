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
#ifndef TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_
#define TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_
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
class MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh {
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
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh() {
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


#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// This little structure holds the offset and the size for a dynamic memory
// allocation in the memory arena as well as first_node and last_node that use
// corresponding tensor. It means that continuous part of memory with this size
// needs to be allocated before execution of operation in the first node and can
// be deallocated after execution of the operation in the last_node. When the
// arena is committed and the underlying buffer is set, the alloc can be
// resolved into an actual memory pointer.
struct ArenaAllocWithUsageInterval {
  ArenaAllocWithUsageInterval() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/simple_memory_arena.h", "ArenaAllocWithUsageInterval");
 reset(); }

  size_t offset;
  size_t size;
  int32_t tensor;
  int32_t first_node;
  int32_t last_node;

  inline void reset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_1(mht_1_v, 217, "", "./tensorflow/lite/simple_memory_arena.h", "reset");

    offset = 0;
    size = 0;
    tensor = -1;
    first_node = -1;
    last_node = -1;
  }

  inline bool operator<(const ArenaAllocWithUsageInterval& other) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_2(mht_2_v, 228, "", "./tensorflow/lite/simple_memory_arena.h", "operator<");

    return offset < other.offset;
  }
};

// This small class is responsible for allocating, deallocating and reusing
// dynamic memory from a common underlying buffer. The arena can be used in
// scenarios when the pattern of memory allocations and deallocations is
// repetitive, e.g. running NN inference in multiple iterations. Note that
// zero-sized allocations are explicitly allowed, and will resolve to null.
class SimpleMemoryArena {
 public:
  explicit SimpleMemoryArena(size_t arena_alignment)
      : committed_(false),
        arena_alignment_(arena_alignment),
        high_water_mark_(0),
        underlying_buffer_size_(0),
        ordered_allocs_() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_3(mht_3_v, 248, "", "./tensorflow/lite/simple_memory_arena.h", "SimpleMemoryArena");
}

  // Schedule memory allocation for a tensor with a given size, assuming that it
  // needs to be allocated before the execution of first_node, and deallocated
  // after the execution of last_node.
  TfLiteStatus Allocate(TfLiteContext* context, size_t alignment, size_t size,
                        int32_t tensor, int32_t first_node, int32_t last_node,
                        ArenaAllocWithUsageInterval* new_alloc);

  TfLiteStatus Deallocate(TfLiteContext* context,
                          const ArenaAllocWithUsageInterval& alloc);

  inline size_t RequiredBufferSize() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_4(mht_4_v, 263, "", "./tensorflow/lite/simple_memory_arena.h", "RequiredBufferSize");

    // Add in a small amount of padding to reduce the chance of resize events
    // for small allocations.
    size_t padding = arena_alignment_;
    return arena_alignment_ + high_water_mark_ + padding;
  }

  TfLiteStatus Commit(TfLiteContext* context);

  TfLiteStatus ResolveAlloc(TfLiteContext* context,
                            const ArenaAllocWithUsageInterval& alloc,
                            char** output_ptr);

  // This clears allocation details but does not release the underlying buffer.
  // New allocations should be committed & resolved before using this arena
  // again.
  TfLiteStatus ClearPlan();

  // This releases the underlying buffer but does not clear the allocation plan.
  // Since all associated pointers are invalidated, the arena cannot be used
  // again until Commit() is called & tensor allocations are resolved.
  TfLiteStatus ReleaseBuffer();

  size_t GetBufferSize() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_5(mht_5_v, 289, "", "./tensorflow/lite/simple_memory_arena.h", "GetBufferSize");
 return underlying_buffer_size_; }

  std::intptr_t BasePointer() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSsimple_memory_arenaDTh mht_6(mht_6_v, 294, "", "./tensorflow/lite/simple_memory_arena.h", "BasePointer");

    return reinterpret_cast<std::intptr_t>(underlying_buffer_aligned_ptr_);
  }

  // Dumps the memory allocation information of this memory arena (which could
  // be differentiated from others by the `name`) against the specified op node
  // execution plan (i.e. `execution_plan`) for the purpose of debugging.
  // Note: in order to have minimal binary increase caused by this debug info
  // dump implementation for the TfLite library, and allow users to plug-in
  // their own memory planner debugger, we have utilized weak symbols to meet
  // these two requirementsements. By default, there is no debugging info
  // dumped. To override this, provide a strong defintion of
  // tflite::DumpArenaInfo(...) whose weak defintion is in
  // simple_memory_arena.cc. TfLite provides a sample one as
  // "lite:simple_memory_arena_debug_dump". When this dep is added to the
  // program, calling this function will output information of this memory arena
  // about tenosrs and ops, such as memory arena utilization rate, live tensors
  // at each op etc.
  void DumpDebugInfo(const std::string& name,
                     const std::vector<int>& execution_plan) const;

 private:
  bool committed_;
  size_t arena_alignment_;
  size_t high_water_mark_;
  std::unique_ptr<char[]> underlying_buffer_;
  size_t underlying_buffer_size_;
  char* underlying_buffer_aligned_ptr_;
  std::vector<ArenaAllocWithUsageInterval> ordered_allocs_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIMPLE_MEMORY_ARENA_H_
