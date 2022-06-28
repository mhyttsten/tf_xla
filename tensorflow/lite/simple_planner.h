/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SIMPLE_PLANNER_H_
#define TENSORFLOW_LITE_SIMPLE_PLANNER_H_
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
class MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh {
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
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh() {
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


#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/util.h"

namespace tflite {

// A structure to keep heap allocation records. This structure is used by
// SimplePlanner::allocs_.
struct SimpleAlloc {
  SimpleAlloc() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_0(mht_0_v, 202, "", "./tensorflow/lite/simple_planner.h", "SimpleAlloc");
 reset(); }

  // Size of allocation.
  size_t size;
  // The index of the node that first needs to use this tensor.
  int32_t node;
  // Allocated heap memory address of allocation.
  char* ptr;

  // Reset member variables.
  inline void reset() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_1(mht_1_v, 215, "", "./tensorflow/lite/simple_planner.h", "reset");

    size = 0;
    node = 0;
    ptr = nullptr;
  }

  // Allocate heap memory for a tensor with the given size and first_node
  // information.
  inline bool alloc(size_t new_size, int32_t new_first_node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_2(mht_2_v, 226, "", "./tensorflow/lite/simple_planner.h", "alloc");

    if (new_size == 0) {
      return false;
    }
    size = new_size;
    node = new_first_node;
    assert(ptr == nullptr);
    ptr = static_cast<char*>(malloc(new_size));
    return true;
  }

  // Free allocated heap memory and reset member variables.
  inline void free() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_3(mht_3_v, 241, "", "./tensorflow/lite/simple_planner.h", "free");

    if (ptr) {
      ::free(ptr);
    }
    reset();
  }
};

// A memory planner that makes all the allocations using malloc()/free().
//
// This is simple implementation of MemoryPlanner which uses malloc()/free()
// instead of memory areana. This planner is designed for AddressSanitizer.
class SimplePlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain util the
  // ArenaPlanner is destroyed. The inputs to the graph will not share
  // memory with any other tensor, effectively preserving them until the end
  // of inference.
  SimplePlanner(TfLiteContext* context, std::unique_ptr<GraphInfo> graph_info);
  ~SimplePlanner() override;
  SimplePlanner(const SimplePlanner&) = delete;
  SimplePlanner& operator=(const SimplePlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus ResetAllocationsAfter(int node) override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_4(mht_4_v, 273, "", "./tensorflow/lite/simple_planner.h", "HasNonPersistentMemory");
 return true; };
  void DumpDebugInfo(const std::vector<int>& execution_plan) const override{
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSsimple_plannerDTh mht_5(mht_5_v, 277, "", "./tensorflow/lite/simple_planner.h", "DumpDebugInfo");
};

 private:
  // Free all the all allocations.
  void FreeAllAllocations();

  // Assign absolute memory location to a tensor.
  TfLiteStatus ResolveTensorAllocation(int tensor_index);

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;

  // Stores allocation data for all tensors.
  std::vector<SimpleAlloc> allocs_;

  // First node, that uses the tensor. It needs to be allocated before
  // execution of the node's operation.
  std::vector<int32_t> alloc_node_;

  // Last node, that uses the tensor. It can be deallocated after execution of
  // the node's operation.
  std::vector<int32_t> dealloc_node_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SIMPLE_PLANNER_H_
