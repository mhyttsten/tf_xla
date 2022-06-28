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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.h"

#include "tensorflow/compiler/xla/service/heap_simulator.h"

namespace xla {

namespace {

using AllocationBlock = MemorySpaceAssignmentRepacker::AllocationBlock;
using Type = GlobalDecreasingSizeBestFitHeap<AllocationBlock>::Type;

// This class inherits GlobalDecreasingSizeBestFitHeap and converts
// AllocationBlock objects into BufferIntervals that the heap algorithm
// understands.
class BestFitRepacker
    : public GlobalDecreasingSizeBestFitHeap<AllocationBlock> {
 public:
  BestFitRepacker(int64_t max_size, int64_t alignment, Type type)
      : GlobalDecreasingSizeBestFitHeap<AllocationBlock>(alignment, type),
        max_size_(max_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.cc", "BestFitRepacker");
}

  void ImportAllocationBlocks(absl::Span<AllocationBlock*> allocations) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.cc", "ImportAllocationBlocks");

    allocation_blocks_ = allocations;
    for (AllocationBlock* allocation_block : allocations) {
      // Check if any of the colocations are already added to buffer_intervals_.
      bool need_allocation = true;
      auto aliased_it = absl::c_find_if(
          allocation_block->colocations, [&](AllocationBlock* search) {
            return buffer_intervals_.contains(search);
          });
      if (aliased_it != allocation_block->colocations.end()) {
        buffer_intervals_[*aliased_it].colocations.push_back(allocation_block);
        need_allocation = false;
      }
      buffer_intervals_[allocation_block] = {allocation_block,
                                             allocation_block->size,
                                             allocation_block->start_time,
                                             allocation_block->end_time,
                                             {},
                                             need_allocation};
    }
  }

  bool Repack() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.cc", "Repack");

    Finish();
    bool success = result_.heap_size <= max_size_;
    if (success) {
      for (AllocationBlock* block : allocation_blocks_) {
        auto chunk_it = result_.chunk_map.find(block);
        if (chunk_it != result_.chunk_map.end()) {
          block->offset = chunk_it->second.offset;
        }
      }
    }
    return success;
  }

 private:
  int64_t max_size_;
  absl::Span<AllocationBlock*> allocation_blocks_;
};

}  // namespace

StatusOr<bool> MemorySpaceAssignmentBestFitRepacker::Repack(
    absl::Span<AllocationBlock*> allocations) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSmemory_space_assignment_best_fit_repackerDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.cc", "MemorySpaceAssignmentBestFitRepacker::Repack");

  BestFitRepacker best_fit_repacker =
      BestFitRepacker(max_size_, alignment_, type_);
  best_fit_repacker.ImportAllocationBlocks(allocations);
  return best_fit_repacker.Repack();
}

}  // namespace xla
