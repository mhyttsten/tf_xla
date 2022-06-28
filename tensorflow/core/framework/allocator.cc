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
class MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc() {
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

#include "tensorflow/core/framework/allocator.h"

#include <atomic>

#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

string AllocatorStats::DebugString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/framework/allocator.cc", "AllocatorStats::DebugString");

  return strings::Printf(
      "Limit:            %20lld\n"
      "InUse:            %20lld\n"
      "MaxInUse:         %20lld\n"
      "NumAllocs:        %20lld\n"
      "MaxAllocSize:     %20lld\n"
      "Reserved:         %20lld\n"
      "PeakReserved:     %20lld\n"
      "LargestFreeBlock: %20lld\n",
      static_cast<long long>(this->bytes_limit ? *this->bytes_limit : 0),
      static_cast<long long>(this->bytes_in_use),
      static_cast<long long>(this->peak_bytes_in_use),
      static_cast<long long>(this->num_allocs),
      static_cast<long long>(this->largest_alloc_size),
      static_cast<long long>(this->bytes_reserved),
      static_cast<long long>(this->peak_bytes_reserved),
      static_cast<long long>(this->largest_free_block_bytes));
}

constexpr size_t Allocator::kAllocatorAlignment;

Allocator::~Allocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/framework/allocator.cc", "Allocator::~Allocator");
}

// If true, cpu allocator collects full stats.
static bool cpu_allocator_collect_full_stats = false;

void EnableCPUAllocatorFullStats() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/framework/allocator.cc", "EnableCPUAllocatorFullStats");
 cpu_allocator_collect_full_stats = true; }
bool CPUAllocatorFullStatsEnabled() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/framework/allocator.cc", "CPUAllocatorFullStatsEnabled");
 return cpu_allocator_collect_full_stats; }

string AllocatorAttributes::DebugString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_4(mht_4_v, 241, "", "./tensorflow/core/framework/allocator.cc", "AllocatorAttributes::DebugString");

  return strings::StrCat("AllocatorAttributes(on_host=", on_host(),
                         " nic_compatible=", nic_compatible(),
                         " gpu_compatible=", gpu_compatible(), ")");
}

Allocator* cpu_allocator_base() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/framework/allocator.cc", "cpu_allocator_base");

  static Allocator* cpu_alloc =
      AllocatorFactoryRegistry::singleton()->GetAllocator();
  // TODO(tucker): This really seems wrong.  It's only going to be effective on
  // the first call in a process (but the desired effect is associated with a
  // session), and we probably ought to be tracking the highest level Allocator,
  // not the lowest.  Revisit the advertised semantics of the triggering option.
  if (cpu_allocator_collect_full_stats && !cpu_alloc->TracksAllocationSizes()) {
    cpu_alloc = new TrackingAllocator(cpu_alloc, true);
  }
  return cpu_alloc;
}

Allocator* cpu_allocator(int numa_node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_6(mht_6_v, 266, "", "./tensorflow/core/framework/allocator.cc", "cpu_allocator");

  // Correctness relies on devices being created prior to the first call
  // to cpu_allocator, if devices are ever to be created in the process.
  // Device creation in turn triggers ProcessState creation and the availability
  // of the correct access pointer via this function call.
  static ProcessStateInterface* ps =
      AllocatorFactoryRegistry::singleton()->process_state();
  if (ps) {
    return ps->GetCPUAllocator(numa_node);
  } else {
    return cpu_allocator_base();
  }
}

SubAllocator::SubAllocator(const std::vector<Visitor>& alloc_visitors,
                           const std::vector<Visitor>& free_visitors)
    : alloc_visitors_(alloc_visitors), free_visitors_(free_visitors) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_7(mht_7_v, 285, "", "./tensorflow/core/framework/allocator.cc", "SubAllocator::SubAllocator");
}

void SubAllocator::VisitAlloc(void* ptr, int index, size_t num_bytes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/framework/allocator.cc", "SubAllocator::VisitAlloc");

  for (const auto& v : alloc_visitors_) {
    v(ptr, index, num_bytes);
  }
}

void SubAllocator::VisitFree(void* ptr, int index, size_t num_bytes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocatorDTcc mht_9(mht_9_v, 299, "", "./tensorflow/core/framework/allocator.cc", "SubAllocator::VisitFree");

  // Although we don't guarantee any order of visitor application, strive
  // to apply free visitors in reverse order of alloc visitors.
  for (int i = free_visitors_.size() - 1; i >= 0; --i) {
    free_visitors_[i](ptr, index, num_bytes);
  }
}
}  // namespace tensorflow
