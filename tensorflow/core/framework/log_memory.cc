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
class MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc() {
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

#include "tensorflow/core/framework/log_memory.h"

#include "tensorflow/core/framework/log_memory.pb.h"

namespace tensorflow {

const string LogMemory::kLogMemoryLabel = "__LOG_MEMORY__";

bool LogMemory::IsEnabled() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::IsEnabled");
 return VLOG_IS_ON(2); }

namespace {

// Write the proto entry to LOG(INFO).
template <typename T>
void OutputToLog(const T& proto) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/framework/log_memory.cc", "OutputToLog");

  string type_name = proto.GetTypeName();
  const size_t index = type_name.find_last_of('.');
  if (index != string::npos) type_name = type_name.substr(index + 1);
  LOG(INFO) << LogMemory::kLogMemoryLabel << " " << type_name << " { "
            << proto.ShortDebugString() << " }";
}

}  // namespace

void LogMemory::RecordStep(const int64_t step_id, const string& handle) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordStep");

  MemoryLogStep step;
  step.set_step_id(step_id);
  step.set_handle(handle);
  OutputToLog(step);
}

void LogMemory::RecordTensorAllocation(const string& kernel_name,
                                       const int64_t step_id,
                                       const Tensor& tensor) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("kernel_name: \"" + kernel_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordTensorAllocation");

  MemoryLogTensorAllocation allocation;
  allocation.set_step_id(step_id);
  allocation.set_kernel_name(kernel_name);
  tensor.FillDescription(allocation.mutable_tensor());
  OutputToLog(allocation);
}

void LogMemory::RecordTensorDeallocation(const int64_t allocation_id,
                                         const string& allocator_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("allocator_name: \"" + allocator_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordTensorDeallocation");

  MemoryLogTensorDeallocation deallocation;
  deallocation.set_allocation_id(allocation_id);
  deallocation.set_allocator_name(allocator_name);
  OutputToLog(deallocation);
}

void LogMemory::RecordTensorOutput(const string& kernel_name,
                                   const int64_t step_id, const int index,
                                   const Tensor& tensor) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("kernel_name: \"" + kernel_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordTensorOutput");

  MemoryLogTensorOutput output;
  output.set_step_id(step_id);
  output.set_kernel_name(kernel_name);
  output.set_index(index);
  tensor.FillDescription(output.mutable_tensor());
  OutputToLog(output);
}

void LogMemory::RecordRawAllocation(const string& operation,
                                    const int64_t step_id, size_t num_bytes,
                                    void* ptr, Allocator* allocator) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("operation: \"" + operation + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_6(mht_6_v, 270, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordRawAllocation");

  MemoryLogRawAllocation allocation;
  allocation.set_step_id(step_id);
  allocation.set_operation(operation);
  allocation.set_num_bytes(static_cast<int64_t>(num_bytes));
  allocation.set_ptr(reinterpret_cast<uintptr_t>(ptr));
  allocation.set_allocation_id(allocator->AllocationId(ptr));
  allocation.set_allocator_name(allocator->Name());
  OutputToLog(allocation);
}

void LogMemory::RecordRawDeallocation(const string& operation,
                                      const int64_t step_id, void* ptr,
                                      Allocator* allocator, bool deferred) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("operation: \"" + operation + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSlog_memoryDTcc mht_7(mht_7_v, 287, "", "./tensorflow/core/framework/log_memory.cc", "LogMemory::RecordRawDeallocation");

  MemoryLogRawDeallocation deallocation;
  deallocation.set_step_id(step_id);
  deallocation.set_operation(operation);
  deallocation.set_allocation_id(allocator->AllocationId(ptr));
  deallocation.set_allocator_name(allocator->Name());
  deallocation.set_deferred(deferred);
  OutputToLog(deallocation);
}

}  // namespace tensorflow
