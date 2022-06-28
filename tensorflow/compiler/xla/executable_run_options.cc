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
class MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc() {
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

#include "tensorflow/compiler/xla/executable_run_options.h"

#include <atomic>

namespace xla {

RunId::RunId() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_0(mht_0_v, 191, "", "./tensorflow/compiler/xla/executable_run_options.cc", "RunId::RunId");

  static std::atomic<int64_t> counter{0};
  data_ = counter.fetch_add(1);
}

bool operator==(const RunId& a, const RunId& b) { return a.data_ == b.data_; }

std::string RunId::ToString() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_1(mht_1_v, 201, "", "./tensorflow/compiler/xla/executable_run_options.cc", "RunId::ToString");

  return "RunId: " + std::to_string(data_);
}

int64_t RunId::ToInt() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_2(mht_2_v, 208, "", "./tensorflow/compiler/xla/executable_run_options.cc", "RunId::ToInt");
 return data_; }

ExecutableRunOptions& ExecutableRunOptions::set_device_ordinal(
    int device_ordinal) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_3(mht_3_v, 214, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_device_ordinal");

  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableRunOptions::device_ordinal() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_4(mht_4_v, 222, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::device_ordinal");
 return device_ordinal_; }

ExecutableRunOptions& ExecutableRunOptions::set_allocator(
    stream_executor::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_5(mht_5_v, 228, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_allocator");

  allocator_ = allocator;
  return *this;
}

stream_executor::DeviceMemoryAllocator* ExecutableRunOptions::allocator()
    const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_6(mht_6_v, 237, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::allocator");

  return allocator_;
}

ExecutableRunOptions& ExecutableRunOptions::set_stream(
    stream_executor::Stream* stream) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_7(mht_7_v, 245, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_stream");

  stream_ = stream;
  return *this;
}

stream_executor::Stream* ExecutableRunOptions::stream() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_8(mht_8_v, 253, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::stream");

  return stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_host_to_device_stream(
    stream_executor::Stream* stream) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_9(mht_9_v, 261, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_host_to_device_stream");

  host_to_device_stream_ = stream;
  return *this;
}

stream_executor::Stream* ExecutableRunOptions::host_to_device_stream() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_10(mht_10_v, 269, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::host_to_device_stream");

  return host_to_device_stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_intra_op_thread_pool(
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_11(mht_11_v, 277, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_intra_op_thread_pool");

  intra_op_thread_pool_ = intra_op_thread_pool;
  return *this;
}

const Eigen::ThreadPoolDevice* ExecutableRunOptions::intra_op_thread_pool()
    const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_12(mht_12_v, 286, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::intra_op_thread_pool");

  return intra_op_thread_pool_;
}

ExecutableRunOptions& ExecutableRunOptions::set_execution_profile(
    ExecutionProfile* profile) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_13(mht_13_v, 294, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_execution_profile");

  execution_profile_ = profile;
  return *this;
}

ExecutionProfile* ExecutableRunOptions::execution_profile() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_14(mht_14_v, 302, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::execution_profile");

  return execution_profile_;
}

ExecutableRunOptions& ExecutableRunOptions::set_device_assignment(
    const DeviceAssignment* device_assignment) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_15(mht_15_v, 310, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_device_assignment");

  device_assignment_ = device_assignment;
  return *this;
}

const DeviceAssignment* ExecutableRunOptions::device_assignment() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_16(mht_16_v, 318, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::device_assignment");

  return device_assignment_;
}

ExecutableRunOptions& ExecutableRunOptions::set_gpu_executable_run_options(
    const gpu::GpuExecutableRunOptions* gpu_executable_run_options) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_17(mht_17_v, 326, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_gpu_executable_run_options");

  gpu_executable_run_options_ = gpu_executable_run_options;
  return *this;
}

const gpu::GpuExecutableRunOptions*
ExecutableRunOptions::gpu_executable_run_options() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_18(mht_18_v, 335, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::gpu_executable_run_options");

  return gpu_executable_run_options_;
}

ExecutableRunOptions& ExecutableRunOptions::set_rng_seed(int rng_seed) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_19(mht_19_v, 342, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_rng_seed");

  rng_seed_ = rng_seed;
  return *this;
}

int ExecutableRunOptions::rng_seed() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_20(mht_20_v, 350, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::rng_seed");
 return rng_seed_; }

ExecutableRunOptions& ExecutableRunOptions::set_run_id(RunId id) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_21(mht_21_v, 355, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::set_run_id");

  run_id_ = id;
  return *this;
}

RunId ExecutableRunOptions::run_id() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTcc mht_22(mht_22_v, 363, "", "./tensorflow/compiler/xla/executable_run_options.cc", "ExecutableRunOptions::run_id");
 return run_id_; }

}  // namespace xla
