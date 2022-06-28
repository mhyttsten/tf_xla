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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"

namespace xla {
namespace gpu {

KernelThunk::KernelThunk(ThunkInfo thunk_info,
                         absl::Span<const BufferAllocation* const> args,
                         const std::string& kernel_name,
                         const LaunchDimensions& launch_dimensions)
    : Thunk(Kind::kKernel, thunk_info),
      args_(args.begin(), args.end()),
      kernel_name_(kernel_name),
      launch_dimensions_(launch_dimensions) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("kernel_name: \"" + kernel_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/gpu/kernel_thunk.cc", "KernelThunk::KernelThunk");
}

std::string KernelThunk::ToStringExtra(int indent) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/xla/service/gpu/kernel_thunk.cc", "KernelThunk::ToStringExtra");

  return absl::StrFormat(", kernel = %s, launch dimensions = %s", kernel_name_,
                         launch_dimensions_.ToString());
}

Status KernelThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xla/service/gpu/kernel_thunk.cc", "KernelThunk::Initialize");

  absl::MutexLock lock(&mutex_);

  // Load the kernel into the device if necessary.
  //
  // We could alternatively do this within ExecuteOnStream, but doing it here
  // lets the time spent loading the kernel not count towards our execution
  // profiles.
  auto it = kernel_cache_.find(executor);
  if (kernel_cache_.end() == it) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::KernelBase> kernel,
        CreateKernel(kernel_name_, args_.size(), executable.text(),
                     executable.binary(), executor));

    kernel_cache_.emplace(executor, std::move(kernel));
  }

  return Status::OK();
}

static void PrintBufferContents(
    se::Stream* stream, absl::Span<const se::DeviceMemoryBase> buffer_args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/service/gpu/kernel_thunk.cc", "PrintBufferContents");

  int input_idx = 0;
  for (const se::DeviceMemoryBase& buf : buffer_args) {
    auto host_buffer = absl::make_unique<char[]>(buf.size());
    CHECK(stream->ThenMemcpy(host_buffer.get(), buf, buf.size()).ok());
    CHECK(stream->BlockHostUntilDone().ok());

    std::string buffer_contents;
    for (int i = 0; i < buf.size(); i++) {
      absl::StrAppendFormat(&buffer_contents, "%x ",
                            static_cast<unsigned>(host_buffer[i]));
    }
    VLOG(100) << "BUF(" << input_idx++ << ") = " << buffer_contents;
  }
}

Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSkernel_thunkDTcc mht_4(mht_4_v, 270, "", "./tensorflow/compiler/xla/service/gpu/kernel_thunk.cc", "KernelThunk::ExecuteOnStream");

  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  LaunchDimensions launch_dimensions;
  const se::KernelBase* kernel = nullptr;

  {
    absl::MutexLock lock(&mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    launch_dimensions = launch_dimensions_;
    kernel = it->second.get();
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  for (const BufferAllocation* arg : args_) {
    se::DeviceMemoryBase buf =
        params.buffer_allocations->GetDeviceAddress(arg->index());
    VLOG(3) << "  Arg: alloc #" << arg->index() << ": " << buf.opaque() << "  ("
            << buf.size() << "B)";
    buffer_args.push_back(buf);
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(params.stream, buffer_args);
  }

  return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                               params.stream);
}

}  // namespace gpu
}  // namespace xla
