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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/shape_util.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/xla_executor_state.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

constexpr int kMaxInfeedsInFlight = 8;

InfeedManager::InfeedManager(se::StreamExecutor* executor)
    : BlockingXfeedQueue(/*max_pending_xfeeds=*/kMaxInfeedsInFlight),
      stream_(absl::make_unique<se::Stream>(executor)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/service/gpu/infeed_manager.cc", "InfeedManager::InfeedManager");

  stream_->Init();
}

static StatusOr<se::ScopedDeviceMemory<uint8_t>> CopyBufferToDevice(
    se::Stream* stream, int64_t size, const void* source) {
  if (size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument("GPU infeed of %d bytes exceeds maximum of %d bytes",
                           size, std::numeric_limits<int32_t>::max());
  }

  if (size == 0) {
    return InvalidArgument("Infeed shape needs 0 bytes");
  }

  se::StreamExecutor* executor = stream->parent();
  se::ScopedDeviceMemory<uint8_t> buffer(
      executor, executor->AllocateArray<uint8_t>(size));
  stream->ThenMemcpy(buffer.ptr(), source, size);

  return std::move(buffer);
}

Status InfeedManager::TransferLiteralToInfeed(se::StreamExecutor* executor,
                                              const LiteralSlice& literal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/xla/service/gpu/infeed_manager.cc", "InfeedManager::TransferLiteralToInfeed");

  const Shape& literal_shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(literal_shape);

  BlockUntilEnqueueSlotAvailable();

  // For a tuple, we transfer each of its elements to the device and enqueue the
  // resulting destination device addresses with the infeed manager.
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> buffer_tree(literal_shape);
  for (auto& leaf : buffer_tree.leaves()) {
    const Shape& sub_shape = ShapeUtil::GetSubshape(literal_shape, leaf.first);
    CHECK(sub_shape.IsArray()) << ShapeUtil::HumanStringWithLayout(sub_shape);
    TF_ASSIGN_OR_RETURN(
        leaf.second,
        CopyBufferToDevice(stream(), ShapeUtil::ByteSizeOf(sub_shape),
                           literal.untyped_data(leaf.first)));
  }

  // TODO(b/30467474): Since this stream is shared across different infeed
  // requests, blocking on the stream might be heavy-handed. Figure out if
  // finer-grained acknowledgement is possible.
  Status block_status = stream()->BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         stream(), block_status.error_message());
  }

  EnqueueDestination(std::move(buffer_tree));
  return Status::OK();
}

InfeedManager* GetOrCreateInfeedManager(se::StreamExecutor* executor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSinfeed_managerDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/gpu/infeed_manager.cc", "GetOrCreateInfeedManager");

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  stream_executor::gpu::GpuExecutor* gpu_executor =
      stream_executor::gpu::ExtractGpuExecutor(executor);
  auto* xla_state =
      gpu_executor->getOrCreateXLAState<GpuExecutorXLAState>(executor);
  return xla_state->getOrCreateInfeedManager(executor);
#else   // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace gpu
}  // namespace xla
