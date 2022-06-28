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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh() {
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


#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the
// metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction.
//
// Thunk provides the Initialize and ExecuteOnStream interface for GpuExecutable
// to initialize and execute the invocation respectively. Its subclasses are
// supposed to override these interfaces to launch a generated kernel or call an
// external library function (such as operations in cuBLAS).
//
// This is thread-compatible.
class Thunk {
 public:
  enum Kind {
    kCholesky,
    kCollectivePermute,
    kConditional,
    kConvolution,
    kCopy,
    kCustomCall,
    kFft,
    kGemm,
    kInfeed,
    kKernel,
    kMemset32BitValue,
    kMemzero,
    kNcclAllGather,
    kNcclAllReduce,
    kNcclAllReduceStart,
    kNcclAllReduceDone,
    kNcclReduceScatter,
    kNcclAllToAll,
    kOutfeed,
    kReplicaId,
    kPartitionId,
    kSequential,
    kTriangularSolve,
    kWhile,
  };

  struct ThunkInfo {
    absl::optional<int64_t> profile_index;
    std::string profile_annotation;
  };

  // The hlo_instruction argument is meant to be the instruction this thunk was
  // generated from, but Thunk never uses this argument other than to save it
  // to Thunk::hlo_instruction, so it can be null.
  explicit Thunk(Kind kind, ThunkInfo thunk_info)
      : kind_(kind),
        profile_index_(thunk_info.profile_index),
        profile_annotation_(thunk_info.profile_annotation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_0(mht_0_v, 253, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "Thunk");
}
  virtual ~Thunk() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_1(mht_1_v, 257, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "~Thunk");
}
  Thunk(const Thunk&) = delete;
  Thunk& operator=(const Thunk&) = delete;

  virtual std::string ToStringExtra(int indent) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "ToStringExtra");
 return ""; }
  Kind kind() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_3(mht_3_v, 268, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "kind");
 return kind_; }
  std::string profile_annotation() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_4(mht_4_v, 272, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "profile_annotation");
 return profile_annotation_; }

  // Prepares the thunk for execution on the given StreamExecutor.
  //
  // This may be called multiple times.  Its main purpose is to give us a chance
  // to do initialization outside of ExecuteOnStream() so that the
  // time spent initializing doesn't count towards our execution profile.
  virtual Status Initialize(const GpuExecutable& /*executable*/,
                            se::StreamExecutor* /*executor*/) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTh mht_5(mht_5_v, 283, "", "./tensorflow/compiler/xla/service/gpu/thunk.h", "Initialize");

    return Status::OK();
  }

  // Parameters passed to ExecuteOnStream.  Encapsulated in a struct so that
  // when we add something we don't have to change every subclass of Thunk.
  struct ExecuteParams {
    ExecuteParams(const ServiceExecutableRunOptions& run_options,
                  const BufferAllocations& buffer_allocations,
                  se::Stream* stream, se::Stream* async_comms_stream);

    const BufferAllocations* buffer_allocations;  // never null
    se::Stream* stream;
    se::Stream* async_comms_stream;
    RunId run_id;
    const DeviceAssignment* device_assn;                          // never null
    const std::vector<GlobalDeviceId>* gpu_global_device_ids;     // may be null
    const NcclUniqueIdCallback* nccl_unique_id_callback;          // may be null

    StatusOr<GlobalDeviceId> GetGlobalDeviceId() const;
  };

  // Execute the kernel for the thunk on the given stream. This method must be
  // called after Initialize and can be called multiple times over Thunk's
  // lifetime.
  //
  // Precondition: Initialize(stream->parent()) has been called.
  virtual Status ExecuteOnStream(const ExecuteParams& params) = 0;

  static absl::string_view KindToString(Thunk::Kind kind);

 protected:
  absl::optional<int64_t> profile_index() const { return profile_index_; }

 private:
  Kind kind_;
  absl::optional<int64_t> profile_index_;
  std::string profile_annotation_;
};

// A sequence of thunks.
class ThunkSequence : public std::vector<std::unique_ptr<Thunk>> {
 public:
  std::string ToString(int indent = 0,
                       std::function<std::string(const Thunk*)>
                           get_thunk_annotation = nullptr) const;
};

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind);

// A struct that defines a shaped slice, i.e., a BufferAllocation::Slice and its
// shape.
struct ShapedSlice {
  BufferAllocation::Slice slice;
  Shape shape;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_H_
