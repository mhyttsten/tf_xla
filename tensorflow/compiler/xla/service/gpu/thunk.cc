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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

Thunk::ExecuteParams::ExecuteParams(
    const ServiceExecutableRunOptions& run_options,
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    se::Stream* async_comms_stream)
    : buffer_allocations(&buffer_allocations),
      stream(stream),
      async_comms_stream(async_comms_stream),
      run_id(run_options.run_options().run_id()),
      device_assn(run_options.run_options().device_assignment()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/gpu/thunk.cc", "Thunk::ExecuteParams::ExecuteParams");

  const GpuExecutableRunOptions* gpu_options =
      run_options.run_options().gpu_executable_run_options();
  gpu_global_device_ids = gpu_options && gpu_options->gpu_global_device_ids()
                              ? &*gpu_options->gpu_global_device_ids()
                              : nullptr;
  nccl_unique_id_callback =
      gpu_options && gpu_options->nccl_unique_id_callback()
          ? &gpu_options->nccl_unique_id_callback()
          : nullptr;
}

StatusOr<GlobalDeviceId> Thunk::ExecuteParams::GetGlobalDeviceId() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/service/gpu/thunk.cc", "Thunk::ExecuteParams::GetGlobalDeviceId");

  int64_t local_device_ordinal = stream->parent()->device_ordinal();
  if (gpu_global_device_ids) {
    TF_RET_CHECK(0 <= local_device_ordinal &&
                 local_device_ordinal < gpu_global_device_ids->size());
    return (*gpu_global_device_ids)[local_device_ordinal];
  } else {
    // No local -> global mapping was provided; assume the identity mapping.
    return GlobalDeviceId(local_device_ordinal);
  }
}

/*static*/ absl::string_view Thunk::KindToString(Thunk::Kind kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc mht_2(mht_2_v, 228, "", "./tensorflow/compiler/xla/service/gpu/thunk.cc", "Thunk::KindToString");

  switch (kind) {
    case Thunk::kCholesky:
      return "kCholesky";
    case Thunk::kCollectivePermute:
      return "kCollectivePermute";
    case Thunk::kConditional:
      return "kConditional";
    case Thunk::kConvolution:
      return "kConvolution";
    case Thunk::kCopy:
      return "kCopy";
    case Thunk::kCustomCall:
      return "kCustomCall";
    case Thunk::kNcclAllGather:
      return "kNcclAllGather";
    case Thunk::kNcclAllReduce:
      return "kNcclAllReduce";
    case Thunk::kNcclAllReduceStart:
      return "kNcclAllReduceStart";
    case Thunk::kNcclAllReduceDone:
      return "kNcclAllReduceDone";
    case Thunk::kNcclReduceScatter:
      return "kNcclReduceScatter";
    case Thunk::kNcclAllToAll:
      return "kNcclAllToAll";
    case Thunk::kFft:
      return "kFft";
    case Thunk::kGemm:
      return "kGemm";
    case Thunk::kInfeed:
      return "kInfeed";
    case Thunk::kKernel:
      return "kKernel";
    case Thunk::kMemset32BitValue:
      return "kMemset32BitValue";
    case Thunk::kMemzero:
      return "kMemzero";
    case Thunk::kOutfeed:
      return "kOutfeed";
    case Thunk::kReplicaId:
      return "kReplicaId";
    case Thunk::kPartitionId:
      return "kPartitionId";
    case Thunk::kSequential:
      return "kSequential";
    case Thunk::kTriangularSolve:
      return "kTriangularSolve";
    case Thunk::kWhile:
      return "kWhile";
  }
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xla/service/gpu/thunk.cc", "operator<<");

  return os << Thunk::KindToString(kind);
}

std::string ThunkSequence::ToString(
    int indent,
    std::function<std::string(const Thunk*)> get_thunk_annotation) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSthunkDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/service/gpu/thunk.cc", "ThunkSequence::ToString");

  const std::string indent_str(indent * 2, ' ');
  if (empty()) return indent_str + "No thunks.";

  auto thunk_with_longest_kind = absl::c_max_element(
      *this,
      [](const std::unique_ptr<Thunk>& a, const std::unique_ptr<Thunk>& b) {
        return Thunk::KindToString(a->kind()).length() <
               Thunk::KindToString(b->kind()).length();
      });
  int64_t max_thunk_kind_len =
      Thunk::KindToString(thunk_with_longest_kind->get()->kind()).length();
  std::string result;
  for (const std::unique_ptr<Thunk>& thunk : *this) {
    // Write out the thunk kind, padded out to max_thunk_kind_len.
    absl::string_view kind_str = Thunk::KindToString(thunk->kind());
    absl::StrAppend(&result, indent_str, kind_str,
                    std::string(max_thunk_kind_len - kind_str.length(), ' '),
                    "\t");
    if (get_thunk_annotation) {
      absl::StrAppend(&result, get_thunk_annotation(thunk.get()));
    }
    absl::StrAppend(&result, thunk->ToStringExtra(indent));
    absl::StrAppend(&result, "\n");
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
