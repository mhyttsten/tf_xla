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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/random.h"

namespace xla {
namespace gpu {

bool StreamAssignment::HasStreamAssigned(const HloInstruction& hlo) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "StreamAssignment::HasStreamAssigned");

  return hlo_to_stream_number_.contains(&hlo);
}

int StreamAssignment::StreamNumberForHlo(const HloInstruction& hlo) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "StreamAssignment::StreamNumberForHlo");

  return FindOrDie(hlo_to_stream_number_, &hlo);
}

void StreamAssignment::AssignStreamToHlo(const HloInstruction* hlo,
                                         int stream_num) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_2(mht_2_v, 214, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "StreamAssignment::AssignStreamToHlo");

  CHECK_GE(stream_num, 0);
  if (stream_num >= stream_count_) {
    stream_count_ = stream_num + 1;
  }
  InsertOrDie(&hlo_to_stream_number_, hlo, stream_num);
  VLOG(2) << "Assign stream #" << stream_num << " to " << hlo->ToString();
}

namespace {

// Returns whether the two HLOs can run concurrently, i.e., neither is a
// transitive consumer of the other.
bool CanRunConcurrently(const HloInstruction& a, const HloInstruction& b,
                        const HloReachabilityMap& reachability) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "CanRunConcurrently");

  return !reachability.IsConnected(&a, &b);
}

constexpr int kInvalidStreamNum = -1;
//  Returns true iff `stream_num` is an invalid stream number.
inline bool IsStreamNumValid(int stream_num) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_4(mht_4_v, 240, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "IsStreamNumValid");

  return stream_num != kInvalidStreamNum;
}

// Returns which existing stream to assign to `hlo`, or -1 if a stream is not
// needed. `stream_assignment` is the existing stream assignment for all
// instructions topologically before `hlo`. `seen_gemms` contains all GEMMs that
// are topologically before `hlo`.
int ComputeStreamToAssign(
    const HloInstruction& hlo, const StreamAssignment& stream_assignment,
    const HloReachabilityMap& reachability,
    const std::vector<const HloInstruction*>& seen_gemms) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSstream_assignmentDTcc mht_5(mht_5_v, 254, "", "./tensorflow/compiler/xla/service/gpu/stream_assignment.cc", "ComputeStreamToAssign");

  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kConstant) {
    // kParameter and kConstant do not need a thunk.
    return kInvalidStreamNum;
  }

  const auto& debug_options = hlo.GetModule()->config().debug_options();
  if (!debug_options.xla_gpu_disable_multi_streaming()) {
    LOG(ERROR) << "Multi streaming is not supported";
  }
  return 0;
}

}  // namespace

std::unique_ptr<StreamAssignment> AssignStreams(const HloModule& module) {
  auto stream_assignment = absl::make_unique<StreamAssignment>();
  const HloComputation& computation = *module.entry_computation();
  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(&computation);
  std::vector<const HloInstruction*> seen_gemms;
  // The execution of different RNG Hlo instructions in the same module updates
  // a common global variable. To avoid a race condition, we simply assign all
  // RNG kernels to the same stream to make them run sequentially.
  //
  // TODO(b/111791052): If we remove such a common variable, we will need to
  // clean up the code here.
  int stream_num_for_rng = kInvalidStreamNum;
  for (const auto* hlo : computation.MakeInstructionPostOrder()) {
    // If we ever enable fusion of RNG instructions, we will need to extend this
    // code to look inside a fused instruction.
    int stream_num = (hlo->opcode() == HloOpcode::kRng &&
                      IsStreamNumValid(stream_num_for_rng))
                         ? stream_num_for_rng
                         : ComputeStreamToAssign(*hlo, *stream_assignment,
                                                 *reachability, seen_gemms);
    if (IsStreamNumValid(stream_num)) {
      stream_assignment->AssignStreamToHlo(hlo, stream_num);
      if (hlo->opcode() == HloOpcode::kRng &&
          !IsStreamNumValid(stream_num_for_rng)) {
        stream_num_for_rng = stream_num;
      }
    }
    if (IsCublasGemm(*hlo) || IsMatrixMultiplication(*hlo)) {
      seen_gemms.push_back(hlo);
    }
  }
  return stream_assignment;
}

}  // namespace gpu
}  // namespace xla
