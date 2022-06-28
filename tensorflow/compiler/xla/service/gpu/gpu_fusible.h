/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTh() {
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


#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"

// TODO(b/112957171): Extract logic to determine fusibility of HLO ops from
// GpuInstructionFusion, FusionMerger, and GpuMultiOutputFusion.

namespace xla {
namespace gpu {

// Fusion passes frequently do checks across all pairs of "interesting" nodes.
// Computing e.g. FusionFitsInBudget(a, b) requires computing expensive
// properties of `a` and `b` individually.  This cache lets us avoid recomputing
// those properties n^2 times.
//
// Invariant: After modifying or removing a fusion node, call Invalidate(node).
struct FusionInfoCache {
 public:
  // Must be called after modifying or removing a fusion node (or other node
  // that's part of this cache).
  void Invalidate(const HloInstruction* instr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.h", "Invalidate");

    shared_memory_usage.erase(instr);
    num_unnested_reductions.erase(instr);
  }

  // The rest of the members of this this class are for internal use within
  // gpu_fusible. You shouldn't need to use them yourself.
  absl::flat_hash_map<const HloInstruction*, int64_t> shared_memory_usage;
  absl::flat_hash_map<const HloInstruction*, int64_t> num_unnested_reductions;
};

inline constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 64; }

bool IsInputFusible(const HloInstruction& instr);

bool IsLoopFusible(const HloInstruction& instr);

// The code emitted for reduce-rooted input fusions (EmitReductionToVector)
// suffers from poor data locality if the layouts of input parameters differ. In
// such situations it is better not to fuse. Only input params with
// maximum rank are considered. Params with smaller ranks will be broadcasted
// and have not been observed to cause data locality issues.
bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce);

// Note that reduction ops are lowered in different ways. Reduce input fusions
// are lowered by IrEmitterUnnested::EmitReductionToVector and must be rooted at
// reduction-to-vector ops. Other reduction ops are lowered by
// GpuElementalIrEmitter and fused like elementwise ops.

// Whether `instr` is an input fusion rooted at a reduction-to-vector op or a
// multi-output input fusion with at least one reduction-to-vector op root.
bool IsReduceInputFusion(const HloInstruction& instr);

// Whether `instr` is fusible as root of a reduce input fusions, i.e. `instr`
// is either an unfused reduction-to-vector op or a reduce input fusion.
bool IsInputFusibleReduction(const HloInstruction& instr);

// Whether `instr` is fusible as root of a scatter input fusions, i.e. `instr`
// is either an unfused scatter op or a scatter input fusion.
bool IsInputFusibleScatter(const HloInstruction& instr);

// Determines whether the combination of `instr1` and `instr2` into a (possibly
// multi-output) fusion fits within a "budget" -- i.e., does have more operands
// and outputs than is allowed or occupy too much shared memory. If the fusion
// is a producer/consumer fusion and `instr1` is the consumer and `instr2` is
// the producer, set consumer_producer_fusion to true to enable more fusion.
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  bool is_consumer_producer_fusion = false,
                                  FusionInfoCache* cache = nullptr);

// Check if fusing producer and consumer will generate a nested loop, e.g. both
// producer and consumer are `reduce-window` HLO instructions.
bool CreatesNestedLoop(const HloInstruction& producer,
                       const HloInstruction& consumer);

// Returns the instruction that determines the emitter used for lowering,
// sometimes referred to as "the real hero".
const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr);

// Whether instruction shapes are compatible for multi-output fusion, i.e.
// whether the emitters support lowering the resulting fusion.
// This function works for both, sibling and producer-consumer multi-output
// fusion.
// So far, multi-output fusion is supported for loop fusions and reduce
// input fusions only. It is up to the caller to ensure the instructions
// themselves are fusible!
bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2);

// Whether the instructions are compatible for producer-consumer fusion
// i.e. whether the producer and consumer are loop/input fusible and
// they are not library calls.
FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer);

// Whether the instructions are producer-consumer fusible with multiple outputs.
// That is, the root tuple of the multi-output fusion will contain the results
// of both, the producer and consumer.
bool IsProducerConsumerMultiOutputFusible(const HloInstruction& producer,
                                          const HloInstruction& consumer);
// Whether `instr` is a candidate for sibling fusion or as a consumer in
// a producer-consumer multi-output fusion.
bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr);

// Determines the fusion kind to be used when fusing `producer` and `consumer`.
HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& producer,
                                            const HloInstruction& consumer);

// Returns whether `consumer` is the only non-root user of `instr`.
bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer);

// Returns number of instructions in the fusible `instr`. If `instr` is not a
// fusion instruction, 1 is returned.
size_t GetInstrCountOfFusible(const HloInstruction& instr);

// Returns the outputs of the fusible `instr`.
absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr);

// Returns the output size of the fusible `instr`.
size_t GetOutputSizeOfFusible(const HloInstruction& instr);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_FUSIBLE_H_
