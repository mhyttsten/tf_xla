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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include <algorithm>
#include <iterator>
#include <stack>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

// The amount of shared memory a CUDA kernel can use.
//
// Stay on the conservative side, this is smaller than full 64kB, but allows
// some extra space for cache.
int64_t kSharedMemoryBudgetInBytes = 40000;

bool IfFusedReadsElementsMultipleTimes(const HloInstruction& instr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IfFusedReadsElementsMultipleTimes");

  CHECK_NE(instr.opcode(), HloOpcode::kFusion) << "`instr` has to be unfused.";
  if (instr.opcode() == HloOpcode::kReduce &&
      !IsReductionFromOrToContiguousDimensions(instr)) {
    return true;
  }
  // Avoid fusing reduce-window when stride is less than window size to minimize
  // the number of reads of the same elements.
  if (instr.opcode() == HloOpcode::kReduceWindow) {
    for (const auto& dim : instr.window().dimensions()) {
      if (dim.size() > dim.stride()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "LayoutsAreReduceInputFusionFriendly");

  if (producer.opcode() == HloOpcode::kCopy) {
    return false;
  }
  if (producer.opcode() == HloOpcode::kFusion) {
    for (const HloInstruction* instr : producer.fused_instructions()) {
      if (instr->opcode() == HloOpcode::kCopy) {
        // Elementwise copies are only inserted in input fusion for
        // transposition, and those are never friendly to the reduction.
        return false;
      }
    }
  }

  // A fusion iterates over its output in physically-contiguous order. This
  // applies "upwards" to operands.  Only an operator that changes an operand's
  // physical layout can create a "bad" memory access pattern, and layout
  // assignment guarantees kCopy is the only such operator.
  return true;
}

bool IsReduceInputFusion(const HloInstruction& instr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsReduceInputFusion");

  if (instr.IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr.fused_expression_root()->operands()) {
      if (IsReductionFromOrToContiguousDimensions(*operand)) {
        CHECK(instr.IsInputFusion())
            << " Multi-output fusion rooted at reduction-to-vector ops must be "
               "of kind kInput: "
            << instr.ToString();
        return true;
      }
    }
  } else if (instr.opcode() == HloOpcode::kFusion &&
             IsReductionFromOrToContiguousDimensions(
                 *instr.fused_expression_root())) {
    CHECK(instr.IsInputFusion())
        << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
        << instr.ToString();
    return true;
  }
  return false;
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsInputFusibleReduction");

  return IsReduceInputFusion(instr) ||
         IsReductionFromOrToContiguousDimensions(instr);
}

const HloInstruction* GetRealHeroForMultiOutputFusion(
    const HloInstruction& instr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "GetRealHeroForMultiOutputFusion");

  if (instr.opcode() != HloOpcode::kFusion) {
    return &instr;
  }
  auto fused_expression_root = instr.fused_expression_root();
  if (!instr.IsMultiOutputFusion()) {
    return fused_expression_root;
  }
  // If possible, we want to pick a reduction-from-or-to-contiguous-dims
  // operand of the fusion root, because it has the most constraints.
  for (const auto* inst : fused_expression_root->operands()) {
    if (IsReductionFromOrToContiguousDimensions(*inst)) {
      return inst;
    }
  }
  return fused_expression_root->operands()[0];
}

bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_5(mht_5_v, 315, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "ShapesCompatibleForMultiOutputFusion");

  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimensions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_6(mht_6_v, 321, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "lambda");

    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    if (IsReductionFromOrToContiguousDimensions(*element_instr)) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  auto* instr_1 = GetRealHeroForMultiOutputFusion(instr1);
  auto* instr_2 = GetRealHeroForMultiOutputFusion(instr2);
  if (IsReductionFromOrToContiguousDimensions(*instr_1) &&
      IsReductionFromOrToContiguousDimensions(*instr_2) &&
      !AreFusedReductionOutputsConsistent({instr_1, instr_2}, instr_1)) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  return ShapeUtil::EqualIgnoringElementType(get_loop_shape(instr_1),
                                             get_loop_shape(instr_2));
}

bool IsInputFusibleScatter(const HloInstruction& instr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_7(mht_7_v, 349, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsInputFusibleScatter");

  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}

bool IsInputFusible(const HloInstruction& instr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_8(mht_8_v, 362, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsInputFusible");

  // Input fusion only handles non-elemental reduction and scatter operations.
  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) || IsInputFusibleScatter(instr));
}

bool IsLoopFusible(const HloInstruction& instr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_9(mht_9_v, 371, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsLoopFusible");

  // Don't fuse get-tuple-element on GPU: We can, but it's slower than not
  // fusing.  We never generate kernels for unfused GTEs.  Instead, if an
  // unfused GTE is an input to a kernel (including a fusion kernel), we
  // compute the address of the GTE at the top of the kernel.  Often we know the
  // address of the GTE result statically, so we can do this without chasing any
  // pointers.
  return instr.IsFusible() &&
         ((instr.IsElementwise() && instr.operand_count() > 0) ||
          instr.opcode() == HloOpcode::kBitcast ||
          instr.opcode() == HloOpcode::kBroadcast ||
          instr.opcode() == HloOpcode::kConcatenate ||
          instr.opcode() == HloOpcode::kDynamicSlice ||
          instr.opcode() == HloOpcode::kDynamicUpdateSlice ||
          (instr.opcode() == HloOpcode::kFusion &&
           instr.fusion_kind() == HloInstruction::FusionKind::kLoop) ||
          instr.opcode() == HloOpcode::kGather ||
          instr.opcode() == HloOpcode::kIota ||
          instr.opcode() == HloOpcode::kPad ||
          (instr.opcode() == HloOpcode::kReduce &&
           !IsReductionFromOrToContiguousDimensions(instr) &&
           !instr.shape().IsTuple()) ||  // TODO(b/129089333): Don't fuse
                                         // variadic reductions.
          instr.opcode() == HloOpcode::kReduceWindow ||
          instr.opcode() == HloOpcode::kReshape ||
          instr.opcode() == HloOpcode::kReverse ||
          instr.opcode() == HloOpcode::kSlice ||
          instr.opcode() == HloOpcode::kConstant ||
          instr.opcode() == HloOpcode::kTranspose);
}

FusionDecision IsProducerConsumerFusible(const HloInstruction& producer,
                                         const HloInstruction& consumer) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_10(mht_10_v, 406, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsProducerConsumerFusible");

  if (!IsLoopFusible(producer)) {
    return "the producer is not loop-fusible";
  }

  if (!IsInputFusible(consumer) && !IsLoopFusible(consumer)) {
    return "the consumer is not input-fusible and not loop-fusible";
  }

  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return "the producer is not fusible as it is a multi-output fusion";
  }

  if (CreatesNestedLoop(producer, consumer)) {
    return "the fusion would create a nested loop";
  }

  // Do not fuse into reduce input fusions if the resulting kernel would suffer
  // from poor data locality (due to unfriendly input layouts).
  if (IsInputFusibleReduction(consumer) &&
      !LayoutsAreReduceInputFusionFriendly(producer, consumer)) {
    return "the producer layout is not fusion-friendly for the consumer "
           "reduction";
  }

  // Fuse scalar constants into loop fusion nodes. This reduces the number of
  // parameters and makes matching scalar broadcasts easier.
  //
  // Don't fuse other constants: Unfused constants in GPU land can be
  // represented as an external constant (i.e. not emitted in LLVM IR / PTX),
  // but fused constants are handled by shrared CPU/GPU code and always emitted
  // in the IR/PTX.  The external constant representation makes for faster
  // compiles and significantly smaller assembly code.
  if (producer.opcode() == HloOpcode::kConstant &&
      (!ShapeUtil::IsEffectiveScalar(producer.shape()) ||
       consumer.opcode() != HloOpcode::kFusion)) {
    return "not fusing constant";
  }

  return {};
}

bool IsProducerConsumerMultiOutputFusible(const HloInstruction& producer,
                                          const HloInstruction& consumer) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_11(mht_11_v, 453, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsProducerConsumerMultiOutputFusible");

  // Skip multiple output fusion. It's not yet supported.
  if (producer.IsMultiOutputFusion()) {
    return false;
  }

  // Allowing multi-output fusions that contain in-place operations makes code
  // generation more difficult. For the generated loop to iterate over all
  // outputs in parallel, it must find an iteration order that guarantees that
  // no loop iteration writes an element of any in-place operand that is read
  // or written by any other iteration. For example:
  //
  //   %fused_computation {
  //     %param_0 = s32[4,4]{1,0} parameter(0)
  //     ...
  //     %updated = s32[4,4]{1,0} dynamic-update-slice(
  //         %param_0, %add, %constant_1, %constant_0)
  //     %transpose = s32[4,4]{0,1} transpose(%updated), dimensions={1,0}
  //     ROOT %tuple.5 = tuple(%transpose, %updated)
  //   }
  //
  // Iterating 'transpose' and 'updated' in parallel by array index is
  // not valid, because an iteration that produces some element of 'transpose'
  // will read from an element of 'param_0' that has been overwritten by some
  // other iteration (writing to 'updated').
  //
  // To avoid these problems, we simply ban fusion altogether when the producer
  // is in-place. (We can relax this restriction by establishing an explicit
  // contract that describes what multi-output fusion scenarios are supported by
  // codegen and then changing this check to allow exactly those fusions).
  if (HloDataflowAnalysis::HasInPlaceOperations(producer)) {
    return false;
  }
  if (!IsLoopFusible(producer) || !IsFusibleAsMultiOutputFusionRoot(consumer)) {
    return false;
  }
  if (CreatesNestedLoop(producer, consumer)) {
    return false;
  }
  if (!ShapesCompatibleForMultiOutputFusion(producer, consumer)) {
    return false;
  }
  if (!LayoutsAreReduceInputFusionFriendly(producer, consumer)) {
    return false;
  }
  return true;
}

// Returns shared memory usage for a given instruction in bytes.
static int64_t SharedMemoryUsageNoCache(const HloInstruction& instr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_12(mht_12_v, 505, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "SharedMemoryUsageNoCache");

  // For now we are only fusing reductions.
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(
        instr.operand(0)->shape().element_type());
    int num_variadic =
        instr.shape().IsTuple() ? instr.shape().tuple_shapes_size() : 1;
    if (reduction_info.is_row_reduction) {
      // __shared__[32] is used for row reduction.
      return 32 * primitive_size * num_variadic;
    } else {
      // __shared__[2][32][33] cache is used for column reduction ("2" comes
      // from potential x-tiling).
      return 2 * 32 * 33 * primitive_size * num_variadic;
    }
  } else if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo);
    }
    return sum;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

static int64_t SharedMemoryUsage(const HloInstruction& instr,
                                 FusionInfoCache* cache = nullptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_13(mht_13_v, 539, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "SharedMemoryUsage");

  if (!cache) {
    return SharedMemoryUsageNoCache(instr);
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // SharedMemoryUsageNoCache and use the cache *within* the fusion.
  auto it_and_inserted = cache->shared_memory_usage.emplace(&instr, -1);
  auto it = it_and_inserted.first;
  auto inserted = it_and_inserted.second;

  if (inserted) {
    it->second = SharedMemoryUsageNoCache(instr);
  }
  return it->second;
}

// Codegen'ing unnested reductions requires a lot of registers, so a MOF
// combining many of those runs a high risk of spilling.
constexpr int64_t kMaxUnnestedReductionOutputsPerFusion = 8;

// Returns the number of unnested reductions in the instruction output.
static int64_t NumUnnestedReductionsNoCache(const HloInstruction& instr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_14(mht_14_v, 566, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "NumUnnestedReductionsNoCache");

  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr)) {
    return 1;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += NumUnnestedReductionsNoCache(*hlo);
    }
    return sum;
  }
  return 0;
}

static int64_t NumUnnestedReductions(const HloInstruction& instr,
                                     FusionInfoCache* cache) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_15(mht_15_v, 586, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "NumUnnestedReductions");

  if (!cache) {
    return NumUnnestedReductionsNoCache(instr);
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // NumUnnestedReductionsNoCache and use the cache *within* the fusion.
  auto it_and_inserted = cache->num_unnested_reductions.emplace(&instr, -1);
  auto it = it_and_inserted.first;
  auto inserted = it_and_inserted.second;

  if (inserted) {
    it->second = NumUnnestedReductionsNoCache(instr);
  }
  return it->second;
}

// This function limits the maximum number of operands to a fusion, and the
// amount of shared memory which can be consumed by the fusion.
//
// There's a cap on how many parameters we can pass to a CUDA kernel, but
// exactly what that limit is hazy, as it depends on (among other things) how
// much GPU constant memory is in use for other purposes.
//
// Moreover, we don't even know at the point that we're running fusion how many
// arguments the CUDA kernel for a fusion node will have: It depends on buffer
// assignment, where we will decide which of the fusion's operands live in XLA's
// big temp buffer versus in other allocations.
//
// As a heuristic, we simply cap the number of fusion operands plus outputs at
// MaxOperandsAndOutputsPerFusion().  This puts an upper bound on the number of
// parameters to the kernel, working around the correctness problem.
//
// This limit is also often good for performance.  In a fusion with many
// operands, each GPU thread likely has to do a lot of work, and so possibly
// uses a lot of registers, thus limiting occupancy.
//
// If the fusion is a producer/consumer fusion and instr1 is the
// consumer and instr2 is the producer, set is_consumer_producer_fusion
// to true to enable more fusion.
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  bool is_consumer_producer_fusion,
                                  FusionInfoCache* cache /*=nullptr*/) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_16(mht_16_v, 634, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "FusionFitsInBudget");

  if (SharedMemoryUsage(instr1, cache) + SharedMemoryUsage(instr2, cache) >
      kSharedMemoryBudgetInBytes) {
    return FusionDecision{}
           << "shared memory usage would be over the budget of "
           << kSharedMemoryBudgetInBytes << "B";
  }

  if (NumUnnestedReductions(instr1, cache) +
          NumUnnestedReductions(instr2, cache) >
      kMaxUnnestedReductionOutputsPerFusion) {
    return FusionDecision{} << "over " << kMaxUnnestedReductionOutputsPerFusion
                            << " unnested reductions in fusion";
  }

  // Compute the number of outputs of the (possibly multi-output) fusion node
  // we're considering creating.
  //
  // This isn't precise; we may be off by one if
  //  - We're creating a multi-output fusion out of two non-MOFs.  Creating a
  //    MOF adds a new buffer, namely, the tuple buffer.
  //  - We're merging two MOFs.  In this case, we should count the tuple buffer
  //    only once.
  //  - WLOG there's an edge from `a` to `b` and `b` is the only consumer of
  //    `a`.  In this case the result of `a` is not part of the output of the
  //    fusion.
  //
  // But because this is a heuristic and our limit
  // MaxOperandsAndOutputsPerFusion() is a large value (so +/- 1 doesn't make a
  // big difference), we ignore this small inaccuracy in favor of simplicity.
  int64_t num_output_buffers = ShapeUtil::SubshapeCount(instr1.shape()) +
                               ShapeUtil::SubshapeCount(instr2.shape());

  // The new fusion will have no more operands and outputs than
  //   producer_operands + consumer_operands - 1 + num_output_buffers
  // (minus one because we may be fusing a producer->consumer edge between `a`
  // and `b`).
  //
  // This fact may be enough to let us avoid having to compute the true total
  // number of operands, which can be expensive.
  if (instr1.operand_count() + instr2.operand_count() - 1 +
          num_output_buffers <=
      MaxOperandsAndOutputsPerFusion()) {
    return {};
  } else {
    VLOG(5) << "Operand count of "
            << "(" << instr1.ToString() << " ) = " << instr1.operand_count()
            << " and ( " << instr2.ToString()
            << " ) = " << instr2.operand_count()
            << " and num_output_buffers = " << num_output_buffers
            << " is bigger than the bound of "
            << MaxOperandsAndOutputsPerFusion();
  }

  // Compute the precise number of operands to the new fusion.
  absl::flat_hash_set<const HloInstruction*> operands(instr1.operands().begin(),
                                                      instr1.operands().end());
  operands.insert(instr2.operands().begin(), instr2.operands().end());
  // If there's an edge between `a` and `b`, don't count it: We're fusing that
  // producer -> consumer relationship.
  operands.erase(&instr1);
  operands.erase(&instr2);

  // If we generate the same numbers of inputs and outputs as
  // before, it won't be bigger after fusion. So accept the fusion.
  // As this is a consumer_producer fusion, this does not change the
  // consumer numbers of output. So no need to check it.
  if (is_consumer_producer_fusion &&
      operands.size() <= instr1.operands().size()) {
    return {};
  }

  // Does the new fusion have more operands and outputs than the max?
  if (operands.size() + num_output_buffers > MaxOperandsAndOutputsPerFusion()) {
    return "Number of operands and output buffers is larger than allowed "
           "budget per fusion";
  }
  return {};
}

bool CreatesNestedLoop(const HloInstruction& producer,
                       const HloInstruction& consumer) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_17(mht_17_v, 718, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "CreatesNestedLoop");

  // If producer does not have an instruction that codegens a loop then there is
  // nothing to do.
  auto producer_has_loop_codegen = [&](const HloInstruction& instr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_18(mht_18_v, 724, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "lambda");

    if (producer.opcode() != HloOpcode::kFusion) {
      return IfFusedReadsElementsMultipleTimes(producer);
    }
    for (const auto& instr : producer.fused_instructions()) {
      if (IfFusedReadsElementsMultipleTimes(*instr)) {
        return true;
      }
    }
    return false;
  };
  if (!producer_has_loop_codegen(producer)) {
    return false;
  }

  // If consumer is a non-fusion instruction then we have to check if it
  // generates a loop.
  if (consumer.opcode() != HloOpcode::kFusion) {
    return IfFusedReadsElementsMultipleTimes(consumer);
  }

  // If consumer is a fusion then we have to check if the output of producer is
  // used directly or indirectly as an input to an HLO instruction that
  // generates a loop, i.e. there is a path in the graph from an operand
  // corresponding to the producer to an HLO instruction generating a loop in
  // the consumer.
  for (const HloInstruction* operand : consumer.operands()) {
    if (operand != &producer) {
      continue;
    }

    const HloInstruction* root =
        consumer.fused_instructions_computation()->parameter_instruction(
            consumer.operand_index(operand));

    std::stack<const HloInstruction*> dfs;
    dfs.push(root);
    absl::flat_hash_set<const HloInstruction*> visited;
    while (!dfs.empty()) {
      const HloInstruction* cur = dfs.top();
      dfs.pop();

      if (visited.contains(cur)) {
        continue;
      }
      visited.insert(cur);

      if (IfFusedReadsElementsMultipleTimes(*cur)) {
        return true;
      }
      for (const auto& user : cur->users()) {
        if (visited.contains(user)) {
          continue;
        }
        dfs.push(user);
      }
    }
  }
  return false;
}

bool IsFusibleAsMultiOutputFusionRoot(const HloInstruction& instr) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_19(mht_19_v, 788, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsFusibleAsMultiOutputFusionRoot");

  // We can fuse reduces and loop fusions. Elementwise instructions can be fused
  // with any other instruction.
  // Note that scatter cannot be the root of a multi-output fusion because
  // its emitter doesn't support it.

  return instr.IsFusible() &&
         (IsInputFusibleReduction(instr) ||
          instr.IsLoopFusion() ||  // TODO(b/130013493): Use IsLoopFusible here.
          instr.IsElementwise());
}

HloInstruction::FusionKind ChooseFusionKind(const HloInstruction& /*producer*/,
                                            const HloInstruction& consumer) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_20(mht_20_v, 804, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "ChooseFusionKind");

  return IsInputFusible(consumer) ? HloInstruction::FusionKind::kInput
                                  : HloInstruction::FusionKind::kLoop;
}

bool IsConsumerTheOnlyNonRootUser(const HloInstruction& instr,
                                  const HloInstruction& consumer) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_21(mht_21_v, 813, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "IsConsumerTheOnlyNonRootUser");

  return absl::c_all_of(instr.users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      // Skip GTE.
      return IsConsumerTheOnlyNonRootUser(*user, consumer);
    }
    if (user == &consumer) {
      // `user` is `consumer`.
      return true;
    }
    if (user == user->parent()->root_instruction()) {
      // Consumed by ROOT.
      return true;
    }
    return false;
  });
}

size_t GetInstrCountOfFusible(const HloInstruction& instr) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_22(mht_22_v, 834, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "GetInstrCountOfFusible");

  if (instr.opcode() != HloOpcode::kFusion) {
    return 1;
  } else {
    return instr.fused_instruction_count();
  }
}

absl::InlinedVector<const HloInstruction*, 2> GetOutputsOfFusible(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return {&instr};
  }

  HloInstruction* root = instr.fused_expression_root();
  if (root->opcode() != HloOpcode::kTuple) {
    return {root};
  } else {
    auto v = root->operands();
    return absl::InlinedVector<const HloInstruction*, 2>(v.begin(), v.end());
  }
}

size_t GetOutputSizeOfFusible(const HloInstruction& instr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_fusibleDTcc mht_23(mht_23_v, 860, "", "./tensorflow/compiler/xla/service/gpu/gpu_fusible.cc", "GetOutputSizeOfFusible");

  if (!instr.IsMultiOutputFusion()) {
    return 1;
  }
  const HloInstruction* root = instr.fused_expression_root();
  return ShapeUtil::TupleElementCount(root->shape());
}

}  // namespace gpu
}  // namespace xla
