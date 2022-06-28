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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_opcodeDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_opcodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_opcodeDTh() {
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


#include <iosfwd>
#include <string>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// High-level optimizer instruction opcodes -- these are linear-algebra level
// opcodes. They are a flattened form of the UnaryOp, BinaryOp, ... opcodes
// present in the XLA service protobuf.
//
// See the XLA documentation for the semantics of each opcode.
//
// Each entry has the format:
// (enum_name, opcode_name, arity)
//
// Note: Do not use ':' in opcode names. It is used as a special character
// in these places:
// - In extended opcode strings (HloInstruction::ExtendedOpcodeString()), to
//   separate the opcode from the fusion kind
// - In fully qualified names (HloInstruction::FullyQualifiedName()), to
//   separate the qualifiers (name of the computation and potentially the
//   fusion instruction) from the name
#define HLO_OPCODE_LIST(V)                                                     \
  V(kAbs, "abs", 1)                                                            \
  V(kAdd, "add", 2)                                                            \
  V(kAddDependency, "add-dependency", 2)                                       \
  V(kAfterAll, "after-all", kHloOpcodeIsVariadic)                              \
  V(kAllGather, "all-gather", kHloOpcodeIsVariadic)                            \
  V(kAllGatherStart, "all-gather-start", kHloOpcodeIsVariadic)                 \
  V(kAllGatherDone, "all-gather-done", 1)                                      \
  V(kAllReduce, "all-reduce", kHloOpcodeIsVariadic)                            \
  V(kAllReduceStart, "all-reduce-start", kHloOpcodeIsVariadic)                 \
  V(kAllReduceDone, "all-reduce-done", 1)                                      \
  V(kAllToAll, "all-to-all", kHloOpcodeIsVariadic)                             \
  V(kAsyncStart, "async-start", kHloOpcodeIsVariadic)                          \
  V(kAsyncUpdate, "async-update", 1)                                           \
  V(kAsyncDone, "async-done", 1)                                               \
  V(kAtan2, "atan2", 2)                                                        \
  V(kBatchNormGrad, "batch-norm-grad", 5)                                      \
  V(kBatchNormInference, "batch-norm-inference", 5)                            \
  V(kBatchNormTraining, "batch-norm-training", 3)                              \
  V(kBitcast, "bitcast", 1)                                                    \
  V(kBitcastConvert, "bitcast-convert", 1)                                     \
  V(kBroadcast, "broadcast", 1)                                                \
  V(kCall, "call", kHloOpcodeIsVariadic)                                       \
  V(kCeil, "ceil", 1)                                                          \
  V(kCholesky, "cholesky", 1)                                                  \
  V(kClamp, "clamp", 3)                                                        \
  V(kCollectivePermute, "collective-permute", kHloOpcodeIsVariadic)            \
  V(kCollectivePermuteStart, "collective-permute-start", kHloOpcodeIsVariadic) \
  V(kCollectivePermuteDone, "collective-permute-done", 1)                      \
  V(kClz, "count-leading-zeros", 1)                                            \
  V(kCompare, "compare", 2)                                                    \
  V(kComplex, "complex", 2)                                                    \
  V(kConcatenate, "concatenate", kHloOpcodeIsVariadic)                         \
  V(kConditional, "conditional", kHloOpcodeIsVariadic)                         \
  V(kConstant, "constant", 0)                                                  \
  V(kConvert, "convert", 1)                                                    \
  V(kConvolution, "convolution", 2)                                            \
  V(kCopy, "copy", 1)                                                          \
  V(kCopyDone, "copy-done", 1)                                                 \
  V(kCopyStart, "copy-start", 1)                                               \
  V(kCos, "cosine", 1)                                                         \
  V(kCustomCall, "custom-call", kHloOpcodeIsVariadic)                          \
  V(kDivide, "divide", 2)                                                      \
  V(kDomain, "domain", 1)                                                      \
  V(kDot, "dot", 2)                                                            \
  V(kDynamicSlice, "dynamic-slice", kHloOpcodeIsVariadic)                      \
  V(kDynamicUpdateSlice, "dynamic-update-slice", kHloOpcodeIsVariadic)         \
  V(kExp, "exponential", 1)                                                    \
  V(kExpm1, "exponential-minus-one", 1)                                        \
  V(kFft, "fft", 1)                                                            \
  V(kFloor, "floor", 1)                                                        \
  V(kFusion, "fusion", kHloOpcodeIsVariadic)                                   \
  V(kGather, "gather", 2)                                                      \
  V(kGetDimensionSize, "get-dimension-size", 1)                                \
  V(kSetDimensionSize, "set-dimension-size", 2)                                \
  V(kGetTupleElement, "get-tuple-element", 1)                                  \
  V(kImag, "imag", 1)                                                          \
  V(kInfeed, "infeed", 1)                                                      \
  V(kIota, "iota", 0)                                                          \
  V(kIsFinite, "is-finite", 1)                                                 \
  V(kLog, "log", 1)                                                            \
  V(kLog1p, "log-plus-one", 1)                                                 \
  V(kLogistic, "logistic", 1)                                                  \
  V(kAnd, "and", 2)                                                            \
  V(kNot, "not", 1)                                                            \
  V(kOptimizationBarrier, "opt-barrier", 1)                                    \
  V(kOr, "or", 2)                                                              \
  V(kXor, "xor", 2)                                                            \
  V(kMap, "map", kHloOpcodeIsVariadic)                                         \
  V(kMaximum, "maximum", 2)                                                    \
  V(kMinimum, "minimum", 2)                                                    \
  V(kMultiply, "multiply", 2)                                                  \
  V(kNegate, "negate", 1)                                                      \
  V(kOutfeed, "outfeed", 2)                                                    \
  V(kPad, "pad", 2)                                                            \
  V(kParameter, "parameter", 0)                                                \
  V(kPartitionId, "partition-id", 0)                                           \
  V(kPopulationCount, "popcnt", 1)                                             \
  V(kPower, "power", 2)                                                        \
  V(kReal, "real", 1)                                                          \
  V(kRecv, "recv", 1)                                                          \
  V(kRecvDone, "recv-done", 1)                                                 \
  V(kReduce, "reduce", kHloOpcodeIsVariadic)                                   \
  V(kReducePrecision, "reduce-precision", 1)                                   \
  V(kReduceScatter, "reduce-scatter", kHloOpcodeIsVariadic)                    \
  V(kReduceWindow, "reduce-window", kHloOpcodeIsVariadic)                      \
  V(kRemainder, "remainder", 2)                                                \
  V(kReplicaId, "replica-id", 0)                                               \
  V(kReshape, "reshape", 1)                                                    \
  V(kDynamicReshape, "dynamic-reshape", kHloOpcodeIsVariadic)                  \
  V(kReverse, "reverse", 1)                                                    \
  V(kRng, "rng", kHloOpcodeIsVariadic)                                         \
  V(kRngGetAndUpdateState, "rng-get-and-update-state", 0)                      \
  V(kRngBitGenerator, "rng-bit-generator", 1)                                  \
  V(kRoundNearestAfz, "round-nearest-afz", 1)                                  \
  V(kRsqrt, "rsqrt", 1)                                                        \
  V(kScatter, "scatter", 3)                                                    \
  V(kSelect, "select", 3)                                                      \
  V(kSelectAndScatter, "select-and-scatter", 3)                                \
  V(kSend, "send", 2)                                                          \
  V(kSendDone, "send-done", 1)                                                 \
  V(kShiftLeft, "shift-left", 2)                                               \
  V(kShiftRightArithmetic, "shift-right-arithmetic", 2)                        \
  V(kShiftRightLogical, "shift-right-logical", 2)                              \
  V(kSign, "sign", 1)                                                          \
  V(kSin, "sine", 1)                                                           \
  V(kSlice, "slice", 1)                                                        \
  V(kSort, "sort", kHloOpcodeIsVariadic)                                       \
  V(kSqrt, "sqrt", 1)                                                          \
  V(kCbrt, "cbrt", 1)                                                          \
  V(kSubtract, "subtract", 2)                                                  \
  V(kTanh, "tanh", 1)                                                          \
  V(kTrace, "trace", 1)                                                        \
  V(kTranspose, "transpose", 1)                                                \
  V(kTriangularSolve, "triangular-solve", 2)                                   \
  V(kTuple, "tuple", kHloOpcodeIsVariadic)                                     \
  V(kTupleSelect, "tuple-select", 3)                                           \
  V(kWhile, "while", 1)

enum class HloOpcode {
#define DECLARE_ENUM(enum_name, opcode_name, ...) enum_name,
  HLO_OPCODE_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

// Arity value that denotes that an operator is variadic.
enum {
  kHloOpcodeIsVariadic = -1,
};

// Returns a string representation of the opcode.
std::string HloOpcodeString(HloOpcode opcode);

// Retrieves the opcode enum by name if the opcode exists.
StatusOr<HloOpcode> StringToHloOpcode(const std::string& opcode_name);

inline std::ostream& operator<<(std::ostream& os, HloOpcode opcode) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_opcodeDTh mht_0(mht_0_v, 351, "", "./tensorflow/compiler/xla/service/hlo_opcode.h", "operator<<");

  return os << HloOpcodeString(opcode);
}

// Returns true iff the given opcode is a comparison operation.
bool HloOpcodeIsComparison(HloOpcode opcode);

// Returns true iff the given opcode has variadic operands.
bool HloOpcodeIsVariadic(HloOpcode opcode);

// Returns the arity of opcode. If the opcode is variadic,
// returns nullopt.
absl::optional<int> HloOpcodeArity(HloOpcode opcode);

// Returns true if the given opcode is one of kAsyncStart, kAsyncUpdate, or
// kAsyncDone.
bool HloOpcodeIsAsync(HloOpcode opcode);

// Returns the number of HloOpcode values.
inline const uint32_t HloOpcodeCount() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_opcodeDTh mht_1(mht_1_v, 373, "", "./tensorflow/compiler/xla/service/hlo_opcode.h", "HloOpcodeCount");

#define HLO_COUNT_ONE(...) +1
#define HLO_XLIST_LENGTH(list) list(HLO_COUNT_ONE)
  return HLO_XLIST_LENGTH(HLO_OPCODE_LIST);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_OPCODE_H_
