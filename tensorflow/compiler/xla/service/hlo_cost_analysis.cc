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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"

#include <cmath>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace xla {

constexpr const char HloCostAnalysis::kFlopsKey[];
constexpr const char HloCostAnalysis::kTranscendentalsKey[];
constexpr const char HloCostAnalysis::kBytesAccessedKey[];
constexpr const char HloCostAnalysis::kOptimalSecondsKey[];

HloCostAnalysis::HloCostAnalysis(const Options& options) : options_(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HloCostAnalysis");
}
HloCostAnalysis::HloCostAnalysis(ShapeSizeFunction shape_size,
                                 const Properties& per_second_rates)
    : HloCostAnalysis(Options{shape_size, per_second_rates}) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HloCostAnalysis");
}

Status HloCostAnalysis::Preprocess(const HloInstruction* hlo) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::Preprocess");

  // Set current instruction cost values to reasonable default values. Each
  // handler can overwrite these values. In Postprocess, these values are
  // accumulated and written to the per-instruction maps.
  current_properties_.clear();
  current_should_compute_bottleneck_time_ = true;

  // The default number of bytes accessed for an instruction is the sum of the
  // sizes of the inputs and outputs. The default ShapeUtil::ByteSizeOf does not
  // handle opaque types.
  float bytes_accessed = GetShapeSize(hlo->shape());
  SetOutputBytesAccessed(GetShapeSize(hlo->shape()));
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    const HloInstruction* operand = hlo->operand(i);
    bytes_accessed += GetShapeSize(operand->shape());
    SetOperandBytesAccessed(i, GetShapeSize(operand->shape()));
  }
  current_properties_[kBytesAccessedKey] = bytes_accessed;

  return Status::OK();
}

Status HloCostAnalysis::Postprocess(const HloInstruction* hlo) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_3(mht_3_v, 244, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::Postprocess");

  if (current_should_compute_bottleneck_time_) {
    // Compute the time as the time of the bottleneck, i.e. the slowest property
    // given the per-second rate of each property.
    float optimal_seconds = 0.0f;
    for (const auto& property : current_properties_) {
      if (property.first != kOptimalSecondsKey) {
        optimal_seconds = std::max(
            optimal_seconds,
            property.second / GetProperty(property.first,
                                          options_.per_second_rates, INFINITY));
      }
    }
    current_properties_[kOptimalSecondsKey] = optimal_seconds;
  }

  TF_RET_CHECK(hlo_properties_.emplace(hlo, current_properties_).second);
  for (const auto& property : current_properties_) {
    properties_sum_[property.first] += property.second;
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleElementwiseOp(
    const HloInstruction* hlo_instruction) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_4(mht_4_v, 272, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleElementwiseOp");

  const auto& shape = hlo_instruction->shape();
  // For element-wise operations, the number of computations is the same as the
  // number of elements in the output shape.
  auto computation_count = ShapeUtil::ElementsIn(shape);
  auto opcode = hlo_instruction->opcode();
  // We treat transcendental operations separately since one transcendental
  // operation can correspond to several floating point ops.
  // kLogistic is included in "trascendental" as it is implemented using
  // trascendental ops (tanh or exp).
  if (opcode == HloOpcode::kExp || opcode == HloOpcode::kLog ||
      opcode == HloOpcode::kLogistic || opcode == HloOpcode::kPower ||
      opcode == HloOpcode::kSqrt || opcode == HloOpcode::kCbrt ||
      opcode == HloOpcode::kRsqrt || opcode == HloOpcode::kTanh ||
      opcode == HloOpcode::kSin || opcode == HloOpcode::kCos ||
      opcode == HloOpcode::kExpm1 || opcode == HloOpcode::kLog1p ||
      opcode == HloOpcode::kAtan2) {
    current_properties_[kTranscendentalsKey] = computation_count;
  } else {
    // Note: transcendental operations are considered a separate category from
    // FLOPs.
    current_properties_[kFlopsKey] = computation_count;
  }
  return Status::OK();
}

/*static*/ float HloCostAnalysis::GetProperty(absl::string_view key,
                                              const Properties& properties,
                                              const float default_value) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key: \"" + std::string(key.data(), key.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_5(mht_5_v, 304, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetProperty");

  auto key_value = properties.find(key);
  return key_value == properties.end() ? default_value : key_value->second;
}

/*static*/ float HloCostAnalysis::GetPropertyForHlo(
    const HloInstruction& hlo, const std::string& key,
    const HloToProperties& hlo_to_properties) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_6(mht_6_v, 315, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetPropertyForHlo");

  auto it = hlo_to_properties.find(&hlo);
  if (it == hlo_to_properties.end()) {
    return 0.0f;
  } else {
    return GetProperty(key, it->second);
  }
}

int64_t HloCostAnalysis::GetShapeSize(const Shape& shape) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_7(mht_7_v, 327, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetShapeSize");

  if (!LayoutUtil::HasLayout(shape)) {
    return 0;
  }
  return options_.shape_size(shape);
}

int64_t HloCostAnalysis::FusionParameterReadBytes(
    const HloInstruction* hlo) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_8(mht_8_v, 338, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::FusionParameterReadBytes");

  int64_t size = 0;
  bool seen_trivial_user = false;
  CHECK(hlo->IsFused() && (hlo->opcode() == HloOpcode::kParameter ||
                           hlo->opcode() == HloOpcode::kGetTupleElement));
  for (const HloInstruction* user : hlo->users()) {
    switch (user->opcode()) {
      case HloOpcode::kFusion: {
        for (int64_t idx : user->OperandIndices(hlo)) {
          size += FusionParameterReadBytes(user->fused_parameter(idx));
        }
        break;
      }
      case HloOpcode::kSlice:
        size += GetShapeSize(user->shape());
        break;
      case HloOpcode::kDynamicSlice:
        size += hlo == user->operand(0) ? GetShapeSize(user->shape())
                                        : GetShapeSize(hlo->shape());
        break;
      case HloOpcode::kDynamicUpdateSlice:
        // Uses the same shape as 'update' which is operand 1.
        size += hlo == user->operand(0)
                    ? GetShapeSize(user->operand(1)->shape())
                    : GetShapeSize(hlo->shape());
        break;
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
        size += GetShapeSize(hlo->shape());
        break;
      default:
        // Other instructions reading this parameter are assumed to be able to
        // share the read from memory.
        if (!seen_trivial_user) {
          seen_trivial_user = true;
          size += GetShapeSize(hlo->shape());
        }
    }
  }
  return size;
}

Status HloCostAnalysis::HandleElementwiseUnary(const HloInstruction* hlo) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_9(mht_9_v, 383, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleElementwiseUnary");

  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleElementwiseBinary(const HloInstruction* hlo) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_10(mht_10_v, 390, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleElementwiseBinary");

  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleCompare(const HloInstruction* compare) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_11(mht_11_v, 397, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCompare");

  return HandleElementwiseOp(compare);
}

Status HloCostAnalysis::HandleClamp(const HloInstruction* clamp) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_12(mht_12_v, 404, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleClamp");

  return HandleElementwiseOp(clamp);
}

Status HloCostAnalysis::HandleReducePrecision(const HloInstruction* hlo) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_13(mht_13_v, 411, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReducePrecision");

  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleParameter(const HloInstruction*) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_14(mht_14_v, 418, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleParameter");

  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleConstant(const HloInstruction*) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_15(mht_15_v, 429, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleConstant");

  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleIota(const HloInstruction*) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_16(mht_16_v, 440, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleIota");

  return Status::OK();
}

Status HloCostAnalysis::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_17(mht_17_v, 448, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleGetTupleElement");

  // GetTupleElement forwards a pointer and does not touch each element in the
  // output.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  SetOperandBytesAccessed(0, 0);
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleSelect(const HloInstruction* hlo) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_18(mht_18_v, 462, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSelect");

  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleTupleSelect(const HloInstruction*) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_19(mht_19_v, 469, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleTupleSelect");

  return Status::OK();
}

Status HloCostAnalysis::HandleReverse(const HloInstruction*) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_20(mht_20_v, 476, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReverse");

  return Status::OK();
}

Status HloCostAnalysis::HandleSlice(const HloInstruction* slice) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_21(mht_21_v, 483, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSlice");

  current_properties_[kBytesAccessedKey] = GetShapeSize(slice->shape()) * 2;
  SetOutputBytesAccessed(GetShapeSize(slice->shape()));
  SetOperandBytesAccessed(0, GetShapeSize(slice->shape()));
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_22(mht_22_v, 494, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleDynamicSlice");

  current_properties_[kBytesAccessedKey] =
      GetShapeSize(dynamic_slice->shape()) * 2 +
      GetShapeSize(dynamic_slice->operand(1)->shape());
  SetOutputBytesAccessed(GetShapeSize(dynamic_slice->shape()));
  SetOperandBytesAccessed(0, GetShapeSize(dynamic_slice->shape()));
  SetOperandBytesAccessed(1, GetShapeSize(dynamic_slice->operand(1)->shape()));
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicUpdateSlice(
    const HloInstruction* dynamic_update_slice) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_23(mht_23_v, 508, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleDynamicUpdateSlice");

  current_properties_[kBytesAccessedKey] =
      GetShapeSize(dynamic_update_slice->operand(1)->shape()) * 2 +
      GetShapeSize(dynamic_update_slice->operand(2)->shape());
  // Operand 0 aliases with the output.
  SetOutputBytesAccessed(
      GetShapeSize(dynamic_update_slice->operand(1)->shape()));
  SetOperandBytesAccessed(0, 0);
  SetOperandBytesAccessed(
      1, GetShapeSize(dynamic_update_slice->operand(1)->shape()));
  SetOperandBytesAccessed(
      2, GetShapeSize(dynamic_update_slice->operand(2)->shape()));
  return Status::OK();
}

Status HloCostAnalysis::HandleTuple(const HloInstruction* tuple) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_24(mht_24_v, 526, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleTuple");

  // The tuple instruction only gathers pointers from inputs (it doesn't iterate
  // through them). The memory touched is then only the size of the output
  // index table of the tuple.

  current_properties_[kBytesAccessedKey] = GetShapeSize(tuple->shape());
  SetOutputBytesAccessed(GetShapeSize(tuple->shape()));
  for (int i = 0; i < tuple->operand_count(); ++i) {
    SetOperandBytesAccessed(i, 0);
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleConcatenate(const HloInstruction*) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_25(mht_25_v, 542, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleConcatenate");

  return Status::OK();
}

Status HloCostAnalysis::HandleConvert(const HloInstruction* convert) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_26(mht_26_v, 549, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleConvert");

  return HandleElementwiseOp(convert);
}

Status HloCostAnalysis::HandleCopy(const HloInstruction*) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_27(mht_27_v, 556, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCopy");

  return Status::OK();
}

Status HloCostAnalysis::HandleDomain(const HloInstruction* domain) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_28(mht_28_v, 563, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleDomain");

  // Domain does not have any computation or data transfer.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  for (int i = 0; i < domain->operand_count(); ++i) {
    SetOperandBytesAccessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

/* static */
int64_t HloCostAnalysis::GetDotFlops(const Shape& lhs_shape,
                                     const Shape& result_shape,
                                     const DotDimensionNumbers& dnums) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_29(mht_29_v, 581, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetDotFlops");

  // Count of elements along the reduction dimensions.
  int64_t reduction_width = 1;
  for (auto dim : dnums.lhs_contracting_dimensions()) {
    reduction_width *= lhs_shape.dimensions(dim);
  }
  // Each output element requires reduction_width FMA operations.
  return kFmaFlops * ShapeUtil::ElementsIn(result_shape) * reduction_width;
}

Status HloCostAnalysis::HandleDot(const HloInstruction* dot) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_30(mht_30_v, 594, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleDot");

  current_properties_[kFlopsKey] = GetDotFlops(
      dot->operand(0)->shape(), dot->shape(), dot->dot_dimension_numbers());
  return Status::OK();
}

Status HloCostAnalysis::HandleInfeed(const HloInstruction* infeed) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_31(mht_31_v, 603, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleInfeed");

  // Count nested infeed output tuples.
  int64_t size = 0;
  for (const auto& indexed_shape : ShapeUtil::GetLeafShapes(infeed->shape())) {
    size += GetShapeSize(indexed_shape.shape);
    SetOutputBytesAccessed(indexed_shape.index,
                           GetShapeSize(indexed_shape.shape));
  }
  SetOutputBytesAccessed(size);
  current_properties_[kBytesAccessedKey] = size;
  return Status::OK();
}

Status HloCostAnalysis::HandleOutfeed(const HloInstruction* outfeed) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_32(mht_32_v, 619, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleOutfeed");

  // Count nested outfeed operand tuples.
  current_properties_[kBytesAccessedKey] = 0;
  for (int64_t i = 0; i < outfeed->operand_count(); ++i) {
    const HloInstruction* operand = outfeed->operand(i);
    int64_t size = 0;
    for (const auto& indexed_shape :
         ShapeUtil::GetLeafShapes(operand->shape())) {
      size += GetShapeSize(indexed_shape.shape);
      SetOperandBytesAccessed(i, indexed_shape.index,
                              GetShapeSize(indexed_shape.shape));
    }
    SetOperandBytesAccessed(i, size);
    current_properties_[kBytesAccessedKey] += size;
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleMap(const HloInstruction* map) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_33(mht_33_v, 640, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleMap");

  // Compute properties of the mapped function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(map->to_apply()));

  // Compute the cost of all elements for this Map operation.
  const int64_t element_count = ShapeUtil::ElementsIn(map->shape());
  for (const auto& property : sub_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] = property.second * element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduce(const HloInstruction* reduce) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_34(mht_34_v, 658, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReduce");

  HloComputation* function = reduce->to_apply();
  // Compute the cost of the user function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));

  // Compute the cost of all elements for this Reduce operation.
  // This counts the number of times the reduction function is applied, so it
  // does not need to be multiplied by the number of input tensors - that's
  // already "priced in" by the sub-computation doing more work.
  auto arg = reduce->operand(0);
  auto output_shape = reduce->shape().IsArray()
                          ? reduce->shape()
                          : reduce->shape().tuple_shapes(0);
  int64_t reduction_count =
      ShapeUtil::ElementsIn(arg->shape()) - ShapeUtil::ElementsIn(output_shape);
  for (const auto& property : sub_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] = property.second * reduction_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleReduceWindow(
    const HloInstruction* reduce_window) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_35(mht_35_v, 686, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReduceWindow");

  const Window& window = reduce_window->window();
  auto function = reduce_window->to_apply();
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(function));

  // Compute the cost of all elements for this ReduceWindow operation. For each
  // output element there are window_size - 1 reductions to perform.
  int64_t window_element_count = 1;
  for (const auto& dimension : window.dimensions()) {
    window_element_count *= dimension.size();
  }

  const int64_t output_element_count =
      ShapeUtil::ElementsIn(reduce_window->shape().IsArray()
                                ? reduce_window->shape()
                                : reduce_window->shape().tuple_shapes(0));
  const int64_t reduction_count =
      (window_element_count - 1) * output_element_count;
  for (const auto& property : sub_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] = property.second * reduction_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleSelectAndScatter(
    const HloInstruction* instruction) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_36(mht_36_v, 718, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSelectAndScatter");

  // Compute the properties of the select and scatter function.
  // Compute the properties of the reduction function.
  TF_ASSIGN_OR_RETURN(const Properties select_properties,
                      ProcessSubcomputation(instruction->select()));
  TF_ASSIGN_OR_RETURN(const Properties scatter_properties,
                      ProcessSubcomputation(instruction->scatter()));

  // Compute the cost of all elements for this operation. For each scatter
  // source element there are window_size - 1 select computations to perform and
  // 1 scatter computation to perform.
  const auto source = instruction->operand(1);
  const auto source_element_count = ShapeUtil::ElementsIn(source->shape());
  int64_t window_element_count = 1;
  for (const auto& dimension : instruction->window().dimensions()) {
    window_element_count *= dimension.size();
  }
  const int64_t select_count =
      source_element_count * (window_element_count - 1);
  for (const auto& property : select_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] += property.second * select_count;
    }
  }
  for (const auto& property : scatter_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] +=
          property.second * source_element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleBitcast(const HloInstruction*) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_37(mht_37_v, 754, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleBitcast");

  // A bitcast does no computation and touches no memory.
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  SetOperandBytesAccessed(0, 0);
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleBroadcast(const HloInstruction*) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_38(mht_38_v, 766, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleBroadcast");

  return Status::OK();
}

Status HloCostAnalysis::HandlePad(const HloInstruction*) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_39(mht_39_v, 773, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandlePad");

  return Status::OK();
}

Status HloCostAnalysis::HandleAsyncStart(const HloInstruction* async_start) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_40(mht_40_v, 780, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAsyncStart");

  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessSubcomputation(async_start->called_computations()[0]));
  return Status::OK();
}

Status HloCostAnalysis::HandleAsyncUpdate(const HloInstruction*) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_41(mht_41_v, 790, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAsyncUpdate");

  return Status::OK();
}

Status HloCostAnalysis::HandleAsyncDone(const HloInstruction*) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_42(mht_42_v, 797, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAsyncDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleCopyStart(const HloInstruction*) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_43(mht_43_v, 804, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCopyStart");

  return Status::OK();
}

Status HloCostAnalysis::HandleCopyDone(const HloInstruction*) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_44(mht_44_v, 811, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCopyDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleSend(const HloInstruction*) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_45(mht_45_v, 818, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSend");

  return Status::OK();
}

Status HloCostAnalysis::HandleSendDone(const HloInstruction*) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_46(mht_46_v, 825, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSendDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleRecv(const HloInstruction*) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_47(mht_47_v, 832, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleRecv");

  return Status::OK();
}

Status HloCostAnalysis::HandleRecvDone(const HloInstruction*) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_48(mht_48_v, 839, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleRecvDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleReshape(const HloInstruction*) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_49(mht_49_v, 846, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReshape");

  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicReshape(const HloInstruction*) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_50(mht_50_v, 853, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleDynamicReshape");

  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormTraining(const HloInstruction*) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_51(mht_51_v, 860, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleBatchNormTraining");

  // TODO(b/62294698): Implement cost analysis for batch-norm-training.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormInference(const HloInstruction*) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_52(mht_52_v, 868, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleBatchNormInference");

  // TODO(b/62294698): Implement cost analysis for batch-norm-inference.
  return Status::OK();
}

Status HloCostAnalysis::HandleBatchNormGrad(const HloInstruction*) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_53(mht_53_v, 876, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleBatchNormGrad");

  // TODO(b/62294698): Implement cost analysis for batch-norm-grad.
  return Status::OK();
}

Status HloCostAnalysis::HandleTranspose(const HloInstruction* transpose) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_54(mht_54_v, 884, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleTranspose");

  if (transpose->IsEffectiveBitcast()) {
    return HandleBitcast(transpose);
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleAfterAll(const HloInstruction* token) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_55(mht_55_v, 894, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAfterAll");

  // This instruction is used to enforce ordering at compile time. No code is
  // emitted.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  for (int i = 0; i < token->operand_count(); ++i) {
    SetOperandBytesAccessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleAddDependency(
    const HloInstruction* add_dependency) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_56(mht_56_v, 911, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAddDependency");

  // This instruction is used to enforce ordering at compile time. No code is
  // emitted.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  SetOutputBytesAccessed(0);
  for (int i = 0; i < add_dependency->operand_count(); ++i) {
    SetOperandBytesAccessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return Status::OK();
}

int64_t HloCostAnalysis::GetConvolutionFlops(
    const HloInstruction* convolution) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_57(mht_57_v, 928, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetConvolutionFlops");

  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& result_shape = convolution->shape();

  return GetConvolutionFlops(convolution, lhs_shape, rhs_shape, result_shape);
}

/* static */
int64_t HloCostAnalysis::GetConvolutionFlops(const HloInstruction* convolution,
                                             const Shape& lhs_shape,
                                             const Shape& rhs_shape,
                                             const Shape& result_shape) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_58(mht_58_v, 945, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetConvolutionFlops");

  Window window = convolution->window();
  const auto& dnums = convolution->convolution_dimension_numbers();
  const int64_t input_batch_dim = dnums.input_batch_dimension();
  const int64_t input_feature_dim = dnums.input_feature_dimension();
  const int64_t output_feature_dim = dnums.output_feature_dimension();
  const int64_t input_feature =
      ShapeUtil::GetDimension(lhs_shape, input_feature_dim);
  const int64_t output_feature =
      ShapeUtil::GetDimension(result_shape, output_feature_dim);
  const int64_t batch = ShapeUtil::GetDimension(lhs_shape, input_batch_dim);

  DimensionVector kernel_limits;
  DimensionVector output_limits;
  DimensionVector input_limits;
  if (window.dimensions().empty()) {
    window = window_util::MakeWindow({1});
    kernel_limits.push_back(1);
    output_limits.push_back(1);
    input_limits.push_back(1);
  } else {
    for (int64_t spatial_dimension = 0;
         spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
      // Spatial dimension number for kernel (rhs).
      const int64_t kernel_spatial_dim =
          dnums.kernel_spatial_dimensions(spatial_dimension);
      const int64_t kernel_limit = rhs_shape.dimensions(kernel_spatial_dim);
      kernel_limits.push_back(kernel_limit);

      // Spatial dimension number for output.
      const int64_t output_spatial_dim =
          dnums.output_spatial_dimensions(spatial_dimension);
      const int64_t output_limit = result_shape.dimensions(output_spatial_dim);
      output_limits.push_back(output_limit);

      // Spatial dimension number for input (lhs).
      const int64_t input_spatial_dim =
          dnums.input_spatial_dimensions(spatial_dimension);
      const int64_t input_limit = lhs_shape.dimensions(input_spatial_dim);
      input_limits.push_back(input_limit);
    }
  }

  DimensionVector valid_position_counts;

  // Loop over each spatial dimension.
  for (int64_t spatial_dimension = 0;
       spatial_dimension < window.dimensions_size(); ++spatial_dimension) {
    const auto& window_dim = window.dimensions(spatial_dimension);
    // These two conditions will create an N^2 iteration pattern with only N
    // valid elements. This is a performance optimization and produces the same
    // result as the whole loop.
    if (input_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        kernel_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        input_limits[spatial_dimension] == window_dim.base_dilation() &&
        window_dim.window_dilation() == 1 &&
        std::max<int64_t>(1, input_limits[spatial_dimension] - 1) ==
            window_dim.stride() &&
        window_dim.padding_low() == 0 && window_dim.padding_high() == 0) {
      valid_position_counts.push_back(input_limits[spatial_dimension]);
      continue;
    }

    if (input_limits[spatial_dimension] == 1 &&
        kernel_limits[spatial_dimension] == output_limits[spatial_dimension] &&
        window_dim.window_dilation() == 1 && window_dim.base_dilation() == 1 &&
        window_dim.stride() == 1 &&
        window_dim.padding_high() == output_limits[spatial_dimension] - 1 &&
        window_dim.padding_low() == output_limits[spatial_dimension] - 1) {
      valid_position_counts.push_back(output_limits[spatial_dimension]);
      continue;
    }

    int64_t valid_position_count = 0;
    // Loop over each point in the kernel.
    for (int64_t kernel_idx = 0; kernel_idx < kernel_limits[spatial_dimension];
         ++kernel_idx) {
      // Loop over each point in the output.
      for (int64_t output_idx = 0;
           output_idx < output_limits[spatial_dimension]; ++output_idx) {
        // Calculate lhs (input) index without taking base dilation into
        // account.
        const int64_t undilated_index =
            output_idx * window_dim.stride() - window_dim.padding_low() +
            kernel_idx * window_dim.window_dilation();

        // Calculate the actual lhs (input) index after dilation. Avoid the
        // division as an optimization.
        const int64_t lhs_spatial_index =
            window_dim.base_dilation() > 1
                ? undilated_index / window_dim.base_dilation()
                : undilated_index;

        // Skip if the lhs (input) index is to be dilated.
        if (undilated_index != lhs_spatial_index * window_dim.base_dilation()) {
          continue;
        }

        // Skip if input index is not in bound.
        if (lhs_spatial_index < 0 ||
            lhs_spatial_index >= input_limits[spatial_dimension]) {
          continue;
        }

        valid_position_count += 1;
      }
    }
    valid_position_counts.push_back(valid_position_count);
  }

  const int64_t fma_count =
      (input_feature / convolution->feature_group_count()) * output_feature *
      (batch / convolution->batch_group_count()) *
      Product(valid_position_counts);
  return fma_count * kFmaFlops;
}

Status HloCostAnalysis::HandleConvolution(const HloInstruction* convolution) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_59(mht_59_v, 1065, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleConvolution");

  current_properties_[kFlopsKey] = GetConvolutionFlops(convolution);
  return Status::OK();
}

Status HloCostAnalysis::HandleFft(const HloInstruction* fft) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_60(mht_60_v, 1073, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleFft");

  auto real_shape =
      fft->operand(0)->shape().IsTuple()
          ? ShapeUtil::GetTupleElementShape(fft->operand(0)->shape(), 0)
          : fft->operand(0)->shape();
  constexpr int kFmaPerComplexMul = 4;
  int64_t log_factors = 1;
  for (int64_t dim : fft->fft_length()) {
    log_factors *= Log2Floor<uint64_t>(dim);
  }
  current_properties_[kFlopsKey] = kFmaFlops * kFmaPerComplexMul * log_factors *
                                   ShapeUtil::ElementsIn(real_shape);
  return Status::OK();
}

Status HloCostAnalysis::HandleTriangularSolve(const HloInstruction* hlo) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_61(mht_61_v, 1091, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleTriangularSolve");

  // Half of operand 0 is read.
  float bytes_accessed = GetShapeSize(hlo->shape());
  SetOutputBytesAccessed(GetShapeSize(hlo->shape()));
  bytes_accessed += GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  SetOperandBytesAccessed(0, GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  bytes_accessed += GetShapeSize(hlo->operand(1)->shape());
  SetOperandBytesAccessed(0, GetShapeSize(hlo->operand(1)->shape()));
  current_properties_[kBytesAccessedKey] = bytes_accessed;

  const Shape& a_shape = hlo->operand(0)->shape();
  const Shape& b_shape = hlo->operand(1)->shape();
  // Estimate as batch * mn^2 / 2 flops.
  int64_t elems = a_shape.dimensions(a_shape.dimensions_size() - 1);
  elems *= ShapeUtil::ElementsIn(b_shape);
  current_properties_[kFlopsKey] = kFmaFlops * elems;
  return Status::OK();
}

Status HloCostAnalysis::HandleCholesky(const HloInstruction* hlo) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_62(mht_62_v, 1113, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCholesky");

  // Half of operand 0 is read and half of the output will be written.
  float bytes_accessed = GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  SetOutputBytesAccessed(GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  bytes_accessed += GetShapeSize(hlo->operand(0)->shape()) / 2.0f;
  SetOperandBytesAccessed(0, GetShapeSize(hlo->operand(0)->shape()) / 2.0f);
  current_properties_[kBytesAccessedKey] = bytes_accessed;

  const Shape& a_shape = hlo->operand(0)->shape();
  // Estimate as batch * n^3 / 3 flops.
  int64_t elems = a_shape.dimensions(a_shape.dimensions_size() - 1);
  elems *= ShapeUtil::ElementsIn(a_shape);
  current_properties_[kFlopsKey] = elems / 3;
  return Status::OK();
}

Status HloCostAnalysis::HandleOptimizationBarrier(
    const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_63(mht_63_v, 1133, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleOptimizationBarrier");

  return Status::OK();
}

Status HloCostAnalysis::HandleAllGather(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_64(mht_64_v, 1140, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllGather");

  return Status::OK();
}

Status HloCostAnalysis::HandleAllGatherStart(const HloInstruction* hlo) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_65(mht_65_v, 1147, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllGatherStart");

  return HandleAllGather(hlo);
}

Status HloCostAnalysis::HandleAllGatherDone(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_66(mht_66_v, 1154, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllGatherDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleAllReduce(const HloInstruction* crs) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_67(mht_67_v, 1161, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllReduce");

  // We assume 2 replicas, so that each output element is the sum of two input
  // elements.
  //
  // TODO(b/33004697): Compute correct cost here, taking the actual number of
  // replicas into account.
  double flops = 0.0;
  int64_t output_bytes_accessed = 0;
  ShapeUtil::ForEachSubshape(
      crs->shape(), [&](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsArray()) {
          flops += ShapeUtil::ElementsIn(subshape);
          output_bytes_accessed += GetShapeSize(subshape);
        }
      });
  int64_t bytes_accessed = output_bytes_accessed;
  for (const HloInstruction* operand : crs->operands()) {
    bytes_accessed += GetShapeSize(operand->shape());
  }
  current_properties_[kFlopsKey] = flops;
  SetOutputBytesAccessed(output_bytes_accessed);
  current_properties_[kBytesAccessedKey] = bytes_accessed;
  return Status::OK();
}

Status HloCostAnalysis::HandleReduceScatter(const HloInstruction* hlo) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_68(mht_68_v, 1189, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReduceScatter");

  return Status::OK();
}

Status HloCostAnalysis::HandleAllReduceStart(const HloInstruction* hlo) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_69(mht_69_v, 1196, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllReduceStart");

  return HandleAllReduce(hlo);
}

Status HloCostAnalysis::HandleAllReduceDone(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_70(mht_70_v, 1203, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllReduceDone");

  return Status::OK();
}

Status HloCostAnalysis::HandleAllToAll(const HloInstruction* hlo) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_71(mht_71_v, 1210, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleAllToAll");

  return Status::OK();
}

Status HloCostAnalysis::HandleCollectivePermute(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_72(mht_72_v, 1217, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCollectivePermute");

  return Status::OK();
}

Status HloCostAnalysis::HandleCollectivePermuteStart(
    const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_73(mht_73_v, 1225, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCollectivePermuteStart");

  return Status::OK();
}

Status HloCostAnalysis::HandleCollectivePermuteDone(
    const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_74(mht_74_v, 1233, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCollectivePermuteDone");

  return Status::OK();
}

Status HloCostAnalysis::HandlePartitionId(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_75(mht_75_v, 1240, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandlePartitionId");

  return Status::OK();
}

Status HloCostAnalysis::HandleReplicaId(const HloInstruction* /*hlo*/) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_76(mht_76_v, 1247, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleReplicaId");

  return Status::OK();
}

Status HloCostAnalysis::HandleRng(const HloInstruction* random) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_77(mht_77_v, 1254, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleRng");

  // TODO(b/26346211): Implement better estimates for the RNG cost, since the
  // cost changes with the implementation and the distribution. For now, assume
  // the cost of each RNG is same as a transcendental operation.
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsIn(random->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleRngBitGenerator(const HloInstruction* random) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_78(mht_78_v, 1266, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleRngBitGenerator");

  // TODO(b/26346211): Implement better estimates for the RNG cost, since the
  // cost changes with the implementation and the distribution. For now, assume
  // the cost of each RNG is same as a transcendental operation.
  current_properties_[kTranscendentalsKey] =
      ShapeUtil::ElementsInRecursive(random->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleRngGetAndUpdateState(
    const HloInstruction* random) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_79(mht_79_v, 1279, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleRngGetAndUpdateState");

  return Status::OK();
}

Status HloCostAnalysis::HandleFusion(const HloInstruction* fusion) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_80(mht_80_v, 1286, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleFusion");

  if (fusion->IsCustomFusion()) {
    for (const HloInstruction* hlo :
         fusion->fused_instructions_computation()->instructions()) {
      if (hlo->opcode() == HloOpcode::kGather) {
        return HandleGather(hlo);
      }
      if (hlo->opcode() == HloOpcode::kScatter) {
        return HandleScatter(hlo);
      }
    }
  }
  TF_ASSIGN_OR_RETURN(
      current_properties_,
      ProcessSubcomputation(fusion->fused_instructions_computation()));

  // Fusion nodes that produce a tuple also produce the entries in the tuple.
  // Ignore the memory accessed inside fused ops, since fusion is supposed to
  // prevent intermediate data from touching slow memory.
  current_properties_[kBytesAccessedKey] = 0;
  ShapeUtil::ForEachSubshape(
      fusion->shape(),
      [this, fusion](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!subshape.IsArray()) {
          return;
        }
        if (shape_index.empty()) {
          if (fusion->fused_expression_root()->opcode() ==
              HloOpcode::kDynamicUpdateSlice) {
            int64_t size = GetShapeSize(
                fusion->fused_expression_root()->operand(1)->shape());
            current_properties_[kBytesAccessedKey] += size;
            SetOutputBytesAccessed(shape_index, size);
            return;
          }
        } else if (shape_index.size() == 1) {
          if (fusion->fused_expression_root()->opcode() == HloOpcode::kTuple &&
              fusion->fused_expression_root()
                      ->operand(shape_index[0])
                      ->opcode() == HloOpcode::kDynamicUpdateSlice) {
            int64_t size = GetShapeSize(fusion->fused_expression_root()
                                            ->operand(shape_index[0])
                                            ->operand(1)
                                            ->shape());
            current_properties_[kBytesAccessedKey] += size;
            SetOutputBytesAccessed(shape_index, size);
            return;
          }
        }
        current_properties_[kBytesAccessedKey] += GetShapeSize(subshape);
        SetOutputBytesAccessed(shape_index, GetShapeSize(subshape));
      });

  if (fusion->shape().IsTuple()) {
    // Propagate and accumulate the output tuple bytes from the tuple subshapes.
    // This ensures we have the correct output bytes accessed for the shape
    // index
    // {}.
    std::function<float(const Shape&, const ShapeIndex&)>
        propagate_output_size_to_parent;
    propagate_output_size_to_parent = [&](const Shape& shape,
                                          const ShapeIndex& shape_index) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_81(mht_81_v, 1350, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "lambda");

      auto output_bytes_it =
          current_properties_.find(GetOutputBytesAccessedKey(shape_index));
      if (output_bytes_it != current_properties_.end()) {
        return output_bytes_it->second;
      }
      float bytes_accessed = 0;
      for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
        const Shape& subshape = shape.tuple_shapes(i);
        ShapeIndex subshape_index(shape_index);
        subshape_index.push_back(i);
        bytes_accessed +=
            propagate_output_size_to_parent(subshape, subshape_index);
      }
      SetOutputBytesAccessed(shape_index, bytes_accessed);
      return bytes_accessed;
    };
    current_properties_.erase(
        current_properties_.find(GetOutputBytesAccessedKey()));
    propagate_output_size_to_parent(fusion->shape(), {});
  }

  for (int64_t i = 0; i < fusion->fused_parameters().size(); ++i) {
    const HloInstruction* operand = fusion->fused_parameter(i);
    int64_t operand_size = 0;
    if (!fusion->shape().IsTuple()) {
      operand_size = FusionParameterReadBytes(operand);
    } else {
      // If the fusion parameter is a tuple type, find the gte for the leaf
      // shape and calculate the bytes accessed for those array types.
      for (const auto& indexed_shape :
           ShapeUtil::GetLeafShapes(operand->shape())) {
        const HloInstruction* gte = operand;
        for (int64_t index : indexed_shape.index) {
          for (const HloInstruction* user : gte->users()) {
            if (user->opcode() == HloOpcode::kGetTupleElement &&
                user->tuple_index() == index) {
              gte = user;
              break;
            }
          }
        }
        int64_t size = FusionParameterReadBytes(gte);
        operand_size += size;
        SetOperandBytesAccessed(i, indexed_shape.index, size);
      }
    }
    current_properties_[kBytesAccessedKey] += operand_size;
    SetOperandBytesAccessed(i, operand_size);
  }

  return Status::OK();
}

Status HloCostAnalysis::HandleCall(const HloInstruction* call) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_82(mht_82_v, 1407, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCall");

  TF_ASSIGN_OR_RETURN(current_properties_,
                      ProcessSubcomputation(call->to_apply()));
  current_should_compute_bottleneck_time_ = false;
  return Status::OK();
}

Status HloCostAnalysis::HandleCustomCall(const HloInstruction* custom_call) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_83(mht_83_v, 1417, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleCustomCall");

  // Mark applicable fields as "unknown", since we don't know what this
  // CustomCall does.  This is better than returning an error, which would stop
  // iteration, and therefore would prevent us from getting *any* stats for a
  // computation which contains a CustomCall.
  current_properties_[kOptimalSecondsKey] = -1;
  current_properties_[kBytesAccessedKey] = -1;
  SetOutputBytesAccessed(-1);
  for (int i = 0; i < custom_call->operand_count(); ++i) {
    SetOperandBytesAccessed(i, -1);
  }
  current_properties_[kFlopsKey] = -1;
  current_should_compute_bottleneck_time_ = false;
  return Status::OK();
}

Status HloCostAnalysis::HandleSort(const HloInstruction* sort) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_84(mht_84_v, 1436, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSort");

  // This assumes a comparison based N*log(N) algorithm. As for all ops, the
  // actual properties of the op depend on the backend implementation.
  int64_t elements = ShapeUtil::ElementsIn(sort->operand(0)->shape());
  current_properties_[kFlopsKey] = elements * Log2Ceiling<uint64_t>(elements);
  return Status::OK();
}

Status HloCostAnalysis::HandleWhile(const HloInstruction* xla_while) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_85(mht_85_v, 1447, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleWhile");

  // Since the number of iterations of the while node will not always be
  // something that we can statically analyze, we cannot precisely compute the
  // cost of a while node. For now compute the cost of a single iteration.
  TF_ASSIGN_OR_RETURN(const Properties body_properties,
                      ProcessSubcomputation(xla_while->while_body()));

  TF_ASSIGN_OR_RETURN(const Properties condition_properties,
                      ProcessSubcomputation(xla_while->while_condition()));

  current_properties_.clear();
  for (const auto& property : body_properties) {
    current_properties_[property.first] += property.second;
  }
  for (const auto& property : condition_properties) {
    current_properties_[property.first] += property.second;
  }
  current_should_compute_bottleneck_time_ = false;

  return Status::OK();
}

Status HloCostAnalysis::HandleConditional(const HloInstruction* conditional) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_86(mht_86_v, 1472, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleConditional");

  // Compute the cost of the branch computations and take the maximum from those
  // for each property.
  TF_ASSIGN_OR_RETURN(
      const Properties branch0_computation_properties,
      ProcessSubcomputation(conditional->branch_computation(0)));
  current_properties_ = branch0_computation_properties;
  for (int j = 1; j < conditional->branch_count(); ++j) {
    TF_ASSIGN_OR_RETURN(
        const Properties branch_computation_properties,
        ProcessSubcomputation(conditional->branch_computation(j)));
    for (const auto& property : branch_computation_properties) {
      if (!tensorflow::gtl::InsertIfNotPresent(&current_properties_,
                                               property)) {
        auto& current_property = current_properties_[property.first];
        current_property = std::max(current_property, property.second);
      }
    }
  }
  current_should_compute_bottleneck_time_ = false;

  return Status::OK();
}

Status HloCostAnalysis::HandleGather(const HloInstruction* gather) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_87(mht_87_v, 1499, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleGather");

  // Gather doesn't read the whole input buffer, it's equivalent to a copy the
  // size of the output shape and a read of the gather indices.
  int64_t output_size = GetShapeSize(gather->shape());
  current_properties_[kBytesAccessedKey] =
      output_size * 2 + GetShapeSize(gather->operand(1)->shape());
  SetOperandBytesAccessed(0, output_size);
  SetOperandBytesAccessed(1, GetShapeSize(gather->operand(1)->shape()));
  SetOutputBytesAccessed(output_size);
  // Gather does not issue any flops.
  return Status::OK();
}

Status HloCostAnalysis::HandleScatter(const HloInstruction* scatter) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_88(mht_88_v, 1515, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleScatter");

  // Scatter accesses the equivalent of 3 update shapes (input, output, and
  // updates), and the scatter indices.
  int64_t update_size = GetShapeSize(scatter->operand(2)->shape());
  current_properties_[kBytesAccessedKey] =
      update_size * 3 + GetShapeSize(scatter->operand(1)->shape());
  SetOperandBytesAccessed(0, update_size);
  SetOperandBytesAccessed(1, GetShapeSize(scatter->operand(1)->shape()));
  SetOperandBytesAccessed(2, update_size);
  SetOutputBytesAccessed(update_size);
  const int64_t element_count =
      ShapeUtil::ElementsIn(scatter->operand(2)->shape());
  TF_ASSIGN_OR_RETURN(const Properties sub_properties,
                      ProcessSubcomputation(scatter->to_apply()));
  for (const auto& property : sub_properties) {
    if (!absl::StartsWith(property.first, kBytesAccessedKey)) {
      current_properties_[property.first] = property.second * element_count;
    }
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleGetDimensionSize(
    const HloInstruction* /*get_size*/) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_89(mht_89_v, 1541, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleGetDimensionSize");

  return Status::OK();
}

Status HloCostAnalysis::HandleSetDimensionSize(
    const HloInstruction* /*set_size*/) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_90(mht_90_v, 1549, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::HandleSetDimensionSize");

  return Status::OK();
}

Status HloCostAnalysis::FinishVisit(const HloInstruction*) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_91(mht_91_v, 1556, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::FinishVisit");

  return Status::OK();
}

float HloCostAnalysis::flop_count() const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_92(mht_92_v, 1563, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::flop_count");

  return GetProperty(kFlopsKey, properties_sum_);
}

float HloCostAnalysis::transcendental_count() const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_93(mht_93_v, 1570, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::transcendental_count");

  return GetProperty(kTranscendentalsKey, properties_sum_);
}

float HloCostAnalysis::bytes_accessed() const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_94(mht_94_v, 1577, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::bytes_accessed");

  return GetProperty(kBytesAccessedKey, properties_sum_);
}

float HloCostAnalysis::optimal_seconds() const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_95(mht_95_v, 1584, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::optimal_seconds");

  return GetProperty(kOptimalSecondsKey, properties_sum_);
}

int64_t HloCostAnalysis::flop_count(const HloInstruction& hlo) const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_96(mht_96_v, 1591, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::flop_count");

  return GetPropertyForHlo(hlo, kFlopsKey, hlo_properties_);
}

int64_t HloCostAnalysis::transcendental_count(const HloInstruction& hlo) const {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_97(mht_97_v, 1598, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::transcendental_count");

  return GetPropertyForHlo(hlo, kTranscendentalsKey, hlo_properties_);
}

int64_t HloCostAnalysis::bytes_accessed(const HloInstruction& hlo) const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_98(mht_98_v, 1605, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::bytes_accessed");

  return GetPropertyForHlo(hlo, kBytesAccessedKey, hlo_properties_);
}

int64_t HloCostAnalysis::operand_bytes_accessed(const HloInstruction& hlo,
                                                int64_t operand_num,
                                                ShapeIndex index) const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_99(mht_99_v, 1614, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::operand_bytes_accessed");

  return GetPropertyForHlo(hlo, GetOperandBytesAccessedKey(operand_num, index),
                           hlo_properties_);
}

int64_t HloCostAnalysis::output_bytes_accessed(const HloInstruction& hlo,
                                               ShapeIndex index) const {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_100(mht_100_v, 1623, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::output_bytes_accessed");

  return GetPropertyForHlo(hlo, GetOutputBytesAccessedKey(index),
                           hlo_properties_);
}

float HloCostAnalysis::optimal_seconds(const HloInstruction& hlo) const {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_101(mht_101_v, 1631, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::optimal_seconds");

  return GetPropertyForHlo(hlo, kOptimalSecondsKey, hlo_properties_);
}

int64_t HloCostAnalysis::GetBytesRead(
    const HloInstruction& hlo, absl::optional<int64_t> memory_space) const {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_102(mht_102_v, 1639, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetBytesRead");

  int64_t bytes_read = 0;
  for (int operand_number = 0; operand_number < hlo.operand_count();
       ++operand_number) {
    const Shape& shape = hlo.operand(operand_number)->shape();
    ShapeUtil::ForEachSubshape(
        shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
          if (ShapeUtil::IsLeafIndex(shape, index)) {
            absl::optional<int64_t> index_memory_space;
            if (sub_shape.has_layout()) {
              index_memory_space = sub_shape.layout().memory_space();
            }
            if (!memory_space || memory_space == index_memory_space) {
              bytes_read += operand_bytes_accessed(hlo, operand_number, index);
            }
          }
        });
  }
  return bytes_read;
}

int64_t HloCostAnalysis::GetBytesWritten(
    const HloInstruction& hlo, absl::optional<int64_t> memory_space) const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_103(mht_103_v, 1664, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetBytesWritten");

  int64_t bytes_written = 0;
  for (const ShapeUtil::IndexedShape& indexed_shape :
       ShapeUtil::GetLeafShapes(hlo.shape())) {
    absl::optional<int64_t> index_memory_space;
    if (indexed_shape.shape.has_layout()) {
      index_memory_space = indexed_shape.shape.layout().memory_space();
    }
    if (!memory_space || memory_space == index_memory_space) {
      bytes_written += output_bytes_accessed(hlo, indexed_shape.index);
    }
  }
  return bytes_written;
}

StatusOr<HloCostAnalysis::Properties> HloCostAnalysis::ProcessSubcomputation(
    HloComputation* computation) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_104(mht_104_v, 1683, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::ProcessSubcomputation");

  auto visitor = CreateNestedCostAnalysis();
  visitor->ReserveVisitStates(computation->instruction_count());
  TF_RETURN_IF_ERROR(computation->Accept(visitor.get()));
  hlo_properties_.insert(visitor->hlo_properties_.begin(),
                         visitor->hlo_properties_.end());
  return visitor->properties();
}

std::unique_ptr<HloCostAnalysis> HloCostAnalysis::CreateNestedCostAnalysis() {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_105(mht_105_v, 1695, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::CreateNestedCostAnalysis");

  return std::make_unique<HloCostAnalysis>(options_);
}

void HloCostAnalysis::SetOperandBytesAccessed(int64_t operand_num,
                                              float value) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_106(mht_106_v, 1703, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::SetOperandBytesAccessed");

  current_properties_[GetOperandBytesAccessedKey(operand_num).c_str()] = value;
}

void HloCostAnalysis::SetOperandBytesAccessed(int64_t operand_num,
                                              ShapeIndex index, float value) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_107(mht_107_v, 1711, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::SetOperandBytesAccessed");

  current_properties_[GetOperandBytesAccessedKey(operand_num, index).c_str()] =
      value;
}

void HloCostAnalysis::SetOutputBytesAccessed(float value) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_108(mht_108_v, 1719, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::SetOutputBytesAccessed");

  current_properties_[GetOutputBytesAccessedKey()] = value;
}

void HloCostAnalysis::SetOutputBytesAccessed(ShapeIndex index, float value) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_109(mht_109_v, 1726, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::SetOutputBytesAccessed");

  current_properties_[GetOutputBytesAccessedKey(index)] = value;
}

/*static*/ std::string HloCostAnalysis::GetOperandBytesAccessedKey(
    int64_t operand_num, ShapeIndex index) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_110(mht_110_v, 1734, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetOperandBytesAccessedKey");

  return absl::StrCat(kBytesAccessedKey, " operand ", operand_num, " ",
                      index.ToString());
}

/*static*/ std::string HloCostAnalysis::GetOutputBytesAccessedKey(
    ShapeIndex index) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_cost_analysisDTcc mht_111(mht_111_v, 1743, "", "./tensorflow/compiler/xla/service/hlo_cost_analysis.cc", "HloCostAnalysis::GetOutputBytesAccessedKey");

  return absl::StrCat(kBytesAccessedKey, " output ", index.ToString());
}

}  // namespace xla
