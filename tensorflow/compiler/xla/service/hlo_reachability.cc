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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include <queue>

#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : size_(instructions.size()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::HloReachabilityMap");

  bit_vectors_.reserve(size_);
  for (const HloInstruction* hlo : instructions) {
    indices_[GetKey(hlo)] = bit_vectors_.size();
    bit_vectors_.emplace_back(size_);
  }
  CHECK_EQ(size_, indices_.size());  // instructions should be unique
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_1(mht_1_v, 209, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::SetReachabilityToUnion");

  Index index = GetIndex(instruction);
  BitVector& bit_vector = GetBitVector(index);
  tmp_bit_vector_ = bit_vector;
  SetReachabilityToUnionHelper(inputs, index);
  return bit_vector != tmp_bit_vector_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::FastSetReachabilityToUnion");

  Index index = GetIndex(instruction);
  SetReachabilityToUnionHelper(inputs, index);
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const Index> input_indices, Index index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::FastSetReachabilityToUnion");

  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs, Index index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_4(mht_4_v, 239, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::SetReachabilityToUnionHelper");

  absl::InlinedVector<Index, 16> input_indices;
  input_indices.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    input_indices.push_back(GetIndex(input));
  }
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const Index> input_indices, Index index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_5(mht_5_v, 252, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::SetReachabilityToUnionHelper");

  BitVector& bit_vector = GetBitVector(index);
  // If instruction is part of inputs, don't reset the bit_vector.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_vector.SetToZero();
  }
  bit_vector.Set(index.v);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_vector.OrWith(GetBitVector(input_index));
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_6(mht_6_v, 270, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::Replace");

  if (GetKey(original) == GetKey(replacement)) {
    return;
  }
  indices_[GetKey(replacement)] = GetIndex(original).v;
  indices_.erase(GetKey(original));
}

void HloReachabilityMap::SetReachable(Index a, Index b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_7(mht_7_v, 281, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::SetReachable");

  GetBitVector(b).Set(a.v);
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_8(mht_8_v, 292, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::BuildWithRestrictions");

  const auto& all = computation->MakeInstructionPostOrder();
  auto result = absl::make_unique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo, &inputs);
    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::Build");

  const auto& all = computation->MakeInstructionPostOrder();
  auto result = absl::make_unique<HloReachabilityMap>(all);
  auto channel_group = computation->ComputeChannelDependencies();

  std::vector<HloInstruction*> inputs;

  const auto add_input = [&channel_group, &inputs](HloInstruction* input) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_10(mht_10_v, 319, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "lambda");

    inputs.push_back(input);
    if ((input->opcode() == HloOpcode::kAllReduce ||
         input->opcode() == HloOpcode::kReduceScatter) &&
        input->channel_id()) {
      auto it = channel_group.find(*input->channel_id());
      if (it != channel_group.end()) {
        inputs.insert(inputs.end(), it->second.begin(), it->second.end());
      }
    }
  };

  const auto add_dependencies = [&add_input](const HloInstruction* hlo) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_11(mht_11_v, 334, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "lambda");

    for (HloInstruction* operand : hlo->operands()) {
      add_input(operand);
    }
    for (HloInstruction* predecessor : hlo->control_predecessors()) {
      add_input(predecessor);
    }
  };

  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo);

    switch (hlo->opcode()) {
      case HloOpcode::kRecvDone: {
        auto it = channel_group.find(*hlo->channel_id());
        if (it != channel_group.end()) {
          for (HloInstruction* channel : it->second) {
            if (channel->opcode() == HloOpcode::kSend) {
              add_input(channel);
            }
          }
        }
        break;
      }
      case HloOpcode::kAllReduce:
      case HloOpcode::kReduceScatter: {
        auto channel_id = hlo->channel_id();
        if (channel_id) {
          auto it = channel_group.find(channel_id.value());
          if (it != channel_group.end()) {
            for (HloInstruction* all_reduce : it->second) {
              add_dependencies(all_reduce);
            }
          }
        }
        break;
      }
      default:
        break;
    }

    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTcc mht_12(mht_12_v, 385, "", "./tensorflow/compiler/xla/service/hlo_reachability.cc", "HloReachabilityMap::UpdateReachabilityThroughInstruction");

  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
      }
    }
  }
}

}  // namespace xla
