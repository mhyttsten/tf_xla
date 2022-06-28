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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc() {
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

#include "tensorflow/compiler/xla/service/logical_buffer_analysis.h"

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

// Gather fusion instructions from 'instruction' into 'fusion_instructions'.
void GatherFusionInstructions(
    HloInstruction* instruction,
    std::vector<HloInstruction*>* fusion_instructions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "GatherFusionInstructions");

  CHECK_EQ(HloOpcode::kFusion, instruction->opcode());
  for (auto* fused : instruction->fused_instructions()) {
    if (fused->opcode() == HloOpcode::kFusion) {
      GatherFusionInstructions(fused, fusion_instructions);
    }
  }
  fusion_instructions->push_back(instruction);
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<LogicalBufferAnalysis>>
LogicalBufferAnalysis::Run(const HloModule* module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::Run");

  std::unique_ptr<LogicalBufferAnalysis> analysis(
      new LogicalBufferAnalysis(module));
  TF_RETURN_IF_ERROR(analysis->Analyze());
  return std::move(analysis);
}

Status LogicalBufferAnalysis::Analyze() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::Analyze");

  // Empirically we usually have a few more logical buffers than instructions,
  // so reserve 10% more than the number of instructions to avoid frequent
  // resizes.
  logical_buffers_.clear();
  logical_buffers_.reserve((module_->instruction_count() * 11) / 10);

  // We filter out fusion computations, and get to them through fusion
  // instructions. This is because it's possible to have orphaned (unreachable)
  // fusion computations, and we don't want to try to assign buffers to those.
  std::vector<HloInstruction*> fusion_instructions;
  for (auto* computation : module_->MakeNonfusionComputations()) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }
      GatherFusionInstructions(instruction, &fusion_instructions);
    }
  }
  for (auto* instruction : fusion_instructions) {
    TF_RETURN_IF_ERROR(instruction->fused_expression_root()->Accept(this));
  }
  return Status::OK();
}

LogicalBuffer& LogicalBufferAnalysis::GetBuffer(LogicalBuffer::Id id) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::GetBuffer");

  return *logical_buffers_.at(id);
}

LogicalBuffer& LogicalBufferAnalysis::GetBuffer(HloInstruction* instruction,
                                                const ShapeIndex& index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::GetBuffer");

  return *output_buffers_.at(std::make_pair(instruction, index));
}

void LogicalBufferAnalysis::NewLogicalBuffer(HloInstruction* instruction,
                                             const ShapeIndex& index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_5(mht_5_v, 274, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::NewLogicalBuffer");

  LogicalBuffer::Id id = logical_buffers_.size();
  auto buffer = std::make_unique<LogicalBuffer>(instruction, index, id);
  auto position = std::make_pair(instruction, index);
  CHECK(output_buffers_.insert({position, buffer.get()}).second);
  logical_buffers_.push_back(std::move(buffer));
}

Status LogicalBufferAnalysis::DefaultAction(HloInstruction* hlo_instruction) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_6(mht_6_v, 285, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::DefaultAction");

  // Create a logical buffer for each output of the instruction.
  ShapeUtil::ForEachSubshape(
      hlo_instruction->shape(),
      [this, hlo_instruction](const Shape& shape, const ShapeIndex& index) {
        NewLogicalBuffer(hlo_instruction, index);
      });

  return Status::OK();
}

Status LogicalBufferAnalysis::HandleGetTupleElement(HloInstruction*) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_7(mht_7_v, 299, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleGetTupleElement");

  // GetTupleElement does not create buffers.
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleAddDependency(
    HloInstruction* add_dependency) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_8(mht_8_v, 308, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleAddDependency");

  // AddDependency just forwards the value of its zero-th operand and does not
  // create buffers.
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleCopy(HloInstruction* copy) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_9(mht_9_v, 317, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleCopy");

  // The top-level buffer (index={}) for kCopy is newly created, but all other
  // buffers (in the case of a tuple shape) come from the operand
  NewLogicalBuffer(copy, /*index=*/{});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleBitcast(HloInstruction*) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_10(mht_10_v, 327, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleBitcast");

  // A kBitcast instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand.
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleDomain(HloInstruction*) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_11(mht_11_v, 336, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleDomain");

  // A kDomain instruction aliases its operand. That is, the buffer of its
  // result *is* the buffer of its operand.
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleRecvDone(HloInstruction* recv_done) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_12(mht_12_v, 345, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleRecvDone");

  // RecvDone produces a two-element tuple containing the data value (which
  // aliases part of its operand) and a token. Only the tuple index table and
  // the token are defined by the RecvDone.
  NewLogicalBuffer(recv_done, /*index=*/{});
  NewLogicalBuffer(recv_done, /*index=*/{1});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleSend(HloInstruction* send) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_13(mht_13_v, 357, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleSend");

  // Send creates new buffers for the top-level tuple, the context (tuple
  // element at {1}), and the token (tuple element at {2}). Tuple element at {0}
  // is an alias of the Send operand, so we don't need to create a new Logical
  // Buffer for that.
  NewLogicalBuffer(send, /*index=*/{});
  NewLogicalBuffer(send, /*index=*/{1});
  NewLogicalBuffer(send, /*index=*/{2});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleCopyStart(HloInstruction* copy_start) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_14(mht_14_v, 371, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleCopyStart");

  // CopyStart defines the tuple, target buffer at index {0}, and context at
  // index {2}.
  NewLogicalBuffer(copy_start, /*index=*/{});
  NewLogicalBuffer(copy_start, /*index=*/{0});
  NewLogicalBuffer(copy_start, /*index=*/{2});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleCopyDone(HloInstruction* copy_done) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_15(mht_15_v, 383, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleCopyDone");

  // The output of CopyDone aliases with operand {0}. CopyDone doesn't create
  // any buffers.
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleTuple(HloInstruction* tuple) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_16(mht_16_v, 392, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleTuple");

  // A Tuple instruction only creates the top-level buffer.
  NewLogicalBuffer(tuple, /*index=*/{});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleTupleSelect(HloInstruction* tuple_select) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_17(mht_17_v, 401, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleTupleSelect");

  // Select allocates a new buffer and then shallow copies the on_true or
  // on_false buffer into this new buffer.
  NewLogicalBuffer(tuple_select, /*index=*/{});
  return Status::OK();
}

Status LogicalBufferAnalysis::HandleCustomCall(HloInstruction* custom_call) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSlogical_buffer_analysisDTcc mht_18(mht_18_v, 411, "", "./tensorflow/compiler/xla/service/logical_buffer_analysis.cc", "LogicalBufferAnalysis::HandleCustomCall");

  auto ccall = Cast<HloCustomCallInstruction>(custom_call);
  absl::flat_hash_set<ShapeIndex> aliased_outputs;
  for (const auto& pair : ccall->output_to_operand_aliasing()) {
    aliased_outputs.insert(pair.first);
  }
  ShapeUtil::ForEachSubshape(ccall->shape(),
                             [&](const Shape& shape, const ShapeIndex& index) {
                               if (!aliased_outputs.contains(index)) {
                                 NewLogicalBuffer(custom_call, index);
                               }
                             });
  return Status::OK();
}

}  // namespace xla
