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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc() {
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

#include "tensorflow/compiler/xla/tools/hlo_extractor.h"

#include <stdio.h>
#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {
namespace {

// Visitor that build a new HLO module with an entry computation and a root that
// is provided to the visit function. Only HLOs that are reachable from the new
// root instruction are included in the new module.
//
// The constructor allows specifying a set of boundary HLOs to prune the HLO
// graph. HLOs at the boundary are replaced with parameters. Can be nullptr
// which means no boundary, i.e. no HLOs are replaced with parameters.
class ExtractionVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  explicit ExtractionVisitor(
      const HloModule& old_module,
      absl::flat_hash_set<const HloInstruction*>* boundary)
      : old_module_(old_module),
        module_(absl::make_unique<HloModule>("extracted", config_)),
        clone_context_(module_.get()),
        builder_("entry_computation"),
        boundary_(boundary) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "ExtractionVisitor");
}

  Status HandleParameter(const HloInstruction* parameter) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "HandleParameter");

    // Entry parameters need renumbering.
    auto new_parameter = HloInstruction::CreateParameter(
        parameter_number_++, parameter->shape(), parameter->name());
    clone_context_.MapInstruction(parameter, new_parameter.get());
    builder_.AddInstruction(std::move(new_parameter));
    return Status::OK();
  }

  Status DefaultAction(const HloInstruction* hlo) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "DefaultAction");

    // Replace instructions at the boundary with parameters, but leave constants
    // untouched.
    if (boundary_ != nullptr && boundary_->count(hlo) > 0) {
      auto new_parameter = HloInstruction::CreateParameter(
          parameter_number_, hlo->shape(), hlo->name());
      parameter_number_++;
      clone_context_.MapInstruction(hlo, new_parameter.get());
      builder_.AddInstruction(std::move(new_parameter));
      return Status::OK();
    }
    std::vector<HloInstruction*> new_operands;
    for (auto operand : hlo->operands()) {
      new_operands.push_back(clone_context_.GetInstruction(operand));
    }
    auto instruction =
        hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context_);
    builder_.AddInstruction(std::move(instruction));
    return Status::OK();
  }

  Status FinishVisit(const HloInstruction* /*root*/) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "FinishVisit");

    module_->AddEntryComputation(builder_.Build());
    // Rename HLOs so that their name matches the original. By default,
    // HLOs get new unique names when adding a new entry computation to
    // a module.
    for (auto computation : old_module_.MakeComputationPostOrder()) {
      for (auto old_instruction : computation->MakeInstructionPostOrder()) {
        if (auto new_instruction =
                clone_context_.FindInstruction(old_instruction)) {
          new_instruction->SetAndSanitizeName(old_instruction->name());
        }
      }
    }
    return Status::OK();
  }

  HloModule* module() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_4(mht_4_v, 278, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "module");
 return module_.get(); }

  std::unique_ptr<HloModule> ConsumeModule() { return std::move(module_); }

 private:
  const HloModule& old_module_;
  HloModuleConfig config_;
  std::unique_ptr<HloModule> module_;
  HloCloneContext clone_context_;
  HloComputation::Builder builder_;
  absl::flat_hash_set<const HloInstruction*>* boundary_;
  int64_t parameter_number_ = 0;
};

void ComputeBoundary(const HloInstruction* root, int64_t limit,
                     absl::flat_hash_set<const HloInstruction*>* boundary) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_extractorDTcc mht_5(mht_5_v, 296, "", "./tensorflow/compiler/xla/tools/hlo_extractor.cc", "ComputeBoundary");

  std::deque<const HloInstruction*> worklist;
  absl::flat_hash_map<const HloInstruction*, int64_t> visited;
  worklist.push_back(root);
  visited.emplace(root, 0);
  while (!worklist.empty()) {
    auto hlo = worklist.front();
    worklist.pop_front();
    int64_t hops = visited[hlo];
    if (hops > limit) {
      boundary->insert(hlo);
      continue;
    }
    for (const HloInstruction* operand : hlo->operands()) {
      if (visited.count(operand)) {
        continue;
      }
      worklist.push_back(operand);
      visited.emplace(operand, hops + 1);
    }
  }
}

}  // namespace

std::unique_ptr<HloModule> ExtractModule(HloInstruction* instruction,
                                         int64_t height) {
  absl::flat_hash_set<const HloInstruction*> boundary;
  if (height != -1) {
    ComputeBoundary(instruction, height, &boundary);
  }
  ExtractionVisitor visitor(*instruction->GetModule(), &boundary);
  CHECK(instruction->Accept(&visitor).ok());

  // The first pass may leave unused parameter instructions. Do another
  // extraction pass to remove unused parameters. This is done because
  // HloComputation does not allow removing parameters after the computation has
  // been built.
  ExtractionVisitor cleanup_visitor(*visitor.module(), /*boundary=*/nullptr);
  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor));

  HloVerifier verifier(/*layout_sensitive=*/false,
                       /*allow_mixed_precision=*/true);
  TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  return cleanup_visitor.ConsumeModule();
}

}  // namespace xla
