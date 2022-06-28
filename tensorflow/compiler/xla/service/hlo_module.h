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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh() {
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


#include <atomic>
#include <list>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/service/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_module_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Describes a compilation unit at the HLO level.
//
// HloModule is the top-level unit in the HLO IR.  It corresponds to a whole
// "program".  Running a module, from beginning to end, is the only way to run
// an XLA program.
//
// A module contains one "entry computation"; this HloComputation is like main()
// in a C program.  The result of running the module is the result of running
// this computation.
//
// A module also contains some number of "nested computations".  Each nested
// computation is attached to an HloInstruction within some other computation.
// The meaning of the nested computation depends on the instruction it's
// attached to.
class HloModule {
 public:
  // Constructor without a versioned computation handle. This constructor should
  // only be used for HloModules used outside of the XLA service (eg
  // tests). The versioned handle is used by the service in the compilation
  // cache. A default configuration is created for this module.
  explicit HloModule(const std::string& name, HloModuleConfig config);
  virtual ~HloModule() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_0(mht_0_v, 238, "", "./tensorflow/compiler/xla/service/hlo_module.h", "~HloModule");
}

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  HloComputation* AddEntryComputation(
      std::unique_ptr<HloComputation> computation);

  // Same as the AddEntryComputation function above but the module's
  // entry_computation_layout is updated to match the layout of the new entry
  // computation.
  HloComputation* AddEntryComputationWithLayouts(
      std::unique_ptr<HloComputation> computation);

  // Replaces the current entry computation with another computation.
  // The new entry computation must be a computation that is already in the
  // module.
  void ReplaceEntryComputation(HloComputation* entry_computation);

  // Adds an embedded computation to the module.
  HloComputation* AddEmbeddedComputation(
      std::unique_ptr<HloComputation> computation);

  // Removes an embedded computation.
  Status RemoveEmbeddedComputation(HloComputation* to_remove);

  // Removes unused computations.
  Status RemoveUnusedComputations();

  // Replaces all uses of computations that are keys of 'replacements' with
  // the corresponding values in 'replacements'. Replaces the entry computation,
  // if applicable.
  //
  // This function iterates over all instructions in the module to find
  // computations to replace. We could speed it up by keeping track of users of
  // computations.
  void ReplaceComputations(
      const absl::flat_hash_map<HloComputation*, HloComputation*>&
          replacements);

  const std::string& name() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_1(mht_1_v, 280, "", "./tensorflow/compiler/xla/service/hlo_module.h", "name");
 return name_; }
  void set_name(std::string name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_2(mht_2_v, 285, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_name");
 name_ = std::move(name); }

  // Returns a deep copy of this module including all computations.
  std::unique_ptr<HloModule> Clone(const std::string& suffix = "clone") const;
  std::unique_ptr<HloModule> Clone(const HloModuleConfig& config,
                                   const std::string& suffix = "clone") const;

  // Performs a deep clone of the computation, by recursively cloning all
  // the called computations as well. If the clone context is specified, it
  // will be populated with the cloned object mappings.
  HloComputation* DeepCloneComputation(HloComputation* computation,
                                       HloCloneContext* context = nullptr);

  // Return a pointer to the entry computation of the module.
  HloComputation* entry_computation() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_3(mht_3_v, 302, "", "./tensorflow/compiler/xla/service/hlo_module.h", "entry_computation");

    CHECK_NE(nullptr, entry_computation_);
    return entry_computation_;
  }

  bool has_entry_computation() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_4(mht_4_v, 310, "", "./tensorflow/compiler/xla/service/hlo_module.h", "has_entry_computation");
 return entry_computation_ != nullptr; }

  // Returns the root instruction shape of entry computation.
  //
  // Precondition: entry_computation_ is not nullptr.
  const Shape& result_shape() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_5(mht_5_v, 318, "", "./tensorflow/compiler/xla/service/hlo_module.h", "result_shape");

    CHECK_NE(nullptr, entry_computation_);
    return entry_computation()->root_instruction()->shape();
  }

  // Creates the ComputationLayout which describes the current status of the HLO
  // module entry computation.
  ComputationLayout compute_computation_layout() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_6(mht_6_v, 328, "", "./tensorflow/compiler/xla/service/hlo_module.h", "compute_computation_layout");

    return ComputationLayout(entry_computation()->ComputeProgramShape(),
                             /*ignore_layouts=*/false);
  }

  ComputationLayout* mutable_entry_computation_layout() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_7(mht_7_v, 336, "", "./tensorflow/compiler/xla/service/hlo_module.h", "mutable_entry_computation_layout");

    return config_.mutable_entry_computation_layout();
  }

  const ComputationLayout& entry_computation_layout() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_8(mht_8_v, 343, "", "./tensorflow/compiler/xla/service/hlo_module.h", "entry_computation_layout");

    return config_.entry_computation_layout();
  }

  // Generates a hash value of an HLO module. Hash considers
  // information on opcode, shape, operands, and typically a root instruction.
  // This function returns the same hash value for equivalent HLO modules,
  // with respect to HloInstruction::Identical() method.
  template <typename H>
  friend H AbslHashValue(H h, const HloModule& module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_9(mht_9_v, 355, "", "./tensorflow/compiler/xla/service/hlo_module.h", "AbslHashValue");

    h = H::combine(std::move(h), module.entry_computation_layout());
    // Use MakeComputationSorted() instead of MakeComputationPostOrder()
    // because naming may affect the order of MakeComputationPostOrder() but not
    // MakeComputationSorted().
    auto computations = module.MakeComputationSorted();
    for (auto* computation : computations) {
      h = H::combine(std::move(h), *computation);
    }
    return H::combine(std::move(h), computations.size());
  }

  // Gets the computations in this module.
  //
  // Returns a view of HloComputation*s, so you can iterate over this in the
  // natural way:
  //
  //   for (HloComputation* c : module->computations()) { ... }
  //
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::const_iterator>>
  computations() const {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::iterator>>
  computations() {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }

  // Returns the computation in this module that has the name `name`.  Returns
  // null if there is no such computation.
  HloComputation* GetComputationWithName(absl::string_view name);

  // Gets the number of computations in this module.
  int64_t computation_count() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_10(mht_10_v, 395, "", "./tensorflow/compiler/xla/service/hlo_module.h", "computation_count");
 return computations_.size(); }

  // Returns the mutable computation for the given index.
  HloComputation* mutable_computation(int64_t idx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_11(mht_11_v, 401, "", "./tensorflow/compiler/xla/service/hlo_module.h", "mutable_computation");

    CHECK(idx >= 0 && idx < computations_.size());
    return computations_[idx].get();
  }

  // Gets the number of instructions in this module.
  int64_t instruction_count() const;

  // Deallocate removed instructions in each computation.
  void Cleanup() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_12(mht_12_v, 413, "", "./tensorflow/compiler/xla/service/hlo_module.h", "Cleanup");

    for (auto& comp : computations_) {
      comp->Cleanup();
    }
  }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::vector<HloComputation*> MakeComputationPostOrder() const;

  // Same as MakeComputationPostOrder() but only returns the computations
  // that are also found in the passed in allowList
  std::vector<HloComputation*> MakeComputationPostOrder(
      const absl::flat_hash_set<HloComputation*>& allow_list) const;

  // Same as MakeComputationPostOrder() but sorting the computations by their
  // contents. The order is longer post order.
  std::vector<HloComputation*> MakeComputationSorted() const;

  // Gets the computations in this module which aren't for fusion nodes.
  //
  // Postcondition: All computations in the returned list have
  // !IsFusionComputation().
  //
  // Note: Callers can and do rely on the return value here being a *snapshot*
  // of the module's non-fusion computations -- that is, it's OK to add or
  // remove computations from a module while iterating over
  // MakeNonfusionComputations().
  std::vector<HloComputation*> MakeNonfusionComputations() const;

  // Same as MakeNonfusionComputations() but sorting computations by content.
  std::vector<HloComputation*> MakeNonfusionComputationsSorted() const;

  HloModuleConfig& config() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_13(mht_13_v, 450, "", "./tensorflow/compiler/xla/service/hlo_module.h", "config");
 return config_; }
  const HloModuleConfig& config() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_14(mht_14_v, 454, "", "./tensorflow/compiler/xla/service/hlo_module.h", "config");
 return config_; }
  void set_config(const HloModuleConfig& config) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_15(mht_15_v, 458, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_config");
 config_ = config; }

  bool is_dynamic() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_16(mht_16_v, 463, "", "./tensorflow/compiler/xla/service/hlo_module.h", "is_dynamic");
 return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_17(mht_17_v, 467, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_is_dynamic");
 is_dynamic_ = is_dynamic; }

  // Return a string representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  std::string ToString() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_18(mht_18_v, 476, "", "./tensorflow/compiler/xla/service/hlo_module.h", "ToString");
 return ToString(HloPrintOptions()); }
  std::string ToString(const HloPrintOptions& options) const;

  // Returns a Cord representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  absl::Cord ToCord() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_19(mht_19_v, 486, "", "./tensorflow/compiler/xla/service/hlo_module.h", "ToCord");
 return ToCord(HloPrintOptions()); }
  absl::Cord ToCord(const HloPrintOptions& options) const;

  // Convert an HloModule to or from a proto.
  HloModuleProto ToProto() const;
  static StatusOr<std::unique_ptr<HloModule>> CreateFromProto(
      const HloModuleProto& proto, const HloModuleConfig& module_config,
      bool prohibit_empty_literal = true);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static StatusOr<HloModuleConfig> CreateModuleConfigFromProto(
      const HloModuleProto& module, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static StatusOr<HloModuleConfig> CreateModuleConfigFromShape(
      const ProgramShape& program_shape, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Outlines the given expression from the given computation.
  // instructions_to_outline contains the instructions that form the expression.
  //
  // Precondition: instructions in instructions_to_outline are in topological
  // order (root of outlined instructions last). TODO(jingyue): takes a set of
  // instructions and topologically sorts them.
  HloInstruction* OutlineExpressionFromComputation(
      absl::Span<HloInstruction* const> instructions_to_outline,
      const std::string& outlined_computation_name,
      HloComputation* computation);

  // Returns a randomly generated uint64_t.
  uint64_t RandomNew64() const;

  // Returns the NameUniquer for uniquing instruction names in this module.
  NameUniquer& instruction_name_uniquer() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_20(mht_20_v, 525, "", "./tensorflow/compiler/xla/service/hlo_module.h", "instruction_name_uniquer");
 return instruction_name_uniquer_; }

  // Assign a new unique dense id for an instruction
  int NewUniqueInstructionId() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_21(mht_21_v, 531, "", "./tensorflow/compiler/xla/service/hlo_module.h", "NewUniqueInstructionId");

    int result = next_unique_id_;
    next_unique_id_++;
    return result;
  }

  // input_output_alias_config indicates the list of aliased buffers that are
  // expected from the module.
  HloInputOutputAliasConfig& input_output_alias_config() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_22(mht_22_v, 542, "", "./tensorflow/compiler/xla/service/hlo_module.h", "input_output_alias_config");

    return input_output_alias_config_;
  }
  const HloInputOutputAliasConfig& input_output_alias_config() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_23(mht_23_v, 548, "", "./tensorflow/compiler/xla/service/hlo_module.h", "input_output_alias_config");

    return input_output_alias_config_;
  }

  // DynamicParameterBinding holds the list of bindings that indicates which
  // parameter dimensions are dynamic and which parameters represent their
  // runtime value.
  DynamicParameterBinding& dynamic_parameter_binding() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_24(mht_24_v, 558, "", "./tensorflow/compiler/xla/service/hlo_module.h", "dynamic_parameter_binding");

    return dynamic_parameter_binding_;
  }
  const DynamicParameterBinding& dynamic_parameter_binding() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_25(mht_25_v, 564, "", "./tensorflow/compiler/xla/service/hlo_module.h", "dynamic_parameter_binding");

    return dynamic_parameter_binding_;
  }

  // Returns an id that is unique to this module across all modules created over
  // the lifetime of this process.
  int unique_id() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_26(mht_26_v, 573, "", "./tensorflow/compiler/xla/service/hlo_module.h", "unique_id");
 return unique_id_; }

  // Sets the schedule of the module to the given schedule.
  Status set_schedule(HloSchedule schedule);

  // Clears the schedule of the module.
  void clear_schedule() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_27(mht_27_v, 582, "", "./tensorflow/compiler/xla/service/hlo_module.h", "clear_schedule");
 schedule_.reset(); }

  // Returns true if the module has a schedule set.
  bool has_schedule() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_28(mht_28_v, 588, "", "./tensorflow/compiler/xla/service/hlo_module.h", "has_schedule");
 return schedule_.has_value(); }

  // Returns the schedule of the module. CHECK fails if no schedule is set.
  const HloSchedule& schedule() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_29(mht_29_v, 594, "", "./tensorflow/compiler/xla/service/hlo_module.h", "schedule");
 return *schedule_; }
  HloSchedule& schedule() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_30(mht_30_v, 598, "", "./tensorflow/compiler/xla/service/hlo_module.h", "schedule");
 return *schedule_; }

  HloComputation* AddComputationAndUnifyNamesAndIds(
      std::unique_ptr<HloComputation> computation, bool is_entry) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_31(mht_31_v, 604, "", "./tensorflow/compiler/xla/service/hlo_module.h", "AddComputationAndUnifyNamesAndIds");

    computation->ClearUniqueIdInternal();
    for (auto* instruction : computation->instructions()) {
      instruction->ClearUniqueIdInternal();
    }
    return AddComputationInternal(std::move(computation), is_entry,
                                  /*uniquify_identifiers=*/true,
                                  /*preserve_entry_layouts=*/true);
  }

  void SetAndUniquifyInstrName(HloInstruction* instr, absl::string_view name) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_32(mht_32_v, 618, "", "./tensorflow/compiler/xla/service/hlo_module.h", "SetAndUniquifyInstrName");

    instr->SetAndSanitizeName(name);
    instr->UniquifyName(&instruction_name_uniquer_);
  }

  Status CheckUniqueNamesAndIdsForComputationsAndInstructions() const;

  // Checks if this config has a list of entry parameters' HLO shardings for
  // SPMD.
  bool has_spmd_parameters_shardings() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_33(mht_33_v, 630, "", "./tensorflow/compiler/xla/service/hlo_module.h", "has_spmd_parameters_shardings");

    return spmd_parameters_shardings_.has_value();
  }

  // Getter and setter for the list of entry parameters' HLO shardings for SPMD.
  const std::vector<HloSharding>& spmd_parameters_shardings() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_34(mht_34_v, 638, "", "./tensorflow/compiler/xla/service/hlo_module.h", "spmd_parameters_shardings");

    CHECK(spmd_parameters_shardings_.has_value());
    return *spmd_parameters_shardings_;
  }
  void set_spmd_parameters_shardings(
      const std::vector<HloSharding>& shardings) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_35(mht_35_v, 646, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_spmd_parameters_shardings");

    spmd_parameters_shardings_ = shardings;
  }

  // Checks if this config has the entry computation output's HLO sharding for
  // SPMD.
  bool has_spmd_output_sharding() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_36(mht_36_v, 655, "", "./tensorflow/compiler/xla/service/hlo_module.h", "has_spmd_output_sharding");

    return spmd_output_sharding_.has_value();
  }

  // Getter and setter for the entry computation output's HLO shardings for
  // SPMD.
  const HloSharding& spmd_output_sharding() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_37(mht_37_v, 664, "", "./tensorflow/compiler/xla/service/hlo_module.h", "spmd_output_sharding");

    CHECK(spmd_output_sharding_.has_value());
    return *spmd_output_sharding_;
  }
  void set_spmd_output_sharding(const HloSharding& sharding) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_38(mht_38_v, 671, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_spmd_output_sharding");

    spmd_output_sharding_ = sharding;
  }

  // Add a program argument to be prefetched across programs.
  void AddCrossProgramPrefetch(int64_t parameter, const ShapeIndex& index) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_39(mht_39_v, 679, "", "./tensorflow/compiler/xla/service/hlo_module.h", "AddCrossProgramPrefetch");

    cross_program_prefetches_.emplace_back(parameter, index);
  }

  // Get the list of program arguments to be prefetch across programs.
  const absl::Span<const std::pair<int64_t, ShapeIndex>>
  CrossProgramPrefetches() const {
    return cross_program_prefetches_;
  }

  const HloModuleMetadata& metadata() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_40(mht_40_v, 692, "", "./tensorflow/compiler/xla/service/hlo_module.h", "metadata");
 return metadata_; }
  HloModuleMetadata* metadata() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_41(mht_41_v, 696, "", "./tensorflow/compiler/xla/service/hlo_module.h", "metadata");
 return &metadata_; }

  // Moves (not copies) metadata from this HloModule to `module`. To be used
  // in cases like HloModuleGroup::ReplaceModule when metadata should be
  // transferred out of a module before it's destroyed.
  void MoveMetadataToModule(HloModule* module) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_42(mht_42_v, 704, "", "./tensorflow/compiler/xla/service/hlo_module.h", "MoveMetadataToModule");

    module->metadata_ = std::move(metadata_);
  }

  uint64_t profile_handle() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_43(mht_43_v, 711, "", "./tensorflow/compiler/xla/service/hlo_module.h", "profile_handle");
 return profile_handle_; }

  void set_profile_handle(uint64_t profile_handle) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_44(mht_44_v, 716, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_profile_handle");

    profile_handle_ = profile_handle;
  }

  void add_profile_info(const HloModuleProto::ProfileInfo& profile_info) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_45(mht_45_v, 723, "", "./tensorflow/compiler/xla/service/hlo_module.h", "add_profile_info");

    profile_info_list_.push_back(profile_info);
  }

  void set_profile_info(
      const std::vector<HloModuleProto::ProfileInfo>& profile_info) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_46(mht_46_v, 731, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_profile_info");

    profile_info_list_ = profile_info;
  }

  const std::vector<HloModuleProto::ProfileInfo>& profile_info() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_47(mht_47_v, 738, "", "./tensorflow/compiler/xla/service/hlo_module.h", "profile_info");

    return profile_info_list_;
  }

  void set_relative_speedup(double relative_speedup) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_48(mht_48_v, 745, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_relative_speedup");

    relative_speedup_ = relative_speedup;
  }

  // Sets the **unoptimized** fingerprint for the module. This fingerprint is
  // prior to any optimizations.
  void set_autofdo_fingerprint(absl::string_view fingerprint) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("fingerprint: \"" + std::string(fingerprint.data(), fingerprint.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_49(mht_49_v, 755, "", "./tensorflow/compiler/xla/service/hlo_module.h", "set_autofdo_fingerprint");

    autofdo_fingerprint_ = std::string(fingerprint);
  }

  absl::string_view autofdo_fingerprint() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_moduleDTh mht_50(mht_50_v, 762, "", "./tensorflow/compiler/xla/service/hlo_module.h", "autofdo_fingerprint");
 return autofdo_fingerprint_; }

 private:
  HloComputation* AddComputationInternal(
      std::unique_ptr<HloComputation> computation, bool is_entry,
      bool uniquify_identifiers, bool preserve_entry_layouts);

  std::string name_;
  HloModuleConfig config_;
  HloComputation* entry_computation_ = nullptr;
  std::vector<std::unique_ptr<HloComputation>> computations_;

  // Random number generator engine to use when generating random numbers per
  // HloModule compilation.
  // TODO(b/25995601): Replace with better seed setting or dev/random for
  // where we don't need deterministic execution.
  mutable std::mt19937_64 rng_{42};
  mutable absl::Mutex rng_mutex_;

  // Unique name generator for computation and instruction names, which are
  // unique per module.
  NameUniquer computation_name_uniquer_{/*separator=*/"."};
  NameUniquer instruction_name_uniquer_{/*separator=*/"."};
  int next_unique_id_ = 0;

  // Used to keep track of the next unique module id that should be assigned.
  static std::atomic<int> next_unique_module_id_;
  // A unique id to label modules with.
  int unique_id_;

  // The HloSchedule of the module. The schedule if it exists contains a
  // sequential order of instructions for each non-fusion computation in the
  // module.
  absl::optional<HloSchedule> schedule_;

  // alias_config indicates the alias information of input/output buffers that
  // are expected from the module.
  HloInputOutputAliasConfig input_output_alias_config_;

  // Bindings for dynamic parameter mapping.
  DynamicParameterBinding dynamic_parameter_binding_;

  // The HLO shardings of the entry computation's parameters for
  // SPMD-partitioned programs.
  absl::optional<std::vector<HloSharding>> spmd_parameters_shardings_;

  // The HLO sharding of the entry computation's output (root) for
  // SPMD-partitioned programs.
  absl::optional<HloSharding> spmd_output_sharding_;

  // Arguments to be prefetched across programs.
  std::vector<std::pair<int64_t, ShapeIndex>> cross_program_prefetches_;

  // Metadata for this module, such as its canonical id and the HLO passes run.
  HloModuleMetadata metadata_;

  // True if the module contains dynamic computation.
  bool is_dynamic_ = false;

  // Optional compilation profile handle.
  uint64_t profile_handle_ = 0;

  // An array of ProfileInfo specifying what optimization profiles this module
  // contains, along with the relative speedups.
  std::vector<HloModuleProto::ProfileInfo> profile_info_list_;

  // Relative speedup of best config compared to default config.
  double relative_speedup_;

  // The unoptimized module fingerprint.
  std::string autofdo_fingerprint_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
