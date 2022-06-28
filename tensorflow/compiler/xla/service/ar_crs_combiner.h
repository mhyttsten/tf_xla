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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// When the HLO graph contains a cross-module AllReduce (N separate AllReduce
// ops that share the same channel_id for MPMD partitioning, or 1 AllReduce op
// for SPMD partitioning), followed by some simple linear operations, followed
// by a cross-replica AllReduce (also known as cross-replica sum, or CRS), we
// can combine the CMAR and the CRAR, to use an efficient AllReduce
// implementation that fully utilizes the interconnect bandwidth.
//
// Such sequences appear in spatially partitioned models (either MPMD or SPMD).
// This pass must run right after spatial partitioning, when the code is still
// in a single HLO module.
//
// The steps are:
// 1) Find CMARs followed by simple ops followed by CRARs.
// 2) Group CMARs by channel_id. They must all be rewritten. For SPMD
//    partitioning, there will only be a single CMAR for each channel_id.
// 3) Prove that the CMAR patterns in each core produce the same result.
// 4) Eliminate the CMAR, and if it feeds an addition/subtraction, divide the
//    other operand by the number of spatial partitions.
// 5) Turn the CRAR into an all-core AllReduce.
//
// The pass also handles the case where multiple CMARs lead to the same CRAR,
// and eliminates all CMARs. This graph:
//
//        Y
//        |
//  X   CMAR_2   Z
//  |      \    /
// CMAR_1     +
//    \     /
//       +
//       |
//     CRAR
//
// gets rewritten to:
//
//           Z   num_partitions
//            \  /
//       Y    div
//        \   /
//    X     +
//     \   /
//       +
//       |
//  all-core AR
//
class ArCrsCombiner : public HloModulePass {
 public:
  ArCrsCombiner(int num_spatial_partitions, bool spmd_partition)
      : num_spatial_partitions_(num_spatial_partitions),
        spmd_partition_(spmd_partition) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh mht_0(mht_0_v, 246, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.h", "ArCrsCombiner");
}
  absl::string_view name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh mht_1(mht_1_v, 250, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.h", "name");
 return "ar-crs-combiner"; }
  StatusOr<bool> Run(HloModule* module) override;

  // Helper method to allow testing of InstructionsComputeSameValue.
  static bool TestInstructionsComputeSameValue(HloInstruction* i1,
                                               HloInstruction* i2);

 private:
  // We used this struct because multiple ARs could be paired with the same CRS.
  // In this case, we want to select the AR that is furthest from the CRS,
  // because it makes it easier to eliminate all ARs during RewriteGraph.
  struct ArCrsPair {
    HloInstruction* ar;
    HloInstruction* crs;
    // The length of the path from AR to CRS in the HLO graph.
    int64_t distance;

    ArCrsPair(HloInstruction* all_reduce, HloInstruction* cross_replica_sum,
              int64_t dist)
        : ar(all_reduce), crs(cross_replica_sum), distance(dist) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh mht_2(mht_2_v, 272, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.h", "ArCrsPair");
}

    std::string ToString() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSar_crs_combinerDTh mht_3(mht_3_v, 277, "", "./tensorflow/compiler/xla/service/ar_crs_combiner.h", "ToString");

      std::vector<std::string> pieces;
      pieces.push_back("(");
      HloInstruction* instruction = ar;
      while (instruction != crs) {
        pieces.push_back(instruction->name());
        pieces.push_back(",");
        instruction = instruction->users()[0];
      }
      pieces.push_back(instruction->name());
      pieces.push_back(")[id:");
      pieces.push_back(std::to_string(*(ar->channel_id())));
      pieces.push_back(",dist:");
      pieces.push_back(std::to_string(distance));
      pieces.push_back("]");
      return absl::StrJoin(pieces, "");
    }
  };

  absl::optional<ArCrsCombiner::ArCrsPair> MatchesArCrsPattern(
      HloInstruction* instruction);

  // If the passed instruction is a while parameter, and the while body is only
  // called by a single while instruction, return the while instruction.
  absl::optional<HloInstruction*> WhileFromBodyParameter(
      HloInstruction* instruction);

  // If the passed instruction is a parameter in one of the branch computations,
  // and the branch body is only called by a single instruction, return the
  // conditional instruction.
  absl::optional<HloInstruction*> ConditionalFromBodyParameter(
      HloInstruction* instruction);

  // Returns a vector of tuple instructions.
  // If all instructions that flow to "instruction" are tuples, return them.
  // Otherwise, return absl::nullopt. Returns an empty vector if the instruction
  // is already in the visited set.
  absl::optional<std::vector<HloInstruction*>> GetAllTuples(
      HloInstruction* instruction,
      absl::flat_hash_set<HloInstruction*>* visited);

  // Checks whether two different elements in the same tuple compute the same
  // value.
  bool TupleElementsComputeSameValue(
      HloInstruction* tuple_shaped_instruction, int64_t i1, int64_t i2,
      absl::flat_hash_map<int64_t, int64_t>* visited_pairs);

  // Returns whether the instructions i1 and i2 can be shown to evaluate to the
  // same value. Handling WHILE requires recursion, which may cause us to visit
  // the same instruction again. To avoid infinite loops, we pass a cache of
  // visited instruction pairs.
  bool InstructionsComputeSameValue(
      HloInstruction* i1, HloInstruction* i2,
      absl::flat_hash_map<int64_t, int64_t>* visited_pairs);

  // Populates all_reduce_map_.
  void GroupAllReducesById(HloModule* module);

  // Looks at each AllReduce group in all_reduce_map_, and keeps only the
  // groups for which it's safe to move the AllReduce later in the HLO graph.
  Status KeepProvablyEqualInstructionGroupsMPMD();

  // Same as above, but runs on SPMD partitioned module instead of MPMD.
  Status KeepProvablyEqualInstructionGroupsSPMD(HloModule* module);

  // Performs the graph rewrite that eliminates the early AllReduce and turns
  // the later CRS into an AllReduce.
  StatusOr<bool> RewriteGraph();

  int num_spatial_partitions_;

  // Run this combiner pass assuming the input module is an SPMD partitioned
  // module (as opposed to MPMD partitioned).
  //
  // The main difference between the two w.r.t. this pass is that there would be
  // N all-reduce ops for each channel in MPMD mode, whereas there is only 1
  // for each channel in SPMD mode. Also we use HloReplicationAnalysis for HLO
  // equivalence check in SPMD mode.
  bool spmd_partition_;

  // Map from all-reduce ids to the AR/CRS pairs.
  absl::flat_hash_map<int64_t, std::vector<ArCrsPair>> all_reduce_map_;

  // Map from a CRS instruction to the all-reduce ID of the AR paired with the
  // CRS. Sometimes, several ARs in the code could be paired with the same CRS.
  // We use this map to pick a single AR/CRS path to rewrite.
  absl::flat_hash_map<HloInstruction*, int64_t> crs_reserved_map_;

  std::unique_ptr<CallGraph> call_graph_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_AR_CRS_COMBINER_H_
