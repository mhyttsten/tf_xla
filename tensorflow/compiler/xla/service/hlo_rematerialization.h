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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh() {
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
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass which rematerializes instructions to reduce peak memory use, where
// memory use is defined as the total size of all live HLO instruction
// values. Parameters and constants are included in memory use estimates.
//
// CSE will undo the effects of this optimization and should not be run after
// this pass. In general, this pass should be run very late, immediately before
// code generation.
class HloRematerialization : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  using CompactShapeFunction = std::function<StatusOr<Shape>(const Shape&)>;

  // Helper struct that communicates the before / after sizes for the
  // rematerialization process.
  struct RematerializationSizes {
    int64_t before_bytes = -1;
    int64_t after_bytes = -1;
  };

  // Mode in which the rematerialization algorithm should be run.
  enum class RematerializationMode {
    kRecomputeOnly,        // Only consider the kCompress RematStrategy.
    kCompressOnly,         // Only consider the kRecompute RematStrategy.
    kRecomputeAndCompress  // Consider both kRecompute and kRemat.
  };

  // Enum to specify whether this rematerialization pass occurs before or after
  // multi-output fusion.
  enum class RematerializationPass {
    kPreFusion,  // Rematerialization pass before multi-output fusion.
    kPostFusion  // Rematerialization pass after multi-output fusion.
  };

  static Shape DefaultCompactShapeFunction(const Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh mht_0(mht_0_v, 235, "", "./tensorflow/compiler/xla/service/hlo_rematerialization.h", "DefaultCompactShapeFunction");
 return shape; }

  // Constructor parameters:
  //
  //   size_function: Function which returns the size in bytes of the top-level
  //     buffer of the given shape.
  //
  //   memory_limit_bytes: The threshold number of bytes to reduce memory use to
  //     via rematerialization. Size of aliased outputs should be subtracted
  //     from this.
  //
  //   sizes: Pointer to data structure which records the peak memory usage of
  //     the HLO module before/after rematerialization. Value are set during
  //     Run(). Can be nullptr.
  //
  //   compact_shape_function: Function which returns the compact form of a
  //   shape. If nullptr is provided, an default identity function is used.
  explicit HloRematerialization(
      const ShapeSizeFunction& size_function, int64_t memory_limit_bytes,
      RematerializationSizes* sizes, RematerializationPass pass_location,
      int block_size_limit, int block_rematerialization_factor,
      CompactShapeFunction compact_shape_function = nullptr,
      RematerializationMode mode = RematerializationMode::kRecomputeAndCompress,
      int64_t min_remat_size = 0)
      : size_function_(size_function),
        memory_limit_bytes_(memory_limit_bytes),
        sizes_(sizes),
        pass_location_(pass_location),
        block_size_limit_(block_size_limit),
        block_rematerialization_factor_(block_rematerialization_factor),
        compact_shape_function_(compact_shape_function == nullptr
                                    ? DefaultCompactShapeFunction
                                    : std::move(compact_shape_function)),
        mode_(mode),
        min_remat_size_(min_remat_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh mht_1(mht_1_v, 272, "", "./tensorflow/compiler/xla/service/hlo_rematerialization.h", "HloRematerialization");
}
  ~HloRematerialization() override = default;

  absl::string_view name() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh mht_2(mht_2_v, 278, "", "./tensorflow/compiler/xla/service/hlo_rematerialization.h", "name");
 return "rematerialization"; }

  // Get the next available channel id and increment count.
  int64_t NextChannelId() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_rematerializationDTh mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xla/service/hlo_rematerialization.h", "NextChannelId");
 return next_channel_id_++; }

  // Runs rematerialization on the given module. Returns whether the module was
  // changed. Requires that the module has a schedule set
  // (HloModule::has_schedule() is true) before running. Returns whether any
  // instructions were rematerialized. If memory use is already below the limit
  // specified in the constructor then no instructions are rematerialized and
  // false is returned.
  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Rematerializes instructions within the given computation. 'order' is the
  // order in which the computation's instructions will be emitted in the
  // backend. Rematerialized instructions will be added to the HLO computation
  // and inserted into 'order'.
  virtual StatusOr<bool> RematerializeComputation(HloComputation* computation,
                                                  HloSchedule* schedule,
                                                  int64_t memory_limit_bytes,
                                                  int64_t min_remat_size);

  // Computes and returns the peak memory used by the given computation. The
  // peak memory is the maximum total size of all live HLO instruction values at
  // any program point. 'order' is the order in which the HLO instructions will
  // be emitted which is used to determine lifespans of HLO values.
  StatusOr<int64_t> ComputePeakMemory(
      const HloComputation* computation,
      const HloInstructionSequence& order) const;

  // Returns the peak memory usage of the called computations for the given
  // instruction. Zero is returned if the instruction calls no computations.
  StatusOr<int64_t> CalledComputationsMemoryUsage(
      const HloInstruction* instruction) const;

  // Selects an algorithm to use for HLO scheduling.
  MemorySchedulerAlgorithm scheduler_algorithm_;

  // Function which computes the size of the top-level buffer of a shape.
  const ShapeSizeFunction size_function_;

  // The threshold number of bytes to reduce memory use to via
  // rematerialization.
  const int64_t memory_limit_bytes_;

  // Pointer to data structure which records the peak memory usage of the HLO
  // module before/after rematerialization
  RematerializationSizes* sizes_;

  // Specifies whether this rematerialization pass occurs before or after
  // multi-output fusion.
  RematerializationPass pass_location_;

  // Maximum number of consecutive instructions to consider for
  // rematerialization.
  int block_size_limit_;

  // Controls the amount of effort spent trying to find large blocks for
  // rematerialization. Larger values leads to longer compilation times in
  // return for potentially reduced memory consumption.
  int block_rematerialization_factor_ = 1;

  // Converts a shape into compact form, returns the same shape if a shape is
  // already considered compact.
  const CompactShapeFunction compact_shape_function_;

  // Call graph of the hlo_module.
  std::unique_ptr<CallGraph> call_graph_;

  // The peak memory usage of each computation. The map contains only those
  // computations called from sequential context
  // (CallContext::kSequential). These values are updated as rematerialization
  // occurs.
  absl::flat_hash_map<const HloComputation*, int64_t> computation_peak_memory_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  // Set of computations which have had rematerialization
  // applied. Rematerialization is only applied once per computation.
  absl::flat_hash_set<const HloComputation*> rematerialized_computations_;

  // Count of the total instructions rematerialized.
  int64_t instructions_rematerialized_ = 0;

  // Count of the net instructions added to the HLO module by
  // rematerialization. This can be different than instructions_rematerialized_
  // because some rematerializations are effectively moves in the HLO
  // schedule. In these cases, the rematerialization instruction replaces all
  // uses of the original instruction and the original instruction is
  // dead. Hence, no net instructions were added.
  int64_t net_instructions_added_ = 0;

  // Size of the largest block that has been rematerialized. This is actually an
  // upper bound (within a factor of 2) on the block size.
  int max_rematerialized_block_size_ = 0;

  RematerializationMode mode_;

  int64_t min_remat_size_;

  // Tracking available channel id numbers to use to apply to rematerialized
  // channel instructions
  int64_t next_channel_id_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REMATERIALIZATION_H_
