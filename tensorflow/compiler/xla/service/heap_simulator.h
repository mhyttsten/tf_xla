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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh() {
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


#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/buffer_value_containers.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Forward declare classes defined below.
template <typename BufferType>
class HeapAlgorithm;
template <typename BufferType>
class NoFragmentationStatsHeap;

// HeapSimulator assigns buffer offsets by running a simulation of a regular
// memory heap with Alloc and Free calls.  It only works for completely
// sequential instruction sequences.  Unlike regular heaps, we have the
// advantage that the sequence of Alloc and Free calls is known up-front; we
// don't need to return the assignment of buffer offsets until the very end.
class HeapSimulator {
 public:
  // Chunk represents a contiguous piece of memory.  Each BufferValue will be
  // associated with a chunk in the assignment result.
  struct Chunk {
    int64_t offset;
    int64_t size;

    int64_t chunk_end() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_0(mht_0_v, 231, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "chunk_end");
 return offset + size; }

    bool OverlapsWith(Chunk other_chunk) const;

    bool operator==(const Chunk& other) const {
      return offset == other.offset && size == other.size;
    }
  };

  template <typename BufferType>
  struct HeapResult {
    // Returns the updated heap size if `chunk` is added to the heap.
    int64_t UpdatedHeapSize(const Chunk& chunk) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "UpdatedHeapSize");

      return std::max(heap_size, chunk.chunk_end());
    }

    // The assignment of buffers to chunks.
    absl::flat_hash_map<const BufferType*, Chunk> chunk_map;

    // The total size in bytes of the heap, containing all assigned chunks.
    int64_t heap_size = 0;
  };
  // Result represents the result of the heap simulation.
  template <typename BufferType>
  struct Result {
    // Heap results.
    std::vector<HeapResult<BufferType>> heap_results;

    // The total size in bytes of the heaps.
    // heap_size == sum([hr.heap_size for hr in heap_results]).
    int64_t heap_size = 0;

    // The total size in bytes of heap fragmentation.
    int64_t fragmentation_size = 0;

    // A trace of heap simulation events.
    HeapSimulatorTrace debug_trace;
  };

  // The different options to be passed to the Run() APIs.
  struct Options {
    Options()
        : may_reuse_operand_buffers(true),
          alloc_constants(false),
          buffers_to_assign(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_2(mht_2_v, 281, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "Options");
}

    // Whether a buffer about to be Free()-ed, can be recycled for a new born
    // one, hence collapsing Free()+Alloc() calls (default true).
    bool may_reuse_operand_buffers;
    // Whether to issue Alloc() and Free() calls for constants (default false).
    bool alloc_constants;
    // If 'buffers_to_assign' is provided, only those buffers are assigned
    // offsets, otherwise all buffers defined by the instructions are assigned.
    const absl::flat_hash_set<const HloValue*>* buffers_to_assign;
  };

  // Returns the minimum memory required to compute an HLO module where all
  // computations have been scheduled (represented by the given
  // schedule), assuming no fragmentation.
  static StatusOr<int64_t> MinimumMemoryForModule(
      const HloSchedule& schedule,
      const LogicalBuffer::SizeFunction& size_function);

  // Returns the minimum memory required to compute the given computation,
  // assuming no fragmentation.
  static StatusOr<int64_t> MinimumMemoryForComputation(
      const HloComputation& computation, const HloInstructionSequence& sequence,
      const HloAliasAnalysis& alias_analysis,
      const LogicalBuffer::SizeFunction& size_function,
      const absl::flat_hash_map<const HloComputation*, int64_t>*
          memory_by_computation = nullptr);

  static StatusOr<int64_t> MinimumMemoryForComputation(
      const HloComputation& computation, const HloInstructionSequence& sequence,
      const HloAliasAnalysis& alias_analysis,
      const LogicalBuffer::SizeFunction& size_function,
      const HloSchedule* schedule);

  // Run the heap simulation with the given algorithm, assuming the given
  // schedule, which must contain a topologically-consistent total
  // ordering of all instructions within each computation. The result is invalid
  // if instructions are not run in exactly this sequence.
  //
  // Running heap simulation on the whole module tends to save memory, compared
  // to running on a per-computation basis, since we can re-use buffer space for
  // called sub-computations.
  //
  static StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloModule& module, const HloSchedule& schedule,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options());

  // Same as above, but runs on a single computation. The 'instruction_sequence'
  // must contain a topologically-consistent total ordering of all instructions
  // in the computation. The result is invalid if instructions are not run in
  // exactly this sequence.
  static StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloComputation& computation,
      const HloInstructionSequence& instruction_sequence,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options(),
      const absl::flat_hash_map<const HloComputation*, int64_t>*
          memory_by_computation = nullptr);

  // Same as above, but runs on with a schedule that covers all nested
  // computations.
  static StatusOr<Result<HloValue>> Run(
      std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
      const HloComputation& computation,
      const HloInstructionSequence& instruction_sequence,
      const HloAliasAnalysis& alias_analysis,
      const BufferValue::SizeFunction& size_fn, const HloSchedule* schedule,
      const Options& options = Options());

 private:
  // If 'schedule' is non-null, it is used to find kCall and kWhile
  // sub-computations, and the heap simulation for those sub-computations will
  // be run recursively. I.e. the simulation is run over the whole module.
  HeapSimulator(std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
                const BufferValue::SizeFunction& size_fn,
                const Options& options, const HloSchedule* schedule = nullptr,
                const absl::flat_hash_map<const HloComputation*, int64_t>*
                    memory_by_computation = nullptr);
  ~HeapSimulator();

  Status RunComputation(const HloComputation& computation,
                        const HloInstructionSequence& instruction_sequence,
                        const HloAliasAnalysis& alias_analysis,
                        HloLiveRange* live_range);

  bool IgnoreBuffer(const HloValue* buffer) const;
  void Alloc(const HloValue* buffer, const HloInstruction* instruction);
  void Free(const HloValue* buffer, const HloInstruction* instruction);
  // ShareBuffer indicates that a new buffer is defined and it has to be the
  // same address as the shared one.
  void ShareBuffer(const HloValue* buffer, const HloValue* shared,
                   const HloInstruction* instruction);

  // Returns true if:
  //  Two buffers belong to the same shared group.
  //  Eight of the buffer has no shared group assigned.
  bool InSameSharedGroup(const HloValue* left, const HloValue* right);
  Result<HloValue> Finish();

  void FillDebugTrace(HeapSimulatorTrace::Event::Kind kind,
                      const HloValue* buffer, const HloInstruction* instruction,
                      const HloValue* share_with_canonical);

  // Counterintuitive: the algorithm_ itself can be a NoFragmentationStatsHeap,
  // in which case we are calculating the same allocs/frees twice in the
  // simulation.
  const std::unique_ptr<NoFragmentationStatsHeap<HloValue>>
      no_fragmentation_stats_;
  const std::unique_ptr<HeapAlgorithm<HloValue>> algorithm_;
  const BufferValue::SizeFunction size_fn_;
  const Options options_;
  // schedule_ is set by buffer assignment, and memory_by_computation_ is
  // set by hlo scheduling. Then, in RunComputation, we check both in order to
  // handle subcomputations. It would be good to unify the handling of
  // subcomputations, but it's not clear how.
  const HloSchedule* schedule_;
  const absl::flat_hash_map<const HloComputation*, int64_t>*
      memory_by_computation_;

  // Hold some sets for error-checking the sequence of Alloc and Free calls.
  absl::flat_hash_set<const HloValue*> allocated_buffers_;
  absl::flat_hash_set<const HloValue*> freed_buffers_;

  // Debugging information filled in while the heap simulator runs.
  HeapSimulatorTrace debug_trace_;
};

// Abstract base class describing a heap simulation algorithm that assigns
// offsets to buffers.  A sequence of Alloc / Free calls will be made, with the
// same semantics as a regular memory heap.  Finish will be called at the end to
// collect the simulation results.
template <typename BufferType>
class HeapAlgorithm {
 public:
  using Chunk = HeapSimulator::Chunk;
  using Result = HeapSimulator::Result<BufferType>;
  using HeapResult = HeapSimulator::HeapResult<BufferType>;

  virtual ~HeapAlgorithm() = default;

  // Alloc allocates a buffer of 'size' bytes.
  virtual void Alloc(const BufferType* buffer, int64_t size) = 0;

  // Takes memory usage of subcomputations into account when calculating the
  // memory usage of a computation. Currently, we don't handle buffer aliasing
  // between computations entirely correctly. We are careful to not double count
  // for the output buffers of whiles/conds/calls. But we don't take into
  // account other aliases, such as for the while init. A more thorough solution
  // would require something like BufferAssignment::BuildColocatedBufferSets.
  // TODO(b/65835246):
  // Since TuplePointsToAnalysis is being replaced with a module-aware alias
  // analysis, it's not worth making major changes to HeapSimulator now.
  virtual void AccountForSubcomputationMemory(
      const HloInstruction* instruction,
      // The total number of bytes allocated by instruction.
      int64_t alloc_size_by_instruction,
      const absl::flat_hash_map<const HloComputation*, int64_t>&
          memory_by_computation) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_3(mht_3_v, 446, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "AccountForSubcomputationMemory");
}

  // Free de-allocates a previously allocated buffer.
  virtual void Free(const BufferType* buffer, int64_t size) = 0;

  // Indicates that a buffer has to be collocated with another buffer. In
  // addition to Alloc and Free, the heap simulator exposes a concept of buffer
  // sharing.  When ShareBuffer is called, instead of allocating new space for
  // the buffer, it associates the buffer with a previously allocated (or
  // shared) buffer.  Each group of mutually-shared buffers points to a single
  // SharedGroup instance, which is a shared control block.
  virtual void ShareWith(const BufferType* buffer, const BufferType* share_with,
                         int64_t size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_4(mht_4_v, 461, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "ShareWith");

    Alloc(buffer, size);
  }

  // Finish collects the buffer offset assignment results.  Finish may only be
  // called once, after all Alloc and Free calls.
  virtual Result Finish() = 0;
};

// NoFragmentationStatsHeap computes the heap size assuming no fragmentation;
// this is the absolute minimum size for a given instruction sequence.  The
// result.chunk_map returned in Finish is always empty, since we only collect
// stats, and don't actually compute chunk assignments.
template <typename BufferType>
class NoFragmentationStatsHeap : public HeapAlgorithm<BufferType> {
 public:
  using Result = HeapSimulator::Result<BufferType>;

  NoFragmentationStatsHeap() = default;
  ~NoFragmentationStatsHeap() override = default;

  void Alloc(const BufferType* buffer, int64_t size) override;

  void AccountForSubcomputationMemory(
      const HloInstruction* instruction, int64_t alloc_size_by_instruction,
      const absl::flat_hash_map<const HloComputation*, int64_t>&
          memory_by_computation) override;

  void Free(const BufferType* buffer, int64_t size) override;

  Result Finish() override;

 private:
  int64_t current_heap_size_ = 0;
  int64_t max_heap_size_ = 0;
};

// Node in BufferIntervalTree that stores the alloc and free times of a buffer,
// and the chunk assigned to it.
struct BufferIntervalTreeNode {
  // Alloc time.
  int64_t start;
  // Free time.
  int64_t end;
  // Maximum free time of all nodes in the subtree where this node is the root.
  int64_t subtree_end;
  // Allocated chunk for the buffer.
  HeapSimulator::Chunk chunk;
  // Left child.
  BufferIntervalTreeNode* left;
  // Right child.
  BufferIntervalTreeNode* right;
  // parent
  BufferIntervalTreeNode* parent;
};

// An interval tree that can query buffers overlapping in time.
class BufferIntervalTree {
 public:
  using Chunk = HeapSimulator::Chunk;
  // Adds a buffer to the interval tree, with the time interval and allocated
  // chunk specified.
  void Add(int64_t start, int64_t end, const Chunk& chunk);

  // Remove the interval from the tree. Returns true if the chunk is removed.
  bool Remove(int64_t start, int64_t end, const Chunk& chunk);

  // Returns vector of allocated chunks that overlap with the given time
  // interval.
  std::vector<Chunk> ChunksOverlappingInTime(int64_t start, int64_t end) const;

  BufferIntervalTreeNode* GetRoot() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_5(mht_5_v, 535, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "GetRoot");
 return root_; }

 private:
  BufferIntervalTreeNode* root_ = nullptr;
  std::list<BufferIntervalTreeNode> node_storage_;
};

// GlobalDecreasingSizeBestFitHeap collects the live intervals of all buffers,
// then allocates them in decreasing spatial or temporal size regardless of the
// alloc/free time. It internally tracks the allocated buffers and their live
// intervals; when allocating a buffer, it finds the best-fit free chunk during
// its live interval.
template <typename BufferType>
class GlobalDecreasingSizeBestFitHeap : public HeapAlgorithm<BufferType> {
 public:
  using HeapResult = HeapSimulator::HeapResult<BufferType>;
  using Result = HeapSimulator::Result<BufferType>;
  using Chunk = HeapSimulator::Chunk;

  enum Type {
    kSpatial = 0,
    kTemporal,
  };

  // BufferInterval stores a buffer's size and time interval.
  struct BufferInterval {
    const BufferType* buffer;
    int64_t size;
    // Alloc time of the buffer.
    int64_t start;
    // Free time of the buffer.
    int64_t end;

    // Colocation buffers that need to be collocated with this one.
    absl::InlinedVector<const BufferType*, 2> colocations;

    // True if this buffer needs an allocation. False if it is collocated with
    // other buffer.
    bool need_allocation;
  };

  // Comparison function that is used to store buffer intervals.
  using BufferIntervalCompare =
      std::function<bool(const BufferInterval&, const BufferInterval&)>;

  explicit GlobalDecreasingSizeBestFitHeap(int64_t alignment,
                                           Type type = kSpatial);
  ~GlobalDecreasingSizeBestFitHeap() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_6(mht_6_v, 585, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "~GlobalDecreasingSizeBestFitHeap");
}

  void Alloc(const BufferType* buffer, int64_t size) override;
  void Free(const BufferType* buffer, int64_t size) override;

  void ShareWith(const BufferType* buffer, const BufferType* share_with,
                 int64_t size) override;

  Result Finish() override;

  // Return a BufferIntervalCompare function that sort by spatial size. We don't
  // look at co-locates as they should have the same size.
  static BufferIntervalCompare GetSpatialBufferIntervalCompare();

 protected:
  // Returns the buffer intervals sorted according to buffer_interval_compare_.
  std::vector<BufferInterval> GetSortedBufferIntervals() const;

  // These two methods below are exposed to other heap algorithms that inherit
  // from this class. The Finish() method tries to find a candidate chunk for
  // each BufferInterval, after calling GetSortedBufferIntervals. If a
  // non-negative preferred_offset is provided, FindChunkCandidate attempts
  // finding a chunk at this offset. The Finish() method can then call
  // CommitChunk to associate the chunk with the BufferInterval, if the final
  // heap size is within the limits.
  Chunk FindChunkCandidate(const BufferInterval& buffer_interval,
                           int64_t preferred_offset = -1) const;
  void CommitChunk(const BufferInterval& buffer_interval, Chunk chunk);

  // Adds the buffer and the chunk to the result chunk map.
  virtual void AddToChunkMap(const BufferType* buffer, Chunk chunk);

  // Return a BufferIntervalCompare function that sorts by live ranges.  A live
  // range is defined by the range between the start of the first buffer and the
  // end of the last co-located buffer.  There could be "holes" in the live
  // ranges of each co-located buffers, but in this heuristics we think they are
  // contiguous.
  BufferIntervalCompare GetTemporalBufferIntervalCompare() const;

  absl::flat_hash_map<const BufferType*, BufferInterval> buffer_intervals_;
  HeapResult result_;
  BufferIntervalCompare buffer_interval_compare_;
  BufferIntervalTree interval_tree_;

 private:
  int64_t alignment_;

  // The current time represented as an integer. It increments by 1 at each
  // Alloc or Free call.
  int64_t current_time_ = 0;

  // Returns all transitive colocated buffers of this buffer interval. I.e., If
  // a buffer A is colocated with B and B is colocated with C, this function
  // returns all three of them.
  absl::flat_hash_set<const BufferType*> GetTransitiveColocations(
      const BufferInterval& interval) const;
};

// This class implements an algorithm that will produce multiple heaps, where
// each heap size is constrained by a given limit. Note that the constraint is
// soft, meaning that a valid heap result is generated even if there are some
// buffer sizes larger than the given constraint size.
//
// Pseudocode:
//   while( `buffers` is not empty ) {
//     create a new heap `h`
//     for (each buffer `buf` in `buffers` in the size-decreasing order) {
//       if (buf.size() is larger than the heap size limit &&
//           `h` is empty) {
//         h.place(buf)
//         buffers.remove(buf)
//       } else if (placing `buf` into `h` does not violate size
//           constraint) {
//         h.place(buf)
//         buffers.remove(buf)
//       }
//     }
//   }
class ConstrainedGlobalDecreasingSizeBestFitHeap
    : public GlobalDecreasingSizeBestFitHeap<HloValue> {
 public:
  explicit ConstrainedGlobalDecreasingSizeBestFitHeap(
      uint64_t size_limit_per_heap, int64_t alignment, Type type = kSpatial)
      : GlobalDecreasingSizeBestFitHeap<HloValue>(alignment, type),
        size_limit_per_heap_(size_limit_per_heap) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_7(mht_7_v, 672, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "ConstrainedGlobalDecreasingSizeBestFitHeap");
}
  ~ConstrainedGlobalDecreasingSizeBestFitHeap() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_8(mht_8_v, 676, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "~ConstrainedGlobalDecreasingSizeBestFitHeap");
}

  Result Finish() override;

 private:
  uint64_t size_limit_per_heap_;
};

// A heap algorithm that chooses the best results from other algorithms added to
// it.
template <typename BufferType>
class ChooseBestHeapAlgorithm : public HeapAlgorithm<BufferType> {
 public:
  using Result = HeapSimulator::Result<BufferType>;

  ChooseBestHeapAlgorithm(
      std::unique_ptr<std::vector<std::unique_ptr<HeapAlgorithm<BufferType>>>>
          algorithms)
      : algorithms_(std::move(*algorithms)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_9(mht_9_v, 697, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "ChooseBestHeapAlgorithm");
}
  ~ChooseBestHeapAlgorithm() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_10(mht_10_v, 701, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "~ChooseBestHeapAlgorithm");
}

  void Alloc(const BufferType* buffer, int64_t size) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_11(mht_11_v, 706, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "Alloc");

    for (auto& algorithm : algorithms_) {
      algorithm->Alloc(buffer, size);
    }
  }

  void ShareWith(const BufferType* buffer, const BufferType* share_with,
                 int64_t size) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_12(mht_12_v, 716, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "ShareWith");

    for (auto& algorithm : algorithms_) {
      algorithm->ShareWith(buffer, share_with, size);
    }
  }

  void Free(const BufferType* buffer, int64_t size) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSheap_simulatorDTh mht_13(mht_13_v, 725, "", "./tensorflow/compiler/xla/service/heap_simulator.h", "Free");

    for (auto& algorithm : algorithms_) {
      algorithm->Free(buffer, size);
    }
  }

  Result Finish() override;

 private:
  std::vector<std::unique_ptr<HeapAlgorithm<BufferType>>> algorithms_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_
