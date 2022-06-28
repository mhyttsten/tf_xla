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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh() {
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


#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Walk the call graph of the HLO module and place each computation into either
// thread_local_computations or global_computations depending upon whether the
// computation requires thread-local allocations or global allocations. The
// elements in thread_local_computations and global_computations are in post
// order (if computation A has an instruction which calls computation B, then A
// will appear after B in the vector).
Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations);

// This class abstracts an allocation of contiguous memory which can hold the
// values described by LogicalBuffers. Each LogicalBuffer occupies a sub-range
// of the allocation, represented by a Slice. A single BufferAllocation may hold
// LogicalBuffers with disjoint liveness, which may have overlapping Slices. A
// single BufferAllocation may also hold LogicalBuffers with overlapping
// liveness, which must have disjoint Slices.
//
// The abstraction includes information required by the backends for allocation,
// use, and deallocation of the buffer. This includes the LogicalBuffers which
// are held in this allocation through the execution of the computation.
class BufferAllocation {
 public:
  // Holds a unique identifier for each allocation. Values are assigned
  // contiguously and can be used as array indexes.
  using Index = int64_t;

  BufferAllocation(Index index, int64_t size, LogicalBuffer::Color color)
      : index_(index), size_(size), color_(color) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_0(mht_0_v, 242, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "BufferAllocation");
}
  ~BufferAllocation() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "~BufferAllocation");
}

  // Returns the index of this allocation.
  Index index() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_2(mht_2_v, 252, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "index");
 return index_; }

  // Whether this allocation is used in a parallel calling context such as
  // inside of a map or reduce computation. Such allocations need to be thread
  // local.
  bool is_thread_local() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_3(mht_3_v, 260, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_thread_local");
 return is_thread_local_; }
  void set_is_thread_local(bool is_thread_local) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_4(mht_4_v, 264, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_is_thread_local");

    is_thread_local_ = is_thread_local;
  }

  // Whether this allocation can be used by more than one logical buffer.
  bool is_reusable() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_5(mht_5_v, 272, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_reusable");

    // We do not reuse thread-local buffers for now, because they are
    // dynamically allocated and their lifetimes are hard to compute.
    //
    // TODO(b/34669761): Don't reuse tuple buffers because the GPU backend
    // assumes longer buffer liveness than indicated by the analysis.
    return !is_thread_local() && !is_tuple();
  }

  // Whether this allocation is readonly i.e. backed by memory we cannot write
  // to.
  bool is_readonly() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_6(mht_6_v, 286, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_readonly");

    // Entry parameters are generally readonly, except when they are aliased
    // with any output.
    return (is_entry_computation_parameter() &&
            !is_parameter_aliased_with_output_) ||
           is_constant();
  }

  bool is_tuple() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_7(mht_7_v, 297, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_tuple");
 return is_tuple_; }
  void set_is_tuple(bool is_tuple) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_8(mht_8_v, 301, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_is_tuple");
 is_tuple_ = is_tuple; }

  // Whether this allocation holds a LogicalBuffer from a parameter of the entry
  // computation. These buffers have lifetimes which may be longer than the
  // XLA computation.
  bool is_entry_computation_parameter() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_entry_computation_parameter");

    return is_entry_computation_parameter_;
  }

  // Whether this allocation holds a constant.  On the CPU and GPU backends
  // constant allocations are not allocated dynamically, instead we resolve
  // references to these buffer allocations to a global in the readonly section
  // of the binary.
  bool is_constant() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_10(mht_10_v, 320, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "is_constant");
 return is_constant_; }

  // If this allocation holds a Buffer from a parameter of the entry
  // computation, this methods returns the parameter number. CHECKs otherwise.
  int64_t parameter_number() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_11(mht_11_v, 327, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "parameter_number");

    CHECK(is_entry_computation_parameter_);
    return parameter_number_;
  }

  // If this allocation is for a parameter of the entry computation, this
  // function returns which subshape of the parameter the allocation is for.
  const ShapeIndex& param_shape_index() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_12(mht_12_v, 337, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "param_shape_index");

    CHECK(is_entry_computation_parameter_);
    return param_shape_index_;
  }

  // Returns whether this allocation is assigned a LogicalBuffer which may
  // be live out of the entry computation.
  bool maybe_live_out() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_13(mht_13_v, 347, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "maybe_live_out");
 return maybe_live_out_; }

  void set_maybe_live_out(bool value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_14(mht_14_v, 352, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_maybe_live_out");
 maybe_live_out_ = value; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64_t size() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_15(mht_15_v, 359, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "size");
 return size_; }

  // Returns the color of the allocation. Only logical buffers with a matching
  // color can reside in this allocation.
  LogicalBuffer::Color color() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_16(mht_16_v, 366, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "color");
 return color_; }

  struct OffsetSize {
    int64_t offset = 0;
    int64_t size = 0;
  };

  // Access to the logical buffers assigned to this allocation, and their
  // associated logical offsets and sizes.
  const absl::flat_hash_map<const HloValue*, OffsetSize>& assigned_buffers()
      const {
    return assigned_buffers_;
  }

  // A Slice represents a contiguous portion of a memory allocation. It is used
  // to identify the memory range that a LogicalBuffer corresponds to.
  class Slice {
   public:
    Slice() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_17(mht_17_v, 387, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "Slice");
}
    Slice(const BufferAllocation* allocation, int64_t offset, int64_t size)
        : allocation_(allocation), offset_(offset), size_(size) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_18(mht_18_v, 392, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "Slice");
}

    const BufferAllocation* allocation() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_19(mht_19_v, 397, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "allocation");
 return allocation_; }
    Index index() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_20(mht_20_v, 401, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "index");
 return allocation_->index(); }
    int64_t offset() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_21(mht_21_v, 405, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "offset");
 return offset_; }
    int64_t size() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_22(mht_22_v, 409, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "size");
 return size_; }

    bool operator==(const Slice& other) const {
      return index() == other.index() && offset_ == other.offset_ &&
             size_ == other.size_;
    }
    bool operator!=(const Slice& other) const { return !(*this == other); }
    bool operator<(const Slice& other) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_23(mht_23_v, 419, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "operator<");

      if (index() != other.index()) return index() < other.index();
      if (offset_ != other.offset_) return offset_ < other.offset_;
      return size_ < other.size_;
    }

    // Returns true iff this slice's memory range has a non-empty intersection
    // with the other slice's memory range.
    bool OverlapsWith(const Slice& other) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_24(mht_24_v, 430, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "OverlapsWith");

      const int64_t end = offset_ + size_;
      const int64_t other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ < other_end &&
             end > other.offset_;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Slice& s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_25(mht_25_v, 441, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "AbslHashValue");

      return H::combine(std::move(h), s.index(), s.offset(), s.size());
    }

    std::string ToString() const;

   private:
    const BufferAllocation* allocation_ = nullptr;
    int64_t offset_ = 0;
    int64_t size_ = 0;
  };

  // GetSlice returns the Slice of contiguous memory that holds the value
  // described by the given 'buffer'.
  // REQUIRES: 'buffer' must be assigned to this allocation.
  Slice GetSlice(const HloValue& buffer) const;

  std::string ToString() const;
  BufferAllocationProto ToProto() const;

  // Whether the buffer is a parameter to or live out of the entry computation.
  bool IsInputOrOutput() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_26(mht_26_v, 465, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "IsInputOrOutput");

    return is_entry_computation_parameter() || maybe_live_out();
  }

  // Whether the buffer is a temporary buffer allocated before
  // Executable::ExecuteOnStream.
  bool IsPreallocatedTempBuffer() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_27(mht_27_v, 474, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "IsPreallocatedTempBuffer");

    // Parameters do not need temporary buffers.
    return !is_entry_computation_parameter() &&
           // LogicalBuffers that maybe pointed to by the output should live out
           // of the computation.
           !maybe_live_out() &&
           // Thread-local buffers are allocated using `alloca`s.
           !is_thread_local() &&
           // Constant buffers are allocated as global values.
           !is_constant();
  }

  // Add a heap trace which was used to assign slices to logical buffers in this
  // allocation. A single BufferAllocation may include multiple heap traces
  // in the case of the temporary block where there is a heap trace per
  // computation.
  void AddHeapTrace(const HeapSimulatorTrace& heap_trace) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_28(mht_28_v, 493, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "AddHeapTrace");

    heap_traces_.push_back(heap_trace);
    heap_traces_.back().set_buffer_allocation_index(index());
  }

  // Return the set of heap traces used to assign slices to logical buffers in
  // this allocation.
  const std::vector<HeapSimulatorTrace> HeapTraces() const {
    return heap_traces_;
  }

  // Returns the LogicalBuffers which are live at the point of peak memory usage
  // for this allocation. The point of peak memory usage is the point at which
  // the total size of all live logical buffers is maximal. If peak memory is
  // reached at multiple points, the set of logical buffers live at the earliest
  // maximal point is returned. The vector is stably sorted by
  // BufferValue::Index.
  const std::vector<const HloValue*>& PeakMemoryLogicalBuffers() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_29(mht_29_v, 513, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "PeakMemoryLogicalBuffers");

    return peak_buffers_;
  }

  // Get the number of bytes lost to fragmentation. This is equal to the
  // difference between the size of the allocation and the size of the maximal
  // live set.
  int64_t fragmentation_bytes() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_30(mht_30_v, 523, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "fragmentation_bytes");
 return fragmentation_bytes_; }

  bool operator==(const BufferAllocation& other) const {
    return index_ == other.index_;
  }
  bool operator!=(const BufferAllocation& other) const {
    return !(*this == other);
  }
  bool operator<(const BufferAllocation& other) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_31(mht_31_v, 534, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "operator<");

    return index() < other.index();
  }

  void set_entry_computation_parameter(int64_t parameter_number,
                                       ShapeIndex param_shape_index,
                                       bool parameter_aliased_with_output) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_32(mht_32_v, 543, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_entry_computation_parameter");

    is_entry_computation_parameter_ = true;
    is_parameter_aliased_with_output_ = parameter_aliased_with_output;
    parameter_number_ = parameter_number;
    param_shape_index_ = std::move(param_shape_index);
  }

  void set_constant(bool is_constant) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_33(mht_33_v, 553, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_constant");
 is_constant_ = is_constant; }

 private:
  // Only BufferAssigner and BufferAssignment can modify BufferAllocation.
  friend class BufferAssigner;
  friend class BufferAssignment;

  // Adds a LogicalBuffer to the set assigned to this buffer.
  void AddAssignment(const HloValue& buffer, int64_t offset, int64_t size);

  void set_index(Index index) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_34(mht_34_v, 566, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_index");
 index_ = index; }
  void set_size(int64_t size) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_35(mht_35_v, 570, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "set_size");
 size_ = size; }

  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64_t size_;

  // Whether this buffer needs to be thread-local.
  bool is_thread_local_ = false;

  // Whether this buffer holds a tuple.
  bool is_tuple_ = false;

  // Color of the allocation.
  LogicalBuffer::Color color_;

  // Whether this allocation holds an entry computation parameter. Entry
  // computation parameters are special because they have lifetimes which may
  // outlast the computation.
  bool is_entry_computation_parameter_ = false;

  // Whether this entry computation parameter is aliased with output.
  bool is_parameter_aliased_with_output_ = false;

  // If this allocation holds an entry computation parameter, this field
  // indicates the index (starting from 0) of the parameter.
  int64_t parameter_number_ = 0;

  // If this buffer is for an entry computation parameter, which subshape of the
  // parameter is it for?
  ShapeIndex param_shape_index_;

  // Whether the allocation contains a LogicalBuffer which may be live-out of
  // the entry computation. Note that this flag is conservatively computed by
  // TuplePointsToAnalysis.  That is, an allocation marked `maybe_live_out_`
  // might not actually escape.
  bool maybe_live_out_ = false;

  // See comment on the is_constant() accessor.
  bool is_constant_ = false;

  // Mapping from the set of buffers assigned to this allocation to their
  // logical offsets and sizes.
  absl::flat_hash_map<const HloValue*, OffsetSize> assigned_buffers_;

  int64_t fragmentation_bytes_ = 0;
  std::vector<HeapSimulatorTrace> heap_traces_;

  // Set of buffers live at the point of peak memory usage for this allocation.
  std::vector<const HloValue*> peak_buffers_;
};

// Add stream operators for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation& s);
std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s);

// This class encapsulates an assignment of the LogicalBuffers in an XLA
// module to a set of BufferAllocations.
class BufferAssignment {
 public:
  // Returns the vector containing all buffer allocations in this assignment.
  const std::vector<BufferAllocation>& Allocations() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_36(mht_36_v, 635, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "Allocations");

    return allocations_;
  }

  // This is similar to copying Allocations(), but since it's moved out, it
  // preserves the addresses. Since BufferAllocation::Slice keeps a
  // BufferAllocation*, and some backends keep BufferAllocation::Slice in
  // xla::Executables, migrating off the use of addresses can be hard.
  std::vector<BufferAllocation> ReleaseAllocations() {
    return std::move(allocations_);
  }

  // Returns the total size allocation holding all temporary buffers.
  int64_t temp_allocation_total_size() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_37(mht_37_v, 651, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "temp_allocation_total_size");

    return temp_allocation_total_size_;
  }

  uint64_t multiheap_size_constraint_per_heap() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_38(mht_38_v, 658, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "multiheap_size_constraint_per_heap");

    return multiheap_size_constraint_per_heap_;
  }

  // Returns whether the given buffer has been assigned an allocation.
  bool HasAllocation(const HloValue& value) const;

  bool HasAllocation(const HloBuffer& buffer) const;

  // Returns the allocation that a particular LogicalBuffer has been assigned
  // to. CHECKs if buffer has not been assigned an allocation.
  const BufferAllocation& GetAssignedAllocation(const HloValue& value) const;

  const BufferAllocation& GetAssignedAllocation(
      const HloBuffer& hlo_buffer) const;

  // Returns the allocation with the given index. CHECKs if no allocation exists
  // with the given index.
  const BufferAllocation& GetAllocation(BufferAllocation::Index index) const;

  // Returns the allocation with the given instruction and shape index. nullptr
  // if no allocation exists.
  const BufferAllocation* GetInstructionAllocation(
      const HloInstruction* hlo, const ShapeIndex& shape_index) const;

  // Builds and returns a vector containing the slices which might contain the
  // subvalue at the given index of given instruction.
  std::set<BufferAllocation::Slice> GetAllSlices(
      const HloInstruction* instruction, const ShapeIndex& index) const;

  // Convenience function which returns whether the buffer of the
  // instruction at the given index is assigned an allocation.
  bool HasAllocationAt(const HloInstruction* instruction,
                       const ShapeIndex& index) const;

  // Convenience function which returns whether the top-level buffer of the
  // instruction (index == {}) is assigned an allocation.
  bool HasTopLevelAllocation(const HloInstruction* instruction) const;

  // Convenience function which returns the unique slice containing the buffer
  // at the given index of the given instruction. If a slice is not assigned or
  // the slice cannot be determined at compile time then an error is returned.
  StatusOr<BufferAllocation::Slice> GetUniqueSlice(
      const HloInstruction* instruction, const ShapeIndex& index) const;
  // Like GetUniqueSlice but fixes the index to the top-level of the shape
  // (index = {}).
  StatusOr<BufferAllocation::Slice> GetUniqueTopLevelSlice(
      const HloInstruction* instruction) const;
  // Like GetUniqueTopLevelSlice but returns the slice for the output of the
  // entry computation of the HLO module (ie, the result of the XLA
  // computation).
  StatusOr<BufferAllocation::Slice> GetUniqueTopLevelOutputSlice() const;

  // Returns the set BufferValues which may be the source of the value at the
  // given index and instruction.
  const std::vector<const HloValue*>& GetSourceBuffers(
      const HloInstruction* instruction, const ShapeIndex& index) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_39(mht_39_v, 717, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "GetSourceBuffers");

    return dataflow_analysis().GetValueSet(instruction, index).values();
  }

  // Returns true if 'hlo_a{shape_index_a}' and 'hlo_b{shape_index_b}'
  // share the same BufferAllocation::Slice.
  // Returns false otherwise.
  // REQUIRES: BufferAssignment assigned allocations to both instructions.
  bool SharesSliceAtIndex(const HloInstruction* hlo_a,
                          const ShapeIndex& shape_index_a,
                          const HloInstruction* hlo_b,
                          const ShapeIndex& shape_index_b) const;

  // Returns true if the top-level buffers of hlo_a and hlo_b are the same.
  // REQUIRES: HasTopLevelAllocation(hlo_a) && HasTopLevelAllocation(hlo_b).
  bool SharesTopLevelSlice(const HloInstruction* hlo_a,
                           const HloInstruction* hlo_b) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_40(mht_40_v, 736, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "SharesTopLevelSlice");

    return SharesSliceAtIndex(hlo_a, {}, hlo_b, {});
  }

  // Returns true if hlo_a and hlo_b both have at least one buffer assigned for
  // their top-level and each of their nested shape indices, and if hlo_a's
  // buffers are all different from hlo_b's buffers.
  bool HaveDisjointSlices(const HloInstruction* hlo_a,
                          const HloInstruction* hlo_b) const;

  const HloDataflowAnalysis& dataflow_analysis() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_41(mht_41_v, 749, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "dataflow_analysis");

    return alias_analysis_->dataflow_analysis();
  }

  HloAliasAnalysis& alias_analysis() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_42(mht_42_v, 756, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "alias_analysis");
 return *alias_analysis_; }

  const HloOrdering& hlo_ordering() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_43(mht_43_v, 761, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "hlo_ordering");
 return *hlo_ordering_; }

  // Returns the HloLiveRange object used to construct this assignment.
  const HloLiveRange& hlo_live_range() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_44(mht_44_v, 767, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "hlo_live_range");
 return *hlo_live_range_; }

  std::string ToString() const;
  // Verbose string tailored to debugging OOMs, includes the Hlo op metadata for
  // every buffer associated with each allocation.
  std::string ToVerboseString() const;
  std::string BufferInfoString() const;
  BufferAssignmentProto ToProto() const;

  // Statistics for the assignment.  Values initialized to -1 are not always
  // collected; fragmentation is only collected for instructions that have a
  // sequential total ordering.
  struct Stats {
    int64_t parameter_allocation_count = 0;
    int64_t parameter_allocation_bytes = 0;
    int64_t constant_allocation_count = 0;
    int64_t constant_allocation_bytes = 0;
    int64_t maybe_live_out_allocation_count = 0;
    int64_t maybe_live_out_allocation_bytes = 0;
    int64_t preallocated_temp_allocation_count = 0;
    int64_t preallocated_temp_allocation_bytes = 0;
    int64_t preallocated_temp_fragmentation_bytes = -1;
    int64_t total_allocation_count = 0;
    int64_t total_allocation_bytes = 0;
    int64_t total_fragmentation_bytes = -1;

    std::string ToString() const;
  };
  const Stats& GetStats() const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_45(mht_45_v, 798, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "GetStats");
 return stats_; }

 private:
  // Only BufferAssigner can build or modify BufferAssignments.
  friend class BufferAssigner;

  BufferAssignment(const HloModule* module,
                   std::unique_ptr<HloOrdering> hlo_ordering,
                   BufferValue::SizeFunction buffer_size,
                   LogicalBuffer::AlignmentFunction color_alignment,
                   std::unique_ptr<HloAliasAnalysis> alias_analysis,
                   std::unique_ptr<HloLiveRange> hlo_live_range)
      : module_(module),
        hlo_ordering_(std::move(hlo_ordering)),
        buffer_size_(std::move(buffer_size)),
        color_alignment_(std::move(color_alignment)),
        alias_analysis_(std::move(alias_analysis)),
        hlo_live_range_(std::move(hlo_live_range)) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_46(mht_46_v, 818, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "BufferAssignment");

    int32_t raw_value = module->config()
                            .debug_options()
                            .xla_multiheap_size_constraint_per_heap();
    // -1 means no constraint.
    multiheap_size_constraint_per_heap_ =
        (raw_value == -1) ? UINT64_MAX : raw_value;
  }

  // Creates and returns a new BufferAllocation, with no assigned
  // LogicalBuffers. Ownership is maintained internally.
  BufferAllocation* NewEmptyAllocation(int64_t size,
                                       LogicalBuffer::Color color);

  // Helper that calls NewEmptyAllocation and AddAssignment in one call,
  // creating an allocation containing a single LogicalBuffer.
  BufferAllocation* NewAllocation(const HloBuffer& buffer, int64_t size);

  // Adds a LogicalBuffer to the set assigned to the given allocation.
  void AddAssignment(BufferAllocation* allocation, const HloBuffer& buffer,
                     int64_t offset, int64_t size);

  void AddAssignment(BufferAllocation* allocation, const HloValue& value,
                     int64_t offset, int64_t size);

  // Returns the HloModule used to construct this assignment.
  const HloModule& module() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_47(mht_47_v, 847, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "module");
 return *module_; }

  // Mutable accessors for allocations.
  BufferAllocation* GetMutableAssignedAllocation(const HloBuffer& buffer);
  BufferAllocation* GetMutableAllocation(BufferAllocation::Index index);

  int64_t HloBufferSize(const HloBuffer& buffer) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_48(mht_48_v, 856, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "HloBufferSize");

    int64_t result = buffer_size_(*buffer.values()[0]);
    for (const HloValue* value : buffer.values()) {
      DCHECK_EQ(result, buffer_size_(*value));
    }
    return result;
  }

  // Combines allocations of temporary buffers into one big BufferAllocation.
  void CombineTempAllocations();

  // Computes stats for the assignment, to be retrieved by GetStats.
  Status ComputeSummaryStats();

  // The vector of buffer allocations. Indexed by BufferAllocation::Index.
  std::vector<BufferAllocation> allocations_;

  // The total size of all temporary buffers.
  int64_t temp_allocation_total_size_ = 0;

  uint64_t multiheap_size_constraint_per_heap_;

  // Maps Buffers to the index of the BufferAllocation which holds the buffer.
  absl::flat_hash_map<const HloValue*, BufferAllocation::Index>
      allocation_index_for_value_;

  const HloModule* module_;

  const std::unique_ptr<HloOrdering> hlo_ordering_;

  // Function which returns the buffer size for a given logical buffer (shape).
  BufferValue::SizeFunction buffer_size_;

  // Function which returns the alignment for a given logical buffer color.
  LogicalBuffer::AlignmentFunction color_alignment_;

  std::unique_ptr<HloAliasAnalysis> alias_analysis_;

  std::unique_ptr<HloLiveRange> hlo_live_range_;

  Stats stats_;

  BufferAssignment(const BufferAssignment&) = delete;
  BufferAssignment& operator=(const BufferAssignment&) = delete;
};

// A class which constructs a buffer assignment.
class BufferAssigner {
 public:
  using Colorer = std::function<Status(HloAliasAnalysis*, const HloOrdering&)>;
  using MustNotLiveOut =
      std::function<bool(const HloInstruction*, const ShapeIndex&)>;

  static Colorer DefaultColorer() {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_49(mht_49_v, 912, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "DefaultColorer");

    return [](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_50(mht_50_v, 916, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "lambda");

      for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
        const HloPosition& defining_position = value->defining_position();
        if (defining_position.shape().has_layout()) {
          value->set_color(BufferValue::Color(
              defining_position.shape().layout().memory_space()));
        } else {
          value->set_color(BufferValue::Color(0));
        }
      }
      return Status::OK();
    };
  }

  // Returns false if a buffer cannot be assigned to given allocation.

  // Build and return a BufferAssignment for the given module. The given
  // HloOrdering is used to determine buffer liveness. buffer_size and
  // color_alignment are functions which returns the size and alignment of a
  // LogicalBuffer. If preset_assignments is provided, those pre-set assignment
  // offsets will be used. The caller guarantees that those assignments are
  // valid and they do not overwrite each other.
  static StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      BufferValue::SizeFunction buffer_size,
      LogicalBuffer::AlignmentFunction color_alignment,
      bool allocate_buffers_for_constants = false,
      Colorer colorer = DefaultColorer(),
      absl::optional<MustNotLiveOut> must_not_live_out = absl::nullopt,
      HloDataflowAnalysis::CanShareBuffer can_share_buffer = nullptr,
      std::unique_ptr<memory_space_assignment::PresetAssignments>
          preset_assignments = {});

 private:
  BufferAssigner(bool allocate_buffers_for_constants, Colorer colorer,
                 absl::optional<MustNotLiveOut> must_not_live_out,
                 std::unique_ptr<memory_space_assignment::PresetAssignments>
                     preset_assignments)
      : allocate_buffers_for_constants_(allocate_buffers_for_constants),
        colorer_(colorer),
        must_not_live_out_(must_not_live_out),
        preset_assignments_(std::move(preset_assignments)) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbuffer_assignmentDTh mht_51(mht_51_v, 960, "", "./tensorflow/compiler/xla/service/buffer_assignment.h", "BufferAssigner");
}
  virtual ~BufferAssigner() = default;

  // Create a buffer assignment.
  StatusOr<std::unique_ptr<BufferAssignment>> CreateAssignment(
      const HloModule* module, std::unique_ptr<HloOrdering> hlo_ordering,
      BufferValue::SizeFunction buffer_size,
      LogicalBuffer::AlignmentFunction color_alignment,
      HloDataflowAnalysis::CanShareBuffer can_share_buffer);

  // Assigns buffers to the instructions in the given computations. "assignment"
  // is modified to reflect the new buffer assignments. If is_thread_local is
  // true, then all assigned buffers have the is_thread_local flag set to
  // true.
  Status AssignBuffersForComputations(
      const std::vector<const HloComputation*>& computations,
      bool is_thread_local,
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloValue*>>*
          buffers_to_assign_sequentially,
      BufferAssignment* assignment);

  // Returns true if buffer's live range interferences with buffer2's.
  bool LiveRangeInterferes(const HloValue* buffer1, const HloValue* buffer2,
                           BufferAssignment* assignment);

  // Assigns pre-set assignments, if provided. These assignments will be added
  // to assigned_buffers and skip buffer allocation.
  Status AssignPresetBuffers(
      absl::flat_hash_set<const HloBuffer*>* assigned_buffers,
      BufferAssignment* assignment);

  // Assigns a single hlo buffer to an HLO allocation.
  Status AssignSingleHloBuffer(
      const HloBuffer* hlo_buffer, bool is_thread_local,
      absl::flat_hash_map<const HloComputation*,
                          absl::flat_hash_set<const HloValue*>>*
          buffers_to_assign_sequentially,
      std::vector<BufferAllocation::Index>* allocation_indices,
      BufferAssignment* assignment);

  // Assigns 'buffers_to_assign_sequentially' using heap simulation, assuming
  // the HLO instructions will be executed in the sequential order given by
  // assignment->liveness().hlo_ordering().SequentialOrder. If
  // 'run_whole_module_heap_simulation' is true, the heap simulation will be run
  // assuming all global computations are sequentially ordered.
  Status AssignBuffersWithSequentialOrdering(
      const absl::flat_hash_map<const HloComputation*,
                                absl::flat_hash_set<const HloValue*>>&
          buffers_to_assign_sequentially,
      bool run_whole_module_heap_simulation, BufferAssignment* assignment);

  // Uses the results of the heap simulator to create a single allocation, with
  // LogicalBuffers packed to specific offsets.
  void AssignBuffersFromHeapSimulator(
      const HeapSimulator::Result<HloValue>& result,
      BufferAssignment* assignment, LogicalBuffer::Color color);

  // Tries to assign the given instruction to the given buffer. Returns if the
  // assignment was successful.
  bool MaybeAssignBuffer(BufferAllocation* allocation, const HloBuffer& buffer,
                         BufferAssignment* assignment);

  // Split a set of buffers into several sets, each of which contains buffers
  // colored with the same color.
  absl::flat_hash_map<LogicalBuffer::Color,
                      absl::flat_hash_set<const HloValue*>>
  SplitBuffersByColor(const absl::flat_hash_set<const HloValue*>& buffers);

  // If true, allocate buffers for constant instructions.
  bool allocate_buffers_for_constants_;

  // Functor used to assign colors to newly allocated logical buffers.
  Colorer colorer_;

  // An optional function that returns true if the given instruction can't live
  // out of a computation.
  absl::optional<MustNotLiveOut> must_not_live_out_;

  // Description of any buffer offsets that are already set by an earlier pass.
  std::unique_ptr<memory_space_assignment::PresetAssignments>
      preset_assignments_;

  BufferAssigner(const BufferAssigner&) = delete;
  BufferAssigner& operator=(const BufferAssigner&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BUFFER_ASSIGNMENT_H_
