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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <list>

#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace {

/// Returns the length of an userange interval.
size_t computeUserangeSize(const UseInterval &interval) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "computeUserangeSize");

  return interval.end - interval.start + 1;
}

/// Compute the byte size of a given Value.
size_t computeByteSize(const Value &v) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "computeByteSize");

  auto type = v.getType().cast<ShapedType>();
  return type.getSizeInBits() / 8;
}

/// Compute the 64 byte alinged segments of a given Value.
size_t computeAlignedSegments(const Value &v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "computeAlignedSegments");

  size_t padding = 64;
  size_t bytes = computeByteSize(v);
  return std::ceil(bytes / (double)padding);
}

/// The buffer offset information.
struct AllocBufferOffset {
 public:
  AllocBufferOffset(Value source, size_t offset)
      : source(source), offset(offset) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "AllocBufferOffset");
}

  Value source;
  size_t offset;
};

/// Contains the information to create a new buffer, that is used to pack
/// other buffers.
struct PackedBuffer {
 public:
  PackedBuffer(size_t numSegments,
               std::vector<AllocBufferOffset> &packedBuffers)
      : numSegments(numSegments), allocBufferOffsets(packedBuffers) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_4(mht_4_v, 249, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "PackedBuffer");
}

  size_t numSegments;
  std::vector<AllocBufferOffset> allocBufferOffsets;
};

/// Contains the information about a buffers allocation for sorting and checking
/// if it fits into other buffers and vise versa.
/// This structure contains the allocation value, the first and last userangeid
/// of a buffer, the window id, the number of alligned 64 byte segments and all
/// userange intervals.
struct AllocationInfo {
 public:
  AllocationInfo(Value alloc, size_t allocUserangeId, size_t firstUse,
                 size_t lastUse, size_t numSegments, size_t windowId,
                 const UseInterval::Vector *userangeIntervals)
      : alloc(alloc),
        allocUserangeId(allocUserangeId),
        firstUse(firstUse),
        lastUse(lastUse),
        numSegments(numSegments),
        windowId(windowId),
        userangeIntervals(userangeIntervals) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_5(mht_5_v, 274, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "AllocationInfo");
}

  /// The allocation value.
  Value alloc;

  /// The id of allocation based on the Userange Analysis.
  size_t allocUserangeId;

  /// The first use of the buffer.
  size_t firstUse;

  /// The last use of the buffer based on the Userange Analysis.
  size_t lastUse;

  /// The number of 64 byte aligned segments of contigous memory.
  size_t numSegments;

  /// The window id of the allocation position.
  size_t windowId;

  /// The userange intervals of the buffer.
  const UseInterval::Vector *userangeIntervals;

  /// Compute the gaps of the alloc userange with the number of segments. The
  /// maxUserangeId is used to add a dummy gap from the last used id to the
  /// maxUserangeId. By default the maxUserangeId is zero and no gap is added.
  std::list<std::pair<UseInterval, size_t>> computeGaps(
      size_t maxUserangeId = 0) {
    std::list<std::pair<UseInterval, size_t>> gaps;

    // The previous gap ending, initially set to 0.
    size_t gapEnd = 0;

    for (const auto *useRangeIter = userangeIntervals->begin();
         useRangeIter < userangeIntervals->end(); ++useRangeIter) {
      // Add a gap if the end is not equal to the start.
      if (gapEnd < useRangeIter->start)
        gaps.emplace_back(UseInterval(gapEnd, useRangeIter->start - 1),
                          numSegments);
      gapEnd = useRangeIter->end + 1;
    }

    // Add a dummy gap behind the last use of the buffer.
    if (gapEnd < maxUserangeId) {
      gaps.emplace_back(UseInterval(gapEnd, maxUserangeId), numSegments);
    }

    return gaps;
  }

  /// Compute the userange size.
  size_t getUserangeSize() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_6(mht_6_v, 328, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "getUserangeSize");
 return lastUse - firstUse + 1; }
};

// Comparator to sort allocation informations by window id, userange and by
// number of memory segments.
class AllocInfoWinIdComparator {
 public:
  bool operator()(const AllocationInfo &a, const AllocationInfo &b) {
    if (a.windowId == b.windowId) {
      if (a.allocUserangeId == b.allocUserangeId)
        return a.numSegments > b.numSegments;
      return a.allocUserangeId > b.allocUserangeId;
    }
    return a.windowId < b.windowId;
  }
};

// Comparator to sort the allocation informations by number of segments.
class AllocInfoMemSizeCompare {
 public:
  bool operator()(const AllocationInfo &a, const AllocationInfo &b) {
    return a.numSegments > b.numSegments;
  }
};

/// This approach computes an allocation information list and sorts it by
/// a given comparator. From top to bottom the algortihm tries to fill userange
/// gaps with appropriate buffers behind it, to optimze the memory. It is a bin
/// packing approach.
template <typename CompareT>
class SortedPackingStrategy {
 public:
  using AllocInfoList = std::vector<AllocationInfo>;

 public:
  /// Constructs the Sorted Packing Strategy. The window size is used as sliding
  /// window size. Allocation userangepositions that are in the same range are
  /// mapped to the same window id. So the information of the allocation
  /// starting position is blured.
  SortedPackingStrategy(size_t windowSize, CompareT compare)
      : windowSize(windowSize), compare(compare) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_7(mht_7_v, 371, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "SortedPackingStrategy");
}

  /// Optimize the buffer allocations.
  void optimze(const mlir::bufferization::BufferPlacementAllocs &allocs,
               const UserangeAnalysis &userangeAnalysis,
               std::vector<PackedBuffer> &packedBuffers) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_8(mht_8_v, 379, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "optimze");

    AllocInfoList allocInfos;
    allocInfos.reserve(std::distance(allocs.begin(), allocs.end()));

    // Create allocInformations and store them in allocInfos.
    size_t maxUserangeId =
        computeAllocationInfos(allocInfos, userangeAnalysis, allocs);

    // Sort the allocation infos.
    std::sort(allocInfos.begin(), allocInfos.end(), compare);

    for (auto currentIter = allocInfos.begin(); currentIter != allocInfos.end();
         ++currentIter) {
      std::vector<AllocBufferOffset> allocBufferOffsets{
          AllocBufferOffset(currentIter->alloc, 0)};

      // Compute userange gaps.
      std::list<std::pair<UseInterval, size_t>> gaps =
          currentIter->computeGaps(maxUserangeId);

      if (gaps.empty()) continue;

      for (auto checkedAllocInfoIter = std::next(currentIter);
           checkedAllocInfoIter != allocInfos.end();) {
        // Check if a gap exists to pack the memory into.
        // If not continue.
        if (!findGapAndUpdate(gaps, allocBufferOffsets, *checkedAllocInfoIter,
                              *currentIter)) {
          ++checkedAllocInfoIter;
          continue;
        }
        checkedAllocInfoIter = allocInfos.erase(checkedAllocInfoIter);
      }
      // Add the current buffer offets to the packed infos.
      packedBuffers.emplace_back(currentIter->numSegments * 64,
                                 allocBufferOffsets);
    }
  }

 private:
  const size_t windowSize;
  const CompareT compare;

  /// We try to find an appropriate userange gap to pack the buffer into it.
  /// If we find one we update only the gaps and the buffer offset map.
  bool findGapAndUpdate(std::list<std::pair<UseInterval, size_t>> &gaps,
                        std::vector<AllocBufferOffset> &allocBufferOffsets,
                        const AllocationInfo &allocToPack,
                        const AllocationInfo &allocToPackInto) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_9(mht_9_v, 430, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "findGapAndUpdate");

    // Check if the buffer to pack into has enough memory.
    if (allocToPackInto.numSegments < allocToPack.numSegments) return false;
    for (auto gapIter = gaps.begin(); gapIter != gaps.end();) {
      // The list is sorted, so we can break here.
      if (gapIter->first.start > allocToPack.firstUse) break;

      // Checks if enough contiguous memory segments are free or if the current
      // gap is out of bounds.
      if (gapIter->second < allocToPack.numSegments ||
          allocToPack.firstUse < gapIter->first.start ||
          allocToPack.lastUse > gapIter->first.end) {
        ++gapIter;
        continue;
      }

      // Stores the packed buffer with the offset.
      allocBufferOffsets.emplace_back(
          allocToPack.alloc,
          (allocToPackInto.numSegments - gapIter->second) * 64);

      // Update gap segments, will removed later if no free contigous memory
      // exists. It is needed to split the interval, if not the full gap is
      // used.
      size_t freeContiguousMemory = gapIter->second;
      gapIter->second = freeContiguousMemory - allocToPack.numSegments;

      // Check if the gap must be splitted. If so, then the current gap must be
      // trimmed accordingly. Therefore, new gaps are created in front and after
      // the current gap.
      if (computeUserangeSize(gapIter->first) > allocToPack.getUserangeSize()) {
        size_t oldStart = gapIter->first.start;
        size_t oldEnd = gapIter->first.end;
        gapIter->first.end = allocToPack.lastUse;
        gapIter->first.start = allocToPack.firstUse;

        // Insert a new gap behind.
        if (allocToPack.lastUse < oldEnd)
          gaps.insert(
              std::next(gapIter),
              std::make_pair(UseInterval(allocToPack.lastUse + 1, oldEnd),
                             freeContiguousMemory));
        // Insert a new gap before.
        if (allocToPack.firstUse > oldStart)
          gaps.insert(
              gapIter,
              std::make_pair(UseInterval(oldStart, allocToPack.firstUse - 1),
                             freeContiguousMemory));
      }

      // If a gap interval has no free contiguous memory anymore, erease it from
      // list.
      if (gapIter->second <= 0) gapIter = gaps.erase(gapIter);

      return true;
    }
    return false;
  }

  /// Aggreagtes the allocation informations of the allocs and returns the
  /// maximal userange.
  size_t computeAllocationInfos(
      AllocInfoList &allocInfos, const UserangeAnalysis &userangeAnalysis,
      const mlir::bufferization::BufferPlacementAllocs &allocs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_10(mht_10_v, 496, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "computeAllocationInfos");

    // Create allocInformations and store them in allocInfos.
    size_t maxUserangeId = 0;

    for (auto &allocEntry : allocs) {
      Value v = std::get<0>(allocEntry);
      auto userangeIntervals = userangeAnalysis.getUserangeInterval(v);

      if (!userangeIntervals) continue;

      // Computes the userange id of the allocation.
      size_t allocUserangeId = userangeAnalysis.computeId(v, v.getDefiningOp());

      // Computes the last use of the allocated buffer.
      size_t lastUse = std::prev((*userangeIntervals.getValue()).end())->end;

      // Computes the first use of the allocated buffer.
      size_t firstUse = (*userangeIntervals.getValue()).begin()->start;

      // Computes the number of aligend segments of the buffer.
      size_t numSegments = computeAlignedSegments(v);
      maxUserangeId = std::max(maxUserangeId, lastUse);
      allocInfos.emplace_back(v, allocUserangeId, firstUse, lastUse,
                              numSegments, 0, userangeIntervals.getValue());
    }

    // If the window size is zero we need no sorting anymore.
    if (windowSize == 0) return maxUserangeId;
    // Sorts the allocation informations to compute the window id. The window id
    // is used to blur the userange starting position of an allocation.
    std::sort(allocInfos.begin(), allocInfos.end(),
              [](const AllocationInfo &a, const AllocationInfo &b) {
                return a.allocUserangeId < b.allocUserangeId;
              });

    // resize window id
    size_t windowId = 0;
    size_t lastAllocUserangeId = 0;
    for (auto &allocationInfo : allocInfos) {
      if (allocationInfo.allocUserangeId > lastAllocUserangeId + windowSize)
        ++windowId;

      lastAllocUserangeId = allocationInfo.allocUserangeId;
      allocationInfo.windowId = windowId;
    }
    return maxUserangeId;
  }
};

/// Pass to pack buffer together to optimize the memeory consumption and to
/// save allocation operations. A strategy must be passed as a template
/// argument.
class BufferPacking : bufferization::BufferPlacementTransformationBase {
 public:
  template <typename StrategyT>
  BufferPacking(Operation *op, StrategyT strategy)
      : BufferPlacementTransformationBase(op),
        userangeAnalysis(op, allocs, aliases),
        dominators(op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_11(mht_11_v, 557, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "BufferPacking");

    std::vector<PackedBuffer> packedBuffers;
    strategy.optimze(allocs, userangeAnalysis, packedBuffers);

    for (auto &packedBuffer : packedBuffers) {
      // Find common dominators.
      Block *block = findAllocationsDominator(packedBuffer.allocBufferOffsets);
      // Find alloc position operation.
      mlir::OpBuilder packBuilder(&(block->front()));
      auto location = block->front().getLoc();
      auto memrefType =
          MemRefType::get({static_cast<int64_t>(packedBuffer.numSegments)},
                          packBuilder.getIntegerType(8));
      Value targetBuffer =
          packBuilder.create<memref::AllocOp>(location, memrefType);

      for (auto &packInfo : packedBuffer.allocBufferOffsets) {
        Value currentAlloc = packInfo.source;
        size_t offset = packInfo.offset;
        Operation *viewDefOp = currentAlloc.getDefiningOp();
        Location loc = viewDefOp->getLoc();
        mlir::OpBuilder viewBuilder(viewDefOp);

        // Create a arithmetic ConstantOp with the aligned offset.
        Value constantOp = viewBuilder.create<mlir::arith::ConstantOp>(
            loc, viewBuilder.getIndexType(),
            viewBuilder.getIntegerAttr(viewBuilder.getIndexType(), offset));

        // Store the operands for the ViewOp.
        SmallVector<Value, 4> newOperands{targetBuffer};
        newOperands.push_back(constantOp);

        auto shape = currentAlloc.getType().cast<MemRefType>();

        // Create a ViewOp with the shape of the old alloc and use the created
        // packed alloc and the constant for the operands.
        Value viewOp =
            viewBuilder.create<memref::ViewOp>(loc, shape, newOperands);

        // Replace all old allocs references with the created ViewOp and
        // afterwards remove the old allocs.
        currentAlloc.replaceAllUsesWith(viewOp);
        viewDefOp->erase();
      }
    }
  }

 private:
  UserangeAnalysis userangeAnalysis;
  /// The current dominance info.
  DominanceInfo dominators;

  /// Find the block that dominates all buffer allocations.
  Block *findAllocationsDominator(
      const std::vector<AllocBufferOffset> &packingInfos) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_12(mht_12_v, 614, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "findAllocationsDominator");

    SmallPtrSet<Value, 16> allocValues;
    for (auto &packInfo : packingInfos) {
      allocValues.insert(packInfo.source);
    }

    // Find common dominators.
    return findCommonDominator(packingInfos.begin()->source, allocValues,
                               dominators);
  }
};

/// Tries to pack allocated buffer together to save allocation operations and
/// memory. The window size is used as sliding window size. Allocation
/// userangepoitions that are in the same range are mapped to the same window
/// id. The information of the allocation starting position is blured.
struct BufferPackingPass : public BufferPackingBase<BufferPackingPass> {
  explicit BufferPackingPass(unsigned windowSize) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_13(mht_13_v, 634, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "BufferPackingPass");

    this->window_size_ = windowSize;
  }

  void runOnOperation() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_14(mht_14_v, 641, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "runOnOperation");

    if (window_size_ == 0) {
      SortedPackingStrategy<AllocInfoMemSizeCompare> strategy(
          window_size_, AllocInfoMemSizeCompare());
      BufferPacking packing(getOperation(), strategy);
    } else {
      SortedPackingStrategy<AllocInfoWinIdComparator> strategy(
          window_size_, AllocInfoWinIdComparator());
      BufferPacking packing(getOperation(), strategy);
    }
  }
};

/// Pass to find all allocations and to compute memory usage.
struct MemoryCountPass : MemoryCountBase<MemoryCountPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSlibPSTransformsPSbuffer_packingDTcc mht_15(mht_15_v, 659, "", "./tensorflow/compiler/mlir/hlo/lib/Transforms/buffer_packing.cc", "runOnOperation");

    Operation *op = getOperation();
    std::vector<Value> allocs;
    op->walk([&](MemoryEffectOpInterface opInterface) {
      // Try to find a single allocation result.
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      opInterface.getEffects(effects);

      SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
      llvm::copy_if(
          effects, std::back_inserter(allocateResultEffects),
          [=](MemoryEffects::EffectInstance &it) {
            Value value = it.getValue();
            return isa<MemoryEffects::Allocate>(it.getEffect()) && value &&
                   value.isa<OpResult>() &&
                   it.getResource() !=
                       SideEffects::AutomaticAllocationScopeResource::get();
          });

      if (allocateResultEffects.size() != 1) return;
      // Insert allocation.
      allocs.push_back(allocateResultEffects[0].getValue());
    });
    auto output = mlir::hlo::computeMemory(allocs);
    llvm::outs() << "Memory Count Pass:\n"
                 << output.first << ";" << output.second << "\n";
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createBufferPackingPass(
    unsigned window_size) {
  return std::make_unique<BufferPackingPass>(window_size);
}

std::unique_ptr<OperationPass<FuncOp>> createMemoryCountPass() {
  return std::make_unique<MemoryCountPass>();
}

}  // namespace mlir
