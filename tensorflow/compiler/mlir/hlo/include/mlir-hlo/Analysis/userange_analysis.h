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

#ifndef MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H
#define MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H
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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh() {
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


#include <vector>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {

/// Representation of an inclusive Interval for the Userange.
struct UseInterval {
  using Vector = SmallVector<UseInterval, 8>;

 public:
  /// UseInterval Constructor.
  UseInterval();
  /// Empty UseInterval Constructor.
  UseInterval(size_t start, size_t end) : start(start), end(end) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_0(mht_0_v, 205, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "UseInterval");
}

  /// Checks if the given UseInterval overlaps with this UseInterval.
  bool isOverlapping(const UseInterval &other) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_1(mht_1_v, 211, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "isOverlapping");

    return start <= other.end && end >= other.start;
  }

  /// Checks if the given UseInterval is contiguous with this UseInterval in
  /// terms of doubled Ids.
  /// For example: (0, 2) and (4, 6) are contiguous where (0, 2) and (5, 6) are
  ///              not.
  bool isContiguous(const UseInterval &other) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_2(mht_2_v, 222, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "isContiguous");

    return start <= other.end + 2 && end + 2 >= other.start;
  }

  /// Checks if the given position is inside this UseInterval.
  bool contains(size_t position) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_3(mht_3_v, 230, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "contains");

    return start <= position && end >= position;
  }

  /// Merges this UseInterval with the given UseInterval by updating start and
  /// end.
  bool mergeWith(const UseInterval &other) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_4(mht_4_v, 239, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "mergeWith");

    if (!isContiguous(other)) return false;
    start = std::min(start, other.start);
    end = std::max(end, other.end);
    return true;
  }

  /// Performs an interval subtraction => A = A - B.
  static void intervalSubtract(Vector &a, const Vector &b);

  /// Performs an interval intersection => A = A ^ B.
  static void intervalIntersect(Vector &a, const Vector &b);

  /// Performs an interval merge => A = A u B.
  /// Note: All overlapping and contiguous UseIntervals are merged.
  static void intervalMerge(Vector &a, const Vector &b);

  /// Merge the UseIntervals and erase overlapping and contiguouse UseIntervals
  /// of the UseInterval::Vector.
  static void mergeAndEraseContiguousIntervals(Vector &interval,
                                               UseInterval *iter,
                                               const UseInterval &toMerge);

  bool operator<(const UseInterval &other) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_5(mht_5_v, 265, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "operator<");
 return end < other.start; }

  bool operator>(const UseInterval &other) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_6(mht_6_v, 270, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "operator>");
 return start > other.end; }

  bool operator==(const UseInterval &other) const {
    return start == other.start && end == other.end;
  }

  /// The start of this UseInterval.
  size_t start;

  /// The end of this UseInterval.
  size_t end;
};

/// Represents an analysis for computing the useranges of all alloc values
/// inside a given function operation. The analysis uses liveness information to
/// compute intervals starting at the first and ending with the last use of
/// every alloc value.
class UserangeAnalysis {
 public:
  using UsePosition = std::pair<size_t, Operation *>;
  using UsePositionList = std::vector<UsePosition>;

  UserangeAnalysis(Operation *op,
                   const bufferization::BufferPlacementAllocs &allocs,
                   const BufferViewFlowAnalysis &aliases);

  /// Returns the index of the first operation that uses the given value or an
  /// empty Optional if the value has no uses.
  llvm::Optional<size_t> getFirstUseIndex(Value value) const {
    auto &intervals = useIntervalMap.find(value)->second;
    if (intervals.empty()) return llvm::None;
    return intervals.begin()->start;
  }

  /// Returns the UseInterval::Vector of the given value.
  llvm::Optional<const UseInterval::Vector *> getUserangeInterval(
      Value value) const {
    auto intervals = useIntervalMap.find(value);
    if (intervals == useIntervalMap.end()) return llvm::None;
    return &intervals->second;
  }

  /// Returns an UsePositionList* of the given value or an empty Optional
  /// if the value has no uses.
  llvm::Optional<const UsePositionList *> getUserangePositions(
      Value value) const {
    auto usePosition = usePositionMap.find(value);
    if (usePosition == usePositionMap.end() || usePosition->second.empty())
      return llvm::None;
    return &usePosition->second;
  }

  /// Returns the operation associated with a given Id.
  Operation *getOperation(size_t id) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSuserange_analysisDTh mht_7(mht_7_v, 326, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/userange_analysis.h", "getOperation");
 return operations[unwrapId(id)]; };

  /// Computes the doubled Id for the given value inside the operation based on
  /// the program sequence. If the value has only read effects, the returning ID
  /// will be even, otherwise odd.
  size_t computeId(Value v, Operation *op) const;

  /// Checks if the use intervals of the given values interfere.
  bool rangesInterfere(Value itemA, Value itemB) const;

  /// Merges the userange of itemB into the userange of itemA.
  void unionRanges(Value itemA, Value itemB);

  /// Merges listB into listA, sorts the result and removes all duplicates.
  static void mergeUsePositions(UsePositionList &listA,
                                const UsePositionList &listB);

  /// Dumps the liveness information to the given stream.
  void dump(raw_ostream &os);

 private:
  using ValueSetT = BufferViewFlowAnalysis::ValueSetT;
  using OperationListT = Liveness::OperationListT;

  /// Builds an UseInterval::Vector corresponding to the given OperationList.
  UseInterval::Vector computeInterval(
      Value value, const Liveness::OperationListT &operationList);

  /// Computes the UsePositions of the given Value, sorts and inserts them into
  /// the usePositionMap.
  void computeUsePositions(Value v);

  /// Checks each operand within the operation for its memory effects and
  /// separates them into read and write.
  void gatherMemoryEffects(Operation *op);

  /// Computes the doubled Id back to the OperationId.
  size_t unwrapId(size_t id) const;

  /// Maps each Operation to a unique ID according to the program sequence.
  DenseMap<Operation *, size_t> operationIds;

  /// Stores all operations according to the program sequence.
  std::vector<Operation *> operations;

  /// Maps a value to its UseInterval::Vector.
  DenseMap<Value, UseInterval::Vector> useIntervalMap;

  /// Maps an Operation to a pair of read and write Operands.
  DenseMap<Operation *, std::pair<SmallPtrSet<Value, 2>, SmallPtrSet<Value, 2>>>
      opReadWriteMap;

  /// Maps aliasValues to their use ranges. This is necessary to prevent
  /// recomputations of the use range intervals of the aliases.
  DenseMap<Value, OperationListT> aliasUseranges;

  /// Maps a Value to a UsePostionList which contains all uses of the Value and
  /// their userange position.
  DenseMap<Value, UsePositionList> usePositionMap;

  /// Cache the alias lists for all values to avoid recomputation.
  BufferViewFlowAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness liveness;
};

}  // namespace mlir

#endif  // MLIR_HLO_ANALYSIS_USERANGE_ANALYSIS_H
