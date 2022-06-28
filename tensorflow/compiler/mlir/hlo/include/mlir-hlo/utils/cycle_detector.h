/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_UTILS_CYCLE_DETECTOR_H
#define MLIR_HLO_UTILS_CYCLE_DETECTOR_H
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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh() {
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

// -------------------------------------------------------------------

// This file contains a light version of GraphCycles implemented in
// tensorflow/compiler/jit/graphcycles/graphcycles.h
//
// We re-implement it here because we do not want to rely
// on TensorFlow data structures, and hence we can move
// corresponding passes to llvm repo. easily in case necessnary.

// --------------------------------------------------------------------

// This is a set data structure that provides a deterministic iteration order.
// The iteration order of elements only depends on the sequence of
// inserts/deletes, so as long as the inserts/deletes happen in the same
// sequence, the set will have the same iteration order.
//
// Assumes that T can be cheaply copied for simplicity.
template <typename T>
class OrderedSet {
 public:
  // Inserts `value` into the ordered set.  Returns true if the value was not
  // present in the set before the insertion.
  bool Insert(T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Insert");

    bool new_insertion =
        value_to_index_.insert({value, value_sequence_.size()}).second;
    if (new_insertion) {
      value_sequence_.push_back(value);
    }
    return new_insertion;
  }

  // Removes `value` from the set.  Assumes `value` is already present in the
  // set.
  void Erase(T value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Erase");

    auto it = value_to_index_.find(value);

    // Since we don't want to move values around in `value_sequence_` we swap
    // the value in the last position and with value to be deleted and then
    // pop_back.
    value_to_index_[value_sequence_.back()] = it->second;
    std::swap(value_sequence_[it->second], value_sequence_.back());
    value_sequence_.pop_back();
    value_to_index_.erase(it);
  }

  void Reserve(size_t new_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Reserve");

    value_to_index_.reserve(new_size);
    value_sequence_.reserve(new_size);
  }

  void Clear() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_3(mht_3_v, 254, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Clear");

    value_to_index_.clear();
    value_sequence_.clear();
  }

  bool Contains(T value) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_4(mht_4_v, 262, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Contains");
 return value_to_index_.count(value); }
  size_t Size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_5(mht_5_v, 266, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "Size");
 return value_sequence_.size(); }

  const std::vector<T>& GetSequence() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSutilsPScycle_detectorDTh mht_6(mht_6_v, 271, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/cycle_detector.h", "GetSequence");
 return value_sequence_; }

 private:
  // The stable order that we maintain through insertions and deletions.
  std::vector<T> value_sequence_;

  // Maps values to their indices in `value_sequence_`.
  llvm::DenseMap<T, int> value_to_index_;
};

// ---------------------------------------------------------------------

// GraphCycles detects the introduction of a cycle into a directed
// graph that is being built up incrementally.
//
// Nodes are identified by small integers.  It is not possible to
// record multiple edges with the same (source, destination) pair;
// requests to add an edge where one already exists are silently
// ignored.
//
// It is also not possible to introduce a cycle; an attempt to insert
// an edge that would introduce a cycle fails and returns false.
//
// GraphCycles uses no internal locking; calls into it should be
// serialized externally.

// Performance considerations:
//   Works well on sparse graphs, poorly on dense graphs.
//   Extra information is maintained incrementally to detect cycles quickly.
//   InsertEdge() is very fast when the edge already exists, and reasonably fast
//   otherwise.
//   FindPath() is linear in the size of the graph.
// The current implementation uses O(|V|+|E|) space.

class GraphCycles {
 public:
  explicit GraphCycles(int32_t num_nodes);
  ~GraphCycles();

  // Attempt to insert an edge from x to y.  If the
  // edge would introduce a cycle, return false without making any
  // changes. Otherwise add the edge and return true.
  bool InsertEdge(int32_t x, int32_t y);

  // Remove any edge that exists from x to y.
  void RemoveEdge(int32_t x, int32_t y);

  // Return whether there is an edge directly from x to y.
  bool HasEdge(int32_t x, int32_t y) const;

  // Contracts the edge from 'a' to node 'b', merging nodes 'a' and 'b'. One of
  // the nodes is removed from the graph, and edges to/from it are added to
  // the remaining one, which is returned. If contracting the edge would create
  // a cycle, does nothing and return no value.
  llvm::Optional<int32_t> ContractEdge(int32_t a, int32_t b);

  // Return whether dest_node `y` is reachable from source_node `x`
  // by following edges. This is non-thread-safe version.
  bool IsReachable(int32_t x, int32_t y);

  // Return a copy of the successors set. This is needed for code using the
  // collection while modifying the GraphCycles.
  std::vector<int32_t> SuccessorsCopy(int32_t node) const;

  // Returns all nodes in post order.
  //
  // If there is a path from X to Y then X appears after Y in the
  // returned vector.
  std::vector<int32_t> AllNodesInPostOrder() const;

  // ----------------------------------------------------
  struct Rep;

 private:
  GraphCycles(const GraphCycles&) = delete;
  GraphCycles& operator=(const GraphCycles&) = delete;

  Rep* rep_;  // opaque representation
};

}  // namespace mlir

#endif  // MLIR_HLO_UTILS_CYCLE_DETECTOR_H
