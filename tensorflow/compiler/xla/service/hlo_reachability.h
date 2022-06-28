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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REACHABILITY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REACHABILITY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh() {
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


#include <cstdio>
#include <list>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// A class for representing reachability between HloInstructions.
//
// It has an adjacency matrix and it is up to the user of the class to set the
// adjacency matrix such that it represents reachability, i.e. such that it is
// transitive. That the graph be transitive is thus not an invariant of this
// class, but it is required for the name of the class and its methods to make
// sense.
class HloReachabilityMap {
 public:
  // An opaque index that clients can use to make repeated operations for the
  // same instruction faster, by calling GetIndex once for the instruction,
  // and then calling the variants of other interfaces that take Index arguments
  // rather than HloInstruction* arguments.
  struct Index {
   public:
    bool operator==(Index other) const { return v == other.v; }
    bool operator!=(Index other) const { return v != other.v; }

   private:
    friend class HloReachabilityMap;

    // Index assigned for a particular instruction.  The value is used to index
    // into the vector of BitVectors and the BitVectors themselves.
    int v;
  };
  // Sets up a graph with no edges and where the nodes correspond to the given
  // instructions.
  explicit HloReachabilityMap(
      absl::Span<const HloInstruction* const> instructions);

  // Computes and returns the reachability between HLO instructions in the
  // computation. The returned HloReachabilityMap is constructed such that
  // HloReachabilityMap::IsReachable(a, b) returns true iff there exists a
  // directed path (from producer to consumer) from 'a' to 'b'. Both data
  // dependencies (operands) and control dependencies are considered for
  // reachability. Trivially an instruction is reachable from itself.
  static std::unique_ptr<HloReachabilityMap> Build(
      const HloComputation* computation);

  // Similar to the above Build operation except that it tries to identify
  // paths between instructions that do not contain control instructions
  // and multiple operands, i.e., b is_reachable a == true iff
  // b = f(f(f(f(f(a), constant), constant), constant).
  // Further, the only ops allowed in a path are basic math operations such
  // as add, sub, mul, div.
  static std::unique_ptr<HloReachabilityMap> BuildWithRestrictions(
      const HloComputation* computation,
      absl::FunctionRef<void(const HloInstruction*,
                             std::vector<HloInstruction*>*)>
          add_dependencies);

  // Set the reachability set of 'instruction' to the union of the reachability
  // sets of 'inputs'. Upon return, IsReachable(x, instruction) where
  // 'x' is not 'instruction' will return true iff IsReachable(x, input) is true
  // for some 'input' in 'inputs'. Also sets 'instruction' to be reachable from
  // itself. Returns whether the reachability set of 'instruction' changed.
  //
  // !!! THIS FUNCTION DOES NOT COMPUTE REACHABILITY !!! It sets the adjacency
  // vector in the internal graph of this HloReachabilityMap for the given
  // instruction and does not transitively update any other part of the
  // adjacency matrix.
  bool SetReachabilityToUnion(absl::Span<const HloInstruction* const> inputs,
                              const HloInstruction* instruction);

  // As above, but faster because it does not check if the reachability changed.
  void FastSetReachabilityToUnion(
      absl::Span<const HloInstruction* const> inputs,
      const HloInstruction* instruction);
  // As above, but use Index instead if it's already looked up which is even
  // faster since no hash map lookup will occur.
  void FastSetReachabilityToUnion(absl::Span<const Index> input_indices,
                                  Index index);

  Index GetIndex(const HloInstruction* instruction) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_0(mht_0_v, 277, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetIndex");

    Index i;
    i.v = FindOrDie(indices_, GetKey(instruction));
    return i;
  }

  // Sets entry so that IsReachable(a, b) will return true
  //
  // !!! THIS FUNCTION DOES NOT COMPUTE REACHABILITY !!! It sets the adjacency
  // matrix in the internal graph of this HloReachabilityMap to have an edge
  // from a to b and does not transitively update any other part of the
  // adjacency matrix.
  void SetReachable(const HloInstruction* a, const HloInstruction* b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_1(mht_1_v, 292, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "SetReachable");

    SetReachable(GetIndex(a), GetIndex(b));
  }
  void SetReachable(Index a, Index b);

  // Updates the given reachability map after the immediate predecessor set
  // (operands and control predecessors) of 'instruction' has changed.
  void UpdateReachabilityThroughInstruction(const HloInstruction* instruction);

  // Returns true if "b" is reachable from "a"
  //
  // Note that this function only correctly answers queries about reachability
  // if the set of edges that have been provided to this class are transitive.
  bool IsReachable(const HloInstruction* a, const HloInstruction* b) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_2(mht_2_v, 308, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "IsReachable");

    return IsReachable(GetIndex(a), GetIndex(b));
  }
  bool IsReachable(Index a, Index b) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_3(mht_3_v, 314, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "IsReachable");
 return GetBitVector(b).Get(a.v); }

  // Returns true if "b" is reachable from "a" or "a" is reachable from "b"
  //
  // Note that this function only correctly answers queries about reachability
  // if the set of edges that have been provided to this class are transitive.
  bool IsConnected(const HloInstruction* a, const HloInstruction* b) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_4(mht_4_v, 323, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "IsConnected");

    return IsConnected(GetIndex(a), GetIndex(b));
  }
  bool IsConnected(Index a, Index b) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_5(mht_5_v, 329, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "IsConnected");

    return IsReachable(a, b) || IsReachable(b, a);
  }

  // Checks if an instruction is in the Reachability map.
  bool IsPresent(const HloInstruction* a) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_6(mht_6_v, 337, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "IsPresent");

    return indices_.contains(GetKey(a));
  }

  // Replace the instruction "original" with "replacement" in the reachability
  // map.
  void Replace(const HloInstruction* original,
               const HloInstruction* replacement);

 private:
  // A bit-vector implementation specialized for this use case which provides a
  // fast bitwise OR operation not available in tensorflow::gtl::BitMap.
  class BitVector {
   public:
    BitVector() = default;
    BitVector(size_t size)
        : size_(size), vector_((size + kBits - 1) / kBits, 0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_7(mht_7_v, 356, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "BitVector");
}

    // Return the bit at the given index.
    bool Get(size_t index) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_8(mht_8_v, 362, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "Get");

      DCHECK(index >= 0 && index < size_);
      return vector_[index / kBits] & (1ull << (index % kBits));
    }

    // Set the bit at the given index.
    void Set(size_t index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_9(mht_9_v, 371, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "Set");

      DCHECK(index >= 0 && index < size_);
      vector_[index / kBits] |= 1ull << (index % kBits);
    }

    // Set this bitvector to the Logical OR of this bitvector and 'other'.
    void OrWith(const BitVector& other) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_10(mht_10_v, 380, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "OrWith");

      for (size_t i = 0; i < vector_.size(); ++i) {
        vector_[i] |= other.vector_[i];
      }
    }

    // Set the bitvector to all zeros.
    void SetToZero() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_11(mht_11_v, 390, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "SetToZero");
 std::fill(vector_.begin(), vector_.end(), 0); }

    bool operator==(const BitVector& other) const {
      return vector_ == other.vector_;
    }
    bool operator!=(const BitVector& other) const {
      return vector_ != other.vector_;
    }

   private:
    using Word = uint64_t;
    static constexpr size_t kBits = 64;

    // Number of bits in the bitvector.
    size_t size_;

    std::vector<Word> vector_;
  };

  // Return the bitvector storing the reachability-to of the given instruction.
  const BitVector& GetBitVector(const HloInstruction* instruction) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_12(mht_12_v, 413, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetBitVector");

    return GetBitVector(GetIndex(instruction));
  }
  BitVector& GetBitVector(const HloInstruction* instruction) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_13(mht_13_v, 419, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetBitVector");

    return GetBitVector(GetIndex(instruction));
  }

  const BitVector& GetBitVector(Index index) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_14(mht_14_v, 426, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetBitVector");

    return bit_vectors_[index.v];
  }
  BitVector& GetBitVector(Index index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_15(mht_15_v, 432, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetBitVector");
 return bit_vectors_[index.v]; }

  // Helper for SetReachabilityToUnion/FastSetReachabilityToUnion.
  void SetReachabilityToUnionHelper(
      absl::Span<const HloInstruction* const> inputs, Index index);
  void SetReachabilityToUnionHelper(absl::Span<const Index> input_indices,
                                    Index index);

  uint64_t GetKey(const HloInstruction* instruction) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_16(mht_16_v, 443, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetKey");

    uint64_t unique_id = absl::bit_cast<uint32_t>(instruction->unique_id());
    uint64_t module_id =
        absl::bit_cast<uint32_t>(instruction->parent()->parent()->unique_id());
    return (module_id << 32) | unique_id;
  }
  // Return the index of the given instruction.
  int GetIndexInternal(const HloInstruction* instruction) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_reachabilityDTh mht_17(mht_17_v, 453, "", "./tensorflow/compiler/xla/service/hlo_reachability.h", "GetIndexInternal");

    return FindOrDie(indices_, GetKey(instruction));
  }

  // The number of instructions in the reachability map.
  const size_t size_;

  // Dense assignment from HloInstruction::unique_id to number. These numbers
  // index into the bit_vectors_ vector and into the bits within a BitVector.
  absl::flat_hash_map<uint64_t, int> indices_;

  // Bitvectors holding the reachability to each instruction. The bit vector for
  // instruction X includes ones for each instruction which X is reachable from.
  std::vector<BitVector> bit_vectors_;

  // A temporary used by SetReachabilityToUnion to avoid an allocation with each
  // call to the method.
  BitVector tmp_bit_vector_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_REACHABILITY_H_
