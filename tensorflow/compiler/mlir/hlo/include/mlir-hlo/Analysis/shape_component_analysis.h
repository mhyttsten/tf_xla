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

#ifndef MLIR_HLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H
#define MLIR_HLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H
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
class MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh() {
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


#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

// Analysis to infer shape information.
//
// This lazily analyzes the individual components of a shape (e.g., the
// dimensions of a tensor) or value (e.g, the elements of a shape tensor).
// Results are cached but the cache is not consistent across IR mutations and
// needs to be reset in that case.
class ShapeComponentAnalysis {
 public:
  // Represents the analysis request for a specific value. We are either
  // interested in the shape of a value or the value itself.
  class ShapeOrValueInfo {
    llvm::PointerIntPair<Value, 1, bool> p;

    explicit ShapeOrValueInfo(decltype(p) p) : p(p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "ShapeOrValueInfo");
}
    ShapeOrValueInfo(Value v, bool isValueInfo) : p(v, isValueInfo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_1(mht_1_v, 212, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "ShapeOrValueInfo");
}

   public:
    static ShapeOrValueInfo getShapeInfoOf(Value v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_2(mht_2_v, 218, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getShapeInfoOf");
 return {v, false}; }
    static ShapeOrValueInfo getValueInfoOf(Value v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_3(mht_3_v, 222, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getValueInfoOf");
 return {v, true}; }
    Value value() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_4(mht_4_v, 226, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "value");
 return p.getPointer(); }
    bool isValueInfo() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_5(mht_5_v, 230, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "isValueInfo");
 return p.getInt(); }
    bool isShapeInfo() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_6(mht_6_v, 234, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "isShapeInfo");
 return !isValueInfo(); }

    bool operator==(ShapeOrValueInfo rhs) const { return p == rhs.p; }
    bool operator!=(ShapeOrValueInfo rhs) const { return !(*this == rhs); }

    // Forward p's DenseMapInfo.
    struct DenseMapInfo {
      using PairInfo = llvm::DenseMapInfo<decltype(p)>;
      static inline ShapeOrValueInfo getEmptyKey() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_7(mht_7_v, 245, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getEmptyKey");

        return ShapeOrValueInfo(PairInfo::getEmptyKey());
      }
      static inline ShapeOrValueInfo getTombstoneKey() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_8(mht_8_v, 251, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getTombstoneKey");

        return ShapeOrValueInfo(PairInfo::getTombstoneKey());
      }
      static unsigned getHashValue(ShapeOrValueInfo val) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_9(mht_9_v, 257, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getHashValue");

        return PairInfo::getHashValue(val.p);
      }
      static bool isEqual(ShapeOrValueInfo lhs, ShapeOrValueInfo rhs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_10(mht_10_v, 263, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "isEqual");

        return lhs == rhs;
      }
    };
  };

  // Symbolically represents one component of a shape (e.g., the dimensions of a
  // tensor) or value (e.g, the elements of a shape tensor). This is used to tie
  // symbolic expressions to components of shapes or values.
  struct Symbol {
    ShapeOrValueInfo source;
    size_t index;

    bool operator==(const Symbol &rhs) const {
      return source == rhs.source && index == rhs.index;
    }
    bool operator!=(const Symbol &rhs) const { return !(*this == rhs); }
  };

  // Represents the analysis result for a one component of a shape (e.g., the
  // dimensions of a tensor) or value (e.g, the elements of a shape tensor).
  // This can be a constant or an expression over symbols.
  struct SymbolicExpr {
    SmallVector<Symbol, 1> symbols;
    AffineExpr expr;

    // Returns true if this symbolic expression is known to be a constant equal
    // to `value`.
    bool isConstant(int64_t value) const;
    // Returns true if this symbolic expression is known to be different from
    // `-1`. This is useful for reshapes.
    bool isKnownNotNegativeOne() const;
    // Returns true if thus symbolic expression is known to be different from
    // `1`. This is useful for broadcasts.
    bool isKnownNotOne() const;
    // If this is a reference to a singular symbol, return it.
    Optional<Symbol> singleton() const;

    bool operator==(const SymbolicExpr &rhs) const {
      return expr == rhs.expr && symbols == rhs.symbols;
    }
    bool operator!=(const SymbolicExpr &rhs) const { return !(*this == rhs); }

    void dump(llvm::raw_ostream &os = llvm::outs()) const;
  };

  using SymbolicExprsMap = DenseMap<ShapeOrValueInfo, std::vector<SymbolicExpr>,
                                    ShapeOrValueInfo::DenseMapInfo>;
  using SymbolicShapeConstraintsMap = DenseMap<int, Symbol>;

 private:
  // Mapping from the analysis requests to the results, i.e. to an array of
  // symbolic expressions. This is essentially a cache for all the results of
  // this analysis.
  SymbolicExprsMap symbolicExprsMap;

  // Mapping from symbolic shape constraints, derived from the argument
  // attributes, to the symbols used in this analysis.
  SymbolicShapeConstraintsMap symbolicShapeConstraintsMap;

  // Run the analysis to request either shape or value information.
  void compute(ShapeOrValueInfo v);

 public:
  // Return the computed components for the shape of a value, e.g., the
  // dimensions of a tensor.
  Optional<ArrayRef<SymbolicExpr>> GetShapeInfo(Value value);
  // Return the computed components for the value of a value, e.g, the elements
  // of a shape tensor.
  Optional<ArrayRef<SymbolicExpr>> GetValueInfo(Value shape);

  // Clear analysis data structures.
  void reset();
};
}  // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::ShapeComponentAnalysis::Symbol> {
  static inline mlir::ShapeComponentAnalysis::Symbol getEmptyKey() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_11(mht_11_v, 346, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getEmptyKey");

    return {mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
                getEmptyKey(),
            llvm::DenseMapInfo<size_t>::getEmptyKey()};
  }
  static inline mlir::ShapeComponentAnalysis::Symbol getTombstoneKey() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_12(mht_12_v, 354, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getTombstoneKey");

    return {mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
                getTombstoneKey(),
            llvm::DenseMapInfo<size_t>::getTombstoneKey()};
  }
  static unsigned getHashValue(mlir::ShapeComponentAnalysis::Symbol symbol) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_13(mht_13_v, 362, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "getHashValue");

    return llvm::hash_combine(
        mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
            getHashValue(symbol.source),
        llvm::DenseMapInfo<size_t>::getHashValue(symbol.index));
  }
  static bool isEqual(mlir::ShapeComponentAnalysis::Symbol lhs,
                      mlir::ShapeComponentAnalysis::Symbol rhs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPShloPSincludePSmlirSLhloPSAnalysisPSshape_component_analysisDTh mht_14(mht_14_v, 372, "", "./tensorflow/compiler/mlir/hlo/include/mlir-hlo/Analysis/shape_component_analysis.h", "isEqual");

    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // MLIR_HLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H
