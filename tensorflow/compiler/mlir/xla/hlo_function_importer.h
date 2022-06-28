/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_FUNCTION_IMPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_FUNCTION_IMPORTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_function_importerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_function_importerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_function_importerDTh() {
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


#include <unordered_map>

#include "absl/types/optional.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloModule;
class HloComputation;
class HloInstruction;
class Shape;

// HLO bounded dynamic shapes can be converted to either MLIR dynamic shapes
// (which lose the bound information) or casted to static shape using the
// bounds.
enum class DynamicShapeHandlingMode { kDynamic, kConvertToStatic };

// Helper class for importing HloComputations.
class HloFunctionImporter {
 public:
  // Imports the given computation as a function in the given module. This also
  // imports any computations referred by instructions in this computation.
  static Status ImportAsFunc(
      const xla::HloComputation& computation, mlir::ModuleOp module,
      std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>*
          function_map,
      mlir::Builder* builder);

  // Imports the given hlo computation to the specified region. If
  // 'flatten_region_arg_tuple' is true, then flatten the tuple-typed region
  // argument(s) and return value(s).
  static Status ImportAsRegion(const xla::HloComputation& computation,
                               mlir::Region* region, mlir::Builder* builder,
                               bool flatten_region_arg_tuple = false);

  // Imports the given computation to the given place specified by `builder`.
  // `arguments` contains values for all parameters.
  static StatusOr<mlir::Value> ImportInstructions(
      const xla::HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::OpBuilder* builder);

  static StatusOr<mlir::Operation*> ImportInstruction(
      const xla::HloInstruction* instr,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  static void SetLayoutForMlir(mlir::Operation* op, const Shape& shape,
                               llvm::StringRef attr_name);

  // TODO(b/179166199): move this to attribute_importer.h.
  // Converts XLA instruction source target pairs to MLIR attribute.
  static mlir::NamedAttribute ConvertSourceTargetPairs(
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      mlir::Builder* builder);

  // TODO(b/179166199): move this to attribute_importer.h.
  // Converts replica groups to attribute
  static mlir::NamedAttribute ConvertReplicaGroups(
      absl::Span<const ReplicaGroup> replica_groups, mlir::Builder* builder);

  // For mlir::IfOp or mlir::CaseOp, replace the uses of their region's block
  // arguments with 'implicit_operands'. Here | implicit_operands | == sum of
  // the number of arguments in all the regions in IfOp or CaseOp.
  void ReplaceBlockArgumentsWithImplicitOperands(
      mlir::Operation* op, llvm::ArrayRef<mlir::Value> implicit_operands);

  // Create a TupleOp using the results of 'op' if 'type' is a mlir::TupleType.
  // Otherwise, return 'op'.
  mlir::Operation* CreateTupleFromOpResults(mlir::OpBuilder* func_builder,
                                            mlir::Location loc,
                                            mlir::Operation* op,
                                            mlir::Type type);

  // FlattenTupleType flattens the types in (nested) tuple-type 'type' and
  // stores them in 'types'.
  static void FlattenTupleType(
      mlir::Type type, llvm::SmallVectorImpl<mlir::Type>& flattened_types);

  // FlattenTupleValue flattens the values in (nested) tuple-typed 'value' and
  // stores them in 'flattened_values'.
  static void FlattenTupleValue(
      mlir::OpBuilder* func_builder, mlir::Location loc, mlir::Value value,
      llvm::SmallVectorImpl<mlir::Value>& flattened_values);

  // CreateTupleValue creates a root TupleOp of (nested) tuple-type 'type' using
  // the non-tuple-typed values in 'flatten_values'.
  //
  // e.g., Given 'flatten_values': [V1, V2, V3] &'type': tuple<T1,tuple<T1,T2>>,
  //      The function returns %t2 such that:
  //       %t1 = mhlo.tuple(V2,V3) : (T2,T3) -> tuple<T2,T3>
  //       %t2 = mhlo.tuple(V1,%t1): (T1,tuple<T2,T3>) -> tuple<T1,tuple<T1,T2>>
  //
  // Note: 1. FlattenTupleValue and CreateTupleValue is a pair of functions to
  //          resp. flatten and create tuples in the exact same order.
  //       2. `flatten_values`, initially storing the flattened values, will be
  //          mutated to a 0-length array by the end of function invocation.
  static mlir::Value CreateTupleValue(
      mlir::OpBuilder* func_builder, mlir::Location loc,
      llvm::MutableArrayRef<mlir::Value>& flatten_values, mlir::Type type);

 private:
  HloFunctionImporter(mlir::ModuleOp module,
                      std::unordered_map<const xla::HloComputation*,
                                         mlir::func::FuncOp>* function_map,
                      mlir::Builder* builder)
      : context_(module.getContext()),
        module_(module),
        builder_(builder),
        function_map_(function_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPShlo_function_importerDTh mht_0(mht_0_v, 311, "", "./tensorflow/compiler/mlir/xla/hlo_function_importer.h", "HloFunctionImporter");

    context_->loadDialect<mlir::arith::ArithmeticDialect>();
    context_->loadDialect<mlir::func::FuncDialect>();
    context_->loadDialect<mlir::mhlo::MhloDialect>();
  }

  // Imports the given computation as a new function, if it hasn't been already
  // imported.
  StatusOr<mlir::func::FuncOp> ImportAsFunc(
      const xla::HloComputation& computation);

  // Imports the given computation in the specified region.
  tensorflow::Status ImportAsRegion(const HloComputation& computation,
                                    mlir::Region* region,
                                    bool flatten_region_arg_tuple = false);

  // Imports instructions from the given computation in the specified block.
  // Assumes that the block already has correct arguments populated.
  tensorflow::Status ImportInstructions(const HloComputation& computation,
                                        mlir::Block* block,
                                        bool flatten_region_arg_tuple);
  StatusOr<mlir::Value> ImportInstructionsImpl(
      const xla::HloComputation& computation,
      const llvm::SmallVectorImpl<mlir::Value>& arguments,
      mlir::OpBuilder* builder);

  // Imports an instruction.
  StatusOr<mlir::Operation*> ImportInstructionWithLayout(
      const xla::HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  StatusOr<mlir::Operation*> ImportInstructionImpl(
      const HloInstruction* instruction,
      const llvm::SmallVectorImpl<mlir::Value>& operands,
      mlir::OpBuilder* func_builder,
      DynamicShapeHandlingMode mode = DynamicShapeHandlingMode::kDynamic);

  // Gets the MLIR operand values from an HLO Instruction.
  StatusOr<llvm::SmallVector<mlir::Value, 4>> GetOperands(
      const xla::HloInstruction* instruction);

  // Converts xla Tensor type to the corresponding MLIR type.
  StatusOr<mlir::RankedTensorType> ConvertTensorType(const xla::Shape& shape);

  // Converts an XLA shape/layout to the corresponding MLIR layout, in
  // flattened_attr, while flattening the tuple layout.
  Status ConvertShapeToMlirLayout(
      const xla::Shape& shape,
      llvm::SmallVectorImpl<mlir::Attribute>& flattened_attr);

  // Returns the output type of an HloInstruction.
  StatusOr<mlir::Type> GetReturnType(const xla::HloInstruction* instruction);

  // Takes a list of HloInstructions and generates the list of types used for
  // input, bypassing tuples to subsets.
  Status GetMlirTypes(const std::vector<xla::HloInstruction*>& instructions,
                      llvm::SmallVectorImpl<mlir::Type>* types);

  // Returns the Mlir Value for the corresponding HloInstruction.
  StatusOr<mlir::Value> GetMlirValue(const xla::HloInstruction* instruction);

  // Converts an XLA ComparisonDirection to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonDirection(
      ComparisonDirection direction);

  // Converts an XLA Comparison::Type to the corresponding MLIR attribute.
  mlir::NamedAttribute ConvertComparisonType(Comparison::Type type);

  // Converts the dimensions of an HLO instruction into an MLIR attribute.
  mlir::DenseIntElementsAttr ConvertDimensions(
      absl::Span<const int64_t> op_dimensions);

  // Converts Array ref to an DenseIntElementsAttr.
  mlir::DenseIntElementsAttr Convert(llvm::ArrayRef<int64_t> elements);

  // Converts Array ref to padding attribute. Input is a flattened list of
  // padding low and padding high for each of the spatial dimensions.
  mlir::NamedAttribute ConvertPadding(llvm::ArrayRef<int64_t> padding);

  // Converts channel id to attribute
  mlir::NamedAttribute ConvertChannelHandle(absl::optional<int64_t> channel_id);

  // Converts channel handle to attribute
  mlir::NamedAttribute ConvertChannelHandle(const xla::ChannelHandle& channel);

  mlir::MLIRContext* context_;
  mlir::ModuleOp module_;
  mlir::Builder* builder_;

  // Mapping from HloComputation to the created MLIR function.
  std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>*
      function_map_;

  // Mapping from HloInstructions to the associative MLIR values.
  std::unordered_map<const xla::HloInstruction*, mlir::Value>
      instruction_value_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_FUNCTION_IMPORTER_H_
