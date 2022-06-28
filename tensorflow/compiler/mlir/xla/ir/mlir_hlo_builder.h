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
#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh() {
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


#include <memory>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

// Provides a way to construct mhlo dialect ops in MLIR using XlaBuilder
// interface.
//
// Requires that all XlaOp arguments are either returned by any of the builder
// method or constructed using MakeXlaOp method in this builder.
//
// TODO(hinsu): Support more ops and utility functions to set special attributes
// like OpMetadata and Sharding.
class MlirHloBuilder : public XlaBuilder {
 public:
  // Constructs builder for the given function. New operations are added to the
  // beginning of the function, if it is non empty and has a block.
  explicit MlirHloBuilder(mlir::func::FuncOp func)
      : XlaBuilder(func.getName().str()),
        builder_(&func.getBody()),
        loc_(builder_.getUnknownLoc()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh mht_0(mht_0_v, 222, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h", "MlirHloBuilder");
}

  // TODO(hinsu): Add a constructor to build a new MLIR function from scratch
  // and override Build methods.

  MlirHloBuilder(std::string name, mlir::OpBuilder builder, mlir::Location loc)
      : XlaBuilder(name), builder_(builder), loc_(loc) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h", "MlirHloBuilder");
}

  MlirHloBuilder(const MlirHloBuilder&) = delete;
  MlirHloBuilder& operator=(const MlirHloBuilder&) = delete;

  ~MlirHloBuilder() override;

  // Wraps the given MLIR value under an XlaOp instance. Note that all HLO
  // operations returns exactly one result therefore each op has an XlaOp
  // wrapping result of the op.
  //
  // Returns an error if the HLO dialect doesn't support type of the given
  // value.
  StatusOr<XlaOp> MakeXlaOp(mlir::Value val);

  // Returns value corresponding to the given op.
  //
  // Requires that the op was created by this builder.
  mlir::Value GetValue(XlaOp op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h", "GetValue");

    void* ptr = reinterpret_cast<void*>(op.handle());
    return mlir::Value::getFromOpaquePointer(ptr);
  }

  // Returns MLIR values corresponding to the given XLA ops.
  //
  // Requires that the ops were created by this builder.
  std::vector<mlir::Value> GetValues(absl::Span<const XlaOp> ops) {
    std::vector<mlir::Value> values;
    for (auto xla_op : ops) {
      values.push_back(GetValue(xla_op));
    }
    return values;
  }

  // Sets location for newly built ops, until reset.
  void SetLocation(mlir::Location loc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh mht_3(mht_3_v, 273, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h", "SetLocation");
 loc_ = loc; }

  // Update insertion point so that newly built ops are inserted before the
  // given op in order, until reset.
  void setInsertionPoint(mlir::Operation* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSirPSmlir_hlo_builderDTh mht_4(mht_4_v, 280, "", "./tensorflow/compiler/mlir/xla/ir/mlir_hlo_builder.h", "setInsertionPoint");

    builder_.setInsertionPoint(op);
  }

  // Returns the shape of the given op.
  StatusOr<const Shape*> GetShapePtr(XlaOp op) const override;

  // Creates the given op at the current location.
  template <typename OpTy, typename... Args>
  OpTy create(Args&&... args) {
    return builder_.create<OpTy>(loc_, std::forward<Args>(args)...);
  }

 private:
  XlaOp ConstantLiteral(const LiteralSlice& literal) override;

  StatusOr<XlaOp> ConvGeneralDilatedInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs, const Window& window,
      absl::Span<const int64_t> window_strides,
      absl::Span<const std::pair<int64_t, int64_t>> padding,
      absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64_t feature_group_count, int64_t batch_group_count,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> FftInternal(const Shape& shape, XlaOp operand,
                              FftType fft_type,
                              absl::Span<const int64_t> fft_length) override;

  StatusOr<XlaOp> TriangularSolveInternal(
      const Shape& shape, XlaOp a, XlaOp b,
      TriangularSolveOptions options) override;

  StatusOr<XlaOp> CholeskyInternal(const Shape& shape, XlaOp a,
                                   bool lower) override;

  StatusOr<XlaOp> CustomCallInternal(
      const std::string& call_target_name, absl::Span<const XlaOp> operands,
      const Shape& shape, const std::string& opaque,
      absl::optional<absl::Span<const Shape>> operand_shapes_with_layout,
      bool has_side_effect,
      absl::Span<const std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          output_operand_aliasing,
      const Literal* literal, absl::optional<Window> window,
      absl::optional<ConvolutionDimensionNumbers> dnums,
      CustomCallSchedule schedule, CustomCallApiVersion api_version) override;

  StatusOr<XlaOp> ReduceInternal(
      const Shape& shape, absl::Span<const XlaOp> all_operands,
      const XlaComputation& computation,
      absl::Span<const int64_t> dimensions_to_reduce) override;

  StatusOr<XlaOp> ReduceWindowInternal(const Shape& shape, XlaOp operand,
                                       XlaOp init_value,
                                       const XlaComputation& computation,
                                       Window window) override;

  XlaOp Iota(const Shape& shape, int64_t iota_dimension) override;

  StatusOr<XlaOp> BitcastConvertTypeInternal(const Shape& shape,
                                             XlaOp operand) override;

  StatusOr<XlaOp> TransposeInternal(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64_t> permutation) override;

  StatusOr<XlaOp> RevInternal(const Shape& shape, XlaOp operand,
                              absl::Span<const int64_t> dimensions) override;

  StatusOr<XlaOp> SortInternal(const Shape& shape,
                               absl::Span<const XlaOp> operands,
                               const XlaComputation& comparator,
                               int64_t dimension, bool is_stable) override;

  StatusOr<XlaOp> WhileInternal(const Shape& shape,
                                const XlaComputation& condition,
                                const XlaComputation& body,
                                XlaOp init) override;

  StatusOr<XlaOp> ReducePrecisionInternal(const Shape& shape, XlaOp operand,
                                          const int exponent_bits,
                                          const int mantissa_bits) override;

  StatusOr<XlaOp> GatherInternal(
      const Shape& shape, XlaOp input, XlaOp start_indices,
      const GatherDimensionNumbers& dimension_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) override;

  StatusOr<XlaOp> ScatterInternal(
      const Shape& shape, XlaOp input, XlaOp scatter_indices, XlaOp updates,
      const XlaComputation& update_computation,
      const ScatterDimensionNumbers& dimension_numbers, bool indices_are_sorted,
      bool unique_indices) override;

  StatusOr<XlaOp> SetDimensionSizeInternal(const Shape& shape, XlaOp operand,
                                           XlaOp val,
                                           int64_t dimension) override;

  StatusOr<XlaOp> RngOpInternal(RandomDistribution distribution,
                                absl::Span<const XlaOp> parameters,
                                const Shape& shape) override;
  StatusOr<XlaOp> RngBitGeneratorInternal(const Shape& full_result_shape,
                                          RandomAlgorithm algorithm,
                                          XlaOp initial_state) override;

  StatusOr<XlaOp> ReshapeInternal(const Shape& shape, XlaOp operand,
                                  int64_t inferred_dimension) override;

  StatusOr<XlaOp> DotGeneralInternal(
      const Shape& shape, XlaOp lhs, XlaOp rhs,
      const DotDimensionNumbers& dimension_number,
      const PrecisionConfig* precision_config) override;

  StatusOr<XlaOp> InDimBroadcast(
      const Shape& shape, XlaOp operand,
      absl::Span<const int64_t> broadcast_dimensions) override;

  StatusOr<XlaOp> AddInstruction(HloInstructionProto&& instr, HloOpcode opcode,
                                 absl::Span<const XlaOp> operands) override;

  StatusOr<XlaOp> Compare(const Shape& shape, XlaOp lhs, XlaOp rhs,
                          ComparisonDirection direction,
                          Comparison::Type type) override;

  XlaOp BinaryOpNoBroadcast(HloOpcode binop, const Shape& shape, XlaOp lhs,
                            XlaOp rhs) override;

  StatusOr<XlaOp> AddOpWithShape(HloOpcode opcode, const Shape& shape,
                                 absl::Span<const XlaOp> operands) override;

  XlaOp CreateToken() override;

  StatusOr<XlaOp> InfeedWithTokenInternal(const Shape& infeed_instruction_shape,
                                          XlaOp token,
                                          const std::string& config) override;
  StatusOr<XlaOp> OutfeedWithTokenInternal(
      XlaOp operand, XlaOp token, const Shape& shape_with_layout,
      const std::string& outfeed_config) override;

  StatusOr<XlaOp> ConcatInDimInternal(const Shape& shape,
                                      absl::Span<const XlaOp> operands,
                                      int64_t dimension) override;

  StatusOr<XlaOp> GetTupleElementInternal(const Shape& shape, XlaOp tuple_data,
                                          int64_t index) override;

  StatusOr<XlaOp> SliceInternal(const Shape& shape, XlaOp operand,
                                absl::Span<const int64_t> start_indices,
                                absl::Span<const int64_t> limit_indices,
                                absl::Span<const int64_t> strides) override;

  StatusOr<XlaOp> DynamicSliceInternal(
      const Shape& shape, XlaOp operand, absl::Span<const XlaOp> start_indices,
      absl::Span<const int64_t> slice_sizes) override;

  StatusOr<XlaOp> DynamicUpdateSliceInternal(
      const Shape& shape, XlaOp operand, XlaOp update,
      absl::Span<const XlaOp> start_indices) override;

  StatusOr<XlaOp> PadInternal(const Shape& shape, XlaOp operand,
                              XlaOp padding_value,
                              const PaddingConfig& padding_config) override;

  StatusOr<XlaOp> TupleInternal(const Shape& shape,
                                absl::Span<const XlaOp> elements) override;

  // Creates HLO dialect op and returns the result as an XlaOp.
  StatusOr<XlaOp> CreateOp(
      const std::string& op_name, const Shape& shape,
      llvm::ArrayRef<XlaOp> operands,
      llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

  Status ImportComputation(const HloModuleProto& computation,
                           mlir::Region* region,
                           bool flatten_region_arg_tuple = false);

  mlir::OpBuilder builder_;
  mlir::Location loc_;

  absl::flat_hash_map<int64_t, std::unique_ptr<Shape>> handle_to_shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_MLIR_HLO_BUILDER_H_
