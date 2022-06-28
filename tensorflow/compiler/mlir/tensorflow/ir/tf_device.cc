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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// TF Device Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_0(mht_0_v, 235, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "isLegalToInline");

    return true;
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF Device operation.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "isLegalToInline");

    return true;
  }

  // Defines the legality of inlining TF Device operations.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_2(mht_2_v, 254, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "isLegalToInline");

    // For now, enable inlining all operations.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  // This is just re-using the same logic as the TensorFlow dialect right now.
  Operation* materializeCallConversion(OpBuilder& builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_3(mht_3_v, 274, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "materializeCallConversion");

    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }
};

// Checks if a block wraps a single operation and the single operation results
// are perfectly forwarded to the block's terminator.
bool BlockWrapsSingleOp(Block* block) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_4(mht_4_v, 287, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "BlockWrapsSingleOp");

  auto body = block->without_terminator();
  if (!hasSingleElement(body)) return false;

  Operation& wrapped_op = *body.begin();
  Operation* terminator = block->getTerminator();
  return wrapped_op.getNumResults() == terminator->getNumOperands() &&
         std::equal(wrapped_op.getResults().begin(),
                    wrapped_op.getResults().end(),
                    terminator->getOperands().begin());
}
}  // end anonymous namespace

TensorFlowDeviceDialect::TensorFlowDeviceDialect(MLIRContext* context)
    : Dialect(/*name=*/"tf_device", context,
              TypeID::get<TensorFlowDeviceDialect>()) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_5(mht_5_v, 305, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "TensorFlowDeviceDialect::TensorFlowDeviceDialect");

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
      >();

  addInterfaces<TFInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// tf_device.launch
//===----------------------------------------------------------------------===//

// Checks if a tf_device.launch wraps a single operation and the single
// operation results are perfectly forwarded to the launch return.
bool LaunchOp::WrapsSingleOp() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_6(mht_6_v, 323, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "LaunchOp::WrapsSingleOp");
 return BlockWrapsSingleOp(&GetBody()); }

//===----------------------------------------------------------------------===//
// tf_device.parallel_execute
//===----------------------------------------------------------------------===//

LogicalResult ParallelExecuteOp::verify() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_7(mht_7_v, 332, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParallelExecuteOp::verify");

  ParallelExecuteOp op = *this;
  const auto& regions = op.getOperation()->getRegions();
  if (regions.size() < 2) {
    return op.emitOpError() << "must have at least two regions.";
  }

  int output_index = 0;
  for (auto& region_and_index : llvm::enumerate(regions)) {
    auto& region = region_and_index.value();
    auto* region_terminator = region.front().getTerminator();

    // Check that output types of regions match return operand types.
    for (auto result_type : region_terminator->getOperandTypes()) {
      if (result_type !=
          op.getOperation()->getResult(output_index++).getType()) {
        return op.emitOpError() << "output types must be a concatenated "
                                << "list of output types for each regions.";
      }
    }
  }

  // Check that total number of outputs from regions match the output types of
  // the parallel_execute op.
  const int num_output_types = op.getOperation()->getNumResults();
  if (num_output_types != output_index) {
    return op.emitOpError()
           << "number of output types (" << num_output_types << ") "
           << "must match the total number of outputs from all "
           << "regions (" << output_index << ").";
  }

  return success();
}

// static
void ParallelExecuteOp::build(OpBuilder& builder, OperationState& state,
                              int num_regions, TypeRange output_types) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_8(mht_8_v, 372, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParallelExecuteOp::build");

  DCHECK_GE(num_regions, 2);
  for (int i = 0; i < num_regions; ++i) {
    Region* region = state.addRegion();
    region->push_back(new Block);
  }
  state.addTypes(output_types);
}

Block& ParallelExecuteOp::GetRegionBlockWithIndex(unsigned index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_9(mht_9_v, 384, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParallelExecuteOp::GetRegionBlockWithIndex");

  return getOperation()->getRegion(index).front();
}

Operation::result_range ParallelExecuteOp::GetRegionOutputs(
    unsigned region_index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_10(mht_10_v, 392, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParallelExecuteOp::GetRegionOutputs");

  int num_region_results =
      GetRegionBlockWithIndex(region_index).getTerminator()->getNumOperands();

  int return_value_offset = 0;
  for (int region_id = 0; region_id < region_index; ++region_id)
    return_value_offset +=
        GetRegionBlockWithIndex(region_id).getTerminator()->getNumOperands();

  return getResults().slice(return_value_offset, num_region_results);
}

bool ParallelExecuteOp::RegionWrapsSingleOp(unsigned index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_11(mht_11_v, 407, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParallelExecuteOp::RegionWrapsSingleOp");

  return BlockWrapsSingleOp(&GetRegionBlockWithIndex(index));
}

//===----------------------------------------------------------------------===//
// tf_device.replicate
//===----------------------------------------------------------------------===//

namespace {
ParseResult ParseReplicateOpOperands(
    OpAsmParser* parser, OperationState* state,
    llvm::SmallVectorImpl<llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8>>*
        replicated_inputs,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand>* packed_inputs,
    llvm::SmallVectorImpl<OpAsmParser::UnresolvedOperand>* region_args,
    llvm::SmallVectorImpl<Type>* region_arg_types) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_12(mht_12_v, 425, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ParseReplicateOpOperands");

  // No operands or empty operand list.
  bool parsed_l_paren = succeeded(parser->parseOptionalLParen());
  if (!parsed_l_paren || succeeded(parser->parseOptionalRParen()))
    return success();

  // Parse comma separated operands of the following format:
  //   replicated_input
  //     [%a, ...] as %block_arg0: type
  //   packed_input
  //     %b as %block_arg1: type
  //
  // Replicated inputs are placed before packed inputs when forming the op.
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> replicated_region_args;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> packed_region_args;
  llvm::SmallVector<Type, 8> replicated_region_arg_types;
  llvm::SmallVector<Type, 8> packed_region_arg_types;
  do {
    OpAsmParser::UnresolvedOperand operand_type;
    if (parser->parseOptionalOperand(operand_type).hasValue()) {
      packed_inputs->emplace_back(operand_type);
      if (parser->parseKeyword("as",
                               " between packed input and block argument") ||
          parser->parseRegionArgument(packed_region_args.emplace_back()) ||
          parser->parseColonType(packed_region_arg_types.emplace_back()))
        return failure();
    } else if (parser->parseOperandList(replicated_inputs->emplace_back(),
                                        OpAsmParser::Delimiter::Square) ||
               parser->parseKeyword(
                   "as", " between replicated inputs and block argument") ||
               parser->parseRegionArgument(
                   replicated_region_args.emplace_back()) ||
               parser->parseColonType(
                   replicated_region_arg_types.emplace_back())) {
      return failure();
    }
  } while (succeeded(parser->parseOptionalComma()));

  region_args->reserve(replicated_region_args.size() +
                       packed_region_args.size());
  region_args->append(replicated_region_args.begin(),
                      replicated_region_args.end());
  region_args->append(packed_region_args.begin(), packed_region_args.end());

  region_arg_types->reserve(replicated_region_arg_types.size() +
                            packed_region_arg_types.size());
  region_arg_types->append(replicated_region_arg_types.begin(),
                           replicated_region_arg_types.end());
  region_arg_types->append(packed_region_arg_types.begin(),
                           packed_region_arg_types.end());

  // Parse remaining `)` surrounding operands.
  return parser->parseRParen();
}

ParseResult SetReplicateOpOperands(
    llvm::SMLoc loc, OpAsmParser* parser, OperationState* state,
    llvm::ArrayRef<llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8>>
        replicated_inputs,
    llvm::ArrayRef<OpAsmParser::UnresolvedOperand> packed_inputs,
    llvm::ArrayRef<Type> region_arg_types, int32_t* n) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_13(mht_13_v, 488, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "SetReplicateOpOperands");

  for (const auto& attr : state->attributes)
    if (attr.getName().strref() == "n")
      if (auto n_attr = attr.getValue().dyn_cast<IntegerAttr>())
        *n = n_attr.getInt();

  if (*n < 2)
    return parser->emitError(loc) << "expects 'n' to be at least 2, got " << *n;

  if (replicated_inputs.empty() && packed_inputs.empty()) return success();

  for (auto replicated_input_and_idx : llvm::enumerate(replicated_inputs)) {
    const int32_t idx = replicated_input_and_idx.index();
    const auto& replicated_input = replicated_input_and_idx.value();
    // Check if replicated input matches `n`.
    if (replicated_input.size() != *n)
      return parser->emitError(loc)
             << "expects number of operands for replicated input " << idx
             << " to be 'n' (" << *n << "), got " << replicated_input.size();

    // Resolve replicated input and block argument type.
    if (parser->resolveOperands(replicated_input, region_arg_types[idx],
                                state->operands))
      return failure();
  }

  const int32_t num_replicated_block_args = replicated_inputs.size();
  for (auto packed_input_and_idx : llvm::enumerate(packed_inputs)) {
    const int32_t idx = packed_input_and_idx.index();
    const auto& packed_input = packed_input_and_idx.value();

    // Resolve packed input and block argument type.
    if (parser->resolveOperand(
            packed_input, region_arg_types[idx + num_replicated_block_args],
            state->operands))
      return failure();
  }

  return success();
}

}  // namespace

static constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

ParseResult ReplicateOp::parse(OpAsmParser& parser, OperationState& result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_14(mht_14_v, 536, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::parse");

  llvm::SMLoc loc = parser.getCurrentLocation();

  // Parse operands, attributes, and region of op.
  llvm::SmallVector<llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8>, 8>
      replicated_inputs;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> packed_inputs;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> region_args;
  llvm::SmallVector<Type, 8> region_arg_types;
  int32_t n = 0;
  Region& body = *result.addRegion();
  if (ParseReplicateOpOperands(&parser, &result, &replicated_inputs,
                               &packed_inputs, &region_args,
                               &region_arg_types) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      SetReplicateOpOperands(loc, &parser, &result, replicated_inputs,
                             packed_inputs, region_arg_types, &n) ||
      parser.parseRegion(body, region_args, region_arg_types))
    return failure();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  if (!result.attributes.get(kOperandSegmentSizesAttr)) {
    int32_t num_replicated_inputs = replicated_inputs.size() * n;
    int32_t num_packed_inputs = packed_inputs.size();
    auto attr = DenseIntElementsAttr::get(
        VectorType::get({2}, parser.getBuilder().getI32Type()),
        {num_replicated_inputs, num_packed_inputs});
    result.addAttribute(kOperandSegmentSizesAttr, attr);
  }

  // Ensure that the region is well formed: it contains at least a block with
  // a ReturnOp terminator.
  ReplicateOp::ensureTerminator(body, parser.getBuilder(), result.location);

  if (!llvm::hasSingleElement(body))
    return parser.emitError(loc) << "expects a single block region";

  Operation& terminator = body.front().back();
  if (!isa<ReturnOp>(terminator))
    return parser.emitError(loc) << "expects a tf_device.return terminator";

  // Get the results type from the terminator type inside the replicate,
  // replicated each by `n`.
  result.types.reserve(terminator.getNumOperands() * n);
  for (const auto& type : terminator.getOperandTypes())
    result.types.append(n, type);

  return success();
}

void ReplicateOp::print(OpAsmPrinter& p) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_15(mht_15_v, 589, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::print");

  // Print comma separated operands of the following format:
  //   replicated_input
  //     [%a, ...] as %block_arg0: type
  //   packed_input
  //     %b as %block_arg1: type
  const int32_t n = this->n();
  const int32_t num_replicated_inputs =
      (*operand_segment_sizes().value_begin<APInt>()).getSExtValue();
  const int32_t num_replicated_block_args = num_replicated_inputs / n;

  if (getNumOperands()) {
    p << '(';
    Block& block = body().front();
    interleaveComma(block.getArguments(), p, [&](BlockArgument arg) {
      const int block_arg_num = arg.getArgNumber();
      if (block_arg_num < num_replicated_block_args) {
        p << '[';
        p.printOperands(
            std::next(replicated_inputs().begin(), block_arg_num * n),
            std::next(replicated_inputs().begin(), (block_arg_num + 1) * n));
        p << "]";
      } else {
        p.printOperand(*std::next(packed_inputs().begin(),
                                  block_arg_num - num_replicated_block_args));
      }
      p << " as " << arg << ": " << arg.getType();
    });
    p << ')';
  }

  // Skip derived `operand_segment_sizes` attribute as custom print format of
  // operands holds enough information to calculate these variadic operand list
  // lengths.
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/ArrayRef<StringRef>{kOperandSegmentSizesAttr});
  p << ' ';
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

namespace {

// Checks if two types are compatible (compatible shapes and same elemental
// type).
LogicalResult VerifyCompatibleTypes(Type a, Type b) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_16(mht_16_v, 637, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "VerifyCompatibleTypes");

  if (failed(verifyCompatibleShape(a, b)) ||
      getElementTypeOrSelf(a) != getElementTypeOrSelf(b))
    return failure();

  return success();
}

void BuildReplicateOp(
    Builder* builder, OperationState* state, int n,
    llvm::Optional<DictionaryAttr> devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_17(mht_17_v, 652, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "BuildReplicateOp");

  DCHECK_GE(n, 2);
  state->addAttribute("n", builder->getI32IntegerAttr(n));

  if (devices.hasValue()) state->addAttribute("devices", devices.getValue());

  Region* region = state->addRegion();
  region->push_back(new Block);
  Block& block = region->front();

  for (auto& replicated_input : replicated_inputs) {
    DCHECK_EQ(llvm::size(replicated_input.first), n);
    for (auto input : replicated_input.first) {
      DCHECK(succeeded(
          VerifyCompatibleTypes(input.getType(), replicated_input.second)));
      state->addOperands(input);
    }
    block.addArgument(replicated_input.second, state->location);
  }

  for (auto packed_input : packed_inputs) {
    state->addOperands(packed_input);
    block.addArgument(packed_input.getType(), state->location);
  }

  // Add derived `operand_segment_sizes` attribute.
  int32_t num_replicated_inputs = replicated_inputs.size() * n;
  int32_t num_packed_inputs = packed_inputs.size();
  auto operand_segment_sizes =
      DenseIntElementsAttr::get(VectorType::get({2}, builder->getI32Type()),
                                {num_replicated_inputs, num_packed_inputs});
  state->addAttribute(kOperandSegmentSizesAttr, operand_segment_sizes);

  for (const auto& output_type : replica_output_types)
    state->addTypes(llvm::SmallVector<Type, 8>(n, output_type));
}

}  // anonymous namespace

LogicalResult ReplicateOp::verify() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_18(mht_18_v, 694, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::verify");

  ReplicateOp op = *this;
  int32_t n = op.n();

  // Check number of devices, if set, matches `n`.
  if (op.devices().hasValue()) {
    for (auto device_attr : op.devices().getValue().getValue()) {
      auto device_list = device_attr.getValue().dyn_cast_or_null<ArrayAttr>();
      if (!device_list)
        return op.emitError()
               << "expects 'devices' to be a map alias and device name list.";

      bool is_device_string = llvm::all_of(device_list, [](Attribute attr) {
        return attr.dyn_cast_or_null<StringAttr>();
      });
      if (!is_device_string)
        return op.emitOpError() << "expects 'devices' to be a consists of "
                                   "string list as values.";

      if (device_list.size() != n)
        return op.emitOpError()
               << "expects number of devices (" << device_list.size()
               << ") to be equal to 'n' (" << n << ")";
    }
  }

  Block& block = op.body().front();

  auto operand_segment_sizes = op.operand_segment_sizes();
  const int32_t num_replicated_inputs =
      operand_segment_sizes.getValues<APInt>()[0].getSExtValue();
  const int32_t num_packed_inputs =
      operand_segment_sizes.getValues<APInt>()[1].getSExtValue();

  if (num_replicated_inputs % n != 0)
    return op.emitOpError()
           << "expects number of replicated inputs (" << num_replicated_inputs
           << ") to be evenly divisible by 'n' (" << n << ")";

  const int32_t num_replicated_block_args = num_replicated_inputs / n;
  if (num_replicated_block_args + num_packed_inputs != block.getNumArguments())
    return op.emitOpError()
           << "expects number of block arguments (" << block.getNumArguments()
           << ") to be equal to number of replicated inputs ("
           << num_replicated_inputs << ") / 'n' (" << n
           << ") + number of packed inputs (" << num_packed_inputs << ")";

  // Check input types match block argument types.
  auto verify_operand_types = [&](BlockArgument block_arg,
                                  int32_t op_operand_idx) -> LogicalResult {
    Type op_operand_type = op.getOperand(op_operand_idx).getType();
    if (failed(VerifyCompatibleTypes(block_arg.getType(), op_operand_type)))
      return op.emitOpError()
             << "expects operand " << op_operand_idx << " (" << op_operand_type
             << ") and block argument " << block_arg.getArgNumber() << " ("
             << block_arg.getType() << ") to have compatible types";

    return success();
  };
  for (auto block_arg : block.getArguments()) {
    if (block_arg.getArgNumber() < num_replicated_block_args) {
      for (int32_t i = n * block_arg.getArgNumber(), e = i + n; i < e; ++i)
        if (failed(verify_operand_types(block_arg, i))) return failure();
    } else {
      const int32_t idx = block_arg.getArgNumber() - num_replicated_block_args +
                          num_replicated_inputs;
      if (failed(verify_operand_types(block_arg, idx))) return failure();
    }
  }

  Operation& terminator = block.back();

  // Check number of results matches `n` * number of return operands.
  if (op.getNumResults() != n * terminator.getNumOperands())
    return op.emitOpError()
           << "expects number of results (" << op.getNumResults()
           << ") to be equal to 'n' * number of terminator operands (" << n
           << " * " << terminator.getNumOperands() << ")";

  // Check replicated output types match return operand types.
  for (auto operand_type_and_idx :
       llvm::enumerate(terminator.getOperandTypes())) {
    Type operand_type = operand_type_and_idx.value();
    int32_t operand_idx = operand_type_and_idx.index();
    for (int32_t i = n * operand_idx, e = i + n; i < e; ++i)
      if (failed(VerifyCompatibleTypes(operand_type, op.getType(i))))
        return op.emitOpError() << "incompatible types for result " << i
                                << " and terminator operand " << operand_idx;
  }

  return success();
}

void ReplicateOp::build(
    OpBuilder& builder, OperationState& state, int n,
    const llvm::SmallDenseMap<StringRef, llvm::SmallVector<StringRef, 4>>&
        devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_19(mht_19_v, 795, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::build");

  llvm::Optional<DictionaryAttr> devices_attr;
  if (!devices.empty()) {
    llvm::SmallVector<mlir::NamedAttribute, 1> device_list;
    device_list.reserve(devices.size());
    for (auto alias_and_devices : devices) {
      NamedAttribute device_name_attr = builder.getNamedAttr(
          alias_and_devices.getFirst(),
          builder.getStrArrayAttr(alias_and_devices.getSecond()));
      device_list.emplace_back(device_name_attr);
    }
    devices_attr.emplace(builder.getDictionaryAttr(device_list));
  }

  BuildReplicateOp(&builder, &state, n, devices_attr, replicated_inputs,
                   packed_inputs, replica_output_types);
}

void ReplicateOp::build(
    OpBuilder& builder, OperationState& state, int n,
    llvm::Optional<DictionaryAttr> devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_20(mht_20_v, 820, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::build");

  BuildReplicateOp(&builder, &state, n, devices, replicated_inputs,
                   packed_inputs, replica_output_types);
}

// Returns the number of packed block arguments.
unsigned ReplicateOp::GetNumPackedBlockArguments() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_21(mht_21_v, 829, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetNumPackedBlockArguments");

  return packed_inputs().size();
}

// Returns the number of replicated block arguments.
unsigned ReplicateOp::GetNumReplicatedBlockArguments() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_22(mht_22_v, 837, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetNumReplicatedBlockArguments");

  return GetBody().getNumArguments() - GetNumPackedBlockArguments();
}

// Returns the replicated block arguments. A copy should be made if the
// replicate op is being modified.
llvm::ArrayRef<BlockArgument> ReplicateOp::GetReplicatedBlockArguments() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_23(mht_23_v, 846, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetReplicatedBlockArguments");

  return GetBody().getArguments().drop_back(GetNumPackedBlockArguments());
}

// Returns the packed block arguments. A copy should be made if the replicate op
// is being modified.
llvm::ArrayRef<BlockArgument> ReplicateOp::GetPackedBlockArguments() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_24(mht_24_v, 855, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetPackedBlockArguments");

  return GetBody().getArguments().take_back(GetNumPackedBlockArguments());
}

// Checks if a block argument is replicated (forwarding replicated inputs).
bool ReplicateOp::IsReplicatedBlockArgument(BlockArgument block_arg) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_25(mht_25_v, 863, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::IsReplicatedBlockArgument");

  assert(block_arg.getOwner() == &GetBody());
  return block_arg.getArgNumber() < GetNumReplicatedBlockArguments();
}

// Checks if a block argument is packed (forwarding a packed input).
bool ReplicateOp::IsPackedBlockArgument(BlockArgument block_arg) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_26(mht_26_v, 872, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::IsPackedBlockArgument");

  return !IsReplicatedBlockArgument(block_arg);
}

// Returns the operand index of the operand being forwarded as a
// replicated/packed block argument for a given replica. This assumes a valid
// block argument (of the replicate op) and a valid replica is provided.
unsigned ReplicateOp::GetReplicaOperandIndexForBlockArgument(
    BlockArgument block_arg, unsigned replica) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_27(mht_27_v, 883, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetReplicaOperandIndexForBlockArgument");

  MutableArrayRef<OpOperand> operands = GetOperandsForBlockArgument(block_arg);
  if (operands.size() == 1) return operands.front().getOperandNumber();

  return operands[replica].getOperandNumber();
}

// Returns the operand being forwarded as a replicated/packed block argument for
// a given replica. This assumes a valid block argument (of the replicate op)
// and a valid replica is provided.
Value ReplicateOp::GetReplicaOperandForBlockArgument(BlockArgument block_arg,
                                                     unsigned replica) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_28(mht_28_v, 897, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetReplicaOperandForBlockArgument");

  MutableArrayRef<OpOperand> operands = GetOperandsForBlockArgument(block_arg);
  if (operands.size() == 1) return operands.front().get();

  return operands[replica].get();
}

// Returns the list of replica op operands that maps to the given block
// argument. Returns list with num_replicas elements for replicated operands
// and list with a single element for packed operands.
//
// Requires that block argument is of this replicate op.
MutableArrayRef<OpOperand> ReplicateOp::GetOperandsForBlockArgument(
    BlockArgument block_arg) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_29(mht_29_v, 913, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::GetOperandsForBlockArgument");

  assert(block_arg.getOwner() == &GetBody());

  unsigned arg_number = block_arg.getArgNumber();
  unsigned num_replicated_args = GetNumReplicatedBlockArguments();
  int32_t num_replicas = nAttr().getInt();
  MutableArrayRef<OpOperand> operands = getOperation()->getOpOperands();

  // All replicated arguments are before packed arguments so return replicated
  // operands if the given argument is one of the replicated arguments.
  if (arg_number < num_replicated_args)
    return operands.slice(arg_number * num_replicas, num_replicas);

  operands = operands.drop_front(num_replicated_args * num_replicas);
  arg_number -= num_replicated_args;
  return operands.slice(arg_number, 1);
}

// Checks if a tf_device.replicate wraps a single operation and the single
// operation results are perfectly forwarded to the replicate return.
bool ReplicateOp::WrapsSingleOp() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_30(mht_30_v, 936, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ReplicateOp::WrapsSingleOp");
 return BlockWrapsSingleOp(&GetBody()); }

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_device.cluster
//===----------------------------------------------------------------------===//

namespace {

// Eliminates cluster op results that are not defined within the cluster and are
// defined outside. cluster op can be rewritten to remove those results.
static LogicalResult EliminatePassThroughResults(ClusterOp op,
                                                 PatternRewriter& rewriter) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_31(mht_31_v, 954, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "EliminatePassThroughResults");

  mlir::Block& body = op.GetBody();
  Operation* return_op = body.getTerminator();
  int num_results = return_op->getNumOperands();

  // Values defined within the cluster.
  llvm::SmallVector<Value, 4> cluster_vals;
  cluster_vals.reserve(num_results);

  // New results stores values to use while replacing the old cluster op.
  llvm::SmallVector<Value, 4> new_results;
  new_results.reserve(num_results);
  for (OpOperand& operand : return_op->getOpOperands()) {
    // If the corresponding result of the cluster op is used in some resource
    // update op, do not eliminate the result. Such assignment ops could be for
    // device resources and are required during fusing of the execute op and
    // the resource update ops.
    bool is_used_for_resource_write = llvm::any_of(
        op.getResult(operand.getOperandNumber()).getUsers(),
        [](Operation* user) { return isa<TF::AssignVariableOp>(user); });

    // TODO(b/186717563): Eliminate all pass through results once XLA correctly
    // handles empty computations. Another approach could be to drop empty
    // clusters within MLIR but that seems to trigger other failures but can be
    // considered again.
    // Old bridge only removes unsupported TPU types (only string for now)
    // during outside compilation extraction so this should be enough for
    // the parity.
    bool is_unsupported_type = getElementTypeOrSelf(operand.get().getType())
                                   .isa<mlir::TF::StringType>();
    Value result = operand.get();
    if (is_unsupported_type && result.getParentBlock() != &body &&
        !is_used_for_resource_write) {
      // Pass through result.
      new_results.push_back(result);
    } else {
      // This result will be populated with the new result after rewriting the
      // cluster op.
      new_results.push_back(nullptr);
      cluster_vals.push_back(result);
    }
  }

  // Return failure if there are no pass through results and op is already
  // canonical.
  if (cluster_vals.size() == num_results) return failure();

  // Rewrite return op in the cluster.
  rewriter.setInsertionPoint(return_op);
  auto new_return =
      rewriter.replaceOpWithNewOp<tf_device::ReturnOp>(return_op, cluster_vals);

  // Rewrite the cluster op.
  rewriter.setInsertionPoint(op);
  auto new_op = rewriter.create<tf_device::ClusterOp>(
      op->getLoc(), new_return.getOperandTypes(), op->getOperands(),
      op->getAttrs());
  rewriter.inlineRegionBefore(op.getBodyRegion(), new_op.getBodyRegion(),
                              new_op.getBodyRegion().end());

  int idx = 0;
  for (Value& result : new_results) {
    if (result == nullptr) result = new_op.getResult(idx++);
  }
  rewriter.replaceOp(op, new_results);
  return success();
}
}  // anonymous namespace

void ClusterOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_32(mht_32_v, 1027, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "ClusterOp::getCanonicalizationPatterns");

  results.add(EliminatePassThroughResults);
}

//===----------------------------------------------------------------------===//
// tf_device.launch
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches LaunchOps with only one ReturnOp (empty) and remaps the
// results of the LaunchOp to the operands of the ReturnOp.
struct DropEmptyLaunch : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_33(mht_33_v, 1045, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "matchAndRewrite");

    Block& block = op.GetBody();
    // Check if launch only has a return.
    if (&block.front() != &block.back()) return failure();

    // Map launch results to return operands.
    rewriter.replaceOp(op, block.front().getOperands());

    return success();
  }
};
}  // anonymous namespace

void LaunchOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_deviceDTcc mht_34(mht_34_v, 1062, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc", "LaunchOp::getCanonicalizationPatterns");

  results.add<DropEmptyLaunch>(context);
}

}  // namespace tf_device
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
