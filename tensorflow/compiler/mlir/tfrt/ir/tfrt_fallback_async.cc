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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc() {
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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/sync/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_async {

namespace {

struct FallbackInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *dest, bool would_be_cloned,
                       BlockAndValueMapping &) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "isLegalToInline");

    return true;
  }
};

}  // namespace

FallbackAsyncDialect::FallbackAsyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback_async", context,
              TypeID::get<FallbackAsyncDialect>()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "FallbackAsyncDialect::FallbackAsyncDialect");

  context->getOrLoadDialect<tfrt::fallback::FallbackDialect>();
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<corert::CoreRTDialect>();

  allowUnknownTypes();

  allowUnknownOperations();

  addInterfaces<FallbackInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cpp.inc"
      >();
}

static Type GetChainType(Builder *builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "GetChainType");

  return builder->getType<compiler::ChainType>();
}

LogicalResult CreateOp::verify() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "CreateOp::verify");

  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOp::verify() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_4(mht_4_v, 263, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOp::verify");

  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOpSeq::verify() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_5(mht_5_v, 269, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeq::verify");

  return fallback_common::VerifyFallbackExecuteOp(*this);
}
LogicalResult ExecuteOpWithAllocator::verify() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_6(mht_6_v, 275, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpWithAllocator::verify");

  return fallback_common::VerifyExecuteOpCommon(*this);
}
LogicalResult ExecuteOpSeqWithAllocator::verify() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_7(mht_7_v, 281, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeqWithAllocator::verify");

  return fallback_common::VerifyExecuteOpCommon(*this);
}
LogicalResult BatchFunctionOp::verify() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_8(mht_8_v, 287, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "BatchFunctionOp::verify");

  return fallback_common::VerifyExecuteOpCommon(*this);
}

ParseResult CreateOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_9(mht_9_v, 294, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "CreateOp::parse");

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = true;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = false;

  auto &builder = parser.getBuilder();
  if (mlir::failed(fallback_common::ParseExecuteOpCommon(
          parser, builder, result, builder.getType<fallback::TFTensorType>(),
          parse_options)))
    return mlir::failure();

  mlir::IntegerAttr num_args;
  if (parser.parseKeyword("num_args") || parser.parseLParen() ||
      parser.parseAttribute(num_args, "num_args", result.attributes) ||
      parser.parseRParen())
    return mlir::failure();

  return mlir::success();
}
ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_10(mht_10_v, 319, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOp::parse");

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpSeq::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_11(mht_11_v, 335, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeq::parse");

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = true;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpWithAllocator::parse(OpAsmParser &parser,
                                          OperationState &result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_12(mht_12_v, 352, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpWithAllocator::parse");

  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> allocator;
  if (parser.parseOperandList(allocator,
                              /*requiredOperandCount=*/1,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  if (parser.resolveOperands(allocator.front(),
                             builder.getType<fallback::TFAllocatorType>(),
                             result.operands))
    return mlir::failure();

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}
ParseResult ExecuteOpSeqWithAllocator::parse(OpAsmParser &parser,
                                             OperationState &result) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_13(mht_13_v, 380, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeqWithAllocator::parse");

  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2>
      chain_and_allocator;
  if (parser.parseOperandList(chain_and_allocator,
                              /*requiredOperandCount=*/2,
                              mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  auto &chain = chain_and_allocator[0];
  auto &allocator = chain_and_allocator[1];

  if (parser.resolveOperands(chain, builder.getType<compiler::ChainType>(),
                             result.operands))
    return mlir::failure();

  if (parser.resolveOperands(allocator,
                             builder.getType<fallback::TFAllocatorType>(),
                             result.operands))
    return mlir::failure();

  // The first result is a chain.
  result.types.push_back(builder.getType<compiler::ChainType>());

  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = true;
  parse_options.has_device = true;
  parse_options.has_func_attr = true;
  parse_options.has_cost = true;

  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, builder.getType<fallback::TFTensorType>(),
      parse_options);
}

ParseResult BatchFunctionOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_14(mht_14_v, 420, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "BatchFunctionOp::parse");

  auto &builder = parser.getBuilder();
  auto chain_type = GetChainType(&builder);
  auto tensorhandle_type = builder.getType<corert::TensorHandleType>();

  FlatSymbolRefAttr f;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> in_chains;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  NamedAttrList op_attrs;
  auto loc = parser.getNameLoc();

  if (parser.parseOperandList(in_chains,
                              /*requiredOperandCount=*/1,
                              OpAsmParser::Delimiter::Paren))
    return failure();

  if (parser.parseAttribute(f, "f", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(op_attrs))
    return failure();

  int64_t num_results = 0;
  if (succeeded(parser.parseOptionalColon())) {
    IntegerAttr attr;
    mlir::NamedAttrList attrs;
    if (failed(parser.parseAttribute(attr, "num_results", attrs)))
      return failure();
    num_results = attr.getValue().getSExtValue();
  }

  SmallVector<Type, 4> operand_types;
  operand_types.push_back(chain_type);
  if (parser.resolveOperands(in_chains, operand_types, loc, result.operands) ||
      parser.resolveOperands(operands, tensorhandle_type, result.operands))
    return failure();

  result.types.push_back(chain_type);
  result.types.append(num_results, tensorhandle_type);

  SmallVector<Attribute, 4> op_attr_array;
  for (const auto &key_value : op_attrs) {
    auto key = key_value.getName();
    auto value = key_value.getValue();
    op_attr_array.push_back(builder.getArrayAttr({key, value}));
  }

  result.attributes.push_back(
      builder.getNamedAttr("op_attrs", builder.getArrayAttr(op_attr_array)));

  return success();
}

void CreateOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_15(mht_15_v, 475, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "CreateOp::print");

  CreateOp op = *this;
  p << "(" << op.in_ch() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") device("
    << op->getAttr("device") << ") " << op->getAttr("op_name") << "()";

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);

  p << " num_args(" << op->getAttrOfType<mlir::IntegerAttr>("num_args").getInt()
    << ')';
}

void ExecuteOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_16(mht_16_v, 491, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOp::print");

  ExecuteOp op = *this;
  p << " key(" << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt()
    << ") cost(" << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void ExecuteOpSeq::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_17(mht_17_v, 506, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeq::print");

  ExecuteOpSeq op = *this;
  p << "(" << op.in_op_chain() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void ExecuteOpWithAllocator::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_18(mht_18_v, 522, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpWithAllocator::print");

  ExecuteOpWithAllocator op = *this;
  p << "(" << op.allocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void ExecuteOpSeqWithAllocator::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_19(mht_19_v, 538, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOpSeqWithAllocator::print");

  ExecuteOpSeqWithAllocator op = *this;
  p << "(" << op.in_op_chain() << ", " << op.allocator() << ") key("
    << op->getAttrOfType<mlir::IntegerAttr>("op_key").getInt() << ") cost("
    << op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt()
    << ") device(" << op->getAttr("device") << ") " << op->getAttr("op_name")
    << '(' << op.operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, op);
  fallback_common::PrintExecuteOpFuncAttribute(p, op);
  if (!op.results().empty()) p << " : " << op.results().size();
}

void BatchFunctionOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_20(mht_20_v, 554, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "BatchFunctionOp::print");

  p << "(" << in_op_chain() << ") " << getOperation()->getAttr("f") << " ("
    << operands() << ") ";

  fallback_common::PrintExecuteOpCommon(p, *this);
  if (!results().empty()) p << " : " << results().size();
}

void ExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_21(mht_21_v, 566, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ExecuteOp::getOpAttrs");

  fallback_common::GetExecuteOpAttrsCommon(
      this->getContext(), this->op_attrs().getValue(), op_attrs);
}

//===----------------------------------------------------------------------===//
// ConstDenseTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstDenseTensorOp::fold(ArrayRef<Attribute> operands) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_22(mht_22_v, 578, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "ConstDenseTensorOp::fold");

  return value();
}

//===----------------------------------------------------------------------===//
// CoreRTTensorHandleToFallbackTensorOp
//===----------------------------------------------------------------------===//

namespace {

// Simplifies pattern containing a corert const tensor op followed by a
// `tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor` op to a single
// tfrt_fallback_async const tensor.
struct ConstCoreRTTensorHandleToFallbackTensorCanonicalization
    : public OpRewritePattern<CoreRTTensorHandleToFallbackTensorOp> {
  using OpRewritePattern<
      CoreRTTensorHandleToFallbackTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoreRTTensorHandleToFallbackTensorOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_23(mht_23_v, 600, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "matchAndRewrite");

    SmallVector<Value, 1> new_values;
    bool should_rewrite = false;
    for (auto operand : op.operands()) {
      if (auto corert_const_dense_tensor_op =
              operand.getDefiningOp<corert::ConstDenseTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstDenseTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_dense_tensor_op.value()));
        should_rewrite = true;
        continue;
      }
      if (auto corert_const_string_tensor_op =
              operand.getDefiningOp<corert::ConstStringTensorOp>()) {
        new_values.push_back(
            rewriter.create<fallback_async::ConstStringTensorOp>(
                op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                corert_const_string_tensor_op.shape(),
                corert_const_string_tensor_op.value()));
        should_rewrite = true;
        continue;
      }
      // To guarantee that the new values are in the same order as the old
      // ones, we create individual ops for the non-canonicalizable operands.
      // For simplicity, we don't consolidate these ops when all the
      // non-canonicalizable operands are adjacent.
      new_values.push_back(
          rewriter
              .create<fallback_async::CoreRTTensorHandleToFallbackTensorOp>(
                  op.getLoc(), rewriter.getType<fallback::TFTensorType>(),
                  operand, op->getAttrOfType<mlir::StringAttr>("device"))
              .getResult(0));
    }

    if (!should_rewrite) return failure();
    rewriter.replaceOp(op, new_values);
    return success();
  }
};

// Removes the following double tensor conversion:
//  %1 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %0
//  %2 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %1
struct RemoveDoubleTensorConversion
    : mlir::OpRewritePattern<CoreRTTensorHandleToFallbackTensorOp> {
  using OpRewritePattern<
      CoreRTTensorHandleToFallbackTensorOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      CoreRTTensorHandleToFallbackTensorOp op,
      mlir::PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_24(mht_24_v, 654, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "matchAndRewrite");

    // Currently only handles the case where there is only one value in the
    // conversion op. This should be enough for most of the cases.
    if (op.getNumOperands() > 1) return mlir::failure();

    auto def =
        op.getOperand(0).getDefiningOp<FallbackTensorToCoreRTTensorHandleOp>();
    if (!def) return mlir::failure();

    if (def.getNumResults() > 1) return mlir::failure();

    rewriter.replaceOp(op, def.getOperand(0));

    return mlir::success();
  }
};

}  // namespace

void CoreRTTensorHandleToFallbackTensorOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSirPStfrt_fallback_asyncDTcc mht_25(mht_25_v, 677, "", "./tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cc", "CoreRTTensorHandleToFallbackTensorOp::getCanonicalizationPatterns");

  results.add<ConstCoreRTTensorHandleToFallbackTensorCanonicalization,
              RemoveDoubleTensorConversion>(context);
}

}  // namespace fallback_async
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.cpp.inc"
