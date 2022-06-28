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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc() {
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

#include <sstream>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project

using llvm::interleaveComma;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::Attribute;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::NamedTypeConstraint;
using mlir::tblgen::Operator;

static std::string GetDefaultAttrExport(
    const mlir::tblgen::NamedAttribute& named_attr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/mlir/xla/operator_writer_gen.cc", "GetDefaultAttrExport");

  Attribute attr = named_attr.attr;
  StringRef storage_type = attr.getStorageType();
  // For some attribute types we have a general conversion, so use that.
  if (!attr.isEnumAttr() && (storage_type.endswith("BoolAttr") ||
                             storage_type.endswith("FloatAttr") ||
                             storage_type.endswith("IntegerAttr") ||
                             storage_type.endswith("StringAttr"))) {
    // The return type may contains qualified namespaces. Split to remove them.
    std::pair<StringRef, StringRef> splits = attr.getReturnType().rsplit("::");
    StringRef symbol = splits.second;
    if (symbol.empty()) symbol = splits.first;
    return "Convert" + symbol.str();
  }
  return "Convert_" + named_attr.name.str();
}

static StringRef GetClientBuilder(const Operator& op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/mlir/xla/operator_writer_gen.cc", "GetClientBuilder");

  static const auto* kOpToXLABuilderMap =
      new llvm::StringMap<StringRef>{{"ReverseOp", "Rev"},
                                     {"ConcatenateOp", "ConcatInDim"},
                                     {"ConvOp", "ConvGeneralDilated"}};

  StringRef op_name = op.getCppClassName();

  // Default case where the client builder method names closely follow the op
  // names in the dialect. For e.g., AddOp -> xla::Add method.
  if (!kOpToXLABuilderMap->count(op_name)) return op_name.drop_back(2);

  // Otherwise, if the op to client builder method mapping is provided.
  return kOpToXLABuilderMap->lookup(op_name);
}

static void BuildOperator(const Operator& op, raw_ostream& os) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/mlir/xla/operator_writer_gen.cc", "BuildOperator");

  os << "mlir::LogicalResult ExportXlaOp(mlir::mhlo::" << op.getCppClassName()
     << " op, OpLoweringContext ctx) {\n"
     << "  auto& value_map = *ctx.values;\n"
     << "  auto result = op.getResult();\n";

  // Build a conversion for each of the arguments.
  int operand_number = 0;
  for (int index : llvm::seq<int>(0, op.getNumArgs())) {
    auto arg = op.getArg(index);

    // Emit an argument for an operand.
    if (auto* operand_cst = arg.dyn_cast<NamedTypeConstraint*>()) {
      std::string xla_arg = "xla_arg_" + std::to_string(index);
      // Handle a non-variadic operand.
      if (!operand_cst->isVariableLength()) {
        os << "  xla::XlaOp " << xla_arg << ";\n";
        os << "  if (failed(GetXlaOp(*op.getODSOperands(" << operand_number++
           << ").begin(), value_map, &" << xla_arg << ", op)))\n";
        os << "    return mlir::failure();\n";
        continue;
      }

      // Otherwise, this is a varidiac operand list.
      os << "  std::vector<xla::XlaOp> " << xla_arg << ";\n"
         << "  for (auto operand : op.getODSOperands(" << operand_number++
         << ")) {\n";
      os << "    xla::XlaOp result;\n";
      os << "    if (failed(GetXlaOp(operand, value_map, &result, op)))\n";
      os << "      return mlir::failure();\n";
      os << "    " << xla_arg << ".push_back(result);\n";
      os << "  }\n";
      continue;
    }

    // Otherwise, this is an attribute.
    auto named_attr = arg.get<NamedAttribute*>();
    os << "  auto xla_arg_" << index << " = "
       << GetDefaultAttrExport(*named_attr) << "(op." << op.getArgName(index)
       << "());\n";
  }

  // Emit call to client API
  os << "  auto xla_result = xla::" << GetClientBuilder(op) << "(";

  // If all operands are variadic, then pass the builder explicitly to xla
  // client API call
  if (op.getNumOperands() == op.getNumVariableLengthOperands()) {
    os << "ctx.builder";
    if (op.getNumArgs() != 0) os << ", ";
  }

  // Emit each of the arguments.
  interleaveComma(llvm::seq<int>(0, op.getNumArgs()), os,
                  [&](int i) { os << "Unwrap(xla_arg_" << i << ')'; });
  os << ");\n";

  os << "  value_map[result] = xla_result;\n";
  os << "  return mlir::success();\n";
  os << "}\n";
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OperatorWritersMain(raw_ostream& os, RecordKeeper& records) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc mht_3(mht_3_v, 318, "", "./tensorflow/compiler/mlir/xla/operator_writer_gen.cc", "OperatorWritersMain");

  emitSourceFileHeader("MLIR XLA Builders", os);

  // Emit all the helper functions.
  for (const auto* def : records.getAllDerivedDefinitions("HLO_Op")) {
    Operator op(def);

    // Skip operations that have a custom exporter.
    if (!def->getValueAsBit("hasCustomHLOConverter")) BuildOperator(op, os);
  }

  // Emit a function to generate an XLA operation for the operations with
  // auto-generated builders.
  os << "mlir::LogicalResult ExportXlaOperator(\n"
        "mlir::Operation* op, OpLoweringContext lowering_context) {\n\n";

  // Create a scoped object to assign sharding to generated XLA ops. Any HLO
  // can have an attribute of "sharding".
  os << "  xla::XlaScopedShardingAssignment sharding(lowering_context.builder, "
        "CreateOpShardingFromAttribute(op));\n\n";

  // Create a scoped object to assign frontend attributes to generated XLA ops.
  // Any HLO can have an attribute of "frontend_attributes", which are used to
  // pass hints / configuration options.
  os << "  xla::XlaScopedFrontendAttributesAssignment "
        "frontend_attributes(lowering_context.builder, "
        "CreateOpFrontendAttributesFromAttribute(op));\n\n";

  // Create a scoped object to assign op metadata to generated XLA ops.
  os << "  xla::XlaScopedOpMetadataAssignment "
        "op_metadata(lowering_context.builder, "
        "CreateOpMetadataFromLocation(op, "
        "lowering_context.converter->GetOptions()));\n\n";

  // Retrieve all the definitions derived from HLO_Op and sort by record name.
  for (const auto* def : records.getAllDerivedDefinitions("HLO_Op")) {
    // Skip operations that have a custom exporter.
    Operator op(def);

    // Cast to the current operation and build the exporter.
    os << "  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::"
       << op.getCppClassName() << ">(op)) {\n";
    os << "    return ";
    // The autogenerated converters aren't in the same namespace.
    // TODO(jpienaar): Reconsider this.
    if (def->getValueAsBit("hasCustomHLOConverter")) os << "mlir::mhlo::";
    os << "ExportXlaOp(xla_op, lowering_context);\n";
    os << "  }\n";
  }

  os << "  return mlir::failure();\n"
        "}\n";
  return false;
}

int main(int argc, char** argv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSoperator_writer_genDTcc mht_4(mht_4_v, 376, "", "./tensorflow/compiler/mlir/xla/operator_writer_gen.cc", "main");

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OperatorWritersMain);
}
