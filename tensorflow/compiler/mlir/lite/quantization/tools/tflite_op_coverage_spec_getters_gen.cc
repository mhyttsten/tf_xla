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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <regex>  // NOLINT
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

enum class InputDataType { INT8, UINT8, INT16 };

// One InputDataType will be likely mapped to multiple types in near future so
// the two structures are separated.
const std::map<std::string, std::string> &GetTypeToStringRepresentation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "GetTypeToStringRepresentation");

  static auto *entries = new std::map<std::string, std::string>({
      {"F32", "32-bit float"},
      {"I32", "32-bit signless integer"},
      {"I64", "64-bit signless integer"},
      {"QI16", "QI16 type"},
      {"I8", "8-bit signless integer"},
      {"UI8", "8-bit unsigned integer"},
      {"QI8", "QI8 type"},
      {"QUI8", "QUI8 type"},
      {"TFL_Quint8", "TFLite quint8 type"},
  });

  return *entries;
}

void EmitDynamicRangeOp(std::vector<Record *> &defs, raw_ostream *ostream) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_1(mht_1_v, 226, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitDynamicRangeOp");

  std::string dynamic_quant_kernel_support_regex =
      "bool GetDynamicRangeQuantKernelSupport() { return true; }";
  raw_ostream &os = *ostream;
  std::vector<std::string> weight_only;
  llvm::sort(defs, LessRecord());

  os.indent(0) << "const std::set<std::string> &ExportDynamicRangeSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve all the ops that have DynamicRangeQuantizedOpInterface trait.
  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("DynamicRangeQuantizedOpInterface::Trait")) continue;

    auto op_name = op.getCppClassName();
    auto op_extra_declaration = op.getExtraClassDeclaration().str();

    bool kernel_support = absl::StrContains(
        absl::StrReplaceAll(op_extra_declaration, {{"\n", " "}}),
        dynamic_quant_kernel_support_regex);

    // Classify dynamic range and weight-only fallback
    if (kernel_support) {
      os.indent(6) << "\"" << op_name << "\",\n";
    } else {
      weight_only.push_back(op_name.str());
    }
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";

  os.indent(0)
      << "const std::set<std::string> &ExportDynamicRangeWeightOnlySpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve weight-only fallback.
  for (const auto &op_name : weight_only) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitSparseOp(std::vector<Record *> &defs, raw_ostream *ostream) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_2(mht_2_v, 279, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitSparseOp");

  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  os.indent(0) << "const std::set<std::string> &ExportSparsitySpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  // Retrieve all the ops that have SparseOp trait.
  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("SparseOpInterface::Trait")) {
      continue;
    }
    os.indent(6) << "\"" << op.getCppClassName() << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

bool CheckTypeConstraints(llvm::Init *input_value,
                          std::list<std::string> required_types,
                          bool per_axis) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_3(mht_3_v, 306, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "CheckTypeConstraints");

  auto *def_init = llvm::cast<llvm::DefInit>(input_value);
  auto *val = def_init->getDef()->getValue("tflRuntimeTypePredicate");

  // For non-per-axis op, no predicate means accepting AnyTensor.
  if (!val) return !per_axis;

  llvm::StringRef supported_types =
      def_init->getDef()->getValueAsString("tflRuntimeTypeDescription");

  for (const std::string &type : required_types) {
    if (!absl::StrContains(supported_types.str(), type)) return false;
  }
  return true;
}

void GenerateStaticQuantOp(std::vector<Record *> &defs,
                           std::vector<std::string> &result,
                           InputDataType act_type, bool per_axis) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_4(mht_4_v, 327, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "GenerateStaticQuantOp");

  std::list<std::string> required_types = {
      GetTypeToStringRepresentation().at("F32")};

  switch (act_type) {
    case InputDataType::INT8: {
      required_types.push_back(GetTypeToStringRepresentation().at("QI8"));
      break;
    }
    case InputDataType::UINT8: {
      required_types.push_back(GetTypeToStringRepresentation().at("QUI8"));
      break;
    }
    case InputDataType::INT16: {
      required_types.push_back(GetTypeToStringRepresentation().at("QI16"));
      break;
    }
    default: {
      // Quantization not applied.
      return;
    }
  }

  // Dimension equals to -1 means per-channel quantization is not supported for
  // the op. Therefore check whether the return value is positive integer as
  // well.
  std::regex per_channel_support_regex(
      "(.*)(int GetQuantizationDimIndex\\(\\) \\{ return (\\d*); \\})(.*)");

  for (const auto *def : defs) {
    Operator op(def);
    if (!op.getTrait("::mlir::OpTrait::quant::QuantizableResult")) continue;

    llvm::DagInit *args_in_dag = def->getValueAsDag("arguments");
    // Assumes argument name is "input" for input activations. Otherwise, assume
    // the first argument is the input activation.
    int input_idx = 0;
    for (int i = 0; i < args_in_dag->getNumArgs(); i++) {
      if (args_in_dag->getArgName(i)->getAsString() == "\"input\"")
        input_idx = i;
    }
    if (CheckTypeConstraints(args_in_dag->getArg(input_idx), required_types,
                             per_axis)) {
      std::string op_name = op.getCppClassName().str();

      if (per_axis) {
        std::string op_extra_declaration = op.getExtraClassDeclaration().str();
        bool per_axis_support = std::regex_match(
            absl::StrReplaceAll(op_extra_declaration, {{"\n", " "}}),
            per_channel_support_regex);
        if (per_axis_support) result.emplace_back(op_name);
      } else {
        result.emplace_back(op_name);
      }
    }
  }
}

void EmitStaticInt8PerAxisQuantOp(std::vector<Record *> &defs,
                                  raw_ostream &os) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_5(mht_5_v, 389, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticInt8PerAxisQuantOp");

  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8PerAxisSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT8, true);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticInt8PerTensorQuantOp(std::vector<Record *> &defs,
                                    raw_ostream &os) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_6(mht_6_v, 411, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticInt8PerTensorQuantOp");

  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8PerTensorSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT8, false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticUInt8PerAxisQuantOp(std::vector<Record *> &defs,
                                   raw_ostream &os) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_7(mht_7_v, 433, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticUInt8PerAxisQuantOp");

  os.indent(0)
      << "const std::set<std::string> &ExportStaticUInt8PerAxisSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::UINT8, true);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticUInt8PerTensorQuantOp(std::vector<Record *> &defs,
                                     raw_ostream &os) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_8(mht_8_v, 455, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticUInt8PerTensorQuantOp");

  os.indent(0)
      << "const std::set<std::string> &ExportStaticUInt8PerTensorSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::UINT8, false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

void EmitStaticQuantOp(std::vector<Record *> &defs, raw_ostream *ostream) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_9(mht_9_v, 476, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticQuantOp");

  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  EmitStaticInt8PerAxisQuantOp(defs, os);
  EmitStaticInt8PerTensorQuantOp(defs, os);
  EmitStaticUInt8PerAxisQuantOp(defs, os);
  EmitStaticUInt8PerTensorQuantOp(defs, os);
}

void EmitStaticQuantWithInt16ActOp(std::vector<Record *> &defs,
                                   raw_ostream *ostream) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_10(mht_10_v, 490, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "EmitStaticQuantWithInt16ActOp");

  raw_ostream &os = *ostream;
  llvm::sort(defs, LessRecord());

  os.indent(0)
      << "const std::set<std::string> &ExportStaticInt8WithInt16ActSpec() {\n";
  os.indent(2) << "static const std::set<std::string> * result =\n";
  os.indent(4) << "new std::set<std::string>({\n";

  std::vector<std::string> result;
  GenerateStaticQuantOp(defs, result, InputDataType::INT16, false);

  for (const auto &op_name : result) {
    os.indent(6) << "\"" << op_name << "\",\n";
  }

  os.indent(4) << "});";
  os.indent(2) << "return *result;\n";
  os.indent(0) << "}\n";
}

static bool TFLiteOpCoverageSpecWritersMain(raw_ostream &os,
                                            RecordKeeper &records) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_11(mht_11_v, 515, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "TFLiteOpCoverageSpecWritersMain");

  std::vector<Record *> op_defs = records.getAllDerivedDefinitions("TFL_Op");
  EmitStaticQuantOp(op_defs, &os);
  EmitDynamicRangeOp(op_defs, &os);
  EmitStaticQuantWithInt16ActOp(op_defs, &os);
  EmitSparseOp(op_defs, &os);
  return false;
}

int main(int argc, char **argv) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPStflite_op_coverage_spec_getters_genDTcc mht_12(mht_12_v, 527, "", "./tensorflow/compiler/mlir/lite/quantization/tools/tflite_op_coverage_spec_getters_gen.cc", "main");

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &TFLiteOpCoverageSpecWritersMain);
}
