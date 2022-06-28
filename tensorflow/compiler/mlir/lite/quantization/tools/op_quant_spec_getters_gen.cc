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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPSop_quant_spec_getters_genDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPSop_quant_spec_getters_genDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPSop_quant_spec_getters_genDTcc() {
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Operator.h"  // from @llvm-project

using llvm::LessRecord;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::Operator;

// Helper macro that returns indented os.
#define OUT(X) os.indent((X))

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
static bool OpQuantSpecWriter(raw_ostream &os, RecordKeeper &records) {
  llvm::Regex acc_uniform_trait_regex{"AccumulatorUniformScale<([0-9]*),"};
  llvm::Regex coeff_index_trait_regex{"AffineOpCoefficient<(-?[0-9]*),"};
  llvm::Regex fixed_uniform_trait_regex{
      "FixedResultUniformScale<([0-9]+).*(true|false)>"};
  emitSourceFileHeader("Generated Ops Quant Spec Getters", os);

  // Retrieve all the definitions derived from Op definition and sort by record
  // name.
  std::vector<Record *> defs = records.getAllDerivedDefinitions("Op");
  llvm::sort(defs, LessRecord());

  OUT(0) << "static std::unique_ptr<quant::OpQuantSpec> "
            "GetOpQuantSpec(mlir::Operation *op) {\n";
  // TODO(b/176258587): Move to OpTrait if this should be generalized.
  // Add special handling for LSTM.
  OUT(2) << "if (auto lstm_op = llvm::dyn_cast<TFL::LSTMOp>(op)) {\n";
  OUT(4) << "return GetLstmOpQuantSpec<TFL::LSTMOp>(lstm_op);\n";
  OUT(2) << "} else if (auto lstm_op = "
            "llvm::dyn_cast<TFL::UnidirectionalSequenceLSTMOp>(op)) {\n";
  OUT(4) << "return "
            "GetLstmOpQuantSpec<TFL::UnidirectionalSequenceLSTMOp>(lstm_op);\n";
  OUT(2) << "}\n";

  OUT(2) << "auto spec = absl::make_unique<quant::OpQuantSpec>();\n";
  llvm::SmallVector<llvm::StringRef, 3> matches;
  for (auto *def : defs) {
    Operator op(def);
    for (const auto t : op.getTraits()) {
      if (auto opTrait = llvm::dyn_cast<mlir::tblgen::NativeTrait>(&t)) {
        auto trait_str = opTrait->getFullyQualifiedTraitName();
        if (!llvm::StringRef{trait_str}.consume_front(
                "::mlir::OpTrait::quant::"))
          continue;

        OUT(2) << "if (auto tfl = llvm::dyn_cast<" << op.getQualCppClassName()
               << ">(op)) {\n";
        // There is a "FixedResultUniformScale" trait, set the type for result.
        if (fixed_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "for (int i = 0, e = op->getNumResults(); i != e; ++i)\n";
          OUT(6) << "spec->restricted_output_params[std::make_pair("
                 << matches[1] << ", " << matches[2]
                 << ")].push_back(tfl.::mlir::OpTrait::quant::" << trait_str
                 << "<" << op.getQualCppClassName()
                 << ">::GetResultQuantizedType(i));\n";
          matches.clear();
        }
        // There is a "AccumulatorUniformScale" trait, set the type for bias.
        if (acc_uniform_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->biases_params.emplace(std::make_pair(" << matches[1]
                 << ", std::make_pair(tfl.GetAllNonBiasOperands(),"
                 << "quant::GetUniformQuantizedTypeForBias)));\n";
          matches.clear();
        }
        // There is a "QuantChannelDim" trait, set the quantization dimension.
        if (coeff_index_trait_regex.match(trait_str, &matches)) {
          OUT(4) << "spec->coeff_op_quant_dim[tfl.GetCoefficientOperandIndex()"
                 << "] = tfl.GetQuantizationDim();\n";
          matches.clear();
        }

        OUT(2) << "}\n";
      }
    }
  }
  OUT(2) << "return spec;\n";
  OUT(0) << "}\n";
  return false;
}

int main(int argc, char **argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPStoolsPSop_quant_spec_getters_genDTcc mht_0(mht_0_v, 277, "", "./tensorflow/compiler/mlir/lite/quantization/tools/op_quant_spec_getters_gen.cc", "main");

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OpQuantSpecWriter);
}
