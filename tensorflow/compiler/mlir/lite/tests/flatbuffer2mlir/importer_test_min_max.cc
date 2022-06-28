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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStestsPSflatbuffer2mlirPSimporter_test_min_maxDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStestsPSflatbuffer2mlirPSimporter_test_min_maxDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStestsPSflatbuffer2mlirPSimporter_test_min_maxDTcc() {
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

#include <iostream>
#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

using llvm::Optional;
using llvm::cl::opt;

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s.mlir -o - \
// RUN:   | %p/importer_test_min_max - \
// RUN:   | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - \
// RUN:   | FileCheck %s

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s.mlir -o - \
// RUN:   | %p/importer_test_min_max - \
// RUN:   | flatbuffer_to_string - \
// RUN:   | FileCheck --check-prefix=FB %s

// Tests for verifying the tflite model with min/max can be imported
// correctly.

// NOLINTNEXTLINE
static opt<std::string> inputFileName(llvm::cl::Positional,
                                      llvm::cl::desc("<input file>"),
                                      llvm::cl::init("-"));

namespace mlir {
namespace {
Optional<std::unique_ptr<tflite::ModelT>> InjectStatsToFullyConnected(
    llvm::StringRef buffer) {
  auto model_ptr = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer.data(), buffer.size());
  if (nullptr == model_ptr) {
    return llvm::None;
  }
  std::unique_ptr<tflite::ModelT> model(model_ptr->GetModel()->UnPack());

  // FB-LABEL:     name: "arg0",
  // FB-NEXT:      quantization: {
  // FB-NEXT:              min: [ -1.0 ],
  // FB-NEXT:              max: [ 1.0 ]
  // FB-NEXT:      }

  // FB-LABEL:     name: "arg1",
  // FB-NEXT:            quantization: {
  // FB-EMPTY:
  // FB-NEXT:            }

  // FB-LABEL:     name: "tfl.fully_connected",
  // FB-NEXT:      quantization: {
  // FB-NEXT:        min: [ -0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0,
  // FB-SAME:  -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0,
  // FB-SAME:  -17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0, -24.0, -25.0,
  // FB-SAME:  -26.0, -27.0, -28.0, -29.0, -30.0, -31.0, -32.0, -33.0, -34.0,
  // FB-SAME:  -35.0, -36.0, -37.0, -38.0, -39.0 ],
  // FB-NEXT:        max: [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  // FB-SAME:  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
  // FB-SAME:  21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
  // FB-SAME:  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0 ],
  // FB-NEXT:        quantized_dimension: 1
  // FB-NEXT:      }

  // FB-LABEL:     name: "tfl.fully_connected:1",
  // FB-NEXT:      quantization: {
  // FB-EMPTY:
  // FB-NEXT:      }

  // FB-LABEL:      operators: [ {
  // FB-NEXT:             inputs: [ 0, 1, 2 ],
  // FB-NEXT:             outputs: [ 3, 4 ],
  // FB-NEXT:             builtin_options_type: FullyConnectedOptions,
  // FB-NEXT:             builtin_options: {
  // FB-EMPTY:
  // FB-NEXT:             }
  // FB-NEXT:       } ],

  // CHECK-LABEL: func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>)
  // CHECK-SAME:      -> tensor<40x40xf32>
  // CHECK:         %[[stat:.*]] = "quant.stats"(%arg0) {layerStats = dense<
  // CHECK-SAME:      [-1.000000e+00, 1.000000e+00]> : tensor<2xf32>}
  // CHECK-SAME:      : (tensor<40x37xf32>) -> tensor<40x37xf32>
  // CHECK-NEXT:    %[[cst:.*]] = "tfl.pseudo_const"() {value = dense<
  // CHECK-SAME:      1.000000e+00> : tensor<40xf32>} : () -> tensor<40xf32>
  // CHECK-NEXT:    %[[fc:.*]]:2 = "tfl.fully_connected"(%[[stat]], %arg1,
  // CHECK-NEXT:    %[[stat1:.*]] = "quant.stats"(%[[fc]]#0) {axis = 1 : i64,
  // CHECK-SAME:      axisStats = dense<{{\[}}[-0.000000e+00, 0.000000e+00],
  // CHECK-SAME:      [-1.000000e+00, 1.000000e+00],
  // CHECK-SAME:      [-2.000000e+00, 2.000000e+00]
  // CHECK-NEXT:    return %[[stat1]] : tensor<40x40xf32>
  // CHECK-NEXT:  }

  // Find the tensors and inject the min and max to the input and output
  for (auto& sub_graph : model->subgraphs) {
    for (auto& op : sub_graph->operators) {
      if (tflite::GetBuiltinCode(
              model->operator_codes[op->opcode_index].get()) ==
          tflite::BuiltinOperator_FULLY_CONNECTED) {
        // inject min/max to the input and output tensors
        auto& input_tensor = sub_graph->tensors[op->inputs[0]];
        input_tensor->quantization->scale.clear();
        input_tensor->quantization->zero_point.clear();
        input_tensor->quantization->min.push_back(-1.0);
        input_tensor->quantization->max.push_back(1.0);

        auto& output_tensor = sub_graph->tensors[op->outputs[0]];
        auto shape = output_tensor->shape;
        output_tensor->quantization->scale.clear();
        output_tensor->quantization->zero_point.clear();
        for (int i = 0; i < shape.back(); ++i) {
          output_tensor->quantization->min.push_back(-1.0 * i);
          output_tensor->quantization->max.push_back(1.0 * i);
        }
        output_tensor->quantization->quantized_dimension = shape.size() - 1;
      }
    }
  }
  return model;
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStestsPSflatbuffer2mlirPSimporter_test_min_maxDTcc mht_0(mht_0_v, 315, "", "./tensorflow/compiler/mlir/lite/tests/flatbuffer2mlir/importer_test_min_max.cc", "main");

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName.c_str());
  if (std::error_code error = file_or_err.getError()) {
    llvm::errs() << argv[0] << ": could not open input file '" << inputFileName
                 << "': " << error.message() << "\n";
    return 1;
  }
  auto buffer = file_or_err->get();
  auto maybe_module =
      mlir::InjectStatsToFullyConnected(buffer->getBuffer().str());
  if (!maybe_module.hasValue()) {
    return 1;
  }
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::Model> output_model_location =
      tflite::Model::Pack(builder, maybe_module.getValue().get());
  tflite::FinishModelBuffer(builder, output_model_location);
  std::string output_model_content(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  std::cout << output_model_content << "\n";
  return 0;
}
