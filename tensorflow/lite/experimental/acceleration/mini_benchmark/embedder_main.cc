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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSembedder_mainDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSembedder_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSembedder_mainDTcc() {
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
// Command line tool for embedding validation data in tflite models.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/call_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace acceleration {
struct EmbedderOptions {
  std::string schema, main_model, metrics_model, output, jpegs_arg;
  float scale = 0.;
  int64_t zero_point = -1;
  bool use_ondevice_cpu_for_golden = false;
};

int RunEmbedder(const EmbedderOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSembedder_mainDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/embedder_main.cc", "RunEmbedder");

  // Load schema.
  std::string fbs_contents;
  if (!flatbuffers::LoadFile(options.schema.c_str(), false, &fbs_contents)) {
    std::cerr << "Unable to load schema file " << options.schema << std::endl;
    return 1;
  }
  const char* include_directories[] = {nullptr};
  flatbuffers::Parser schema_parser;
  if (!schema_parser.Parse(fbs_contents.c_str(), include_directories)) {
    std::cerr << "Unable to parse schema " << schema_parser.error_ << std::endl;
    return 2;
  }
  schema_parser.Serialize();
  const reflection::Schema* schema =
      reflection::GetSchema(schema_parser.builder_.GetBufferPointer());

  // Load main model.
  std::string main_model_contents;
  if (!flatbuffers::LoadFile(options.main_model.c_str(), false,
                             &main_model_contents)) {
    std::cerr << "Unable to load main model file " << options.main_model
              << std::endl;
    return 3;
  }
  const Model* main_model =
      flatbuffers::GetRoot<Model>(main_model_contents.data());

  // Load metrics model.
  std::string metrics_model_contents;
  if (!flatbuffers::LoadFile(options.metrics_model.c_str(), false,
                             &metrics_model_contents)) {
    std::cerr << "Unable to load metrics model file " << options.metrics_model
              << std::endl;
    return 4;
  }
  const Model* metrics_model =
      flatbuffers::GetRoot<Model>(metrics_model_contents.data());

  // Load sample images.
  std::vector<std::string> jpeg_paths = absl::StrSplit(options.jpegs_arg, ',');
  std::vector<std::string> jpeg_data;
  for (const std::string& jpeg_path : jpeg_paths) {
    std::string data;
    if (!flatbuffers::LoadFile(jpeg_path.c_str(), false, &data)) {
      std::cerr << "Unable to load jpeg file '" << jpeg_path << "'"
                << std::endl;
      return 5;
    }
    jpeg_data.push_back(data);
  }

  // Create model with embedded validation.
  tflite::acceleration::Embedder embedder(
      main_model, jpeg_data, options.scale, options.zero_point, metrics_model,
      schema, options.use_ondevice_cpu_for_golden);
  flatbuffers::FlatBufferBuilder fbb;
  ::tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom("validation/call",
                     ::tflite::acceleration::ops::Register_CALL(), 1);
  resolver.AddCustom(
      "validation/decode_jpeg",
      ::tflite::acceleration::decode_jpeg_kernel::Register_DECODE_JPEG(), 1);
  auto status = embedder.CreateModelWithEmbeddedValidation(&fbb, &resolver);
  if (!status.ok()) {
    std::cerr << "Creating model with embedded validation failed: "
              << status.ToString() << std::endl;
    return 6;
  }

  // Write created model to output path.
  std::string binary(reinterpret_cast<const char*>(fbb.GetBufferPointer()),
                     fbb.GetSize());
  std::ofstream f;
  f.open(options.output);
  if (!f.good()) {
    std::cerr << "Opening " << options.output
              << " for writing failed: " << strerror(errno) << std::endl;
    return 7;
  }
  f << binary;
  f.close();
  if (!f.good()) {
    std::cerr << "Writing to " << options.output
              << " failed: " << strerror(errno) << std::endl;
    return 8;
  }

  const Model* model = flatbuffers::GetRoot<Model>(fbb.GetBufferPointer());
  std::unique_ptr<Interpreter> interpreter;
  if (InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Loading the created model failed" << std::endl;
    return 9;
  }

  return 0;
}

}  // namespace acceleration
}  // namespace tflite

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSembedder_mainDTcc mht_1(mht_1_v, 320, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/embedder_main.cc", "main");

  tflite::acceleration::EmbedderOptions options;
  std::vector<tflite::Flag> flags = {
      tflite::Flag::CreateFlag("schema", &options.schema,
                               "Path to tflite schema.fbs"),
      tflite::Flag::CreateFlag("main_model", &options.main_model,
                               "Path to main inference tflite model"),
      tflite::Flag::CreateFlag("metrics_model", &options.metrics_model,
                               "Path to metrics tflite model"),
      tflite::Flag::CreateFlag("output", &options.output,
                               "Path to tflite output file"),
      tflite::Flag::CreateFlag(
          "jpegs", &options.jpegs_arg,
          "Comma-separated list of jpeg files to use as input"),
      tflite::Flag::CreateFlag("scale", &options.scale,
                               "Scale to use when dequantizing input images"),
      tflite::Flag::CreateFlag(
          "zero_point", &options.zero_point,
          "Zero-point to use when dequantizing input images"),
      tflite::Flag::CreateFlag(
          "use_ondevice_cpu_for_golden", &options.use_ondevice_cpu_for_golden,
          "Use on-device CPU as golden data (rather than embedding golden "
          "data)"),
  };
  if (!tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flags) ||
      options.schema.empty() || options.main_model.empty() ||
      options.output.empty() || options.jpegs_arg.empty()) {
    std::cerr << tflite::Flags::Usage("embedder_cmdline", flags);
    return 1;
  }
  return tflite::acceleration::RunEmbedder(options);

  return 0;
}
