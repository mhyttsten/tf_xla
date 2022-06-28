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
class MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/util.h"

const char kInputModelFlag[] = "input_models";
const char kNamespace[] = "namespace";
const char kOutputRegistrationFlag[] = "output_registration";
const char kTfLitePathFlag[] = "tflite_path";
const char kForMicro[] = "for_micro";

void ParseFlagAndInit(int* argc, char** argv, std::string* input_models,
                      std::string* output_registration,
                      std::string* tflite_path, std::string* namespace_flag,
                      bool* for_micro) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/tools/gen_op_registration_main.cc", "ParseFlagAndInit");

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kInputModelFlag, input_models,
                               "path to the tflite models, separated by comma"),
      tflite::Flag::CreateFlag(kOutputRegistrationFlag, output_registration,
                               "filename for generated registration code"),
      tflite::Flag::CreateFlag(kTfLitePathFlag, tflite_path,
                               "Path to tensorflow lite dir"),
      tflite::Flag::CreateFlag(
          kNamespace, namespace_flag,
          "Namespace in which to put RegisterSelectedOps."),
      tflite::Flag::CreateFlag(
          kForMicro, for_micro,
          "By default this script generate TFL registration file, but can "
          "also generate TFLM files when this flag is set to true"),
  };

  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
}

namespace {

void GenerateFileContent(const std::string& tflite_path,
                         const std::string& filename,
                         const std::string& namespace_flag,
                         const tflite::RegisteredOpMap& builtin_ops,
                         const tflite::RegisteredOpMap& custom_ops,
                         const bool for_micro) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tflite_path: \"" + tflite_path + "\"");
   mht_1_v.push_back("filename: \"" + filename + "\"");
   mht_1_v.push_back("namespace_flag: \"" + namespace_flag + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/tools/gen_op_registration_main.cc", "GenerateFileContent");

  std::ofstream fout(filename);

  if (for_micro) {
    if (!builtin_ops.empty()) {
      fout << "#include \"" << tflite_path << "/micro/kernels/micro_ops.h\"\n";
    }
    fout << "#include \"" << tflite_path
         << "/micro/micro_mutable_op_resolver.h\"\n";
  } else {
    if (!builtin_ops.empty()) {
      fout << "#include \"" << tflite_path
           << "/kernels/builtin_op_kernels.h\"\n";
    }
    fout << "#include \"" << tflite_path << "/model.h\"\n";
    fout << "#include \"" << tflite_path << "/op_resolver.h\"\n";
  }

  if (!custom_ops.empty()) {
    fout << "namespace tflite {\n";
    fout << "namespace ops {\n";
    fout << "namespace custom {\n";
    fout << "// Forward-declarations for the custom ops.\n";
    for (const auto& op : custom_ops) {
      // Skips Tensorflow ops, only TFLite custom ops can be registered here.
      if (tflite::IsFlexOp(op.first.c_str())) continue;
      fout << "TfLiteRegistration* Register_"
           << ::tflite::NormalizeCustomOpName(op.first) << "();\n";
    }
    fout << "}  // namespace custom\n";
    fout << "}  // namespace ops\n";
    fout << "}  // namespace tflite\n";
  }

  if (!namespace_flag.empty()) {
    fout << "namespace " << namespace_flag << " {\n";
  }
  if (for_micro) {
    fout << "void RegisterSelectedOps(::tflite::MicroMutableOpResolver* "
            "resolver) {\n";
  } else {
    fout << "void RegisterSelectedOps(::tflite::MutableOpResolver* resolver) "
            "{\n";
  }
  for (const auto& op : builtin_ops) {
    fout << "  resolver->AddBuiltin(::tflite::BuiltinOperator_" << op.first;
    if (for_micro) {
      fout << ", ::tflite::ops::micro::Register_" << op.first << "()";
    } else {
      fout << ", ::tflite::ops::builtin::Register_" << op.first << "()";
    }
    if (op.second.first != 1 || op.second.second != 1) {
      fout << ", " << op.second.first << ", " << op.second.second;
    }
    fout << ");\n";
  }
  for (const auto& op : custom_ops) {
    // Skips Tensorflow ops, only TFLite custom ops can be registered here.
    if (tflite::IsFlexOp(op.first.c_str())) continue;
    fout << "  resolver->AddCustom(\"" << op.first
         << "\", ::tflite::ops::custom::Register_"
         << ::tflite::NormalizeCustomOpName(op.first) << "()";
    if (op.second.first != 1 || op.second.second != 1) {
      fout << ", " << op.second.first << ", " << op.second.second;
    }
    fout << ");\n";
  }
  fout << "}\n";
  if (!namespace_flag.empty()) {
    fout << "}  // namespace " << namespace_flag << "\n";
  }
  fout.close();
}

void AddOpsFromModel(const std::string& input_model,
                     tflite::RegisteredOpMap* builtin_ops,
                     tflite::RegisteredOpMap* custom_ops) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input_model: \"" + input_model + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc mht_2(mht_2_v, 319, "", "./tensorflow/lite/tools/gen_op_registration_main.cc", "AddOpsFromModel");

  std::ifstream fin(input_model);
  std::stringstream content;
  content << fin.rdbuf();
  // Need to store content data first, otherwise, it won't work in bazel.
  std::string content_str = content.str();
  const ::tflite::Model* model = ::tflite::GetModel(content_str.data());
  ::tflite::ReadOpsFromModel(model, builtin_ops, custom_ops);
}

}  // namespace

int main(int argc, char** argv) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSgen_op_registration_mainDTcc mht_3(mht_3_v, 334, "", "./tensorflow/lite/tools/gen_op_registration_main.cc", "main");

  std::string input_models;
  std::string output_registration;
  std::string tflite_path;
  std::string namespace_flag;
  bool for_micro = false;
  ParseFlagAndInit(&argc, argv, &input_models, &output_registration,
                   &tflite_path, &namespace_flag, &for_micro);

  tflite::RegisteredOpMap builtin_ops;
  tflite::RegisteredOpMap custom_ops;
  if (!input_models.empty()) {
    std::vector<std::string> models = absl::StrSplit(input_models, ',');
    for (const std::string& input_model : models) {
      AddOpsFromModel(input_model, &builtin_ops, &custom_ops);
    }
  }
  for (int i = 1; i < argc; i++) {
    AddOpsFromModel(argv[i], &builtin_ops, &custom_ops);
  }

  GenerateFileContent(tflite_path, output_registration, namespace_flag,
                      builtin_ops, custom_ops, for_micro);
  return 0;
}
