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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc() {
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

// Dumps a TFLite flatbuffer to a textual output format.
// This tool is intended to be used to simplify unit testing/debugging.

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/minireflect.h"  // from @flatbuffers
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace tflite {
namespace {

// Reads a model from a provided file path and verifies if it is a valid
// flatbuffer, and returns false with the model in serialized_model if valid
// else true.
bool ReadAndVerify(const std::string& file_path,
                   std::string* serialized_model) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "ReadAndVerify");

  if (file_path == "-") {
    *serialized_model = std::string{std::istreambuf_iterator<char>(std::cin),
                                    std::istreambuf_iterator<char>()};
  } else {
    std::ifstream t(file_path);
    if (!t.is_open()) {
      std::cerr << "Failed to open input file.\n";
      return true;
    }
    *serialized_model = std::string{std::istreambuf_iterator<char>(t),
                                    std::istreambuf_iterator<char>()};
  }

  flatbuffers::Verifier model_verifier(
      reinterpret_cast<const uint8_t*>(serialized_model->c_str()),
      serialized_model->length());
  if (!model_verifier.VerifyBuffer<Model>()) {
    std::cerr << "Verification failed.\n";
    return true;
  }
  return false;
}

// A FlatBuffer visitor that outputs a FlatBuffer as a string with proper
// indention for sequence fields.
// TODO(wvo): ToStringVisitor already has indentation functionality, use
// that directly instead of this sub-class?
struct IndentedToStringVisitor : flatbuffers::ToStringVisitor {
  std::string indent_str;
  int indent_level;

  IndentedToStringVisitor(const std::string& delimiter,
                          const std::string& indent)
      : ToStringVisitor(delimiter), indent_str(indent), indent_level(0) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("delimiter: \"" + delimiter + "\"");
   mht_1_v.push_back("indent: \"" + indent + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "IndentedToStringVisitor");
}

  void indent() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "indent");

    for (int i = 0; i < indent_level; ++i) s.append(indent_str);
  }

  // Adjust indention for fields in sequences.

  void StartSequence() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_3(mht_3_v, 260, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "StartSequence");

    s += "{";
    s += d;
    ++indent_level;
  }

  void EndSequence() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_4(mht_4_v, 269, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "EndSequence");

    s += d;
    --indent_level;
    indent();
    s += "}";
  }

  void Field(size_t /*field_idx*/, size_t set_idx,
             flatbuffers::ElementaryType /*type*/, bool /*is_vector*/,
             const flatbuffers::TypeTable* /*type_table*/, const char* name,
             const uint8_t* val) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_5(mht_5_v, 283, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "Field");

    if (!val) return;
    if (set_idx) {
      s += ",";
      s += d;
    }
    indent();
    if (name) {
      s += name;
      s += ": ";
    }
  }

  void StartVector() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "StartVector");
 s += "[ "; }
  void EndVector() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_7(mht_7_v, 303, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "EndVector");
 s += " ]"; }

  void Element(size_t i, flatbuffers::ElementaryType /*type*/,
               const flatbuffers::TypeTable* /*type_table*/,
               const uint8_t* /*val*/) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_8(mht_8_v, 310, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "Element");

    if (i) s += ", ";
  }
};

void ToString(const std::string& serialized_model) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("serialized_model: \"" + serialized_model + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_9(mht_9_v, 319, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "ToString");

  IndentedToStringVisitor visitor(/*delimiter=*/"\n", /*indent=*/"  ");
  IterateFlatBuffer(reinterpret_cast<const uint8_t*>(serialized_model.c_str()),
                    ModelTypeTable(), &visitor);
  std::cout << visitor.s << "\n\n";
}

}  // end namespace
}  // end namespace tflite

int main(int argc, char** argv) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSflatbuffer_to_stringDTcc mht_10(mht_10_v, 332, "", "./tensorflow/compiler/mlir/lite/flatbuffer_to_string.cc", "main");

  if (argc < 2) {
    std::cerr << "Missing input argument. Usage:\n"
              << argv[0] << " <filename or - for stdin>\n\n"
              << "Converts TensorFlowLite flatbuffer to textual output format. "
              << "One positional input argument representing the source of the "
              << "flatbuffer is supported.\n";
    return 1;
  }

  std::string serialized_model;
  if (tflite::ReadAndVerify(argv[1], &serialized_model)) return 1;
  tflite::ToString(serialized_model);
  return 0;
}
