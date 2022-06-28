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
class MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/schema/builtin_ops_header/generator.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace builtin_ops_header {

namespace {
const char* kFileHeader =
    R"(/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_BUILTIN_OPS_H_
#define TENSORFLOW_LITE_BUILTIN_OPS_H_

// DO NOT EDIT MANUALLY: This file is automatically generated by
// `schema/builtin_ops_header/generator.cc`.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The enum for builtin operators.
// Note: CUSTOM, DELEGATE, and PLACEHOLDER_FOR_GREATER_OP_CODES are 3 special
// ops which are not real built-in ops.
typedef enum {
)";

const char* kFileFooter =
    R"(} TfLiteBuiltinOperator;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_LITE_BUILTIN_OPS_H_
)";
}  // anonymous namespace

bool IsValidInputEnumName(const std::string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc mht_0(mht_0_v, 234, "", "./tensorflow/lite/schema/builtin_ops_header/generator.cc", "IsValidInputEnumName");

  const char* begin = name.c_str();
  const char* ch = begin;
  while (*ch != '\0') {
    // If it's not the first character, expect an underscore.
    if (ch != begin) {
      if (*ch != '_') {
        return false;
      }
      ++ch;
    }

    // Expecting a word with upper case letters or digits, like "CONV",
    // "CONV2D", "2D"...etc.
    bool empty = true;
    while (isupper(*ch) || isdigit(*ch)) {
      // It's not empty if at least one character is consumed.
      empty = false;
      ++ch;
    }
    if (empty) {
      return false;
    }
  }
  return true;
}

std::string ConstantizeVariableName(const std::string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc mht_1(mht_1_v, 265, "", "./tensorflow/lite/schema/builtin_ops_header/generator.cc", "ConstantizeVariableName");

  std::string result = "kTfLiteBuiltin";
  bool uppercase = true;
  for (char input_char : name) {
    if (input_char == '_') {
      uppercase = true;
    } else if (uppercase) {
      result += toupper(input_char);
      uppercase = false;
    } else {
      result += tolower(input_char);
    }
  }

  return result;
}

bool GenerateHeader(std::ostream& os) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSschemaPSbuiltin_ops_headerPSgeneratorDTcc mht_2(mht_2_v, 285, "", "./tensorflow/lite/schema/builtin_ops_header/generator.cc", "GenerateHeader");

  auto enum_names = tflite::EnumNamesBuiltinOperator();

  // Check if all the input enum names are valid.
  for (auto enum_value : EnumValuesBuiltinOperator()) {
    auto enum_name = enum_names[enum_value];
    if (!IsValidInputEnumName(enum_name)) {
      std::cerr << "Invalid input enum name: " << enum_name << std::endl;
      return false;
    }
  }

  os << kFileHeader;
  for (auto enum_value : EnumValuesBuiltinOperator()) {
    auto enum_name = enum_names[enum_value];
    os << "  ";
    os << ConstantizeVariableName(enum_name);
    os << " = ";
    os << enum_value;
    os << ",\n";
  }
  os << kFileFooter;
  return true;
}

}  // namespace builtin_ops_header
}  // namespace tflite
