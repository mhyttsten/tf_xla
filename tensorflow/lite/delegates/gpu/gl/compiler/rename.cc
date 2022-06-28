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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/rename.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Rewrites names of all variables according to returned values from the
// given NameFunctor.
class VariableRewriter : public InlineRewrite {
 public:
  VariableRewriter(const std::string& inline_delimiter,
                   const NameFunctor& name_func)
      : inline_delimiter_(inline_delimiter), name_func_(name_func) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("inline_delimiter: \"" + inline_delimiter + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "VariableRewriter");
}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "Rewrite");

    auto ref = variable_accessor_internal::Parse(input);
    if (ref.name.empty()) {
      absl::StrAppend(output, "INVALID_SYNTAX");
      return RewriteStatus::ERROR;
    }

    auto it =
        name_to_variable_.find(std::string(ref.name.data(), ref.name.size()));
    if (it == name_to_variable_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }

    // reconstruct access using the new name.
    absl::StrAppend(output, inline_delimiter_, it->second.name);
    if (!ref.index.empty()) {
      absl::StrAppend(output, "[", ref.index, "]");
    }
    absl::StrAppend(output, ref.field, inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  // Return true if variable was successfully added.
  bool AddVariable(Variable&& variable) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_2(mht_2_v, 246, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "AddVariable");

    std::string old_name = variable.name;
    variable.name = name_func_(old_name);
    return name_to_variable_.insert({old_name, std::move(variable)}).second;
  }

  // Returns a collection of uniform parameters with updated names.
  std::vector<Variable> GetUniformParameters() const {
    std::vector<Variable> variables;
    variables.reserve(name_to_variable_.size());
    for (const auto& variable : name_to_variable_) {
      variables.push_back(variable.second);
    }
    return variables;
  }

 private:
  const std::string inline_delimiter_;
  const NameFunctor name_func_;

  absl::flat_hash_map<std::string, Variable> name_to_variable_;
};

// Rewrites names of all objects according to returned values from the
// given NameFunctor.
class ObjectRewriter : public InlineRewrite {
 public:
  ObjectRewriter(const std::string& inline_delimiter,
                 const NameFunctor& name_func)
      : inline_delimiter_(inline_delimiter), name_func_(name_func) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("inline_delimiter: \"" + inline_delimiter + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_3(mht_3_v, 279, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "ObjectRewriter");
}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_4(mht_4_v, 285, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "Rewrite");

    // Splits 'a = b' into {'a','b'}.
    std::pair<absl::string_view, absl::string_view> n =
        absl::StrSplit(input, absl::MaxSplits('=', 1), absl::SkipWhitespace());
    if (n.first.empty()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }

    if (n.second.empty()) {
      return RewriteRead(absl::StripAsciiWhitespace(n.first), output);
    }
    return RewriteWrite(absl::StripAsciiWhitespace(n.first),
                        absl::StripAsciiWhitespace(n.second), output);
  }

  // Return true if an object was successfully added.
  bool AddObject(const std::string& name, Object object) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_5(mht_5_v, 305, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "AddObject");

    std::string new_name = name_func_(name);
    return name_to_object_.insert({name, {new_name, std::move(object)}}).second;
  }

  // Returns a collection of registered objects with updated names.
  std::vector<std::pair<std::string, Object>> GetObjects() const {
    std::vector<std::pair<std::string, Object>> objects;
    objects.reserve(name_to_object_.size());
    for (const auto& o : name_to_object_) {
      objects.push_back(o.second);
    }
    return objects;
  }

 private:
  RewriteStatus RewriteRead(absl::string_view location, std::string* output) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("location: \"" + std::string(location.data(), location.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_6(mht_6_v, 325, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "RewriteRead");

    auto element = object_accessor_internal::ParseElement(location);
    if (element.object_name.empty()) {
      absl::StrAppend(output, "UNABLE_TO_PARSE_INDEXED_ELEMENT");
      return RewriteStatus::ERROR;
    }
    auto it = name_to_object_.find(
        std::string(element.object_name.data(), element.object_name.size()));
    if (it == name_to_object_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }
    absl::StrAppend(output, inline_delimiter_, it->second.first, "[",
                    absl::StrJoin(element.indices, ","), "]",
                    inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  RewriteStatus RewriteWrite(absl::string_view location,
                             absl::string_view value, std::string* output) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("location: \"" + std::string(location.data(), location.size()) + "\"");
   mht_7_v.push_back("value: \"" + std::string(value.data(), value.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_7(mht_7_v, 348, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "RewriteWrite");

    // name[index1, index2...] = value
    auto element = object_accessor_internal::ParseElement(location);
    if (element.object_name.empty()) {
      absl::StrAppend(output, "UNABLE_TO_PARSE_INDEXED_ELEMENT");
      return RewriteStatus::ERROR;
    }
    auto it = name_to_object_.find(
        std::string(element.object_name.data(), element.object_name.size()));
    if (it == name_to_object_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }
    absl::StrAppend(output, inline_delimiter_, it->second.first, "[",
                    absl::StrJoin(element.indices, ","), "] = ", value,
                    inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  const std::string inline_delimiter_;
  const NameFunctor name_func_;

  absl::flat_hash_map<std::string, std::pair<std::string, Object>>
      name_to_object_;
};

}  // namespace

absl::Status Rename(const NameFunctor& name_func, GeneratedCode* code) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSrenameDTcc mht_8(mht_8_v, 378, "", "./tensorflow/lite/delegates/gpu/gl/compiler/rename.cc", "Rename");

  VariableRewriter variable_rewriter("$", name_func);
  ObjectRewriter object_rewriter("$", name_func);
  for (auto&& uniform_parameter : code->parameters) {
    if (!variable_rewriter.AddVariable(std::move(uniform_parameter))) {
      return absl::InternalError("Variable name already exists");
    }
  }
  for (auto&& object : code->objects) {
    if (!object_rewriter.AddObject(object.first, std::move(object.second))) {
      return absl::InternalError("Object name already exists");
    }
  }
  TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/true);
  preprocessor.AddRewrite(&variable_rewriter);
  preprocessor.AddRewrite(&object_rewriter);
  std::string source_code;
  RETURN_IF_ERROR(preprocessor.Rewrite(code->source_code, &source_code));
  code->source_code = source_code;
  code->parameters = variable_rewriter.GetUniformParameters();
  code->objects = object_rewriter.GetObjects();
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
