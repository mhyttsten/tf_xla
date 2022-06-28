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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/arguments.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {
namespace {
bool IsWordSymbol(char symbol) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("symbol: '" + std::string(1, symbol) + "'");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "IsWordSymbol");

  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("old_word: \"" + old_word + "\"");
   mht_1_v.push_back("new_word: \"" + new_word + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "ReplaceAllWords");

  size_t position = str->find(old_word);
  while (position != std::string::npos) {
    char prev = position == 0 ? '.' : (*str)[position - 1];
    char next = position + old_word.size() < str->size()
                    ? (*str)[position + old_word.size()]
                    : '.';
    if (IsWordSymbol(prev) || IsWordSymbol(next)) {
      position = str->find(old_word, position + 1);
      continue;
    }
    str->replace(position, old_word.size(), new_word);
    position = str->find(old_word, position + new_word.size());
  }
}

std::string GetNextWord(const std::string& code, size_t first_position) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("code: \"" + code + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "GetNextWord");

  size_t pos = first_position;
  char t = code[pos];
  while (IsWordSymbol(t)) {
    pos++;
    t = code[pos];
  }
  return code.substr(first_position, pos - first_position);
}

bool HasWord(const std::string& word, const std::string& text) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("word: \"" + word + "\"");
   mht_3_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "HasWord");

  size_t pos = text.find(word);
  while (pos != std::string::npos) {
    char prev = pos == 0 ? '.' : text[pos - 1];
    char next = pos + word.size() < text.size() ? text[pos + word.size()] : '.';
    if (!IsWordSymbol(prev) && !IsWordSymbol(next)) {
      return true;
    }
    pos = text.find(word, pos + 1);
  }
  return false;
}

std::string RenameArg(const std::vector<std::string>& object_names,
                      const std::string& postfix, const std::string& arg_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("postfix: \"" + postfix + "\"");
   mht_4_v.push_back("arg_name: \"" + arg_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_4(mht_4_v, 270, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "RenameArg");

  for (const auto& object_name : object_names) {
    if (absl::StartsWith(arg_name, object_name) &&
        arg_name.size() > object_name.size() &&
        arg_name[object_name.size()] == '_') {
      return object_name + postfix +
             arg_name.substr(object_name.size(),
                             arg_name.size() - object_name.size());
    }
  }
  return arg_name + postfix;
}

size_t FindEnclosingBracket(const std::string& text, size_t first_pos,
                            char bracket) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("text: \"" + text + "\"");
   mht_5_v.push_back("bracket: '" + std::string(1, bracket) + "'");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_5(mht_5_v, 289, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "FindEnclosingBracket");

  const std::map<char, char> brackets = {
      {'(', ')'},
      {'{', '}'},
      {'[', ']'},
      {'<', '>'},
  };
  char b_open = bracket;
  auto it = brackets.find(b_open);
  if (it == brackets.end()) {
    return -1;
  }
  char b_close = it->second;
  size_t pos = first_pos;
  int opened = 1;
  int closed = 0;
  while (opened != closed && pos < text.size()) {
    if (text[pos] == b_open) {
      opened++;
    } else if (text[pos] == b_close) {
      closed++;
    }
    pos++;
  }
  if (opened == closed) {
    return pos;
  } else {
    return -1;
  }
}

absl::Status ParseArgsInsideBrackets(const std::string& text,
                                     size_t open_bracket_pos,
                                     size_t* close_bracket_pos,
                                     std::vector<std::string>* args) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_6(mht_6_v, 327, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "ParseArgsInsideBrackets");

  *close_bracket_pos =
      FindEnclosingBracket(text, open_bracket_pos + 1, text[open_bracket_pos]);
  if (*close_bracket_pos == -1) {
    return absl::NotFoundError("Not found enclosing bracket");
  }
  std::string str_args = text.substr(open_bracket_pos + 1,
                                     *close_bracket_pos - open_bracket_pos - 2);
  std::vector<absl::string_view> words = absl::StrSplit(str_args, ',');
  args->reserve(words.size());
  for (const auto& word : words) {
    absl::string_view arg = absl::StripAsciiWhitespace(word);
    if (!arg.empty()) {
      args->push_back(std::string(arg));
    }
  }
  return absl::OkStatus();
}

std::string DataTypeToGlType(DataType data_type, int vec_size,
                             bool explicit_f16) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_7(mht_7_v, 350, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "DataTypeToGlType");

  if (data_type == DataType::FLOAT32) {
    if (vec_size == 1) {
      return "float";
    } else {
      return "vec" + std::to_string(vec_size);
    }
  } else if (data_type == DataType::FLOAT16) {
    if (vec_size == 1) {
      return explicit_f16 ? "float16_t" : "float";
    } else {
      if (explicit_f16) {
        return "f16vec" + std::to_string(vec_size);
      } else {
        return "vec" + std::to_string(vec_size);
      }
    }
  } else if (data_type == DataType::INT32) {
    if (vec_size == 1) {
      return "int";
    } else {
      return "ivec" + std::to_string(vec_size);
    }
  } else if (data_type == DataType::UINT32) {
    if (vec_size == 1) {
      return "uint";
    } else {
      return "uvec" + std::to_string(vec_size);
    }
  }
  return "unsupported_type";
}

absl::Status BufferToKernelLanguage(const GpuInfo& gpu_info,
                                    const std::string& buffer_name,
                                    const BufferDescriptor* buffer_desc,
                                    std::string* result) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("buffer_name: \"" + buffer_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_8(mht_8_v, 390, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "BufferToKernelLanguage");

  if (buffer_desc->element_size != 1) {
    return absl::UnimplementedError("No support of vector types.");
  }
  const int elements_count =
      buffer_desc->size /
      (buffer_desc->element_size * SizeOf(buffer_desc->element_type));
  if (gpu_info.IsGlsl()) {
    const std::string gl_type =
        DataTypeToGlType(buffer_desc->element_type, buffer_desc->element_size,
                         gpu_info.IsGlslSupportsExplicitFp16());
    *result = "const ";
    if (buffer_desc->element_type == DataType::FLOAT16 &&
        !gpu_info.IsGlslSupportsExplicitFp16()) {
      *result += "mediump ";
    }
    *result += gl_type + " " + buffer_name + "_buffer[] = " + gl_type + "[](\n";
  } else if (gpu_info.IsApiMetal()) {
    const std::string metal_type =
        ToMetalDataType(buffer_desc->element_type, buffer_desc->element_size);
    *result = "constant " + metal_type + " " + buffer_name + "_buffer[" +
              std::to_string(elements_count) + "] = {\n";
  } else if (gpu_info.IsApiOpenCl()) {
    const std::string cl_type =
        ToCLDataType(buffer_desc->element_type, buffer_desc->element_size);
    *result = "__constant " + cl_type + " " + buffer_name + "_buffer[" +
              std::to_string(elements_count) + "] = {\n";
  } else {
    return absl::UnimplementedError("Not supported API.");
  }
  if (buffer_desc->element_type == DataType::FLOAT16) {
    std::string postfix = "f";
    if (gpu_info.IsGlsl() && gpu_info.IsGlslSupportsExplicitFp16()) {
      postfix = "hf";
    }
    const half* data_ptr =
        reinterpret_cast<const half*>(buffer_desc->data.data());
    for (int i = 0; i < elements_count; ++i) {
      *result += "  " +
                 absl::StrFormat("%.10f", static_cast<float>(data_ptr[i])) +
                 postfix;
      if (i != elements_count - 1) {
        *result += ",\n";
      }
    }
  } else if (buffer_desc->element_type == DataType::FLOAT32) {
    const float* data_ptr =
        reinterpret_cast<const float*>(buffer_desc->data.data());
    for (int i = 0; i < elements_count; ++i) {
      *result += "  " + absl::StrFormat("%.10f", data_ptr[i]) + "f";
      if (i != elements_count - 1) {
        *result += ",\n";
      }
    }
  } else {
    return absl::UnimplementedError("Not supported type.");
  }
  if (gpu_info.IsGlsl()) {
    *result += ");\n";
  } else {
    *result += "};\n";
  }

  return absl::OkStatus();
}

}  // namespace

// Static
constexpr char Arguments::kArgsPrefix[];

void Arguments::AddFloat(const std::string& name, float value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_9(mht_9_v, 465, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddFloat");

  float_values_[name].value = value;
}
void Arguments::AddHalf(const std::string& name, half value) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_10(mht_10_v, 472, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddHalf");

  half_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_11(mht_11_v, 479, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddInt");

  int_values_[name].value = value;
}

absl::Status Arguments::SetInt(const std::string& name, int value) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_12(mht_12_v, 487, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::SetInt");

  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}
absl::Status Arguments::SetFloat(const std::string& name, float value) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_13(mht_13_v, 500, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::SetFloat");

  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}

absl::Status Arguments::SetHalf(const std::string& name, half value) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_14(mht_14_v, 514, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::SetHalf");

  auto it = half_values_.find(name);
  if (it == half_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  return absl::OkStatus();
}

void Arguments::AddObjectRef(const std::string& name, AccessType access_type,
                             GPUObjectDescriptorPtr&& descriptor_ptr) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_15(mht_15_v, 529, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddObjectRef");

  descriptor_ptr->SetAccess(access_type);
  object_refs_[name] = {std::move(descriptor_ptr)};
}

void Arguments::AddObject(const std::string& name,
                          GPUObjectDescriptorPtr&& descriptor_ptr) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_16(mht_16_v, 539, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddObject");

  descriptor_ptr->SetAccess(AccessType::READ);
  objects_[name] = {std::move(descriptor_ptr)};
}

void Arguments::RenameArgs(const std::string& postfix,
                           std::string* code) const {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("postfix: \"" + postfix + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_17(mht_17_v, 549, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::RenameArgs");

  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position + strlen(kArgsPrefix);
    std::string arg_name = GetNextWord(*code, arg_pos);
    code->replace(arg_pos, arg_name.size(), arg_name + postfix);
    next_position = code->find(kArgsPrefix, arg_pos + arg_name.size());
  }
}

absl::Status Arguments::Merge(Arguments&& args, const std::string& postfix,
                              const std::vector<std::string>& exception_names) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("postfix: \"" + postfix + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_18(mht_18_v, 564, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::Merge");

  std::vector<std::string> object_names;
  object_names.reserve(args.object_refs_.size() + args.objects_.size());
  for (auto& v : args.object_refs_) {
    if (std::find(exception_names.begin(), exception_names.end(), v.first) !=
        exception_names.end()) {
      continue;
    }
    object_names.push_back(v.first);
    const std::string name = v.first + postfix;
    if (object_refs_.find(name) != object_refs_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Object reference name collision. Name - ", name));
    }
    object_refs_[name] = {std::move(v.second)};
  }
  for (auto& v : args.objects_) {
    if (std::find(exception_names.begin(), exception_names.end(), v.first) !=
        exception_names.end()) {
      continue;
    }
    object_names.push_back(v.first);
    const std::string name = v.first + postfix;
    if (objects_.find(name) != objects_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Object name collision. Name - ", name));
    }
    objects_[name] = {std::move(v.second)};
  }
  for (const auto& v : args.int_values_) {
    AddInt(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  for (const auto& v : args.float_values_) {
    AddFloat(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  for (const auto& v : args.half_values_) {
    AddHalf(RenameArg(object_names, postfix, v.first), v.second.value);
  }
  return absl::OkStatus();
}

absl::Status Arguments::GetDescriptor(const std::string& name,
                                      GPUObjectDescriptor** descriptor) const {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_19(mht_19_v, 610, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::GetDescriptor");

  auto it_ref = object_refs_.find(name);
  if (it_ref != object_refs_.end()) {
    *descriptor = it_ref->second.get();
    return absl::OkStatus();
  }
  auto it = objects_.find(name);
  if (it != objects_.end()) {
    *descriptor = it->second.get();
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat("No GPU object with name - ", name));
}

void Arguments::ReleaseCPURepresentation() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_20(mht_20_v, 627, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ReleaseCPURepresentation");

  for (auto& t : objects_) {
    t.second->Release();
  }
}

void Arguments::GetActiveArguments(const std::string& code) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("code: \"" + code + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_21(mht_21_v, 637, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::GetActiveArguments");

  for (auto& float_val : float_values_) {
    float_val.second.active = HasWord(kArgsPrefix + float_val.first, code);
  }
  for (auto& int_val : int_values_) {
    int_val.second.active = HasWord(kArgsPrefix + int_val.first, code);
  }
  for (auto& half_val : half_values_) {
    half_val.second.active = HasWord(kArgsPrefix + half_val.first, code);
  }
}

int Arguments::GetReadTexturesCount(const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_22(mht_22_v, 652, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::GetReadTexturesCount");

  int counter = 0;
  for (auto& t : objects_) {
    counter += t.second->GetGPUResources(gpu_info).GetReadImagesCount();
  }
  for (auto& t : object_refs_) {
    counter += t.second->GetGPUResources(gpu_info).GetReadImagesCount();
  }
  return counter;
}

int Arguments::GetWriteTexturesCount(const GpuInfo& gpu_info) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_23(mht_23_v, 666, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::GetWriteTexturesCount");

  int counter = 0;
  for (auto& t : objects_) {
    counter += t.second->GetGPUResources(gpu_info).GetWriteImagesCount();
  }
  for (auto& t : object_refs_) {
    counter += t.second->GetGPUResources(gpu_info).GetWriteImagesCount();
  }
  return counter;
}

void Arguments::SetStateValueForAllObjects(const std::string& key,
                                           const std::string& value) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("key: \"" + key + "\"");
   mht_24_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_24(mht_24_v, 683, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::SetStateValueForAllObjects");

  for (auto& obj : object_refs_) {
    obj.second->SetStateVar(key, value);
  }
  for (auto& obj : objects_) {
    obj.second->SetStateVar(key, value);
  }
}

absl::Status Arguments::Compile(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables, std::string* code) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_25(mht_25_v, 697, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::Compile");

  RETURN_IF_ERROR(AddObjectsScalarArgs(gpu_info));
  RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, code));
  RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, linkables, code));
  GetActiveArguments(*code);
  RETURN_IF_ERROR(ResolveKernelGlobalSpaceBuffers(gpu_info, code));
  return absl::OkStatus();
}

absl::Status Arguments::ResolveConstExprPass(const GpuInfo& gpu_info,
                                             std::string* code) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_26(mht_26_v, 710, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveConstExprPass");

  std::string result;
  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    if (next_position + object_name.size() > code->size() - 2) {
      next_position = code->find(kArgsPrefix, next_position);
      continue;
    }
    char next0 = (*code)[next_position + object_name.size()];
    char next1 = (*code)[next_position + object_name.size() + 1];
    if (next0 == ':' && next1 == ':') {
      next_position += object_name.size() + 2;
      std::string const_expr_name = GetNextWord(*code, next_position);
      next_position += const_expr_name.size();
      std::string patch;
      RETURN_IF_ERROR(
          ResolveConstExpr(gpu_info, object_name, const_expr_name, &patch));
      code->replace(arg_pos, next_position - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

absl::Status Arguments::ResolveConstExpr(const GpuInfo& gpu_info,
                                         const std::string& object_name,
                                         const std::string& const_expr,
                                         std::string* result) const {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("object_name: \"" + object_name + "\"");
   mht_27_v.push_back("const_expr: \"" + const_expr + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_27(mht_27_v, 749, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveConstExpr");

  tflite::gpu::GPUObjectDescriptor* desc_ptr;
  RETURN_IF_ERROR(GetDescriptor(object_name, &desc_ptr));
  RETURN_IF_ERROR(desc_ptr->PerformConstExpr(gpu_info, const_expr, result));
  return absl::OkStatus();
}

absl::Status Arguments::ResolveSelectorsPass(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables,
    std::string* code) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_28(mht_28_v, 762, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveSelectorsPass");

  std::string result;
  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    char next = (*code)[next_position + object_name.size()];
    if (next == '.') {
      next_position += object_name.size() + 1;
      std::string selector_name = GetNextWord(*code, next_position);
      next_position += selector_name.size();
      next = (*code)[next_position];
      std::vector<std::string> template_args;
      if (next == '<') {
        size_t close_bracket_pos;
        RETURN_IF_ERROR(ParseArgsInsideBrackets(
            *code, next_position, &close_bracket_pos, &template_args));
        next_position = close_bracket_pos;
        next = (*code)[next_position];
      }
      if (next != '(') {
        return absl::NotFoundError(absl::StrCat(
            "Expected ( after ", object_name, ".", selector_name, " call"));
      }
      std::vector<std::string> function_args;
      size_t close_bracket_pos;
      RETURN_IF_ERROR(ParseArgsInsideBrackets(
          *code, next_position, &close_bracket_pos, &function_args));
      for (auto& arg : function_args) {
        RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, &arg));
      }
      std::string patch;
      RETURN_IF_ERROR(ResolveSelector(gpu_info, linkables, object_name,
                                      selector_name, function_args,
                                      template_args, &patch));
      code->replace(arg_pos, close_bracket_pos - arg_pos, patch);
      position = arg_pos + patch.size();
    } else {
      position = arg_pos + strlen(kArgsPrefix);
    }
    next_position = code->find(kArgsPrefix, position);
  }
  return absl::OkStatus();
}

absl::Status Arguments::ResolveSelector(
    const GpuInfo& gpu_info,
    const std::map<std::string, std::string>& linkables,
    const std::string& object_name, const std::string& selector,
    const std::vector<std::string>& function_args,
    const std::vector<std::string>& template_args, std::string* result) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("object_name: \"" + object_name + "\"");
   mht_29_v.push_back("selector: \"" + selector + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_29(mht_29_v, 819, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveSelector");

  GPUObjectDescriptor* desc_ptr;
  RETURN_IF_ERROR(GetDescriptor(object_name, &desc_ptr));
  auto names = desc_ptr->GetGPUResources(gpu_info).GetNames();
  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc_ptr);
  if (tensor_desc && (selector == "Write" || selector == "Linking")) {
    auto it = linkables.find(object_name);
    if (it != linkables.end()) {
      if (desc_ptr->GetAccess() != AccessType::WRITE &&
          desc_ptr->GetAccess() != AccessType::READ_WRITE) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Object with name - ", object_name, " should have Write access."));
      }
      std::string value_name, x_coord, y_coord, s_coord;
      RETURN_IF_ERROR(tensor_desc->GetLinkingContextFromWriteSelector(
          function_args, &value_name, &x_coord, &y_coord, &s_coord));
      // x_coord can have batch size property of link_object
      ResolveObjectNames(object_name, names, &x_coord);
      *result = it->second;
      ReplaceAllWords("in_out_value", value_name, result);
      ReplaceAllWords("X_COORD", x_coord, result);
      ReplaceAllWords("Y_COORD", y_coord, result);
      ReplaceAllWords("S_COORD", s_coord, result);
      RETURN_IF_ERROR(ResolveConstExprPass(gpu_info, result));
      RETURN_IF_ERROR(ResolveSelectorsPass(gpu_info, {}, result));
      if (selector == "Linking") {
        return absl::OkStatus();
      }
    }
  }
  std::string patch;
  RETURN_IF_ERROR(desc_ptr->PerformSelector(gpu_info, selector, function_args,
                                            template_args, &patch));
  ResolveObjectNames(object_name, names, &patch);
  *result += patch;
  return absl::OkStatus();
}

void Arguments::ResolveObjectNames(const std::string& object_name,
                                   const std::vector<std::string>& member_names,
                                   std::string* code) const {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("object_name: \"" + object_name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_30(mht_30_v, 863, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveObjectNames");

  for (const auto& member_name : member_names) {
    const std::string new_name = kArgsPrefix + object_name + "_" + member_name;
    ReplaceAllWords(member_name, new_name, code);
  }
}

absl::Status Arguments::AddObjectsScalarArgs(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_31(mht_31_v, 873, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::AddObjectsScalarArgs");

  for (auto& t : objects_) {
    const auto resources = t.second->GetGPUResources(gpu_info);
    for (const auto& r : resources.ints) {
      AddInt(absl::StrCat(t.first, "_", r));
    }
    for (const auto& r : resources.floats) {
      AddFloat(absl::StrCat(t.first, "_", r));
    }
  }
  for (auto& t : object_refs_) {
    const auto resources = t.second->GetGPUResources(gpu_info);
    for (const auto& r : resources.ints) {
      AddInt(absl::StrCat(t.first, "_", r));
    }
    for (const auto& r : resources.floats) {
      AddFloat(absl::StrCat(t.first, "_", r));
    }
  }
  return absl::OkStatus();
}

void Arguments::ResolveArgsPass(std::string* code) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_32(mht_32_v, 898, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveArgsPass");

  size_t position = 0;
  size_t next_position = code->find(kArgsPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kArgsPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    std::string new_name = object_name;
    code->replace(arg_pos, object_name.size() + strlen(kArgsPrefix), new_name);
    position = arg_pos + new_name.size();
    next_position = code->find(kArgsPrefix, position);
  }
}

absl::Status Arguments::ResolveKernelGlobalSpaceBuffers(const GpuInfo& gpu_info,
                                                        std::string* code) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStaskPSargumentsDTcc mht_33(mht_33_v, 916, "", "./tensorflow/lite/delegates/gpu/common/task/arguments.cc", "Arguments::ResolveKernelGlobalSpaceBuffers");

  for (auto it = objects_.begin(); it != objects_.end();) {
    const auto* buffer_desc =
        dynamic_cast<const BufferDescriptor*>(it->second.get());
    if (!buffer_desc || buffer_desc->memory_type != MemoryType::CONSTANT) {
      ++it;
      continue;
    }
    bool is_kernel_global_space = false;
    for (const auto& attribute : buffer_desc->attributes) {
      if (attribute == "kernel_global_space") {
        is_kernel_global_space = true;
        break;
      }
    }
    if (!is_kernel_global_space) {
      ++it;
      continue;
    }
    std::string declaration;
    if (!BufferToKernelLanguage(gpu_info, it->first, buffer_desc, &declaration)
             .ok()) {
      ++it;
      continue;
    }
    *code = declaration + *code;
    objects_.erase(it++);
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
