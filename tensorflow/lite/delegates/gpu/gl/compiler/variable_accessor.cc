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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace variable_accessor_internal {

// Parse the following regex manually
// name(\[index\])?(\.field)?
VariableReference Parse(absl::string_view input) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "Parse");

  VariableReference ref;
  auto start_index = input.find('[');
  if (start_index != std::string::npos) {
    auto end_index = input.rfind(']');
    if (end_index == std::string::npos) {
      return ref;
    }
    ref.index = input.substr(start_index + 1, end_index - start_index - 1);
    ref.name = input.substr(0, start_index);
    ref.field = input.substr(end_index + 1);
  } else {
    auto dot = input.find('.');
    if (dot != std::string::npos) {
      ref.name = input.substr(0, dot);
      ref.field = input.substr(dot);
    } else {
      ref.name = input;
    }
  }
  return ref;
}

}  // namespace variable_accessor_internal

namespace {

struct VariableTypeGetter {
  std::string operator()(int) const { return "int"; }
  std::string operator()(const int2&) const { return "ivec2"; }
  std::string operator()(const std::vector<int2>&) const { return "ivec2"; }
  std::string operator()(const int4&) const { return "ivec4"; }
  std::string operator()(unsigned int) const { return "uint"; }
  std::string operator()(const uint4&) const { return "uvec4"; }
  std::string operator()(float) const { return "float"; }
  std::string operator()(const float2&) const { return "vec2"; }
  std::string operator()(const float4&) const { return "vec4"; }
  std::string operator()(const std::vector<float4>&) const { return "vec4"; }
};

// Returns GLSL uniform type of the given variable.
std::string GetVariableType(const Variable::ValueType& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_1(mht_1_v, 245, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GetVariableType");

  return absl::visit(VariableTypeGetter(), value);
}

struct LengthGetter {
  template <typename T>
  int operator()(const T& param) const {
    return 1;
  }
  template <typename T>
  int operator()(const std::vector<T>& param) const {
    return param.size();
  }
};

int GetLength(const Variable::ValueType& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GetLength");

  return absl::visit(LengthGetter(), value);
}

template <typename T>
void FormatValue(std::string* result, T t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_3(mht_3_v, 271, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "FormatValue");

  absl::StrAppend(result, t);
}

template <>
void FormatValue(std::string* result, float t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_4(mht_4_v, 279, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "FormatValue");

  absl::StrAppend(result, absl::StrFormat("%.9ff", t));
}

// Unfortunately absl::StrJoin with custom formatter requires formatter to use
// string, not std::string. Therefore, due to this compatibility issue data
// needs to be converted to string representation first and then joined.
template <typename T, int N>
std::vector<std::string> ToString(const std::array<T, N>& data) {
  std::vector<std::string> result(N);
  for (int i = 0; i < N; ++i) {
    FormatValue(&result[i], data[i]);
  }
  return result;
}

struct ConstGenerator {
  template <typename T>
  void operator()(T t) const {
    FormatValue(result, t);
  }

  template <typename T>
  void operator()(const Vec2<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 2>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec3<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 3>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const Vec4<T>& v) const {
    absl::StrAppend(result, VariableTypeGetter()(v), "(",
                    absl::StrJoin(ToString<T, 4>(v.data_), ","), ")");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    std::string type = VariableTypeGetter()(v);
    absl::StrAppend(result, type, "[", v.size(), "](");
    bool first = true;
    for (const auto& i : v) {
      if (first) {
        first = false;
      } else {
        absl::StrAppend(result, ",");
      }
      (*this)(i);
    }
    absl::StrAppend(result, ")");
  }

  std::string* result;
};

// Appends string representation of a variable value.
void GetValue(const Variable::ValueType& value, std::string* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_5(mht_5_v, 342, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GetValue");

  absl::visit(ConstGenerator{result}, value);
}

struct SharedVariableDeclarationGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "shared highp ", GetVariableType(variable.value),
                    " ", variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "shared highp ", GetVariableType(variable.value),
                    " ", variable.name);
    if (v.empty()) {
      // Normalize the size of the shared array to that of the WorkGroupSize
      absl::StrAppend(
          result,
          "[gl_WorkGroupSize.z * gl_WorkGroupSize.y * gl_WorkGroupSize.x];\n");
    } else {
      // Use the specified size
      absl::StrAppend(result, "[", v.size(), "];\n");
    }
  }

  const Variable& variable;
  std::string* result;
};

void GenerateSharedVariableDeclaration(const Variable& variable,
                                       std::string* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_6(mht_6_v, 376, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GenerateSharedVariableDeclaration");

  absl::visit(SharedVariableDeclarationGenerator{variable, result},
              variable.value);
}

struct UniformParameterDeclarationGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "uniform ", GetVariableType(variable.value), " ",
                    variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "uniform ", GetVariableType(variable.value), " ",
                    variable.name, "[", v.size(), "];\n");
  }

  const Variable& variable;
  std::string* result;
};

void GenerateUniformParameterDeclaration(const Variable& variable,
                                         std::string* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_7(mht_7_v, 402, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GenerateUniformParameterDeclaration");

  absl::visit(UniformParameterDeclarationGenerator{variable, result},
              variable.value);
}

struct VulkanPushConstantGenerator {
  template <typename T>
  void operator()(const T&) const {
    absl::StrAppend(result, "  ", GetVariableType(variable.value), " ",
                    variable.name, ";\n");
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    absl::StrAppend(result, "  ", GetVariableType(variable.value), " ",
                    variable.name, "[", v.size(), "];\n");
  }

  const Variable& variable;
  std::string* result;
};

void GenerateVulkanPushConstant(const Variable& variable, std::string* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_8(mht_8_v, 427, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GenerateVulkanPushConstant");

  absl::visit(VulkanPushConstantGenerator{variable, result}, variable.value);
}

struct VariableLengthGetter {
  template <typename T>
  bool operator()(const T&) const {
    return false;
  }
  template <typename T>
  bool operator()(const std::vector<T>&) const {
    return true;
  }
};

struct VulkanConstantGenerator {
  template <typename T>
  void operator()(const T&) const {
    const std::string variable_type = GetVariableType(variable.value);

    // Vulkan specialization constants are used for scalar types, all other
    // types go in push (uniform) constants.
    if (variable_type == "int" || variable_type == "uint" ||
        variable_type == "float") {
      absl::StrAppend(result, "layout(constant_id = ", *constant_id, ") const ",
                      variable_type, " ", variable.name, " = ");
      // Always set the default values to zero to generate generic cacheable
      // shaders.
      absl::StrAppend(result, (variable_type == "float" ? "0.0" : "0"), ";\n");
      (*constant_id)++;
    } else {
      non_scalar_variables->push_back(variable);
    }
  }

  template <typename T>
  void operator()(const std::vector<T>& v) const {
    non_scalar_variables->push_back(variable);
  }

  const Variable& variable;
  int* const constant_id;
  std::vector<Variable>* non_scalar_variables;
  std::string* result;
};

void GenerateVulkanConstant(const Variable& variable, int* constant_id,
                            std::vector<Variable>* non_scalar_variables,
                            std::string* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_9(mht_9_v, 478, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GenerateVulkanConstant");

  absl::visit(VulkanConstantGenerator{variable, constant_id,
                                      non_scalar_variables, result},
              variable.value);
}

class VulkanConstantsProcessor {
 public:
  void ProcessVulkanConstant(const Variable& variable, std::string* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_10(mht_10_v, 489, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "ProcessVulkanConstant");

    GenerateVulkanConstant(variable, &constant_id_, &non_scalar_variables_,
                           result);
  }

  void GeneratePushConstantsDeclarations(std::string* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_11(mht_11_v, 497, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GeneratePushConstantsDeclarations");

    if (!non_scalar_variables_.empty()) {
      *result += "\nlayout(push_constant) uniform pushConstants {\n";
      for (const auto& variable : non_scalar_variables_) {
        GenerateVulkanPushConstant(variable, result);
      }
      *result += "};\n";
    }
  }

 protected:
  // Reserve the first three specialization constants slots for the
  // workgroup size.
  int constant_id_ = 3;
  std::vector<Variable> non_scalar_variables_;
};

// Returns true if value is a vector
bool IsVariableLength(const Variable::ValueType& value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_12(mht_12_v, 518, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "IsVariableLength");

  return absl::visit(VariableLengthGetter(), value);
}

enum Field : uint8_t { UNKNOWN = 4, X = 0, Y = 1, Z = 2, W = 3 };

Field ToField(absl::string_view field_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("field_name: \"" + std::string(field_name.data(), field_name.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_13(mht_13_v, 528, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "ToField");

  if (field_name.size() == 2 && field_name[0] == '.') {
    switch (field_name[1]) {
      case 'x':
        return Field::X;
      case 'y':
        return Field::Y;
      case 'z':
        return Field::Z;
      case 'w':
        return Field::W;
    }
  }
  return Field::UNKNOWN;
}

struct FieldAccessor {
  template <typename T>
  void operator()(const T&) const {}

  template <typename T>
  void operator()(const Vec2<T>& v) const {
    FormatValue(result, v[field]);
  }

  template <typename T>
  void operator()(const Vec3<T>& v) const {
    FormatValue(result, v[field]);
  }

  template <typename T>
  void operator()(const Vec4<T>& v) const {
    FormatValue(result, v[field]);
  }

  Field field;
  std::string* result;
};

// Appends formatted value of the given field.
void GetValue(const Variable::ValueType& value, Field field,
              std::string* result) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_14(mht_14_v, 572, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "GetValue");

  absl::visit(FieldAccessor{field, result}, value);
}

struct FieldChecker {
  // For trivial as well as variable-length types indexed access is not allowed.
  template <typename T>
  bool operator()(const T&) const {
    return false;
  }

  template <typename T>
  bool operator()(const Vec2<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const Vec3<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const Vec4<T>& v) const {
    return field < v.size();
  }

  template <typename T>
  bool operator()(const std::vector<T>&) const {
    // technically accessing [0] element of an empty vector is UB, but we need
    // only type information for this check. Therefore, construct default T and
    // use it instead.
    T t;
    return (*this)(t);
  }

  Field field;
};

// Returns true if field has field access and field is not out of bounds.
bool HasField(const Variable::ValueType& value, Field field) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_15(mht_15_v, 614, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "HasField");

  return absl::visit(FieldChecker{field}, value);
}

void AssembleAccessor(absl::string_view name, absl::string_view index,
                      absl::string_view field, std::string* result) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   mht_16_v.push_back("index: \"" + std::string(index.data(), index.size()) + "\"");
   mht_16_v.push_back("field: \"" + std::string(field.data(), field.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_16(mht_16_v, 625, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "AssembleAccessor");

  if (index.empty()) {
    absl::StrAppend(result, name, field);
  } else {
    absl::StrAppend(result, name, "[", index, "]", field);
  }
}

}  // namespace

RewriteStatus VariableAccessor::Rewrite(absl::string_view input,
                                        std::string* output) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_17(mht_17_v, 640, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::Rewrite");

  auto ref = variable_accessor_internal::Parse(input);
  if (ref.name.empty()) {
    absl::StrAppend(output, "INVALID_SYNTAX");
    return RewriteStatus::ERROR;
  }

  auto it =
      name_to_variable_.find(std::string(ref.name.data(), ref.name.size()));
  if (it == name_to_variable_.end()) {
    // Uniform with this name is not registered.
    return RewriteStatus::NOT_RECOGNIZED;
  }
  const auto& value = it->second.value;

  if (!ref.index.empty() && !IsVariableLength(value)) {
    // Trying to access variable by index, but it is not variable-length.
    absl::StrAppend(output, "INVALID_ACCESS_BY_INDEX");
    return RewriteStatus::ERROR;
  }

  Field f = ToField(ref.field);
  if (!ref.field.empty() && !HasField(value, f)) {
    // Trying to access a variable by field, but it does not have it.
    absl::StrAppend(output, "INVALID_ACCESS_BY_FIELD");
    return RewriteStatus::ERROR;
  }

  // Error checks are complete now.

  // All variable-length variables are encoded as-is without inlining.
  if (!inline_values_ || IsVariableLength(value)) {
    AssembleAccessor(it->second.name, ref.index, ref.field, output);
  } else {
    // Parameter + field is replaced with field value.
    if (f != Field::UNKNOWN) {
      GetValue(value, f, output);
    } else {
      // Parameter is accessed directly.
      GetValue(value, output);
    }
  }
  return RewriteStatus::SUCCESS;
}

bool VariableAccessor::AddSharedVariable(Variable&& variable) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_18(mht_18_v, 688, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::AddSharedVariable");

  const std::string name = variable.name;
  if (!name_to_variable_.insert({name, std::move(variable)}).second) {
    return false;
  }
  shared_variables_.insert(name);
  return true;
}

bool VariableAccessor::AddUniformParameter(Variable&& variable) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_19(mht_19_v, 700, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::AddUniformParameter");

  const std::string name = variable.name;
  if (!name_to_variable_.insert({name, std::move(variable)}).second) {
    return false;
  }
  uniform_parameters_.insert(name);
  return true;
}

bool VariableAccessor::IsEmptyVariableLength(const Variable& variable) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_20(mht_20_v, 712, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::IsEmptyVariableLength");

  const auto& value = variable.value;
  return IsVariableLength(value) && GetLength(value) == 0;
}

std::string VariableAccessor::GetConstDeclarations() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_21(mht_21_v, 720, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::GetConstDeclarations");

  // Variable length variables are declared as const and accessed via variable
  // with index.
  std::string declarations;
  for (const auto& variable : name_to_variable_) {
    // Skip shared variables.
    const std::string& variable_name = variable.second.name;
    if (shared_variables_.find(variable_name) != shared_variables_.end()) {
      continue;
    }

    const auto& value = variable.second.value;
    if (IsVariableLength(value)) {
      absl::StrAppend(&declarations, "const ", GetVariableType(value), " ",
                      variable_name, "[] = ");
      GetValue(value, &declarations);
      absl::StrAppend(&declarations, ";\n");
    }
  }
  return declarations;
}

std::string VariableAccessor::GetSharedVariableDeclarations() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_22(mht_22_v, 745, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::GetSharedVariableDeclarations");

  std::string declarations;
  for (const auto& name : shared_variables_) {
    const auto& variable = name_to_variable_.at(name);
    GenerateSharedVariableDeclaration(variable, &declarations);
  }
  return declarations;
}

std::string VariableAccessor::GetUniformParameterDeclarations() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_23(mht_23_v, 757, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::GetUniformParameterDeclarations");

  std::string declarations;
  if (!inline_values_) {
    if (vulkan_support_) {
      VulkanConstantsProcessor processor;
      for (const auto& name : uniform_parameters_) {
        const auto& variable = name_to_variable_.at(name);
        processor.ProcessVulkanConstant(variable, &declarations);
      }
      processor.GeneratePushConstantsDeclarations(&declarations);
    } else {
      for (const auto& name : uniform_parameters_) {
        const auto& variable = name_to_variable_.at(name);
        GenerateUniformParameterDeclaration(variable, &declarations);
      }
    }
  }
  return declarations;
}

std::vector<Variable> VariableAccessor::GetUniformParameters() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSvariable_accessorDTcc mht_24(mht_24_v, 780, "", "./tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.cc", "VariableAccessor::GetUniformParameters");

  std::vector<Variable> variables;
  if (!inline_values_) {
    variables.reserve(name_to_variable_.size());
    // Keep the order of the variables consistent with that of the declarations
    for (const auto& name : uniform_parameters_) {
      const auto& variable = name_to_variable_.at(name);
      variables.push_back(variable);
    }
  }
  return variables;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
