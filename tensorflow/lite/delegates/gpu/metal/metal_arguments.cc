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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc() {
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
#include "tensorflow/lite/delegates/gpu/metal/metal_arguments.h"

#include <cstring>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/buffer.h"
#include "tensorflow/lite/delegates/gpu/metal/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/texture2d.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
bool IsWordSymbol(char symbol) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("symbol: '" + std::string(1, symbol) + "'");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "IsWordSymbol");

  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("old_word: \"" + old_word + "\"");
   mht_1_v.push_back("new_word: \"" + new_word + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "ReplaceAllWords");

  if (!str) {
    return;
  }
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

void AppendArgument(const std::string& arg, std::string* args) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("arg: \"" + arg + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "AppendArgument");

  if (!args->empty()) {
    absl::StrAppend(args, ",\n");
  }
  absl::StrAppend(args, arg);
}

absl::Status CreateMetalObject(id<MTLDevice> device, GPUObjectDescriptor* desc,
                            GPUObjectPtr* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "CreateMetalObject");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(desc);
  if (buffer_desc) {
    Buffer gpu_buffer;
    RETURN_IF_ERROR(
        gpu_buffer.CreateFromBufferDescriptor(*buffer_desc, device));
    *result = absl::make_unique<Buffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(desc);
  if (texture_desc) {
    Texture2D gpu_texture;
    RETURN_IF_ERROR(
        gpu_texture.CreateFromTexture2DDescriptor(*texture_desc, device));
    *result = absl::make_unique<Texture2D>(std::move(gpu_texture));
    return absl::OkStatus();
  }

  const auto* linear_desc = dynamic_cast<const TensorLinearDescriptor*>(desc);
  if (linear_desc) {
    LinearStorage gpu_storage;
    RETURN_IF_ERROR(
        gpu_storage.CreateFromTensorLinearDescriptor(*linear_desc, device));
    *result = absl::make_unique<LinearStorage>(std::move(gpu_storage));
    return absl::OkStatus();
  }

  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc);
  if (tensor_desc) {
    MetalSpatialTensor gpu_tensor;
    RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(*tensor_desc, device));
    *result = absl::make_unique<MetalSpatialTensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

std::string AccessToMetalTextureAccess(AccessType access_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_4(mht_4_v, 289, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "AccessToMetalTextureAccess");

  if (access_type == AccessType::READ) {
    return "access::read";
  } else if (access_type == AccessType::READ_WRITE) {
    return "access::read_write";
  } else if (access_type == AccessType::WRITE) {
    return "access::write";
  } else {
    return "access::unknown";
  }
}
}  // namespace

// Static
constexpr char MetalArguments::kArgsPrefix[];

absl::Status MetalArguments::Init(
    bool use_arguments_buffer, MetalDevice* device, Arguments* args,
    std::string* code) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_5(mht_5_v, 310, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::Init");

  RETURN_IF_ERROR(AllocateObjects(*args, device->device()));
  RETURN_IF_ERROR(AddObjectArgs(device->GetInfo(), *args));
  args->MoveObjectRefs(&object_refs_);
  std::string call_prefix = use_arguments_buffer ? "args." : "";
  std::string struct_desc =
      CopyScalarArgumentsToStructWithVec4Fields(*args, call_prefix, code);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  if (!use_arguments_buffer) {
    args->ResolveArgsPass(code);
  }
  std::string header = R"(
#include <metal_stdlib>
using namespace metal;

)";
  header += struct_desc + "\n";
  if (use_arguments_buffer) {
    const std::string arg_buf_struct =
        GetArgumentBufferStructDefinition(!struct_desc.empty());
    header += arg_buf_struct + "\n";
  }
  *code = header + *code;
  std::string arguments;
  if (use_arguments_buffer) {
    arguments = "device ArgBuffer& args[[buffer(0)]]";
  } else {
    arguments = GetListOfArgs(/*buffer_offset*/ 0);
  }
  const bool use_global_id = code->find("GLOBAL_ID_") != std::string::npos;
  const bool use_local_id = code->find("LOCAL_ID_") != std::string::npos;
  const bool use_group_id = code->find("GROUP_ID_") != std::string::npos;
  const bool use_group_size = code->find("GROUP_SIZE_") != std::string::npos;
  const bool use_simd_id =
      code->find("SUB_GROUP_LOCAL_ID") != std::string::npos;
  if (use_global_id) {
    AppendArgument("uint3 reserved_gid[[thread_position_in_grid]]", &arguments);
  }
  if (use_local_id) {
    AppendArgument("uint3 reserved_lid[[thread_position_in_threadgroup]]",
                   &arguments);
  }
  if (use_group_id) {
    AppendArgument("uint3 reserved_group_id[[threadgroup_position_in_grid]]",
                   &arguments);
  }
  if (use_group_size) {
    AppendArgument("uint3 reserved_group_size[[threads_per_threadgroup]]",
                   &arguments);
  }
  if (use_simd_id) {
    AppendArgument("uint reserved_simd_id[[thread_index_in_simdgroup]]",
                   &arguments);
  }
  if (!use_global_id && !use_local_id && !use_group_id && !use_group_size &&
      !arguments.empty()) {
    arguments += ",\n";
  }
  *code = absl::Substitute(*code, arguments);
  return absl::OkStatus();
}

absl::Status MetalArguments::Init(bool use_arguments_buffer,
                                  MetalDevice* device, Arguments* args) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_6(mht_6_v, 376, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::Init");

  RETURN_IF_ERROR(AllocateObjects(*args, device->device()));
  RETURN_IF_ERROR(AddObjectArgs(device->GetInfo(), *args));
  args->MoveObjectRefs(&object_refs_);
  CopyScalarArgumentsToStructWithVec4Fields(*args);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  return absl::OkStatus();
}

std::string MetalArguments::CopyScalarArgumentsToStructWithScalarFields(
    const Arguments& args, const std::string& call_prefix, std::string* code) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("call_prefix: \"" + call_prefix + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_7(mht_7_v, 390, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::CopyScalarArgumentsToStructWithScalarFields");

  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  for (auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + fvalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + fvalue.first,
                      call_prefix + "U." + fvalue.first, code);
    }
  }
  for (const auto& hfvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  float " + hfvalue.first + ";\n";
      ReplaceAllWords(
          kArgsPrefix + hfvalue.first,
          "static_cast<half>(" + call_prefix + "U." + hfvalue.first + ")",
          code);
    }
  }
  for (auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      pos++;
      struct_desc += "  int " + ivalue.first + ";\n";
      ReplaceAllWords(kArgsPrefix + ivalue.first,
                      call_prefix + "U." + ivalue.first, code);
    }
  }
  if (pos != 0) {
    int aligned_pos = AlignByN(pos, 4);
    for (int i = pos; i < aligned_pos; i++) {
      struct_desc += "  int dummy" + std::to_string(i - pos) + ";\n";
    }
    struct_desc += "};";
    const_data_.resize(aligned_pos * 4);
    for (auto& it : float_values_) {
      if (it.second.active) {
        float* ptr =
            reinterpret_cast<float*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : int_values_) {
      if (it.second.active) {
        int32_t* ptr =
            reinterpret_cast<int32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
  } else {
    struct_desc = "";
  }
  return struct_desc;
}

std::string MetalArguments::CopyScalarArgumentsToStructWithVec4Fields(
    const Arguments& args, const std::string& call_prefix, std::string* code) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("call_prefix: \"" + call_prefix + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_8(mht_8_v, 463, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::CopyScalarArgumentsToStructWithVec4Fields");

  std::string struct_desc = "struct uniforms_buffer {\n";
  int pos = 0;
  std::string channels[4] = {".x", ".y", ".z", ".w"};
  for (auto& fvalue : args.GetFloatValues()) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = call_prefix + "U.cmp_float4_" +
                             std::to_string(pos / 4) + channels[pos % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
      pos++;
    }
  }
  for (const auto& hfvalue : args.GetHalfValues()) {
    auto& new_val = float_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  float4 cmp_float4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = "static_cast<half>(" + call_prefix +
                             "U.cmp_float4_" + std::to_string(pos / 4) +
                             channels[pos % 4] + ")";
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
      pos++;
    }
  }
  pos = AlignByN(pos, 4);
  for (auto& ivalue : args.GetIntValues()) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.bytes_offset = pos * 4;
      if (pos % 4 == 0) {
        struct_desc += "  int4 cmp_int4_" + std::to_string(pos / 4) + ";\n";
      }
      std::string new_name = call_prefix + "U.cmp_int4_" +
                             std::to_string(pos / 4) + channels[pos % 4];
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
      pos++;
    }
  }
  if (pos != 0) {
    int aligned_pos = AlignByN(pos, 4);
    struct_desc += "};";
    const_data_.resize(aligned_pos * 4);
    for (auto& it : float_values_) {
      if (it.second.active) {
        float* ptr =
            reinterpret_cast<float*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
    for (auto& it : int_values_) {
      if (it.second.active) {
        int32_t* ptr =
            reinterpret_cast<int32_t*>(&const_data_[it.second.bytes_offset]);
        *ptr = it.second.value;
      }
    }
  } else {
    struct_desc = "";
  }
  return struct_desc;
}

std::string MetalArguments::GetArgumentBufferStructDefinition(
    bool add_constants_struct) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_9(mht_9_v, 542, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::GetArgumentBufferStructDefinition");

  std::string result;
  result = "struct ArgBuffer {\n";
  int index = 0;
  for (auto& t : buffers_) {
    std::string mem_type = MemoryTypeToMetalType(t.second.desc.memory_type);
    std::string metal_type =
        ToMetalDataType(t.second.desc.data_type, t.second.desc.element_size);
    result += absl::StrCat("  ", mem_type, " ", metal_type, "* ", t.first,
                           "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : images2d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture2d<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : image2d_arrays_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture2d_array<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : images3d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture3d<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  for (auto& t : image_buffers_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    result += absl::StrCat("  texture_buffer<", data_type, ", ", access, "> ",
                           t.first, "[[id(", index, ")]];\n");
    index++;
  }
  if (add_constants_struct) {
    result += "  uniforms_buffer U;\n";
  }
  result += "};";
  return result;
}

absl::Status MetalArguments::SetInt(const std::string& name, int value) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_10(mht_10_v, 597, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetInt");

  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    int32_t* ptr =
        reinterpret_cast<int32_t*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}
absl::Status MetalArguments::SetFloat(const std::string& name, float value) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_11(mht_11_v, 615, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetFloat");

  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::SetHalf(const std::string& name, half value) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_12(mht_12_v, 634, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetHalf");

  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    float* ptr =
        reinterpret_cast<float*>(&const_data_[it->second.bytes_offset]);
    *ptr = value;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::SetObjectRef(const std::string& name,
                                          const GPUObject& object) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_13(mht_13_v, 654, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetObjectRef");

  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  RETURN_IF_ERROR(object.GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

void MetalArguments::Encode(id<MTLComputeCommandEncoder> encoder,
                            int buffer_offset, int texture_offset) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_14(mht_14_v, 669, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::Encode");

  for (auto& b : buffers_) {
    [encoder setBuffer:b.second.handle
                offset:b.second.offset
               atIndex:buffer_offset];
    buffer_offset++;
  }
  for (auto& image : images2d_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : image2d_arrays_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : images3d_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }
  for (auto& image : image_buffers_) {
    [encoder setTexture:image.second.handle atIndex:texture_offset];
    texture_offset++;
  }

  if (!const_data_.empty()) {
    [encoder setBytes:const_data_.data()
               length:const_data_.size()
              atIndex:buffer_offset];
  }
}

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void MetalArguments::AddResourcesToEncoder(
    id<MTLComputeCommandEncoder> encoder) const {
  for (auto& b : buffers_) {
    [encoder useResource:b.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : images2d_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : image2d_arrays_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : images3d_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  for (auto& image : image_buffers_) {
    [encoder useResource:image.second.handle
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
}

API_AVAILABLE(ios(11.0), macos(10.13), tvos(11.0))
void MetalArguments::EncodeArguments(id<MTLArgumentEncoder> arguments_encoder) {
  int index = 0;
  for (auto& b : buffers_) {
    [arguments_encoder setBuffer:b.second.handle
                          offset:b.second.offset
                         atIndex:index];
    index++;
  }
  for (auto& image : images2d_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : image2d_arrays_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : images3d_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  for (auto& image : image_buffers_) {
    [arguments_encoder setTexture:image.second.handle atIndex:index];
    index++;
  }
  if (!const_data_.empty()) {
    std::memcpy([arguments_encoder constantDataAtIndex:index],
                const_data_.data(), const_data_.size());
  }
}

absl::Status MetalArguments::AllocateObjects(const Arguments& args,
                                          id<MTLDevice> device) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_15(mht_15_v, 760, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AllocateObjects");

  objects_.resize(args.GetObjects().size());
  int i = 0;
  for (auto& t : args.GetObjects()) {
    RETURN_IF_ERROR(CreateMetalObject(device, t.second.get(), &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status MetalArguments::AddObjectArgs(const GpuInfo& gpu_info,
                                           const Arguments& args) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_16(mht_16_v, 774, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddObjectArgs");

  for (const auto& t : args.GetObjects()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  for (const auto& t : args.GetObjectRefs()) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  return absl::OkStatus();
}

std::string MetalArguments::GetListOfArgs(int buffer_offset,
                                          int textures_offset) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_17(mht_17_v, 788, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::GetListOfArgs");

  std::string result;
  for (auto& t : buffers_) {
    AppendArgument(
        absl::StrCat(MemoryTypeToMetalType(t.second.desc.memory_type), " ",
                     ToMetalDataType(t.second.desc.data_type,
                                     t.second.desc.element_size),
                     "* ", t.first, "[[buffer(", buffer_offset, ")]]"),
        &result);
    buffer_offset++;
  }
  for (auto& t : images2d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    if (t.second.desc.normalized) {
      data_type = ToMetalDataType(t.second.desc.normalized_type);
    }
    AppendArgument(absl::StrCat("texture2d<", data_type, ", ", access, "> ",
                                t.first, "[[texture(", textures_offset, ")]]"),
                   &result);
    textures_offset++;
  }
  for (auto& t : image2d_arrays_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    AppendArgument(
        absl::StrCat("texture2d_array<", data_type, ", ", access, "> ", t.first,
                     "[[texture(", textures_offset, ")]]"),
        &result);
    textures_offset++;
  }
  for (auto& t : images3d_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    AppendArgument(absl::StrCat("texture3d<", data_type, ", ", access, "> ",
                                t.first, "[[texture(", textures_offset, ")]]"),
                   &result);
    textures_offset++;
  }
  for (auto& t : image_buffers_) {
    std::string access = AccessToMetalTextureAccess(t.second.desc.access_type);
    std::string data_type =
        ToMetalDataType(ToMetalTextureType(t.second.desc.data_type));
    AppendArgument(
        absl::StrCat("texture_buffer<", data_type, ", ", access, "> ", t.first,
                     "[[texture(", textures_offset, ")]]"),
        &result);
    textures_offset++;
  }
  if (!const_data_.empty()) {
    AppendArgument(absl::StrCat("constant uniforms_buffer& U[[buffer(",
                                buffer_offset, ")]]"),
                   &result);
    buffer_offset++;
  }
  return result;
}

absl::Status MetalArguments::SetGPUResources(
    const std::string& name, const GPUResourcesWithValue& resources) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_18(mht_18_v, 854, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetGPUResources");

  for (const auto& r : resources.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.buffers) {
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second.handle,
                              r.second.offset));
  }
  for (const auto& r : resources.images2d) {
    RETURN_IF_ERROR(SetImage2D(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.image2d_arrays) {
    RETURN_IF_ERROR(
        SetImage2DArray(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.images3d) {
    RETURN_IF_ERROR(SetImage3D(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.image_buffers) {
    RETURN_IF_ERROR(SetImageBuffer(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

void MetalArguments::AddBuffer(const std::string& name,
                               const GPUBufferDescriptor& desc) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_19(mht_19_v, 886, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddBuffer");

  buffers_[name].desc = desc;
}

void MetalArguments::AddImage2D(const std::string& name,
                                const GPUImage2DDescriptor& desc) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_20(mht_20_v, 895, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddImage2D");

  images2d_[name].desc = desc;
}

void MetalArguments::AddImage2DArray(const std::string& name,
                                     const GPUImage2DArrayDescriptor& desc) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_21(mht_21_v, 904, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddImage2DArray");

  image2d_arrays_[name].desc = desc;
}

void MetalArguments::AddImage3D(const std::string& name,
                                const GPUImage3DDescriptor& desc) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_22(mht_22_v, 913, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddImage3D");

  images3d_[name].desc = desc;
}

void MetalArguments::AddImageBuffer(const std::string& name,
                                    const GPUImageBufferDescriptor& desc) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_23(mht_23_v, 922, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddImageBuffer");

  image_buffers_[name].desc = desc;
}

void MetalArguments::AddGPUResources(const std::string& name,
                                     const GPUResources& resources) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_24(mht_24_v, 931, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::AddGPUResources");

  for (const auto& r : resources.buffers) {
    AddBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images2d) {
    AddImage2D(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.image2d_arrays) {
    AddImage2DArray(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images3d) {
    AddImage3D(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.image_buffers) {
    AddImageBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status MetalArguments::SetBuffer(const std::string& name,
                                       id<MTLBuffer> handle, uint64_t offset) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_25(mht_25_v, 954, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetBuffer");

  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.handle = handle;
  it->second.offset = offset;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImage2D(const std::string& name,
                                        id<MTLTexture> handle) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_26(mht_26_v, 970, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetImage2D");

  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImage2DArray(const std::string& name,
                                             id<MTLTexture> handle) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_27(mht_27_v, 985, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetImage2DArray");

  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2d array argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImage3D(const std::string& name,
                                        id<MTLTexture> handle) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_28(mht_28_v, 1000, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetImage3D");

  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3d argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetImageBuffer(const std::string& name,
                                            id<MTLTexture> handle) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_29(mht_29_v, 1015, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetImageBuffer");

  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image buffer argument with name - ", name));
  }
  it->second.handle = handle;
  return absl::OkStatus();
}

absl::Status MetalArguments::SetObjectsResources(const Arguments& args) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSmetalPSmetal_argumentsDTcc mht_30(mht_30_v, 1028, "", "./tensorflow/lite/delegates/gpu/metal/metal_arguments.cc", "MetalArguments::SetObjectsResources");

  int i = 0;
  for (const auto& t : args.GetObjects()) {
    GPUResourcesWithValue resources;
    RETURN_IF_ERROR(objects_[i]->GetGPUResources(t.second.get(), &resources));
    RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
