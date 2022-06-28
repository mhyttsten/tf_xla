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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc() {
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

#include "tensorflow/lite/delegates/gpu/cl/cl_arguments.h"

#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/gpu_object.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
bool IsWordSymbol(char symbol) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("symbol: '" + std::string(1, symbol) + "'");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "IsWordSymbol");

  return absl::ascii_isalnum(symbol) || symbol == '_';
}

void ReplaceAllWords(const std::string& old_word, const std::string& new_word,
                     std::string* str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("old_word: \"" + old_word + "\"");
   mht_1_v.push_back("new_word: \"" + new_word + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "ReplaceAllWords");

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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "AppendArgument");

  if (!args->empty()) {
    absl::StrAppend(args, ",\n  ");
  }
  absl::StrAppend(args, arg);
}

std::string GetImageModifier(AccessType access) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_3(mht_3_v, 246, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "GetImageModifier");

  switch (access) {
    case AccessType::READ:
      return "__read_only";
    case AccessType::WRITE:
      return "__write_only";
    case AccessType::READ_WRITE:
      return "__read_write";
  }
}

std::string GetDefaultSamplers(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_4(mht_4_v, 260, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "GetDefaultSamplers");

  std::string result;
  result +=
      "__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | "
      "CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  if (gpu_info.IsAdreno() && gpu_info.adreno_info.IsAdreno3xx()) {
    // Unfortunately, CLK_ADDRESS_CLAMP is very slow on Adreno3xx and
    // we can observe huge register overhead when compared to other modes.

    // While using CLK_ADDRESS_NONE with out-of-range image coordinates is
    // undefined in the OpenCL specification, we have observed that
    // CLK_ADDRESS_NONE works like CLK_ADDRESS_CLAMP for out-of-range image
    // coordinates for RGBA F16/F32 textures on Adreno3xx devices. Using
    // CLK_ADDRESS_NONE is significantly faster than CLK_ADDRESS_CLAMP on Adreno
    // 3xx.
    result +=
        "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | "
        "CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n";
  } else {
    result +=
        "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | "
        "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
  }

  return result;
}

absl::Status CreateCLObject(GPUObjectDescriptor* desc, CLContext* context,
                            GPUObjectPtr* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_5(mht_5_v, 291, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CreateCLObject");

  const auto* buffer_desc = dynamic_cast<const BufferDescriptor*>(desc);
  if (buffer_desc) {
    Buffer gpu_buffer;
    RETURN_IF_ERROR(
        gpu_buffer.CreateFromBufferDescriptor(*buffer_desc, context));
    *result = absl::make_unique<Buffer>(std::move(gpu_buffer));
    return absl::OkStatus();
  }

  const auto* texture_desc = dynamic_cast<const Texture2DDescriptor*>(desc);
  if (texture_desc) {
    Texture2D gpu_texture;
    RETURN_IF_ERROR(
        gpu_texture.CreateFromTexture2DDescriptor(*texture_desc, context));
    *result = absl::make_unique<Texture2D>(std::move(gpu_texture));
    return absl::OkStatus();
  }

  const auto* linear_desc = dynamic_cast<const TensorLinearDescriptor*>(desc);
  if (linear_desc) {
    LinearStorage gpu_storage;
    RETURN_IF_ERROR(
        gpu_storage.CreateFromTensorLinearDescriptor(*linear_desc, context));
    *result = absl::make_unique<LinearStorage>(std::move(gpu_storage));
    return absl::OkStatus();
  }

  const auto* tensor_desc = dynamic_cast<const TensorDescriptor*>(desc);
  if (tensor_desc) {
    Tensor gpu_tensor;
    RETURN_IF_ERROR(gpu_tensor.CreateFromDescriptor(*tensor_desc, context));
    *result = absl::make_unique<Tensor>(std::move(gpu_tensor));
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unknown GPU descriptor.");
}

}  // namespace

// Static
constexpr char CLArguments::kArgsPrefix[];

absl::Status CLArguments::Init(const GpuInfo& gpu_info, CLContext* context,
                               Arguments* args, std::string* code) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_6(mht_6_v, 339, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::Init");

  RETURN_IF_ERROR(AllocateObjects(*args, context));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  object_refs_ = std::move(args->object_refs_);
  const bool use_f32_for_halfs = gpu_info.IsPowerVR();
  CopyArguments(*args, use_f32_for_halfs);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  RenameArgumentsInCode(code);
  args->ResolveArgsPass(code);
  *code = absl::Substitute(*code, GetListOfArgs());
  if (gpu_info.SupportsImages()) {
    *code = GetDefaultSamplers(gpu_info) + *code;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::Init(const GpuInfo& gpu_info, Arguments* args,
                               CLContext* context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_7(mht_7_v, 359, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::Init");

  RETURN_IF_ERROR(AllocateObjects(*args, context));
  RETURN_IF_ERROR(AddObjectArgs(gpu_info, *args));
  object_refs_ = std::move(args->object_refs_);
  const bool use_f32_for_halfs = gpu_info.IsPowerVR();
  CopyArguments(*args, use_f32_for_halfs);
  RETURN_IF_ERROR(SetObjectsResources(*args));
  return absl::OkStatus();
}

absl::Status CLArguments::AllocateObjects(const Arguments& args,
                                          CLContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_8(mht_8_v, 373, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AllocateObjects");

  objects_.resize(args.objects_.size());
  int i = 0;
  for (auto& t : args.objects_) {
    RETURN_IF_ERROR(CreateCLObject(t.second.get(), context, &objects_[i]));
    i++;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::AddObjectArgs(const GpuInfo& gpu_info,
                                        const Arguments& args) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_9(mht_9_v, 387, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddObjectArgs");

  for (const auto& t : args.objects_) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  for (const auto& t : args.object_refs_) {
    AddGPUResources(t.first, t.second->GetGPUResources(gpu_info));
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetObjectsResources(const Arguments& args) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_10(mht_10_v, 400, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetObjectsResources");

  int i = 0;
  for (const auto& t : args.objects_) {
    GPUResourcesWithValue resources;
    RETURN_IF_ERROR(objects_[i]->GetGPUResources(t.second.get(), &resources));
    RETURN_IF_ERROR(SetGPUResources(t.first, resources));
    i++;
  }
  return absl::OkStatus();
}

void CLArguments::CopyArguments(const Arguments& args, bool use_f32_for_halfs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_11(mht_11_v, 414, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::CopyArguments");

  for (const auto& fvalue : args.float_values_) {
    auto& new_val = float_values_[fvalue.first];
    new_val.value = fvalue.second.value;
    new_val.active = fvalue.second.active;
    if (fvalue.second.active) {
      new_val.offset = shared_float4s_data_.size();
      shared_float4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& ivalue : args.int_values_) {
    auto& new_val = int_values_[ivalue.first];
    new_val.value = ivalue.second.value;
    new_val.active = ivalue.second.active;
    if (ivalue.second.active) {
      new_val.offset = shared_int4s_data_.size();
      shared_int4s_data_.push_back(new_val.value);
    }
  }
  for (const auto& hfvalue : args.half_values_) {
    auto& new_val = half_values_[hfvalue.first];
    new_val.value = hfvalue.second.value;
    new_val.active = hfvalue.second.active;
    if (hfvalue.second.active) {
      if (use_f32_for_halfs) {
        new_val.store_as_f32 = true;
        new_val.offset = shared_float4s_data_.size();
        shared_float4s_data_.push_back(new_val.value);
      } else {
        new_val.store_as_f32 = false;
        new_val.offset = shared_half4s_data_.size();
        shared_half4s_data_.push_back(new_val.value);
      }
    }
  }
  int shared_int4s_aligned_size = AlignByN(shared_int4s_data_.size(), 4);
  shared_int4s_data_.resize(shared_int4s_aligned_size);
  int shared_float4s_aligned_size = AlignByN(shared_float4s_data_.size(), 4);
  shared_float4s_data_.resize(shared_float4s_aligned_size);
  int shared_half4s_aligned_size = AlignByN(shared_half4s_data_.size(), 4);
  shared_half4s_data_.resize(shared_half4s_aligned_size);
}

void CLArguments::RenameArgumentsInCode(std::string* code) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_12(mht_12_v, 460, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::RenameArgumentsInCode");

  const std::string postfixes[4] = {"x", "y", "z", "w"};
  for (const auto& fvalue : float_values_) {
    if (fvalue.second.active) {
      std::string index = std::to_string(fvalue.second.offset / 4);
      std::string new_name =
          "shared_float4_" + index + "." + postfixes[fvalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + fvalue.first, new_name, code);
    }
  }
  for (const auto& ivalue : int_values_) {
    if (ivalue.second.active) {
      std::string index = std::to_string(ivalue.second.offset / 4);
      std::string new_name =
          "shared_int4_" + index + "." + postfixes[ivalue.second.offset % 4];
      ReplaceAllWords(kArgsPrefix + ivalue.first, new_name, code);
    }
  }
  for (const auto& hfvalue : half_values_) {
    if (hfvalue.second.active) {
      std::string index = std::to_string(hfvalue.second.offset / 4);
      std::string new_name;
      if (hfvalue.second.store_as_f32) {
        new_name = "(half)(shared_float4_" + index + "." +
                   postfixes[hfvalue.second.offset % 4] + ")";
      } else {
        new_name = "shared_half4_" + index + "." +
                   postfixes[hfvalue.second.offset % 4];
      }
      ReplaceAllWords(kArgsPrefix + hfvalue.first, new_name, code);
    }
  }
}

void CLArguments::AddBuffer(const std::string& name,
                            const GPUBufferDescriptor& desc) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_13(mht_13_v, 499, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddBuffer");

  buffers_[name].desc = desc;
}
void CLArguments::AddImage2D(const std::string& name,
                             const GPUImage2DDescriptor& desc) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_14(mht_14_v, 507, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddImage2D");

  images2d_[name].desc = desc;
}

void CLArguments::AddImage2DArray(const std::string& name,
                                  const GPUImage2DArrayDescriptor& desc) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_15(mht_15_v, 516, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddImage2DArray");

  image2d_arrays_[name].desc = desc;
}

void CLArguments::AddImage3D(const std::string& name,
                             const GPUImage3DDescriptor& desc) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_16(mht_16_v, 525, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddImage3D");

  images3d_[name].desc = desc;
}

void CLArguments::AddImageBuffer(const std::string& name,
                                 const GPUImageBufferDescriptor& desc) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_17(mht_17_v, 534, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddImageBuffer");

  image_buffers_[name].desc = desc;
}

void CLArguments::AddCustomMemory(const std::string& name,
                                  const GPUCustomMemoryDescriptor& desc) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_18(mht_18_v, 543, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddCustomMemory");

  custom_memories_[name].desc = desc;
}

void CLArguments::AddGPUResources(const std::string& name,
                                  const GPUResources& resources) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_19(mht_19_v, 552, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::AddGPUResources");

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
  for (const auto& r : resources.custom_memories) {
    AddCustomMemory(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status CLArguments::SetInt(const std::string& name, int value) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_20(mht_20_v, 577, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetInt");

  auto it = int_values_.find(name);
  if (it == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_int4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}
absl::Status CLArguments::SetFloat(const std::string& name, float value) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_21(mht_21_v, 593, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetFloat");

  auto it = float_values_.find(name);
  if (it == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    shared_float4s_data_[it->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetHalf(const std::string& name, half value) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_22(mht_22_v, 610, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetHalf");

  auto it = half_values_.find(name);
  if (it == half_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No half argument with name - ", name));
  }
  it->second.value = value;
  if (it->second.active) {
    if (it->second.store_as_f32) {
      shared_float4s_data_[it->second.offset] = value;
    } else {
      shared_half4s_data_[it->second.offset] = value;
    }
  }
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage2D(const std::string& name, cl_mem memory) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_23(mht_23_v, 631, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetImage2D");

  auto it = images2d_.find(name);
  if (it == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetBuffer(const std::string& name, cl_mem memory) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_24(mht_24_v, 645, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetBuffer");

  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage2DArray(const std::string& name,
                                          cl_mem memory) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_25(mht_25_v, 660, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetImage2DArray");

  auto it = image2d_arrays_.find(name);
  if (it == image2d_arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D array argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImage3D(const std::string& name, cl_mem memory) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_26(mht_26_v, 674, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetImage3D");

  auto it = images3d_.find(name);
  if (it == images3d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image3D argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetImageBuffer(const std::string& name,
                                         cl_mem memory) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_27(mht_27_v, 689, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetImageBuffer");

  auto it = image_buffers_.find(name);
  if (it == image_buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetCustomMemory(const std::string& name,
                                          cl_mem memory) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_28(mht_28_v, 704, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetCustomMemory");

  auto it = custom_memories_.find(name);
  if (it == custom_memories_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No custom memory argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status CLArguments::SetObjectRef(const std::string& name,
                                       const GPUObject* object) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_29(mht_29_v, 719, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetObjectRef");

  auto it = object_refs_.find(name);
  if (it == object_refs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No object ref with name - ", name));
  }
  GPUResourcesWithValue resources;
  RETURN_IF_ERROR(object->GetGPUResources(it->second.get(), &resources));
  return SetGPUResources(name, resources);
}

absl::Status CLArguments::SetGPUResources(
    const std::string& name, const GPUResourcesWithValue& resources) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_30(mht_30_v, 735, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::SetGPUResources");

  for (const auto& r : resources.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.buffers) {
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second));
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
  for (const auto& r : resources.custom_memories) {
    RETURN_IF_ERROR(
        SetCustomMemory(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

std::string CLArguments::GetListOfArgs() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_31(mht_31_v, 768, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::GetListOfArgs");

  std::string result;
  for (auto& t : buffers_) {
    const std::string type_name =
        t.second.desc.data_type == DataType::FLOAT32 ? "float" : "half";
    std::string attributes;
    for (const auto& attr : t.second.desc.attributes) {
      attributes += absl::StrCat("  __attribute__((", attr, "))");
    }
    AppendArgument(
        absl::StrCat(
            MemoryTypeToCLType(t.second.desc.memory_type), " ",
            ToCLDataType(t.second.desc.data_type, t.second.desc.element_size),
            "* ", t.first, attributes),
        &result);
  }
  for (auto& t : image_buffers_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image1d_buffer_t ", t.first),
                   &result);
  }
  for (auto& t : images2d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image2d_t ", t.first),
                   &result);
  }
  for (auto& t : image2d_arrays_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image2d_array_t ", t.first),
                   &result);
  }
  for (auto& t : images3d_) {
    AppendArgument(absl::StrCat(GetImageModifier(t.second.desc.access_type),
                                " image3d_t ", t.first),
                   &result);
  }
  for (auto& t : custom_memories_) {
    AppendArgument(absl::StrCat(t.second.desc.type_name, " ", t.first),
                   &result);
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("int4 shared_int4_", i), &result);
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("float4 shared_float4_", i), &result);
  }
  for (int i = 0; i < shared_half4s_data_.size() / 4; ++i) {
    AppendArgument(absl::StrCat("half4 shared_half4_", i), &result);
  }
  return result;
}

absl::Status CLArguments::Bind(cl_kernel kernel, int offset) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSclPScl_argumentsDTcc mht_32(mht_32_v, 823, "", "./tensorflow/lite/delegates/gpu/cl/cl_arguments.cc", "CLArguments::Bind");

  for (auto& t : buffers_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : image_buffers_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : images2d_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : image2d_arrays_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : images3d_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : custom_memories_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_int4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_float4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_half4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int16_t) * 4,
                                          &shared_half4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
