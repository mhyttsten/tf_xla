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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/kernels/converter.h"

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Wraps given SSBO into GlBuffer object that does not have ownership.
absl::Status WrapSSBO(OpenGlBuffer ssbo, GlBuffer* buffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "WrapSSBO");

  int64_t size_bytes;
  RETURN_IF_ERROR(GetSSBOSize(ssbo.id, &size_bytes));
  *buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.id, size_bytes, 0, false);
  return absl::OkStatus();
}

std::string GetShaderHeader(const uint3& localsize) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_1(mht_1_v, 213, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "GetShaderHeader");

  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

class OpenGlConverterImpl : public TensorObjectConverter {
 public:
  explicit OpenGlConverterImpl(CommandQueue* command_queue)
      : command_queue_(command_queue) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "OpenGlConverterImpl");
}

  virtual absl::Status Init(const TensorObjectDef& input_def,
                            const TensorObjectDef& output_def) = 0;

 protected:
  absl::Status InitializeProgram(const uint3& workgroup_size,
                                 const std::string& shader_source) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("shader_source: \"" + shader_source + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_3(mht_3_v, 236, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "InitializeProgram");

    workgroup_size_ = workgroup_size;
    GlShader shader;
    RETURN_IF_ERROR(GlShader::CompileShader(
        GL_COMPUTE_SHADER, GetShaderHeader(workgroup_size) + shader_source,
        &shader));
    return GlProgram::CreateWithShader(shader, &program_);
  }

  absl::Status Dispatch(const uint3& workload) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_4(mht_4_v, 248, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Dispatch");

    uint3 num_workgroups = DivideRoundUp(workload, workgroup_size_);
    if (command_queue_) {
      return command_queue_->Dispatch(program_, num_workgroups);
    }
    return program_.Dispatch(num_workgroups);
  }

  GlProgram program_;
  uint3 workgroup_size_;
  CommandQueue* command_queue_;
};

bool IsSupportedDataType(DataType type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_5(mht_5_v, 264, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupportedDataType");
 return type == DataType::FLOAT32; }

uint32_t SizeInBytesDHWC4(const BHWC& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_6(mht_6_v, 269, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "SizeInBytesDHWC4");

  return shape.b * shape.h * shape.w * AlignByN(shape.c, 4) * sizeof(float);
}

uint32_t SizeInBytesBHWC(const BHWC& shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_7(mht_7_v, 276, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "SizeInBytesBHWC");

  return shape.DimensionsProduct() * sizeof(float);
}

// Implements conversion from OpenGL-specific tensor layout to BHWC.
class FromTensorConverter : public OpenGlConverterImpl {
 public:
  explicit FromTensorConverter(CommandQueue* command_queue)
      : OpenGlConverterImpl(command_queue) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_8(mht_8_v, 287, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "FromTensorConverter");
}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_9(mht_9_v, 292, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupported");

    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Output is always SSBO/BHWC
           output.object_type == ObjectType::OPENGL_SSBO &&
           output.data_layout == DataLayout::BHWC &&
           // SSBO/DHWC4 ->
           input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def) final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_10(mht_10_v, 307, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Init");

    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    if (shape_.b != 1) {
      return absl::UnimplementedError(
          "FromTensorConverter: Batch size != 1 is not supported.");
    }

    return InitializeProgram(uint3(8, 4, 2), R"(
    layout(std430) buffer;
    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      vec4 elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      float elements[];
    } output_data;

    uniform ivec4 sizes;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes.x || gid.y >= sizes.y || gid.z >= sizes.z) {
        return;
      }
      output_data.elements[(gid.y * sizes.x + gid.x) * sizes.z + gid.z] = input_data.elements[(gid.z / 4 * sizes.y + gid.y) * sizes.x + gid.x][gid.z % 4];
    })");
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_11(mht_11_v, 342, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Convert");

    auto output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (!output || !output->id) {
      return absl::InvalidArgumentError("Missing output in converter");
    }
    auto input = absl::get_if<OpenGlBuffer>(&input_obj);
    if (!input || !input->id) {
      return absl::InvalidArgumentError("Missing input in converter");
    }
    if (input->id == output->id) {
      return absl::InvalidArgumentError("Can not execute inplace conversion");
    }
    GlBuffer input_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*input, &input_ssbo));
    GlBuffer output_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*output, &output_ssbo));

    if (input_ssbo.bytes_size() != SizeInBytesDHWC4(shape_)) {
      return absl::InvalidArgumentError(
          "FromTensorConverter: input data size does not match expected size.");
    }
    if (output_ssbo.bytes_size() != SizeInBytesBHWC(shape_)) {
      return absl::InvalidArgumentError(
          "FromTensorConverter: output data size does not match expected "
          "size.");
    }
    RETURN_IF_ERROR(program_.SetParameter(
        {"sizes",
         int4(static_cast<int32_t>(shape_.w), static_cast<int32_t>(shape_.h),
              static_cast<int32_t>(shape_.c), 0)}));
    RETURN_IF_ERROR(input_ssbo.BindToIndex(0));
    RETURN_IF_ERROR(output_ssbo.BindToIndex(1));
    return Dispatch(uint3(shape_.w, shape_.h, shape_.c));
  }

  BHWC shape_;
};

// Implements conversion from BHWC to OpenGL-specific tensor layout.
class ToTensorConverter : public OpenGlConverterImpl {
 public:
  explicit ToTensorConverter(CommandQueue* command_queue)
      : OpenGlConverterImpl(command_queue) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_12(mht_12_v, 387, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "ToTensorConverter");
}

  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_13(mht_13_v, 392, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupported");

    return IsSupportedDataType(input.data_type) &&
           IsSupportedDataType(output.data_type) &&
           // Input is always SSBO/BHWC
           input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_layout == DataLayout::BHWC &&
           // -> SSBO/DHWC4
           output.object_type == ObjectType::OPENGL_SSBO &&
           output.data_layout == DataLayout::DHWC4;
  }

  absl::Status Init(const TensorObjectDef& input_def,
                    const TensorObjectDef& output_def) final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_14(mht_14_v, 407, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Init");

    shape_ = BHWC(output_def.dimensions.b, output_def.dimensions.h,
                  output_def.dimensions.w, output_def.dimensions.c);
    if (shape_.b != 1) {
      return absl::UnimplementedError(
          "FromTensorConverter: Batch size != 1 is not supported.");
    }

    return InitializeProgram(uint3(8, 4, 2), R"(
    layout(std430) buffer;
    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      float elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      vec4 elements[];
    } output_data;

    uniform ivec4 sizes;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes.x || gid.y >= sizes.y || gid.z >= sizes.w) {
        return;
      }
      vec4 v = vec4(0);
      int dst_channel = gid.z * 4;
      int index = (gid.y * sizes.x + gid.x) * sizes.z + dst_channel;
      for (int i = 0; i < 4; ++i, ++index, ++dst_channel) {
        if (dst_channel >= sizes.z) break;
        v[i] = input_data.elements[index];
      }
      output_data.elements[(gid.z * sizes.y + gid.y) * sizes.x + gid.x] = v;
    })");
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_15(mht_15_v, 449, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Convert");

    auto output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (!output || !output->id) {
      return absl::InvalidArgumentError("Missing output in converter");
    }
    auto input = absl::get_if<OpenGlBuffer>(&input_obj);
    if (!input || !input->id) {
      return absl::InvalidArgumentError("Missing input in converter");
    }
    if (input->id == output->id) {
      return absl::InvalidArgumentError("Can not execute inplace conversion");
    }
    GlBuffer input_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*input, &input_ssbo));
    GlBuffer output_ssbo;
    RETURN_IF_ERROR(WrapSSBO(*output, &output_ssbo));

    if (input_ssbo.bytes_size() != SizeInBytesBHWC(shape_)) {
      return absl::InvalidArgumentError(
          "ToTensorConverter: input data size does not match expected size.");
    }
    if (output_ssbo.bytes_size() != SizeInBytesDHWC4(shape_)) {
      return absl::InvalidArgumentError(
          "ToTensorConverter: output data size does not match expected size.");
    }
    auto d = DivideRoundUp(shape_.c, 4);
    RETURN_IF_ERROR(program_.SetParameter(
        {"sizes",
         int4(static_cast<int32_t>(shape_.w), static_cast<int32_t>(shape_.h),
              static_cast<int32_t>(shape_.c), static_cast<int32_t>(d))}));
    RETURN_IF_ERROR(input_ssbo.BindToIndex(0));
    RETURN_IF_ERROR(output_ssbo.BindToIndex(1));
    return Dispatch(uint3(shape_.w, shape_.h, d));
  }

  BHWC shape_;
};

// Copies data from one object of the same type and layout to another object.
class TrivialCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_16(mht_16_v, 493, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupported");

    return input.object_type == ObjectType::OPENGL_SSBO &&
           input.data_type == output.data_type &&
           input.object_type == output.object_type &&
           input.data_layout == output.data_layout;
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_17(mht_17_v, 504, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Convert");

    auto ssbo_input = absl::get_if<OpenGlBuffer>(&input_obj);
    auto ssbo_output = absl::get_if<OpenGlBuffer>(&output_obj);
    if (ssbo_input && ssbo_output) {
      return Copy(*ssbo_input, *ssbo_output);
    }
    return absl::InternalError("Unexpected object");
  }

  absl::Status Copy(OpenGlBuffer input, OpenGlBuffer output) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_18(mht_18_v, 516, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Copy");

    if (input.id == output.id) {
      return absl::OkStatus();
    }
    GlBuffer input_obj;
    RETURN_IF_ERROR(WrapSSBO(input, &input_obj));
    GlBuffer output_obj;
    RETURN_IF_ERROR(WrapSSBO(output, &output_obj));
    return CopyBuffer(input_obj, output_obj);
  }
};

// Copies data from/to CPU into a tensor.
class CpuCopier : public TensorObjectConverter {
 public:
  static bool IsSupported(const ObjectDef& input, const ObjectDef& output) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_19(mht_19_v, 534, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupported");

    return input.data_type == output.data_type &&
           input.data_layout == output.data_layout &&
           ((input.object_type == ObjectType::CPU_MEMORY &&
             output.object_type == ObjectType::OPENGL_SSBO) ||
            (output.object_type == ObjectType::CPU_MEMORY &&
             input.object_type == ObjectType::OPENGL_SSBO));
  }

  absl::Status Convert(const TensorObject& input_obj,
                       const TensorObject& output_obj) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_20(mht_20_v, 547, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "Convert");

    auto cpu_input = absl::get_if<CpuMemory>(&input_obj);
    auto cpu_output = absl::get_if<CpuMemory>(&output_obj);
    if (cpu_input) {
      auto ssbo_output = absl::get_if<OpenGlBuffer>(&output_obj);
      if (ssbo_output) {
        GlBuffer gl_buffer;
        RETURN_IF_ERROR(WrapSSBO(*ssbo_output, &gl_buffer));
        return gl_buffer.Write(
            absl::MakeConstSpan(static_cast<const uint8_t*>(cpu_input->data),
                                cpu_input->size_bytes));
      }
    } else if (cpu_output) {
      auto ssbo_input = absl::get_if<OpenGlBuffer>(&input_obj);
      if (ssbo_input) {
        GlBuffer gl_buffer;
        RETURN_IF_ERROR(WrapSSBO(*ssbo_input, &gl_buffer));
        return gl_buffer.Read(absl::MakeSpan(
            static_cast<uint8_t*>(cpu_output->data), cpu_output->size_bytes));
      }
    }
    return absl::InternalError("Unexpected object");
  }
};

class TensorConverterBuilderImpl : public TensorObjectConverterBuilder {
 public:
  explicit TensorConverterBuilderImpl(CommandQueue* command_queue)
      : command_queue_(command_queue) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_21(mht_21_v, 578, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "TensorConverterBuilderImpl");
}

  bool IsSupported(const TensorObjectDef& input,
                   const TensorObjectDef& output) const final {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_22(mht_22_v, 584, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "IsSupported");

    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    return input.dimensions == output.dimensions &&
           (TrivialCopier::IsSupported(input_def, output_def) ||
            CpuCopier::IsSupported(input_def, output_def) ||
            FromTensorConverter::IsSupported(input_def, output_def) ||
            ToTensorConverter::IsSupported(input_def, output_def));
  }

  absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) final {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSkernelsPSconverterDTcc mht_23(mht_23_v, 599, "", "./tensorflow/lite/delegates/gpu/gl/kernels/converter.cc", "MakeConverter");

    std::unique_ptr<OpenGlConverterImpl> impl;
    const auto& input_def = input.object_def;
    const auto& output_def = output.object_def;
    if (TrivialCopier::IsSupported(input_def, output_def)) {
      *converter = absl::make_unique<TrivialCopier>();
      return absl::OkStatus();
    }
    if (CpuCopier::IsSupported(input_def, output_def)) {
      *converter = absl::make_unique<CpuCopier>();
      return absl::OkStatus();
    }
    if (FromTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<FromTensorConverter>(command_queue_);
    } else if (ToTensorConverter::IsSupported(input_def, output_def)) {
      impl = absl::make_unique<ToTensorConverter>(command_queue_);
    } else {
      return absl::UnimplementedError("Unsupported conversion");
    }
    RETURN_IF_ERROR(impl->Init(input, output));
    *converter = std::move(impl);
    return absl::OkStatus();
  }

 private:
  CommandQueue* command_queue_;
};

}  // namespace

std::unique_ptr<TensorObjectConverterBuilder> NewConverterBuilder(
    CommandQueue* command_queue) {
  return absl::make_unique<TensorConverterBuilderImpl>(command_queue);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
