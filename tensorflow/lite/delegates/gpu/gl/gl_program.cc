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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"

#include <string>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

absl::Status CreateNewProgramId(GLuint* program_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "CreateNewProgramId");

  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glCreateProgram, program_id));
  if (!*program_id) {
    return absl::UnknownError("Can't create opengl program: 0 program_id");
  }
  return absl::OkStatus();
}

absl::Status CheckProgramLinked(GLuint program_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "CheckProgramLinked");

  GLint linked;
  glGetProgramiv(program_id, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
    return absl::OkStatus();
  }
  GLint info_size;
  glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &info_size);
  std::string errors;
  errors.resize(info_size + 1 /* plus \0 */);
  glGetProgramInfoLog(program_id, info_size + 1, nullptr, &errors[0]);
  // TODO(akulik): use glValidateProgram to gather more info.
  return absl::UnavailableError("Program is not properly linked: " + errors);
}

struct ParameterSetter {
  absl::Status operator()(int value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform1i, program_id, uniform_id,
                              value);
  }

  absl::Status operator()(const int2& value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform2i, program_id, uniform_id,
                              value.x, value.y);
  }

  absl::Status operator()(const int4& value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform4i, program_id, uniform_id,
                              value.x, value.y, value.z, value.w);
  }

  absl::Status operator()(const std::vector<int2>& value) {
    std::vector<GLint> ints(value.size() * 2, 0);
    for (int i = 0; i < value.size(); ++i) {
      ints[i * 2] = value[i].x;
      ints[i * 2 + 1] = value[i].y;
    }
    return TFLITE_GPU_CALL_GL(glProgramUniform2iv, program_id, uniform_id,
                              ints.size(), ints.data());
  }

  absl::Status operator()(unsigned int value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform1ui, program_id, uniform_id,
                              value);
  }

  absl::Status operator()(const uint4& value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform4ui, program_id, uniform_id,
                              value.x, value.y, value.z, value.w);
  }

  absl::Status operator()(float value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform1f, program_id, uniform_id,
                              value);
  }

  absl::Status operator()(const float2& value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform2f, program_id, uniform_id,
                              value.x, value.y);
  }

  absl::Status operator()(const float4& value) {
    return TFLITE_GPU_CALL_GL(glProgramUniform4f, program_id, uniform_id,
                              value.x, value.y, value.z, value.w);
  }

  absl::Status operator()(const std::vector<float4>& value) {
    std::vector<GLfloat> floats(value.size() * 4, 0);
    for (int i = 0; i < value.size(); ++i) {
      floats[i * 4] = value[i].x;
      floats[i * 4 + 1] = value[i].y;
      floats[i * 4 + 2] = value[i].z;
      floats[i * 4 + 3] = value[i].w;
    }
    return TFLITE_GPU_CALL_GL(glProgramUniform4fv, program_id, uniform_id,
                              floats.size(), floats.data());
  }

  const GLuint program_id;
  const GLint uniform_id;
};

}  // namespace

absl::Status GlProgram::CreateWithShader(const GlShader& shader,
                                         GlProgram* gl_program) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_2(mht_2_v, 300, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::CreateWithShader");

  GLuint program_id;
  RETURN_IF_ERROR(CreateNewProgramId(&program_id));

  // program_id needs to be properly deleted if there will be an error, hense
  // wrap program_id into Program.
  GlProgram program(program_id);

  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_GL(glAttachShader, program.id(), shader.id()));
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glLinkProgram, program.id()));
  RETURN_IF_ERROR(CheckProgramLinked(program.id()));

  *gl_program = std::move(program);
  return absl::OkStatus();
}

absl::Status GlProgram::CreateWithBinaryShader(const BinaryShader& shader,
                                               GlProgram* gl_program) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_3(mht_3_v, 321, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::CreateWithBinaryShader");

  GLuint program_id;
  RETURN_IF_ERROR(CreateNewProgramId(&program_id));

  // program_id needs to be properly deleted if there will be an error, hense
  // wrap program_id into Program.
  GlProgram program(program_id);

  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glProgramBinary, program.id(),
                                     shader.format(), shader.binary().data(),
                                     shader.binary().size()));
  RETURN_IF_ERROR(CheckProgramLinked(program.id()));

  *gl_program = std::move(program);
  return absl::OkStatus();
}

absl::Status GlProgram::GetBinary(BinaryShader* binary_shader) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_4(mht_4_v, 341, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::GetBinary");

  GLint size = 0;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_GL(glGetProgramiv, id_, GL_PROGRAM_BINARY_LENGTH, &size));
  if (!size) {
    return absl::InternalError("Getting binary size failed.");
  }
  // TODO(akulik): call
  // glProgramParameteri(id_, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE)
  // before linking a program to increase chances of retrieving a binary.
  std::vector<uint8_t> binary(size);
  GLsizei returned_size;
  GLenum format;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetProgramBinary, id_, size,
                                     &returned_size, &format,
                                     reinterpret_cast<void*>(&binary[0])));
  if (size != returned_size) {
    return absl::InternalError("Getting binary is failed.");
  }
  *binary_shader = BinaryShader(format, std::move(binary));
  return absl::OkStatus();
}

GlProgram::GlProgram(GlProgram&& program) : id_(program.id_) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_5(mht_5_v, 367, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::GlProgram");

  program.id_ = 0;
}

void GlProgram::Invalidate() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_6(mht_6_v, 374, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::Invalidate");

  if (id_) {
    glDeleteProgram(id_);
    id_ = 0;
  }
}

GlProgram& GlProgram::operator=(GlProgram&& program) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_7(mht_7_v, 384, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "=");

  if (this != &program) {
    Invalidate();
    std::swap(id_, program.id_);
  }
  return *this;
}

GlProgram::~GlProgram() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_8(mht_8_v, 395, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::~GlProgram");
 Invalidate(); }

absl::Status GlProgram::SetParameter(const Variable& param) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_9(mht_9_v, 400, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::SetParameter");

  GLint uniform_location;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetUniformLocation, &uniform_location,
                                     id_, param.name.c_str()));
  return absl::visit(ParameterSetter{id_, uniform_location}, param.value);
}

absl::Status GlProgram::Dispatch(const uint3& workgroups) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSgl_programDTcc mht_10(mht_10_v, 410, "", "./tensorflow/lite/delegates/gpu/gl/gl_program.cc", "GlProgram::Dispatch");

  if (workgroups.x == 0 || workgroups.y == 0 || workgroups.z == 0) {
    return absl::InvalidArgumentError("Invalid workgroups");
  }
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glUseProgram, id_));
  return TFLITE_GPU_CALL_GL(glDispatchCompute, workgroups.x, workgroups.y,
                            workgroups.z);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
