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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.h"

#include <algorithm>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

ShaderCodegen::ShaderCodegen(const CompilationOptions& options,
                             const GpuInfo& gpu_info)
    : options_(options), gpu_type_(gpu_info.vendor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.cc", "ShaderCodegen::ShaderCodegen");
}

absl::Status ShaderCodegen::Build(CompiledNodeAttributes attr,
                                  ShaderCode* shader_code) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.cc", "ShaderCodegen::Build");

  VariableAccessor variable_accessor(options_.inline_parameters,
                                     options_.vulkan_support);
  ObjectAccessor object_accessor(gpu_type_ == GpuVendor::kMali,
                                 options_.sampler_textures, &variable_accessor);

  const auto add_object = [&](const std::string& name, Object&& object) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.cc", "lambda");

    if (!object_accessor.AddObject(name, std::forward<Object>(object))) {
      return absl::AlreadyExistsError(absl::StrCat("Object \"", name, "\""));
    }
    return absl::OkStatus();
  };

  const auto add_uniform_parameter = [&](Variable&& variable) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSshader_codegenDTcc mht_3(mht_3_v, 229, "", "./tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.cc", "lambda");

    const std::string name = variable.name;
    const Variable& const_ref = variable;
    if (variable_accessor.IsEmptyVariableLength(const_ref)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Empty uniform vector value \"", name, "\""));
    }
    if (!variable_accessor.AddUniformParameter(std::move(variable))) {
      return absl::AlreadyExistsError(
          absl::StrCat("Uniform parameter \"", name, "\""));
    }
    return absl::OkStatus();
  };

  for (auto&& object : attr.code.objects) {
    RETURN_IF_ERROR(add_object(object.first, std::move(object.second)));
  }

  for (auto&& variable : attr.code.shared_variables) {
    const std::string name = variable.name;
    if (!variable_accessor.AddSharedVariable(std::move(variable))) {
      return absl::AlreadyExistsError(
          absl::StrCat("Shared variable \"", name, "\""));
    }
  }

  for (auto&& variable : attr.code.parameters) {
    RETURN_IF_ERROR(add_uniform_parameter(std::move(variable)));
  }

  int index = 0;
  for (auto&& input : attr.inputs) {
    RETURN_IF_ERROR(
        add_object(absl::StrCat("input_data_", index++), std::move(input)));
  }
  index = 0;
  for (auto&& output : attr.outputs) {
    RETURN_IF_ERROR(
        add_object(absl::StrCat("output_data_", index++), std::move(output)));
  }

  // TODO(akulik): workload params need to go away and be replaced with
  // output_data_0_w
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_x", static_cast<int32_t>(attr.code.workload.x)}));
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_y", static_cast<int32_t>(attr.code.workload.y)}));
  RETURN_IF_ERROR(add_uniform_parameter(
      {"workload_z", static_cast<int32_t>(attr.code.workload.z)}));

  // NOTE: If the shader has shared variables it will have to use barriers,
  //       which will conflict with a return at this stage.
  // Let the user deal with the geometry constraints.
  const bool has_shared_variables = !attr.code.shared_variables.empty();
  std::string main_source_code = has_shared_variables ? R"(
  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
)"
                                                      : R"(
  ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
  if (gid.x >= $workload_x$ || gid.y >= $workload_y$ || gid.z >= $workload_z$) {
    return;
  }
)";

  switch (attr.code.input) {
    case IOStructure::ONLY_DEFINITIONS:
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&main_source_code, "  highp vec4 value_", i,
                        " = vec4(0);\n");
      }
      break;
    case IOStructure::AUTO: {
      for (int i = 0; i < attr.inputs.size(); ++i) {
        absl::StrAppend(&main_source_code, "  highp vec4 value_", i,
                        " = $input_data_", i, "[gid.x, gid.y, gid.z]$;\n");
      }
      break;
    }
  }

  main_source_code.append(attr.code.source_code);

  if (attr.code.output == IOStructure::AUTO) {
    for (int i = 0; i < attr.outputs.size(); ++i) {
      absl::StrAppend(&main_source_code, "  $output_data_", i,
                      "[gid.x, gid.y, gid.z] = value_", i, "$;\n");
    }
  }

  // At this point main function is already generated. Now we need to process
  // object and variable accessors.

  // process objects first. Object accessor may introduce new uniform
  // parameters that need to be rewritten in the subsequent pass.
  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/true);
    preprocessor.AddRewrite(&object_accessor);
    RETURN_IF_ERROR(preprocessor.Rewrite(main_source_code, &main_source_code));
  }

  {
    TextPreprocessor preprocessor('$', /*keep_unknown_rewrites=*/false);
    preprocessor.AddRewrite(&variable_accessor);
    RETURN_IF_ERROR(preprocessor.Rewrite(main_source_code, &main_source_code));
  }

  if (options_.inline_parameters) {
    main_source_code = absl::StrCat(variable_accessor.GetConstDeclarations(),
                                    main_source_code);
  }

  // partial_source_code is only missing the following which is added later:
  // #version 310 es
  // layout(local_size_x = ..., local_size_y = ..., local_size_z = ...) in;
  const char* precision = options_.allow_precision_loss ? "mediump" : "highp";
  const std::string partial_source_code = absl::StrCat(
      "layout(std430) buffer;\n",                                 //
      "precision ", precision, " float;\n",                       //
      object_accessor.GetFunctionsDeclarations(), "\n",           //
      object_accessor.GetObjectDeclarations(), "\n",              //
      variable_accessor.GetUniformParameterDeclarations(), "\n",  //
      variable_accessor.GetSharedVariableDeclarations(), "\n",    //
      "void main() {\n",                                          //
      main_source_code,                                           //
      "}");
  *shader_code =
      ShaderCode(variable_accessor.GetUniformParameters(),
                 object_accessor.GetObjects(), attr.code.workload,
                 attr.code.workgroup, partial_source_code, attr.node_indices);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
