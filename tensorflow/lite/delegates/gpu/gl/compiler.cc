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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inline.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_codegen.h"
#include "tensorflow/lite/delegates/gpu/gl/float16_conversions.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct ExceedSizeChecker {
  bool operator()(uint32_t v) const { return v > max_size.x; }

  bool operator()(const uint2& v) const {
    return v.x > max_size.x || v.y > max_size.y;
  }

  bool operator()(const uint3& v) const {
    return v.x > max_size.x || v.y > max_size.y || v.z > max_z_size;
  }

  int2 max_size;
  int max_z_size;
};

// Returns true if any size variable exceeds the given limit
bool ExceedsMaxSize(const Object& object, const GpuInfo& gpu_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_0(mht_0_v, 230, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "ExceedsMaxSize");

  ExceedSizeChecker size_checker;
  size_checker.max_size =
      int2(gpu_info.GetMaxImage2DWidth(), gpu_info.GetMaxImage2DHeight());
  size_checker.max_z_size = gpu_info.GetMaxImage2DArrayLayers();
  return absl::visit(size_checker, object.size);
}

ObjectType ChooseFastestObjectType(const GpuInfo& gpu_info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_1(mht_1_v, 241, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "ChooseFastestObjectType");

  return gpu_info.IsAdreno() ? ObjectType::TEXTURE : ObjectType::BUFFER;
}

ObjectType ChooseFastestRefObjectType(const GpuInfo& gpu_info,
                                      const CompilationOptions& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_2(mht_2_v, 249, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "ChooseFastestRefObjectType");

  if (!gpu_info.IsAdreno()) {
    return ObjectType::BUFFER;
  }
  if (gpu_info.adreno_info.adreno_gpu == AdrenoGpu::kAdreno630) {
    return ObjectType::TEXTURE;
  } else {
    return options.allow_precision_loss ? ObjectType::TEXTURE
                                        : ObjectType::BUFFER;
  }
}

// Compiler executes the following steps:
//   1. Runs NodeShader for every node in the input graph.
//   2. Creates a compiled graph that mirrors the input graph and keeps
//      GeneratedCode in operation's attributes.
//   3. Fuses nodes in the compiled graph.
//   4. Generates the full shader code using the nodes in the compiled graph.
class CompilerImpl : public Compiler {
 public:
  // We use const GpuInfo* because it doesn't let you assign temporary object
  CompilerImpl(const NodeShader* node_shader, const GpuInfo* gpu_info,
               const CompilationOptions& options)
      : node_shader_(*node_shader), gpu_info_(*gpu_info), options_(options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_3(mht_3_v, 275, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "CompilerImpl");

    if (options_.preferred_obj_type == ObjectType::UNKNOWN) {
      options_.preferred_obj_type = ChooseFastestObjectType(*gpu_info);
    }
    if (options_.ref_obj_type == ObjectType::UNKNOWN) {
      options_.ref_obj_type = ChooseFastestRefObjectType(*gpu_info, options);
    }
  }

  absl::Status Compile(
      const GraphFloat32& graph,
      const std::unordered_set<int>& tflite_graph_io,  // NOLINT
      const ShaderCodeCallback& callback) final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_4(mht_4_v, 290, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "Compile");

    // It is important to have ids in a compiled graph identical to the given
    // graph.
    RETURN_IF_ERROR(graph.MakeExactCopy(&compiled_graph_));

    // Clear out batch dimension for dynamic batch support.
    if (options_.dynamic_batch) {
      for (auto value : compiled_graph_.values()) {
        value->tensor.shape.b = 1;
      }
    }

    // Generate a shader for a node and all input/output objects.
    for (auto node : compiled_graph_.nodes()) {
      CompiledNodeAttributes attr;
      attr.node_indices.push_back(node->id);
      NodeShader::GenerationContext ctx = {&gpu_info_, options_,
                                           node->operation.type,
                                           node->operation.attributes};
      for (const auto& tensor : graph.FindInputs(node->id)) {
        const auto& shape = tensor->tensor.shape;
        ctx.input_shapes.push_back({shape.b, shape.h, shape.w, shape.c});
      }
      for (const auto& tensor : graph.FindOutputs(node->id)) {
        const auto& shape = tensor->tensor.shape;
        ctx.output_shapes.push_back({shape.b, shape.h, shape.w, shape.c});
      }
      RETURN_IF_ERROR(node_shader_.GenerateCode(ctx, &attr.code));
      node->operation.attributes = std::move(attr);
    }

    ModelTransformer transformer(&compiled_graph_);
    if (options_.fuse_operations) {
      FuseAutoOutputWithInline fuse_inline;
      if (!transformer.Apply("fuse_auto_with_inline", &fuse_inline)) {
        return absl::InternalError("fuse_auto_with_inline failed");
      }
      FuseInplaceUpdate fuse_inplace;
      if (!transformer.Apply("fuse_inplace_update", &fuse_inplace)) {
        return absl::InternalError("fuse_inplace failed");
      }
      if (options_.auto_input_fusion) {
        FuseAutoInput fuse_auto_input;
        if (!transformer.Apply("fuse_auto_input", &fuse_auto_input)) {
          return absl::InternalError("fuse_auto_input failed");
        }
      }
    }
    RemoveUnusedInplaceUpdates remove_inplace_updates;
    if (!transformer.Apply("remove_inplace_updates", &remove_inplace_updates)) {
      return absl::InternalError("remove_inplace_updates failed");
    }

    // Prepare internal objects.
    absl::flat_hash_map<ValueId, Object> objects;
    for (auto value : compiled_graph_.values()) {
      Object object = MakePHWC4Ref(value->id, value->tensor.shape);
      object.data_type = value->tensor.type;
      // External references may not be upgraded to f16 nor be represented as
      // textures.
      const bool is_external =
          graph.IsGraphInput(value->id) || graph.IsGraphOutput(value->id) ||
          tflite_graph_io.find(value->tensor.ref) != tflite_graph_io.end();
      if (is_external) {
        object.object_type = ObjectType::BUFFER;
      } else if (options_.allow_precision_loss) {
        MaybeConvertToFloat16(&object);
      }
      objects[value->id] = std::move(object);
    }

    // Prepare readonly objects and check whether object types are supported.
    for (auto node : compiled_graph_.nodes()) {
      auto& attr =
          absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);

      // Set workload explicitly.
      if (attr.code.workload == uint3()) {
        auto outputs = compiled_graph_.FindOutputs(node->id);
        auto shape = outputs[0]->tensor.shape;
        for (auto output : outputs) {
          if (shape != output->tensor.shape) {
            return absl::FailedPreconditionError(
                "Workload uint3() requires all output sizes to match");
          }
        }
        attr.code.workload = uint3(shape.w, shape.h, DivideRoundUp(shape.c, 4));
      }

      int num_textures = 0;
      // Counts number of used textures and chooses ObjectType for an object.
      auto set_object_type = [&](Object* object) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_5(mht_5_v, 384, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "lambda");

        if (object->object_type == ObjectType::BUFFER) {
          // Don't change from buffer once it is set.
          return;
        }
        bool is_ref = IsRef(*object);
        if (num_textures < gpu_info_.GetMaxImageArguments() &&
            !ExceedsMaxSize(*object, gpu_info_) &&
            (object->object_type == ObjectType::TEXTURE ||
             (is_ref && options_.ref_obj_type == ObjectType::TEXTURE) ||
             (!is_ref && options_.preferred_obj_type == ObjectType::TEXTURE))) {
          object->object_type = ObjectType::TEXTURE;
          num_textures++;
        } else {
          object->object_type = ObjectType::BUFFER;
        }
      };

      for (auto& object : attr.code.objects) {
        // Downgrade readonly objects to F16 is requested.
        if (options_.allow_precision_loss) {
          MaybeConvertToFloat16(&object.second);
        }
        set_object_type(&object.second);
      }

      for (auto ref : compiled_graph_.FindInputs(node->id)) {
        set_object_type(&objects[ref->id]);
      }
      for (auto ref : compiled_graph_.FindOutputs(node->id)) {
        set_object_type(&objects[ref->id]);
      }
    }

    // Generate shaders from the transformed graph.
    ShaderCodegen codegen(options_, gpu_info_);
    for (auto node : compiled_graph_.nodes()) {
      auto& attr =
          absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
      if (attr.code.source_code.empty()) {
        // noop. Skip this node.
        continue;
      }

      // Declare inputs and outputs explicitly.
      for (auto ref : compiled_graph_.FindInputs(node->id)) {
        auto object = objects[ref->id];
        object.access = AccessType::READ;
        attr.inputs.push_back(object);
      }
      for (auto ref : compiled_graph_.FindOutputs(node->id)) {
        auto object = objects[ref->id];
        object.access = AccessType::WRITE;
        attr.outputs.push_back(object);
      }

      // Allocate bindings. Textures must be bound first.
      uint32_t binding = 0;
      auto set_binding = [&](ObjectType type, Object& object) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerDTcc mht_6(mht_6_v, 445, "", "./tensorflow/lite/delegates/gpu/gl/compiler.cc", "lambda");

        if (object.object_type == type) {
          object.binding = binding++;
        }
      };
      for (auto& object : attr.inputs) {
        set_binding(ObjectType::TEXTURE, object);
      }
      for (auto& object : attr.outputs) {
        set_binding(ObjectType::TEXTURE, object);
      }
      for (auto& object : attr.code.objects) {
        set_binding(ObjectType::TEXTURE, object.second);
      }
      for (auto& object : attr.inputs) {
        set_binding(ObjectType::BUFFER, object);
      }
      for (auto& object : attr.outputs) {
        set_binding(ObjectType::BUFFER, object);
      }
      for (auto& object : attr.code.objects) {
        set_binding(ObjectType::BUFFER, object.second);
      }

      // Generate source code.
      ShaderCode shader_code;
      RETURN_IF_ERROR(codegen.Build(std::move(attr), &shader_code));
      RETURN_IF_ERROR(callback(std::move(shader_code)));
    }
    return absl::OkStatus();
  }

 private:
  const NodeShader& node_shader_;
  const GpuInfo& gpu_info_;
  CompilationOptions options_;
  GraphFloat32 compiled_graph_;
};

}  // namespace

std::unique_ptr<Compiler> NewCompiler(const NodeShader* node_shader,
                                      const GpuInfo* gpu_info,
                                      const CompilationOptions& options) {
  return absl::make_unique<CompilerImpl>(node_shader, gpu_info, options);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
