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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/api.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>  // NOLINT
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/delegates/gpu/gl/runtime.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "tensorflow/lite/delegates/gpu/gl/serialization.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

using ObjectsSizes = absl::flat_hash_map<ValueId, size_t>;

enum class InferenceContextState {
  NOT_STARTED,
  IN_PROGRESS,
};

class InferenceContextImpl : public InferenceContext {
 public:
  explicit InferenceContextImpl(std::unique_ptr<Runtime> runtime)
      : runtime_(std::move(runtime)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_0(mht_0_v, 228, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "InferenceContextImpl");
}

  absl::Status Execute() final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Execute");

    std::lock_guard<std::mutex> lock(guard_);
    if (state_ != InferenceContextState::NOT_STARTED) {
      return absl::FailedPreconditionError("InferenceContext is not reset");
    }
    state_ = InferenceContextState::IN_PROGRESS;
    return runtime_->Execute();
  }

  absl::Status Reset() final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_2(mht_2_v, 245, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Reset");

    std::lock_guard<std::mutex> lock(guard_);
    // TODO(akulik): should Reset not return Status?
    state_ = InferenceContextState::NOT_STARTED;
    return absl::OkStatus();
  }

  RuntimeStats stats() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_3(mht_3_v, 255, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "stats");
 return runtime_->stats(); }

 private:
  std::unique_ptr<Runtime> runtime_;

  mutable std::mutex guard_;
  InferenceContextState state_ = InferenceContextState::NOT_STARTED;
};

class InferenceContextWithBatchImpl : public InferenceContext {
 public:
  InferenceContextWithBatchImpl(const ObjectsSizes& sizes,
                                const ObjectManager* objects,
                                std::unique_ptr<ObjectManager> refs,
                                std::unique_ptr<Runtime> runtime)
      : sizes_(sizes),
        objects_(objects),
        refs_(std::move(refs)),
        runtime_(std::move(runtime)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_4(mht_4_v, 276, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "InferenceContextWithBatchImpl");
}

  absl::Status Execute() final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_5(mht_5_v, 281, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Execute");

    std::lock_guard<std::mutex> lock(guard_);
    if (state_ != InferenceContextState::NOT_STARTED) {
      return absl::FailedPreconditionError("InferenceContext is not reset");
    }
    state_ = InferenceContextState::IN_PROGRESS;

    // Calculate expected number of batches and check that all external objects
    // match that number.
    int num_batches = 0;
    for (const auto& s : sizes_) {
      const ValueId id = s.first;
      const size_t byte_size = s.second;

      auto buffer = objects_->FindBuffer(id);
      if (!buffer) continue;

      if (buffer->bytes_size() % byte_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Object ", id, " does not match expected byte size: ", byte_size));
      }

      const size_t b = buffer->bytes_size() / byte_size;
      if (num_batches == 0) {
        num_batches = b;
      } else if (num_batches != b) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Object ", id, " size does not match expected batch size: ", b,
            " vs ", num_batches));
      }
    }

    for (size_t b = 0; b < num_batches; ++b) {
      // slice external objects by batch.
      for (const auto& s : sizes_) {
        const ValueId id = s.first;
        const size_t byte_size = s.second;
        auto buffer = objects_->FindBuffer(id);
        if (buffer) {
          auto ref = refs_->FindBuffer(id);
          if (!ref) {
            return absl::InvalidArgumentError(
                absl::StrCat("Reference to ", id, " is not found"));
          }
          RETURN_IF_ERROR(buffer->MakeView(b * byte_size, byte_size, ref));
        }
      }
      RETURN_IF_ERROR(runtime_->Execute());
    }
    return absl::OkStatus();
  }

  absl::Status Reset() final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_6(mht_6_v, 336, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Reset");

    std::lock_guard<std::mutex> lock(guard_);
    state_ = InferenceContextState::NOT_STARTED;
    // TODO(akulik): should Reset not return Status?
    return absl::OkStatus();
  }

  RuntimeStats stats() const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_7(mht_7_v, 346, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "stats");
 return runtime_->stats(); }

 private:
  const ObjectsSizes sizes_;
  const ObjectManager* objects_;

  // view over external objects provided by a user.
  std::unique_ptr<ObjectManager> refs_;
  std::unique_ptr<Runtime> runtime_;

  mutable std::mutex guard_;
  InferenceContextState state_ = InferenceContextState::NOT_STARTED;
};

struct ProgramParameters {
  // A list of uniform parameters to be set.
  std::vector<Variable> parameters;

  // A list of objects to bind to opengl program.
  std::vector<Object> objects;

  uint3 workgroup_size;
  uint3 num_workgroups;

  size_t shader_idx;
};

std::string GetShaderHeader(uint3 localsize) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_8(mht_8_v, 376, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "GetShaderHeader");

  return absl::StrCat("#version 310 es\nlayout(local_size_x = ", localsize.x,
                      ", local_size_y = ", localsize.y,
                      ", local_size_z = ", localsize.z, ") in;\n");
}

class CompiledModelImpl
#ifndef TFLITE_GPU_BINARY_RELEASE
    : public CompiledModel,
      public DeserializationHandler {
#else
    : public CompiledModel {
#endif  // TFLITE_GPU_BINARY_RELEASE
 public:
  explicit CompiledModelImpl(const GpuInfo& gpu_info) : gpu_info_(gpu_info) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_9(mht_9_v, 393, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "CompiledModelImpl");
}

  // Called while compiling shaders from scratch
  absl::Status Add(const WorkgroupsCalculator& workgroup_calculator,
                   ShaderCode code) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_10(mht_10_v, 400, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Add");

    // Calculate workgroup size.
    uint3 workgroup_size = workgroup_calculator.Calculate(code);
    uint3 num_workgroups = DivideRoundUp(code.workload, workgroup_size);

    for (const auto& object : code.objects) {
      if (IsRef(object)) {
        object_sizes_[GetRef(object)] = ByteSizeOf(object);
      }
    }

    // Store full shader and compile it if necessary.
    size_t shader_idx;
    RETURN_IF_ERROR(
        AddFullShader(code.source_code, workgroup_size, &shader_idx));
    programs_.push_back({
        std::move(code.parameters),
        std::move(code.objects),
        workgroup_size,
        num_workgroups,
        shader_idx,
    });
    return absl::OkStatus();
  }

  // Store full shader and compile it if necessary.
  // Returns full_shader_index
  absl::Status AddFullShader(const std::string& partial_shader,
                             const uint3& workgroup_size, size_t* size) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("partial_shader: \"" + partial_shader + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_11(mht_11_v, 432, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "AddFullShader");

    std::string shader_src = GetShaderHeader(workgroup_size) + partial_shader;
    auto it = shader_to_index_.find(shader_src);
    if (it == shader_to_index_.end()) {
      GlShader shader;
      RETURN_IF_ERROR(
          GlShader::CompileShader(GL_COMPUTE_SHADER, shader_src, &shader));
      shaders_.push_back(std::move(shader));
      shader_to_index_.insert({shader_src, shader_to_index_.size()});
      *size = shader_to_index_.size() - 1;
    } else {
      *size = it->second;
    }
    return absl::OkStatus();
  }

  absl::Status NewRun(
      const RuntimeOptions& options, const ObjectManager* objects,
      CommandQueue* command_queue,
      std::unique_ptr<InferenceContext>* inference_context) const final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_12(mht_12_v, 454, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "NewRun");

    std::unique_ptr<ObjectManager> refs;
    if (dynamic_batch_) {
      // Runtime is using objects from refs that will point to provided objects.
      // At this point just create 0 batch slice references.
      refs = absl::make_unique<ObjectManager>();
      for (const auto& s : object_sizes_) {
        auto buffer = objects->FindBuffer(s.first);
        if (!buffer) continue;
        GlBuffer ref;
        RETURN_IF_ERROR(buffer->MakeView(0, s.second, &ref));
        RETURN_IF_ERROR(refs->RegisterBuffer(s.first, std::move(ref)));
      }
    }
    auto runtime = absl::make_unique<Runtime>(options, gpu_info_, command_queue,
                                              refs ? refs.get() : objects);
    for (auto& program : programs_) {
      RETURN_IF_ERROR(runtime->AddProgram(shaders_[program.shader_idx],
                                          program.parameters, program.objects,
                                          program.num_workgroups));
    }
    RETURN_IF_ERROR(runtime->PrepareForExecution());
    if (dynamic_batch_) {
      *inference_context = absl::make_unique<InferenceContextWithBatchImpl>(
          object_sizes_, objects, std::move(refs), std::move(runtime));
    } else {
      *inference_context =
          absl::make_unique<InferenceContextImpl>(std::move(runtime));
    }
    return absl::OkStatus();
  }

#ifndef TFLITE_GPU_BINARY_RELEASE
  // Called on deserialization
  absl::Status OnProgram(const std::vector<Variable>& parameters,
                         const std::vector<Object>& objects,
                         const uint3& workgroup_size,
                         const uint3& num_workgroups,
                         size_t partial_shader_index) final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_13(mht_13_v, 495, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "OnProgram");

    for (auto& object : objects) {
      if (IsRef(object)) {
        object_sizes_[GetRef(object)] = ByteSizeOf(object);
      }
    }

    size_t shader_idx;
    RETURN_IF_ERROR(AddFullShader(partial_shaders_[partial_shader_index],
                                  workgroup_size, &shader_idx));
    programs_.push_back({
        parameters,
        objects,
        workgroup_size,
        num_workgroups,
        shader_idx,
    });
    return absl::OkStatus();
  }

  absl::Status Serialize(
      std::vector<uint8_t>* serialized_compiled_model) const final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_14(mht_14_v, 519, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Serialize");

    SerializedCompiledModelBuilder builder;

    // sort shaders first. They need to be serialized in order.
    std::vector<std::string> full_shaders(shaders_.size());
    for (const auto& shader : shader_to_index_) {
      full_shaders[shader.second] = shader.first;
    }

    absl::flat_hash_map<std::string, size_t> partial_shader_to_index;
    std::vector<std::string> partial_shaders;
    for (const auto& program : programs_) {
      // Remove a header from a shader.
      std::string shader_without_header = full_shaders[program.shader_idx];
      shader_without_header.erase(0, shader_without_header.find("in;") + 3);

      // Insert shader into partial shaders array.
      auto it = partial_shader_to_index.find(shader_without_header);
      size_t shader_idx;
      if (it == partial_shader_to_index.end()) {
        shader_idx = partial_shaders.size();
        partial_shaders.push_back(shader_without_header);
        builder.AddShader(shader_without_header);
        partial_shader_to_index.insert({shader_without_header, shader_idx});
      } else {
        shader_idx = it->second;
      }
      builder.AddProgram(program.parameters, program.objects,
                         program.workgroup_size, program.num_workgroups,
                         shader_idx);
    }
    CompiledModelOptions options;
    options.dynamic_batch = dynamic_batch_;
    auto data = builder.Finalize(options);
    serialized_compiled_model->insert(serialized_compiled_model->end(),
                                      data.begin(), data.end());
    return absl::OkStatus();
  }

  absl::Status OnShader(absl::Span<const char> shader_src) final {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_15(mht_15_v, 561, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "OnShader");

    std::string source(shader_src.data(), shader_src.size());
    partial_shaders_.push_back(source);
    return absl::OkStatus();
  }

  void OnOptions(const CompiledModelOptions& options) final {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_16(mht_16_v, 570, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "OnOptions");

    dynamic_batch_ = options.dynamic_batch;
  }
#endif  // TFLITE_GPU_BINARY_RELEASE

  CompilerStats stats() const final { return stats_; }

  void set_dynamic_batch(bool dynamic_batch) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_17(mht_17_v, 580, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "set_dynamic_batch");
 dynamic_batch_ = dynamic_batch; }

 private:
  const GpuInfo gpu_info_;
  bool dynamic_batch_ = false;

  std::vector<std::string> partial_shaders_;
  std::vector<GlShader> shaders_;

  // Shaders are serialized in order of their indices.
  absl::flat_hash_map<std::string, size_t> shader_to_index_;
  std::deque<ProgramParameters> programs_;
  absl::flat_hash_map<ValueId, size_t> object_sizes_;
  CompilerStats stats_;
};
}  // namespace

absl::Status Compile(const CompilationOptions& options,
                     const GraphFloat32& model,
                     const std::unordered_set<int>& tflite_graph_io,  // NOLINT
                     const NodeShader& node_shader,
                     const WorkgroupsCalculator& workgroup_calculator,
                     std::unique_ptr<CompiledModel>* compiled_model) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_18(mht_18_v, 605, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "Compile");

  RETURN_IF_ERROR(CheckBatchSizeForAllValues(model));
  GpuInfo gpu_info;
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
  if (!gpu_info.IsApiOpenGl31OrAbove()) {
    return absl::InternalError(
        "OpenGL ES 3.1 or above is required to use OpenGL inference.");
  }
  auto compiled_model_impl = absl::make_unique<CompiledModelImpl>(gpu_info);
  compiled_model_impl->set_dynamic_batch(options.dynamic_batch);
  auto compiler = NewCompiler(&node_shader, &gpu_info, options);
  RETURN_IF_ERROR(compiler->Compile(
      model, tflite_graph_io, [&](ShaderCode code) -> absl::Status {
        return compiled_model_impl->Add(workgroup_calculator, std::move(code));
      }));
  *compiled_model = std::move(compiled_model_impl);
  return absl::OkStatus();
}

#ifndef TFLITE_GPU_BINARY_RELEASE
absl::Status ReadSerializedModel(
    const std::vector<uint8_t>& serialized_model,
    std::unique_ptr<CompiledModel>* compiled_model) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPSapiDTcc mht_19(mht_19_v, 630, "", "./tensorflow/lite/delegates/gpu/gl/api.cc", "ReadSerializedModel");

  GpuInfo gpu_info;
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info));
  if (!gpu_info.IsApiOpenGl31OrAbove()) {
    return absl::InternalError(
        "OpenGL ES 3.1 or above is required to use OpenGL inference.");
  }
  auto compiled_model_impl = absl::make_unique<CompiledModelImpl>(gpu_info);
  RETURN_IF_ERROR(DeserializeCompiledModel(
      absl::MakeConstSpan(serialized_model), compiled_model_impl.get()));
  *compiled_model = std::move(compiled_model_impl);
  return absl::OkStatus();
}
#endif  // TFLITE_GPU_BINARY_RELEASE

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
