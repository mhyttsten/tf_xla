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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_executable.h"

#include "absl/cleanup/cleanup.h"
#include "tensorflow/core/tpu/tpu_executor_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"

namespace ApiConverter {
static SE_ExecutableRunOptions ToC(
    const xla::ServiceExecutableRunOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_0(mht_0_v, 195, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "ToC");

  SE_ExecutableRunOptions se_options;
  se_options.allocator = ApiConverter::ToC(options.run_options().allocator());
  se_options.device_ordinal = options.run_options().device_ordinal();
  if (options.run_options().host_to_device_stream() != nullptr) {
    se_options.host_to_device_stream =
        static_cast<tensorflow::tpu::TpuStream*>(
            options.run_options().host_to_device_stream()->implementation())
            ->se_stream();
  } else {
    se_options.host_to_device_stream = nullptr;
  }

  if (options.run_options().device_assignment() != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    options.run_options()
        .device_assignment()
        ->Serialize(&dev_assign_proto)
        .IgnoreError();
    se_options.device_assignment =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
  } else {
    se_options.device_assignment.bytes = nullptr;
    se_options.device_assignment.size = 0;
  }

  se_options.rng_seed = options.run_options().rng_seed();
  se_options.run_id = options.run_options().run_id().ToInt();
  se_options.launch_id = options.run_options().launch_id();

  CHECK_EQ(options.run_options().then_execute_function(), nullptr)
      << "ThenExecuteFunction not supported by this platform.";

  auto impl =
      const_cast<stream_executor::Stream*>(options.stream())->implementation();
  se_options.stream =
      static_cast<tensorflow::tpu::TpuStream*>(impl)->se_stream();
  return se_options;
}
}  // namespace ApiConverter

namespace xla {

using ::tensorflow::tpu::ExecutorApiFn;

TpuExecutable::~TpuExecutable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_1(mht_1_v, 243, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "TpuExecutable::~TpuExecutable");

  ExecutorApiFn()->TpuExecutable_FreeFn(se_executable_);
}

StatusOr<ExecutionOutput> TpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_2(mht_2_v, 253, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "TpuExecutable::ExecuteAsyncOnStream");

  SE_ExecutableRunOptions se_run_options = ApiConverter::ToC(*run_options);
  SE_ExecutionInput** se_args = new SE_ExecutionInput*[arguments.size()];
  for (int i = 0; i < arguments.size(); ++i) {
    auto& arg = arguments[i];
    se_args[i] = new SE_ExecutionInput;

    ApiConverter::ToC(arg.shape(), &se_args[i]->shape_tree.shape);
    auto* arg_buffers = arg.MutableBuffers();
    absl::InlinedVector<SE_MaybeOwningDeviceMemory, 2> se_buffers;
    for (auto& pair : *arg_buffers) {
      bool aliased = arg.unowned_indices().count(pair.first) > 0;
      se_buffers.push_back(ApiConverter::ToC(pair.second, aliased));
    }
    se_args[i]->shape_tree.buffers =
        new SE_MaybeOwningDeviceMemory[se_buffers.size()];
    for (int j = 0; j < se_buffers.size(); ++j) {
      se_args[i]->shape_tree.buffers[j] = se_buffers[j];
    }

    ApiConverter::ToC(arg.shape(), &se_args[i]->dynamic_shape);
    const auto& unowned_indices = arg.unowned_indices();
    se_args[i]->unowned_indices_size = unowned_indices.size();
    se_args[i]->unowned_indices = new XLA_ShapeIndex[unowned_indices.size()];
    int j = 0;
    for (auto& idx : unowned_indices) {
      se_args[i]->unowned_indices[j] = ApiConverter::ToC(idx);
      ++j;
    }
  }
  SE_ExecutionOutput se_execution_output;
  StatusHelper status;
  ExecutorApiFn()->TpuExecutable_ExecuteAsyncOnStreamFn(
      se_executable_, &se_run_options, se_args, arguments.size(), nullptr,
      &se_execution_output, status.c_status);

  if (se_run_options.device_assignment.bytes != nullptr) {
    stream_executor::tpu::SerializedProto_Free(
        se_run_options.device_assignment);
  }
  for (int i = 0; i < arguments.size(); ++i) {
    ApiConverter::Destroy(&se_args[i]->shape_tree.shape);
    ApiConverter::Destroy(&se_args[i]->dynamic_shape);
    delete[] se_args[i]->unowned_indices;
    delete[] se_args[i]->shape_tree.buffers;
    delete se_args[i];
  }
  delete[] se_args;

  if (!status.ok()) {
    return status.status();
  }

  xla::ScopedShapedBuffer result(
      ApiConverter::FromC(&se_execution_output.result),
      run_options->stream()->parent()->GetAllocator());
  ApiConverter::Destroy(&se_execution_output.result);

  ExecutionOutput output(std::move(result));
  for (int i = 0; i < se_execution_output.aliased_indices_size; ++i) {
    output.AddAliasedIndex(
        ApiConverter::FromC(&se_execution_output.aliased_indices[i]));
  }
  ExecutorApiFn()->TpuExecutable_FreeXlaShapeIndexArrayFn(
      se_execution_output.aliased_indices);

  for (int i = 0; i < se_execution_output.to_be_released_size; ++i) {
    output.AddToBeReleased(
        ApiConverter::FromC(&se_execution_output.to_be_released[i],
                            run_options->stream()->parent()->GetAllocator())
            .Release()
            .value());
  }
  ExecutorApiFn()->TpuExecutable_FreeMaybeOwningDeviceMemoryArrayFn(
      se_execution_output.to_be_released);

  return output;
}

absl::string_view TpuExecutable::fingerprint() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_3(mht_3_v, 335, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "TpuExecutable::fingerprint");

  const char* data;
  size_t size;
  ExecutorApiFn()->TpuExecutable_FingerprintFn(se_executable_, &data, &size);
  return absl::string_view(data, size);
}

StatusOr<std::string> TpuExecutable::Serialize() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_4(mht_4_v, 345, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "TpuExecutable::Serialize");

  SE_ExecutableSerializationHandle* handle = nullptr;
  auto cleanup = absl::MakeCleanup([&handle]() {
    ExecutorApiFn()->TpuExecutableSerialize_FreeHandleFn(handle);
  });
  StatusHelper status;
  ExecutorApiFn()->TpuExecutable_SerializeFn(se_executable_, &handle,
                                             status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  size_t size = ExecutorApiFn()->TpuExecutableSerialize_GetByteSizeFn(handle);
  CHECK_GT(size, 0);
  std::string serialized;
  // NOTE(skyewm): this initializes serialized. If this ever becomes a
  // bottleneck, we could change the return type to std::vector<uint8_t> or
  // similar.
  serialized.resize(size);
  ExecutorApiFn()->TpuExecutableSerialize_WriteToArrayFn(
      handle, size,
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(serialized.data())),
      status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  return serialized;
}

StatusOr<std::unique_ptr<TpuExecutable>> TpuExecutable::Deserialize(
    absl::string_view serialized) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("serialized: \"" + std::string(serialized.data(), serialized.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_executableDTcc mht_5(mht_5_v, 378, "", "./tensorflow/stream_executor/tpu/tpu_executable.cc", "TpuExecutable::Deserialize");

  SE_Executable* se_executable;
  StatusHelper status;
  ExecutorApiFn()->TpuExecutable_DeserializeFn(
      serialized.size(), reinterpret_cast<const uint8_t*>(serialized.data()),
      &se_executable, status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  XLA_HloModule c_module =
      ExecutorApiFn()->TpuExecutable_HloModuleFn(se_executable);
  auto cleanup_c_module =
      absl::MakeCleanup([&c_module]() { ApiConverter::Destroy(&c_module); });
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ApiConverter::FromC(c_module));
  return absl::make_unique<TpuExecutable>(se_executable, std::move(hlo_module));
}

}  // namespace xla
