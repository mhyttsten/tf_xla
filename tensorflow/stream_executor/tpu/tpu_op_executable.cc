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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc() {
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

#include "tensorflow/stream_executor/tpu/tpu_op_executable.h"

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {

TpuOpExecutable::TpuOpExecutable(const XLA_TpuProgram* core_program,
                                 std::unique_ptr<xla::HloModule> hlo_module,
                                 HostCommandHandler host_command_handler)
    : TpuExecutableInterface(std::move(hlo_module)),
      core_program_(core_program),
      host_command_handler_(std::move(host_command_handler)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/tpu/tpu_op_executable.cc", "TpuOpExecutable::TpuOpExecutable");
}

Status TpuOpExecutable::LoadProgramAndEnqueueToStream(
    const xla::ServiceExecutableRunOptions& run_options,
    absl::Span<const se::DeviceMemoryBase> arguments,
    se::DeviceMemoryBase result,
    absl::optional<se::DeviceMemoryBase> cross_program_prefetch_addr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc mht_1(mht_1_v, 213, "", "./tensorflow/stream_executor/tpu/tpu_op_executable.cc", "TpuOpExecutable::LoadProgramAndEnqueueToStream");

  SE_DeviceMemoryBase* arguments_bases = nullptr;
  if (!arguments.empty()) {
    arguments_bases = new SE_DeviceMemoryBase[arguments.size()];
    for (int i = 0; i < arguments.size(); i++) {
      arguments_bases[i] =
          SE_DeviceMemoryBase{const_cast<void*>(arguments[i].opaque()),
                              arguments[i].size(), arguments[i].payload()};
    }
  }

  SE_DeviceMemoryBase result_base{result.opaque(), result.size(),
                                  result.payload()};
  SE_DeviceMemoryBase prefetch_base;
  if (cross_program_prefetch_addr.has_value()) {
    prefetch_base = SE_DeviceMemoryBase{cross_program_prefetch_addr->opaque(),
                                        cross_program_prefetch_addr->size(),
                                        cross_program_prefetch_addr->payload()};
  }
  int32_t rng_seed = run_options.run_options().rng_seed();

  XLA_DeviceAssignment c_dev_assign{/*bytes=*/nullptr, /*size=*/0};
  auto dev_assign = run_options.run_options().device_assignment();
  stream_executor::tpu::SerializedProto dev_assign_serialized;
  if (dev_assign != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    TF_RETURN_IF_ERROR(dev_assign->Serialize(&dev_assign_proto));
    dev_assign_serialized =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
    c_dev_assign.bytes = dev_assign_serialized.bytes;
    c_dev_assign.size = dev_assign_serialized.size;
  }

  auto platform = down_cast<tpu::TpuPlatform*>(
      tpu::TpuPlatformInterface::GetRegisteredPlatform());
  auto stream = platform->LookupStream(
      run_options.run_options().stream()->implementation());
  StatusHelper status;

  TpuExecutable_LoadProgramAndEnqueueToStream_Params params;
  params.struct_size = TpuExecutable_LoadProgramAndEnqueueToStream_Params_SIZE;
  params.priv = nullptr;
  params.program = core_program_;
  params.arguments = arguments_bases;
  params.arguments_len = arguments.size();
  params.result = &result_base;
  params.has_cross_program_prefetch_addr =
      cross_program_prefetch_addr.has_value();
  params.cross_program_prefetch_addr =
      cross_program_prefetch_addr.has_value() ? &prefetch_base : nullptr;
  params.rng_seed = rng_seed;
  params.device_assignment = &c_dev_assign;
  params.stream = stream;
  params.status = status.c_status;

  tpu::OpsApiFn()->TpuExecutable_LoadProgramAndEnqueueToStreamFn(&params);

  if (dev_assign != nullptr) {
    stream_executor::tpu::SerializedProto_Free(dev_assign_serialized);
  }
  delete[] arguments_bases;
  return status.status();
}

xla::Shape TpuOpExecutable::HostShapeToDeviceShape(
    const xla::Shape& host_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc mht_2(mht_2_v, 281, "", "./tensorflow/stream_executor/tpu/tpu_op_executable.cc", "TpuOpExecutable::HostShapeToDeviceShape");

  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;
  ApiConverter::ToC(host_shape, &c_host_shape);
  tpu::OpsApiFn()->HardwareLayout_HostShapeToDeviceShapeFn(&c_host_shape,
                                                           &c_device_shape);
  xla::Shape device_shape = ApiConverter::FromC(&c_device_shape);
  ApiConverter::Destroy(&c_host_shape);
  ApiConverter::Destroy(&c_device_shape);
  return device_shape;
}

int64_t TpuOpExecutable::ShapeSize(const xla::Shape& shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc mht_3(mht_3_v, 296, "", "./tensorflow/stream_executor/tpu/tpu_op_executable.cc", "TpuOpExecutable::ShapeSize");

  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size = tpu::OpsApiFn()->HardwareLayout_ShapeSizeFn(&c_shape);
  ApiConverter::Destroy(&c_shape);
  return size;
}

absl::string_view TpuOpExecutable::fingerprint() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_op_executableDTcc mht_4(mht_4_v, 307, "", "./tensorflow/stream_executor/tpu/tpu_op_executable.cc", "TpuOpExecutable::fingerprint");

  // TODO(skye): the fingerprint can be plumbed through via core_program_
  LOG(FATAL) << "TpuOpExecutable::fingerprint() unimplemented";
}

}  // namespace tensorflow
