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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/client.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

Client::Client(ServiceInterface* stub) : stub_(stub) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Client");
}

Client::~Client() = default;

StatusOr<Literal> Client::Transfer(const GlobalData& data,
                                   const Shape* shape_with_layout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Transfer");

  TransferToClientRequest request;
  *request.mutable_data() = data.handle();
  if (shape_with_layout != nullptr) {
    *request.mutable_shape_with_layout() = shape_with_layout->ToProto();
  }
  TransferToClientResponse response;

  VLOG(1) << "making transfer request";
  VLOG(3) << "TransferToClientRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToClient(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToClientResponse: {" << response.DebugString() << "}";

  if (!response.has_literal()) {
    return FailedPrecondition(
        "server provided response without a literal in "
        "TransferToClient request");
  }
  return Literal::CreateFromProto(*response.mutable_literal());
}

StatusOr<std::unique_ptr<GlobalData>> Client::TransferToServer(
    const LiteralSlice& literal, const DeviceHandle* device_handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_2(mht_2_v, 243, "", "./tensorflow/compiler/xla/client/client.cc", "Client::TransferToServer");

  TransferToServerRequest request;
  *request.mutable_literal() = literal.ToProto();
  if (device_handle) {
    *request.mutable_device_handle() = *device_handle;
  }
  TransferToServerResponse response;

  VLOG(1) << "making transfer to server request";
  VLOG(3) << "TransferToServerRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToServer(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToServerResponse: {" << response.DebugString() << "}";

  if (!response.has_data()) {
    return FailedPrecondition(
        "server provided response without a data handle in "
        "TransferToServer request");
  }

  return absl::make_unique<GlobalData>(stub_, response.data());
}

Status Client::TransferToInfeed(const LiteralSlice& literal, int64_t replica_id,
                                const DeviceHandle* device_handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_3(mht_3_v, 274, "", "./tensorflow/compiler/xla/client/client.cc", "Client::TransferToInfeed");

  TransferToInfeedRequest request;
  *request.mutable_literal() = literal.ToProto();
  if (device_handle) {
    *request.mutable_device_handle() = *device_handle;
  }
  request.set_replica_id(replica_id);
  TransferToInfeedResponse response;

  VLOG(1) << "making transfer to infeed request";
  VLOG(3) << "TransferToInfeedRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferToInfeed(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferToInfeedResponse: {" << response.DebugString() << "}";
  return Status::OK();
}

StatusOr<Literal> Client::TransferFromOutfeed(
    const Shape* shape_with_layout, int64_t replica_id,
    const DeviceHandle* device_handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_4(mht_4_v, 300, "", "./tensorflow/compiler/xla/client/client.cc", "Client::TransferFromOutfeed");

  TransferFromOutfeedRequest request;
  if (device_handle) {
    *request.mutable_device_handle() = *device_handle;
  }
  request.set_replica_id(replica_id);
  if (shape_with_layout != nullptr) {
    *request.mutable_shape_with_layout() = shape_with_layout->ToProto();
  }
  TransferFromOutfeedResponse response;

  VLOG(1) << "making transfer from outfeed request";
  VLOG(3) << "TransferFromOutfeedRequest: {" << request.DebugString() << "}";
  Status s = stub_->TransferFromOutfeed(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "TransferFromOutfeedResponse: {" << response.DebugString() << "}";

  if (!response.has_literal()) {
    return FailedPrecondition(
        "server provided response without a literal in "
        "TransferToClient request");
  }

  return Literal::CreateFromProto(response.literal());
}

Status Client::ResetDevice() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_5(mht_5_v, 333, "", "./tensorflow/compiler/xla/client/client.cc", "Client::ResetDevice");

  ResetDeviceRequest request;
  ResetDeviceResponse response;

  VLOG(1) << "making reset device request";
  VLOG(3) << "ResetDeviceRequest: {" << request.DebugString() << "}";
  Status s = stub_->ResetDevice(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  VLOG(3) << "ResetDeviceResponse: {" << response.DebugString() << "}";
  return Status::OK();
}

StatusOr<Literal> Client::ExecuteAndTransfer(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_6(mht_6_v, 355, "", "./tensorflow/compiler/xla/client/client.cc", "Client::ExecuteAndTransfer");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GlobalData> data,
      Execute(computation, arguments, execution_options, execution_profile));

  absl::optional<Shape> shape_with_output_layout;
  if (execution_options && execution_options->has_shape_with_output_layout()) {
    shape_with_output_layout =
        Shape(execution_options->shape_with_output_layout());
  }
  return Transfer(*data, shape_with_output_layout.has_value()
                             ? &(*shape_with_output_layout)
                             : nullptr);
}

StatusOr<Literal> Client::ComputeConstant(const XlaComputation& computation,
                                          const Layout* output_layout) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_7(mht_7_v, 374, "", "./tensorflow/compiler/xla/client/client.cc", "Client::ComputeConstant");

  ComputeConstantGraphRequest request;
  *request.mutable_computation() = computation.proto();
  if (output_layout != nullptr) {
    *request.mutable_output_layout() = output_layout->ToProto();
  }

  ComputeConstantResponse response;

  VLOG(2) << "making compute-constant-graph request";
  Status s = stub_->ComputeConstantGraph(&request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    return s;
  }

  VLOG(3) << "ComputeConstant: {" << response.DebugString() << "}";

  if (!response.has_literal()) {
    return InternalError(
        "no computed literal in the provided response in ComputeConstantGraph "
        "request");
  }
  return Literal::CreateFromProto(response.literal());
}

StatusOr<XlaComputation> Client::LoadSnapshot(const HloSnapshot& module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_8(mht_8_v, 404, "", "./tensorflow/compiler/xla/client/client.cc", "Client::LoadSnapshot");

  TF_RET_CHECK(module.has_hlo() && module.hlo().has_hlo_module());
  return XlaComputation(module.hlo().hlo_module());
}

StatusOr<ExecutionHandle> Client::Compile(
    const XlaComputation& computation, absl::Span<const Shape> argument_shapes,
    const ExecutionOptions* execution_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_9(mht_9_v, 414, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Compile");

  CompileRequest request;
  *request.mutable_computation() = computation.proto();

  if (execution_options == nullptr) {
    *request.mutable_execution_options() = CreateDefaultExecutionOptions();
  } else {
    *request.mutable_execution_options() = *execution_options;
  }
  if (request.execution_options().device_handles_size() > 1) {
    return InvalidArgument(
        "Compiling with multiple device handles is not supported. Use "
        "'Execute' instead.");
  }

  // The argument shapes affect how the computation is compiled.
  for (const auto& arg_shape : argument_shapes) {
    *request.add_input_shape_with_layout() = arg_shape.ToProto();
  }

  CompileResponse response;
  VLOG(1) << "making compile request: " << request.ShortDebugString();
  Status s = stub_->Compile(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  TF_RET_CHECK(response.has_handle());
  return response.handle();
}

StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const ExecutionHandle& handle, absl::Span<GlobalData* const> arguments,
    ExecutionProfile* execution_profile) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_10(mht_10_v, 451, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Execute");

  ExecuteRequest request;
  *request.mutable_handle() = handle;
  for (GlobalData* argument : arguments) {
    CHECK(argument != nullptr) << "Argument pointers must not be null.";
    *request.add_arguments() = argument->handle();
  }

  ExecuteResponse response;
  VLOG(1) << "making execute request: " << request.ShortDebugString();
  Status s = stub_->Execute(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  if (execution_profile != nullptr) {
    *execution_profile = response.profile();
  }

  return absl::make_unique<GlobalData>(stub_, response.output());
}

StatusOr<std::unique_ptr<GlobalData>> Client::Execute(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const ExecutionOptions* execution_options,
    ExecutionProfile* execution_profile) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_11(mht_11_v, 481, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Execute");

  // Create an ExecutionOptions if necessary, or set its DeviceHandles.
  absl::optional<ExecutionOptions> options_storage;
  if (!execution_options || execution_options->device_handles().empty()) {
    if (execution_options) {
      options_storage.emplace(*execution_options);
    } else {
      options_storage.emplace(CreateDefaultExecutionOptions());
    }
    execution_options = &*options_storage;

    TF_ASSIGN_OR_RETURN(auto device_handles,
                        GetDeviceHandles(/*device_count=*/1));
    TF_RET_CHECK(!device_handles.empty());
    *options_storage->add_device_handles() = std::move(device_handles[0]);
  }

  std::vector<XlaComputationInstance> computation_instances = {
      XlaComputationInstance{
          computation,
          std::vector<GlobalData*>(arguments.begin(), arguments.end()),
          *execution_options, execution_profile}};

  // Instead of invoking Compile() and Execute(), invoke
  // Service::ExecuteParallel() to execute our one computation.  Compile()
  // caches the executable forever, which isn't what we want.
  VLOG(1) << "Making ExecuteParallel request: "
          << execution_options->DebugString();
  TF_ASSIGN_OR_RETURN(auto results, ExecuteParallel(computation_instances));
  VLOG(1) << "ExecuteParallel request done.";

  // The result selection is a bit hacky, but better than assuming it is
  // device 0.
  //
  // TODO(b/118493728): Allow Execute to return one result per computation.
  for (int64_t i = 0, end = results.size(); i < end; i++) {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(*results[i]));
    if (!ShapeUtil::IsEmptyTuple(shape)) {
      VLOG(3) << "Fetching result from device " << i << ": "
              << ShapeUtil::HumanString(shape);
      return std::move(results[i]);
    }
  }
  TF_RET_CHECK(!results.empty());
  VLOG(1) << "Defaulting to device 0 result";
  return std::move(results[0]);
}

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::ExecuteParallel(
    absl::Span<const XlaComputationInstance> computations) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_12(mht_12_v, 533, "", "./tensorflow/compiler/xla/client/client.cc", "Client::ExecuteParallel");

  ExecuteGraphParallelRequest request;

  for (const XlaComputationInstance& computation : computations) {
    ExecuteGraphRequest single_request;
    *single_request.mutable_computation() = computation.computation.proto();
    for (GlobalData* argument : computation.arguments) {
      *single_request.add_arguments() = argument->handle();
    }
    *single_request.mutable_execution_options() = computation.execution_options;
    *request.add_requests() = single_request;
  }

  ExecuteParallelResponse response;
  VLOG(1) << "making execute-graph-parallel request: "
          << request.ShortDebugString();
  Status s = stub_->ExecuteGraphParallel(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<std::unique_ptr<GlobalData>> outputs;
  for (size_t i = 0, end = response.responses_size(); i < end; ++i) {
    outputs.push_back(
        absl::make_unique<GlobalData>(stub_, response.responses(i).output()));
    if (i < computations.size() &&
        computations[i].execution_profile != nullptr) {
      *computations[i].execution_profile = response.responses(i).profile();
    }
  }

  return std::move(outputs);
}

StatusOr<std::vector<DeviceHandle>> Client::GetDeviceHandles(
    int64_t device_count) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_13(mht_13_v, 573, "", "./tensorflow/compiler/xla/client/client.cc", "Client::GetDeviceHandles");

  if (device_count < 1) {
    return InvalidArgument("device_count must be greater than 0");
  }
  GetDeviceHandlesRequest request;
  request.set_device_count(device_count);

  GetDeviceHandlesResponse response;
  VLOG(1) << "making get device request: " << request.ShortDebugString();
  Status s = stub_->GetDeviceHandles(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<DeviceHandle> device_handles;
  const auto& response_device_handles = response.device_handles();
  device_handles.reserve(response_device_handles.size());
  for (const DeviceHandle& device_handle : response_device_handles) {
    device_handles.push_back(device_handle);
  }

  return device_handles;
}

Status Client::Unregister(const GlobalData& data) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_14(mht_14_v, 602, "", "./tensorflow/compiler/xla/client/client.cc", "Client::Unregister");

  UnregisterRequest request;
  *request.add_data() = data.handle();
  UnregisterResponse response;

  VLOG(1) << "making unregister request";
  Status s = stub_->Unregister(&request, &response);
  VLOG(1) << "done with request";

  return s;
}

StatusOr<std::vector<std::unique_ptr<GlobalData>>> Client::DeconstructTuple(
    const GlobalData& data) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_15(mht_15_v, 618, "", "./tensorflow/compiler/xla/client/client.cc", "Client::DeconstructTuple");

  DeconstructTupleRequest request;
  *request.mutable_tuple_handle() = data.handle();
  DeconstructTupleResponse response;

  VLOG(1) << "making DestructTuple request";
  Status s = stub_->DeconstructTuple(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  std::vector<std::unique_ptr<GlobalData>> handles;
  for (auto& handle : response.element_handles()) {
    handles.push_back(absl::make_unique<GlobalData>(stub_, handle));
  }
  return std::move(handles);
}

StatusOr<ComputationStats> Client::GetComputationStats(
    const XlaComputation& computation,
    const DebugOptions& debug_options) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_16(mht_16_v, 643, "", "./tensorflow/compiler/xla/client/client.cc", "Client::GetComputationStats");

  ComputationGraphStatsRequest request;

  // TODO(b/74197823): Find a way to avoid the copy of the hlo proto.
  *request.mutable_computation() = computation.proto();
  *request.mutable_debug_options() = debug_options;
  ComputationStatsResponse response;

  VLOG(1) << "making computation graph stats request";
  Status s = stub_->GetComputationGraphStats(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }
  CHECK(response.has_stats());
  return response.stats();
}

StatusOr<std::unique_ptr<ProgramShape>> Client::GetComputationShape(
    const XlaComputation& computation) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_17(mht_17_v, 666, "", "./tensorflow/compiler/xla/client/client.cc", "Client::GetComputationShape");

  TF_ASSIGN_OR_RETURN(const auto& result, computation.GetProgramShape());
  return absl::make_unique<ProgramShape>(result);
}

StatusOr<Shape> Client::GetShape(const GlobalData& data) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_18(mht_18_v, 674, "", "./tensorflow/compiler/xla/client/client.cc", "Client::GetShape");

  GetShapeRequest request;
  *request.mutable_data() = data.handle();
  GetShapeResponse response;

  VLOG(1) << "making get shape request";
  Status s = stub_->GetShape(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return Shape(response.shape());
}

StatusOr<std::string> Client::ExecutionStatsAsString(
    const XlaComputation& computation, const ExecutionProfile& profile) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_19(mht_19_v, 694, "", "./tensorflow/compiler/xla/client/client.cc", "Client::ExecutionStatsAsString");

  TF_ASSIGN_OR_RETURN(
      auto computation_stats,
      GetComputationStats(computation, GetDebugOptionsFromFlags()));
  int64_t total_flops =
      computation_stats.flop_count() + computation_stats.transcendental_count();
  if (profile.compute_time_ns() > 0) {
    int64_t nanoseconds = profile.compute_time_ns();
    int64_t cycle_count = profile.compute_cycle_count();
    double gflops = total_flops / nanoseconds;
    return absl::StrCat(
        "[Execution Statistics] flop count: ", computation_stats.flop_count(),
        ", transcendental count: ", computation_stats.transcendental_count(),
        ", compute execution time: ", nanoseconds, " nsec",
        ", compute cycles: ", cycle_count, ", performance: ", gflops,
        "gflop/s");
  }
  return std::string("[Execution Statistics] not available.");
}

StatusOr<ChannelHandle> Client::CreateChannelHandleByType(
    ChannelHandle::ChannelType type) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_20(mht_20_v, 718, "", "./tensorflow/compiler/xla/client/client.cc", "Client::CreateChannelHandleByType");

  CreateChannelHandleRequest request;
  request.set_channel_type(type);
  CreateChannelHandleResponse response;

  VLOG(1) << "making create channel handle request";
  Status s = stub_->CreateChannelHandle(&request, &response);
  VLOG(1) << "done with request";

  if (!s.ok()) {
    return s;
  }

  return response.channel();
}

StatusOr<ChannelHandle> Client::CreateChannelHandle() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_21(mht_21_v, 737, "", "./tensorflow/compiler/xla/client/client.cc", "Client::CreateChannelHandle");

  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_DEVICE);
}

StatusOr<ChannelHandle> Client::CreateHostToDeviceChannelHandle() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_22(mht_22_v, 744, "", "./tensorflow/compiler/xla/client/client.cc", "Client::CreateHostToDeviceChannelHandle");

  return CreateChannelHandleByType(ChannelHandle::HOST_TO_DEVICE);
}

StatusOr<ChannelHandle> Client::CreateDeviceToHostChannelHandle() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclientDTcc mht_23(mht_23_v, 751, "", "./tensorflow/compiler/xla/client/client.cc", "Client::CreateDeviceToHostChannelHandle");

  return CreateChannelHandleByType(ChannelHandle::DEVICE_TO_HOST);
}

}  // namespace xla
