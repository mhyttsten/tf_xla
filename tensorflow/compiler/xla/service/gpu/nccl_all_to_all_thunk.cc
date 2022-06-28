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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

/*static*/ NcclAllToAllConfig NcclAllToAllThunk::GetNcclAllToAllConfig(
    mlir::lmhlo::AllToAllOp op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc", "NcclAllToAllThunk::GetNcclAllToAllConfig");

  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, absl::nullopt);
  config.has_split_dimension = op.split_dimension().hasValue();
  return config;
}

/*static*/ bool NcclAllToAllThunk::CanImplement(mlir::lmhlo::AllToAllOp op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc", "NcclAllToAllThunk::CanImplement");

  return absl::c_all_of(op.operands(), [&op](mlir::Value operand) {
    Shape shape = GetShape(operand);
    return LayoutUtil::IsDenseArray(shape) &&
           IsTypeSupportedByNccl(shape.element_type()) &&
           (!op.split_dimension() ||
            LayoutUtil::MinorToMajor(shape).back() == *op.split_dimension());
  });
}

NcclAllToAllThunk::NcclAllToAllThunk(
    ThunkInfo thunk_info, mlir::lmhlo::AllToAllOp op,
    std::vector<NcclAllToAllThunk::Buffer> buffers)
    : NcclCollectiveThunk(Thunk::kNcclAllToAll, thunk_info),
      config_(GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc", "NcclAllToAllThunk::NcclAllToAllThunk");

  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

Status NcclAllToAllThunk::RunNcclCollective(const ExecuteParams& params,
                                            ncclComm_t comm) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_to_all_thunkDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.cc", "NcclAllToAllThunk::RunNcclCollective");

#if XLA_ENABLE_XCCL
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  se::gpu::GpuStreamHandle gpu_stream =
      se::gpu::AsGpuStreamValue(params.stream);

  int num_participants;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (config_.has_split_dimension) {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8_t* send_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8_t* recv_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(element_type));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int element_count = buffer.element_count * dtype_and_multiplier.second;

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes =
          chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);

      for (int rank = 0; rank < num_participants; ++rank) {
        XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer + rank * chunk_bytes,
                                          chunk_elements, dtype, rank, comm,
                                          gpu_stream));
        XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer + rank * chunk_bytes,
                                          chunk_elements, dtype, rank, comm,
                                          gpu_stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers_.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    for (size_t i = 0; i < buffers_.size(); ++i) {
      const Buffer& buffer = buffers_[i];
      const uint8_t* send_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
              .opaque());
      uint8_t* recv_buffer = static_cast<uint8_t*>(
          params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
              .opaque());

      PrimitiveType element_type = config_.config.operand_element_type[i];
      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(element_type));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int element_count = buffer.element_count * dtype_and_multiplier.second;

      XLA_CUDA_RETURN_IF_ERROR(ncclSend(send_buffer, element_count, dtype,
                                        /*rank=*/i, comm, gpu_stream));
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(recv_buffer, element_count, dtype,
                                        /*rank=*/i, comm, gpu_stream));
    }
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-to-all for ordinal: " << device_ordinal;
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla
