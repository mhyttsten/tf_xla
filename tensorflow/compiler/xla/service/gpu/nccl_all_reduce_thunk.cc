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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {
namespace {

Status RunAllReduce(const NcclAllReduceConfig& config,
                    const std::vector<NcclCollectiveThunk::Buffer>& buffers,
                    const BufferAllocations& buffer_allocations,
                    se::Stream& stream, ncclComm_t comm) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "RunAllReduce");

#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(config.reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    const NcclCollectiveThunk::Buffer& buffer = buffers[i];
    const void* send_buffer =
        buffer_allocations.GetDeviceAddress(buffer.source_buffer).opaque();
    void* recv_buffer =
        buffer_allocations.GetDeviceAddress(buffer.destination_buffer).opaque();

    PrimitiveType element_type = config.config.operand_element_type[i];
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(element_type));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int element_count = buffer.element_count * dtype_and_multiplier.second;

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           element_count, dtype, reduce_op,
                                           comm, gpu_stream));
  }
  return XLA_CUDA_STATUS(ncclGroupEnd());
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

bool IsValidOperand(mlir::Value operand) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_1(mht_1_v, 259, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "IsValidOperand");

  Shape shape = TypeToShape(operand.getType());
  return LayoutUtil::IsDenseArray(shape) &&
         IsTypeSupportedByNccl(shape.element_type());
}

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `BFloat16Normalization`
// pass will have converted the op to float32 and added type conversions.
// TODO(cjfj): Can we prevent the bf16 conversion for this computation?
StatusOr<mlir::Operation*> FindReductionOp(mlir::Block& block) {
  TF_RET_CHECK(block.getNumArguments() == 2);
  mlir::Operation* terminator = block.getTerminator();
  TF_RET_CHECK(terminator);
  TF_RET_CHECK(terminator->getNumOperands() == 1);
  mlir::Value result = terminator->getOperand(0);
  TF_RET_CHECK(block.getArgument(0).getType() == result.getType());
  TF_RET_CHECK(block.getArgument(1).getType() == result.getType());

  mlir::Operation* result_op = result.getDefiningOp();
  TF_RET_CHECK(result_op);

  // In the bf16 case, the type conversions and op might be fused.
  if (mlir::isa<mlir::mhlo::FusionOp>(result_op)) {
    return FindReductionOp(result_op->getRegion(0).front());
  }

  // Standard case.
  if (absl::c_is_permutation(result_op->getOperands(), block.getArguments())) {
    return result_op;
  }

  // bf16 case.
  TF_RET_CHECK(mlir::isa<mlir::mhlo::ConvertOp>(result_op));
  TF_RET_CHECK(result_op->getNumOperands() == 1);
  mlir::Operation* reduction_op = result_op->getOperand(0).getDefiningOp();
  TF_RET_CHECK(reduction_op);
  TF_RET_CHECK(reduction_op->getNumOperands() == 2);
  mlir::Value operand0 = reduction_op->getOperand(0);
  mlir::Value operand1 = reduction_op->getOperand(1);
  auto operand0_op = operand0.getDefiningOp<mlir::mhlo::ConvertOp>();
  auto operand1_op = operand1.getDefiningOp<mlir::mhlo::ConvertOp>();
  TF_RET_CHECK(operand0_op);
  TF_RET_CHECK(operand1_op);
  TF_RET_CHECK(operand0_op->getNumOperands() == 1);
  TF_RET_CHECK(operand1_op->getNumOperands() == 1);
  std::array<mlir::Value, 2> operands{operand0_op->getOperand(0),
                                      operand1_op->getOperand(0)};
  TF_RET_CHECK(absl::c_is_permutation(operands, block.getArguments()));
  return reduction_op;
}

}  // namespace

namespace impl {

template <typename OpT>
bool CanImplement(OpT op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_2(mht_2_v, 319, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "CanImplement");

  return absl::c_all_of(op.operands(), IsValidOperand) &&
         NcclAllReduceThunkBase::MatchAllReduceComputation(op.computation())
             .has_value();
}

template <typename OpT>
NcclAllReduceConfig GetNcclAllReduceConfig(OpT op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_3(mht_3_v, 329, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "GetNcclAllReduceConfig");

  absl::optional<ReductionKind> reduction_kind =
      NcclAllReduceThunkBase::MatchAllReduceComputation(op.computation());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
bool IsDegenerate(OpT op, int64_t replica_count, int64_t partition_count) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_4(mht_4_v, 345, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "IsDegenerate");

  return GetNcclCollectiveConfigForMlir(op, op.use_global_device_ids())
      .IsDegenerate(replica_count, partition_count);
}

template <typename OpT>
CollectiveOpGroupMode GetGroupMode(OpT op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_5(mht_5_v, 354, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "GetGroupMode");

  return GetNcclAllReduceConfig(op).config.group_mode;
}

}  // namespace impl

absl::optional<ReductionKind> NcclAllReduceThunkBase::MatchAllReduceComputation(
    mlir::Region& computation) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_6(mht_6_v, 364, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunkBase::MatchAllReduceComputation");

  mlir::Block& block = computation.front();
  StatusOr<mlir::Operation*> reduction_op = FindReductionOp(block);
  if (!reduction_op.ok()) return absl::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(*reduction_op);
  if (!opcode.ok()) return absl::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type =
      TypeToShape(block.getArgument(0).getType()).element_type();
  if (type == PRED) {
    switch (opcode.ValueOrDie()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return absl::nullopt;
    }
  } else if (primitive_util::IsComplexType(type)) {
    // Only addition is supported for complex types.
    if (*opcode == HloOpcode::kAdd) {
      return ReductionKind::SUM;
    } else {
      return absl::nullopt;
    }
  } else {
    switch (*opcode) {
      case HloOpcode::kAdd:
        return ReductionKind::SUM;
      case HloOpcode::kMultiply:
        return ReductionKind::PRODUCT;
      case HloOpcode::kMaximum:
        return ReductionKind::MAX;
      case HloOpcode::kMinimum:
        return ReductionKind::MIN;
      default:
        return absl::nullopt;
    }
  }
}

NcclAllReduceThunkBase::NcclAllReduceThunkBase(Thunk::Kind kind,
                                               ThunkInfo thunk_info,
                                               NcclAllReduceConfig config,
                                               std::vector<Buffer> buffers)
    : NcclCollectiveThunk(kind, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_7(mht_7_v, 415, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunkBase::NcclAllReduceThunkBase");

  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllReduceThunk::NcclAllReduceThunk(ThunkInfo thunk_info,
                                       mlir::lmhlo::AllReduceOp op,
                                       std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduce, thunk_info,
                             impl::GetNcclAllReduceConfig(op), buffers) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_8(mht_8_v, 426, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunk::NcclAllReduceThunk");
}

bool NcclAllReduceThunk::CanImplement(mlir::lmhlo::AllReduceOp op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_9(mht_9_v, 431, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunk::CanImplement");

  return impl::CanImplement(op);
}

bool NcclAllReduceThunk::IsDegenerate(mlir::lmhlo::AllReduceOp op,
                                      int64_t replica_count,
                                      int64_t partition_count) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_10(mht_10_v, 440, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunk::IsDegenerate");

  return impl::IsDegenerate(op, replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceThunk::GetGroupMode(
    mlir::lmhlo::AllReduceOp op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_11(mht_11_v, 448, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunk::GetGroupMode");

  return impl::GetGroupMode(op);
}

Status NcclAllReduceThunk::RunNcclCollective(const ExecuteParams& params,
                                             ncclComm_t comm) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_12(mht_12_v, 456, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceThunk::RunNcclCollective");

  se::Stream& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunAllReduce(config_, buffers_, *params.buffer_allocations,
                                  stream, comm));

  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Done performing all-reduce for ordinal: " << device_ordinal;
  return Status::OK();
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, mlir::lmhlo_gpu::AllReduceStartOp op,
    std::vector<Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclAllReduceStart, thunk_info,
                             impl::GetNcclAllReduceConfig(op), buffers) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_13(mht_13_v, 473, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::NcclAllReduceStartThunk");
}

bool NcclAllReduceStartThunk::CanImplement(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_14(mht_14_v, 479, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::CanImplement");

  return impl::CanImplement(op);
}

bool NcclAllReduceStartThunk::IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                                           int64_t replica_count,
                                           int64_t partition_count) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_15(mht_15_v, 488, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::IsDegenerate");

  return impl::IsDegenerate(op, replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceStartThunk::GetGroupMode(
    mlir::lmhlo_gpu::AllReduceStartOp op) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_16(mht_16_v, 496, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::GetGroupMode");

  return impl::GetGroupMode(op);
}

Status NcclAllReduceStartThunk::RunNcclCollective(const ExecuteParams& params,
                                                  ncclComm_t comm) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_17(mht_17_v, 504, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::RunNcclCollective");

  se::Stream& async_comms_stream = *params.async_comms_stream;
  // Wait until compute inputs are ready.
  async_comms_stream.ThenWaitFor(params.stream);

  TF_RETURN_IF_ERROR(RunAllReduce(config_, buffers_, *params.buffer_allocations,
                                  async_comms_stream, comm));

  // Create an event on the async stream for the completion of the all-reduce.
  se::Event done_event(async_comms_stream.parent());
  TF_RET_CHECK(done_event.Init());
  async_comms_stream.ThenRecordEvent(&done_event);

  int device_ordinal = async_comms_stream.parent()->device_ordinal();

  {
    absl::MutexLock lock(&mu_);
    auto result = done_events_.emplace(device_ordinal, std::move(done_event));
    TF_RET_CHECK(result.second) << "done event has not been consumed";
  }

  VLOG(3) << "Done performing all-reduce-start for ordinal: " << device_ordinal;
  return Status::OK();
}

StatusOr<se::Event> NcclAllReduceStartThunk::TakeDoneEvent(int device_ordinal) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_18(mht_18_v, 532, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceStartThunk::TakeDoneEvent");

  absl::MutexLock lock(&mu_);
  auto it = done_events_.find(device_ordinal);
  TF_RET_CHECK(it != done_events_.end()) << "done event not found";
  // Take ownership of the event.
  se::Event done_event = std::move(it->second);
  done_events_.erase(it);
  return done_event;
}

NcclAllReduceDoneThunk::NcclAllReduceDoneThunk(
    ThunkInfo thunk_info, NcclAllReduceStartThunk& start_thunk)
    : Thunk(Thunk::kNcclAllReduceDone, thunk_info), start_thunk_(start_thunk) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_19(mht_19_v, 547, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceDoneThunk::NcclAllReduceDoneThunk");
}

Status NcclAllReduceDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_20(mht_20_v, 552, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclAllReduceDoneThunk::ExecuteOnStream");

  int device_ordinal = params.stream->parent()->device_ordinal();
  TF_ASSIGN_OR_RETURN(se::Event done_event,
                      start_thunk_.TakeDoneEvent(device_ordinal));
  params.stream->ThenWaitFor(&done_event);
  return Status::OK();
}

NcclReduceScatterThunk::NcclReduceScatterThunk(
    ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : NcclAllReduceThunkBase(Thunk::kNcclReduceScatter, thunk_info,
                             impl::GetNcclAllReduceConfig(op),
                             std::move(buffers)) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_21(mht_21_v, 568, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclReduceScatterThunk::NcclReduceScatterThunk");
}

/*static*/ bool NcclReduceScatterThunk::CanImplement(
    mlir::lmhlo::ReduceScatterOp op) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_22(mht_22_v, 574, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclReduceScatterThunk::CanImplement");

  return impl::CanImplement(op);
}

/*static*/ bool NcclReduceScatterThunk::IsDegenerate(
    mlir::lmhlo::ReduceScatterOp op, int64_t replica_count,
    int64_t partition_count) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_23(mht_23_v, 583, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclReduceScatterThunk::IsDegenerate");

  return impl::IsDegenerate(op, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterThunk::GetGroupMode(
    mlir::lmhlo::ReduceScatterOp op) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_24(mht_24_v, 591, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclReduceScatterThunk::GetGroupMode");

  return impl::GetGroupMode(op);
}

Status NcclReduceScatterThunk::RunNcclCollective(const ExecuteParams& params,
                                                 ncclComm_t comm) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSnccl_all_reduce_thunkDTcc mht_25(mht_25_v, 599, "", "./tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.cc", "NcclReduceScatterThunk::RunNcclCollective");

#if XLA_ENABLE_XCCL
  int device_ordinal = params.stream->parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(config_.reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream =
      se::gpu::AsGpuStreamValue(params.stream);

  int num_participants = 0;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();

    PrimitiveType element_type = config_.config.operand_element_type[i];
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(element_type));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int element_count = buffer.element_count * dtype_and_multiplier.second;

    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(element_count % num_participants == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    int64_t recv_count = element_count / num_participants;
    VLOG(3) << absl::StreamFormat(
        "Calling ncclReduceScatter(send_buffer=%p, recv_buffer=%p, "
        "recvcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, recv_count, static_cast<const void*>(comm),
        gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclReduceScatter(send_buffer, recv_buffer,
                                               recv_count, dtype, reduce_op,
                                               comm, gpu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla
