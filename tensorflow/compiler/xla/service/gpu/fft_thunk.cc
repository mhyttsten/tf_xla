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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

FftScratchAllocator::FftScratchAllocator(
    int device_ordinal, se::DeviceMemoryAllocator* memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftScratchAllocator::FftScratchAllocator");
}

int64_t FftScratchAllocator::GetMemoryLimitInBytes() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftScratchAllocator::GetMemoryLimitInBytes");

  constexpr int64_t kFftScratchSize = 1LL << 32;  // 4GB by default.
  return kFftScratchSize;
}

StatusOr<se::DeviceMemory<uint8_t>> FftScratchAllocator::AllocateBytes(
    int64_t byte_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_2(mht_2_v, 215, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftScratchAllocator::AllocateBytes");

  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8_t>(buffer_addr);
}

namespace {

se::fft::Type FftTypeToSeType(FftType type, bool double_precision) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_3(mht_3_v, 240, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftTypeToSeType");

  switch (type) {
    case FftType::FFT:
      return double_precision ? se::fft::Type::kZ2ZForward
                              : se::fft::Type::kC2CForward;
    case FftType::IFFT:
      return double_precision ? se::fft::Type::kZ2ZInverse
                              : se::fft::Type::kC2CInverse;
    case FftType::IRFFT:
      return double_precision ? se::fft::Type::kZ2D : se::fft::Type::kC2R;
    case FftType::RFFT:
      return double_precision ? se::fft::Type::kD2Z : se::fft::Type::kR2C;
    default:
      LOG(FATAL) << "unsupported fft type";
  }
}

std::string FftTypeToString(se::fft::Type type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_4(mht_4_v, 260, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftTypeToString");

  switch (type) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kZ2ZForward:
      return "FFT";
    case se::fft::Type::kC2CInverse:
    case se::fft::Type::kZ2ZInverse:
      return "IFFT";
    case se::fft::Type::kC2R:
    case se::fft::Type::kZ2D:
      return "IRFFT";
    case se::fft::Type::kR2C:
    case se::fft::Type::kD2Z:
      return "RFFT";
    default:
      LOG(FATAL) << "unknown fft type";
  }
}

}  // namespace

FftThunk::FftThunk(ThunkInfo thunk_info, FftType fft_type,
                   absl::Span<const int64_t> fft_length,
                   const BufferAllocation::Slice& input_buffer,
                   const BufferAllocation::Slice& output_buffer,
                   const Shape& input_shape, const Shape& output_shape)
    : Thunk(Kind::kFft, thunk_info),
      fft_type_(
          FftTypeToSeType(fft_type, input_shape.element_type() == F64 ||
                                        input_shape.element_type() == C128)),
      fft_length_(fft_length.begin(), fft_length.end()),
      scale_factor_(1.0f),
      input_buffer_(input_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      output_shape_(output_shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_5(mht_5_v, 298, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftThunk::FftThunk");
}

Status FftThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSfft_thunkDTcc mht_6(mht_6_v, 303, "", "./tensorflow/compiler/xla/service/gpu/fft_thunk.cc", "FftThunk::ExecuteOnStream");

  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(3) << "FFT type: " << FftTypeToString(fft_type_);
  VLOG(3) << "Input shape: " << ShapeUtil::HumanStringWithLayout(input_shape_);
  VLOG(3) << "Output shape: "
          << ShapeUtil::HumanStringWithLayout(output_shape_);

  FftScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                        buffer_allocations.memory_allocator());
  FftPlan* fft_plan_ptr;
  {
    absl::MutexLock lock(&mu_);
    std::unique_ptr<FftPlan>& plan =
        fft_plans_[buffer_allocations.device_ordinal()];
    if (!plan) {
      plan = std::make_unique<FftPlan>();
    }
    fft_plan_ptr = plan.get();
  }
  // CuFFT thread-safety requires that separate host threads not share plans;
  // protect each plan with a mutex.
  absl::MutexLock lock(&fft_plan_ptr->mu);
  std::unique_ptr<se::fft::Plan>& fft_plan = fft_plan_ptr->plan;
  if (fft_plan == nullptr) {
    const int64_t fft_rank = fft_length_.size();
    CHECK_LE(fft_rank, 3);
    int batch_size = 1;
    for (int i = 0; i < input_shape_.dimensions_size() - fft_rank; ++i) {
      batch_size *= input_shape_.dimensions(i);
    }
    uint64_t fft_length[3];
    uint64_t input_embed[3];
    const uint64_t input_stride = 1;
    uint64_t input_distance = 1;
    uint64_t output_embed[3];
    const uint64_t output_stride = 1;
    uint64_t output_distance = 1;

    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape_.dimensions_size() - fft_rank + i;
      fft_length[i] = static_cast<uint64_t>(fft_length_[i]);
      input_embed[i] = input_shape_.dimensions(dim_offset);
      input_distance *= input_shape_.dimensions(dim_offset);
      output_embed[i] = output_shape_.dimensions(dim_offset);
      output_distance *= output_shape_.dimensions(dim_offset);
    }

    constexpr bool kInPlaceFft = false;
    fft_plan = stream.parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
        &stream, fft_rank, fft_length, input_embed, input_stride,
        input_distance, output_embed, output_stride, output_distance, fft_type_,
        kInPlaceFft, batch_size, &scratch_allocator);
    scale_factor_ = 1.0f / output_distance;
  } else {
    stream.parent()->AsFft()->UpdatePlanWithScratchAllocator(
        &stream, fft_plan.get(), &scratch_allocator);
  }

  bool launch_ok;
  switch (fft_type_) {
    case se::fft::Type::kC2CForward: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kZ2ZForward: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kC2CInverse: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      complex64(scale_factor_), &output_data, 1)
                        .ok();
      }
      break;
    }
    case se::fft::Type::kZ2ZInverse: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok =
            stream
                .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                              complex128(scale_factor_), &output_data, 1)
                .ok();
      }
      break;
    }
    case se::fft::Type::kR2C: {
      se::DeviceMemory<float> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kD2Z: {
      se::DeviceMemory<double> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kC2R: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<float> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      scale_factor_, &output_data, 1)
                        .ok();
      }
      break;
    }
    case se::fft::Type::kZ2D: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<double> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok = stream.ThenFft(fft_plan.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      scale_factor_, &output_data, 1)
                        .ok();
      }
      break;
    }
    default:
      LOG(FATAL) << "unsupported fft type";
  }
  if (launch_ok) {
    return Status::OK();
  }
  return InternalError("Unable to launch fft for thunk %p with type %s", this,
                       FftTypeToString(fft_type_));
}

}  // namespace gpu
}  // namespace xla
