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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/cuda/cuda_fft.h"

#include <complex>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuFftPlugin);

namespace {

// A helper function transforming gpu_fft arguments into cuFFT arguments.
cufftType CUDAFftType(fft::Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_0(mht_0_v, 212, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftType");

  switch (type) {
    case fft::Type::kC2CForward:
    case fft::Type::kC2CInverse:
      return CUFFT_C2C;
    case fft::Type::kC2R:
      return CUFFT_C2R;
    case fft::Type::kR2C:
      return CUFFT_R2C;
    case fft::Type::kZ2ZForward:
    case fft::Type::kZ2ZInverse:
      return CUFFT_Z2Z;
    case fft::Type::kZ2D:
      return CUFFT_Z2D;
    case fft::Type::kD2Z:
      return CUFFT_D2Z;
    default:
      LOG(FATAL) << "Invalid value of fft::Type.";
  }
}

// Associates the given stream with the given cuFFT plan.
bool SetStream(GpuExecutor *parent, cufftHandle plan, Stream *stream) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_1(mht_1_v, 237, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "SetStream");

  cuda::ScopedActivateExecutorContext sac(parent);
  auto ret = cufftSetStream(plan, AsGpuStreamValue(stream));
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine cufftSetStream: " << ret;
    return false;
  }
  return true;
}

}  // namespace

port::Status CUDAFftPlan::Initialize(
    GpuExecutor *parent, Stream *stream, int rank, uint64_t *elem_count,
    uint64_t *input_embed, uint64 input_stride, uint64 input_distance,
    uint64_t *output_embed, uint64 output_stride, uint64 output_distance,
    fft::Type type, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_2(mht_2_v, 256, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftPlan::Initialize");

  if (IsInitialized()) {
    LOG(FATAL) << "Try to repeatedly initialize.";
  }
  is_initialized_ = true;
  scratch_allocator_ = scratch_allocator;
  cuda::ScopedActivateExecutorContext sac(parent);
  int elem_count_[3], input_embed_[3], output_embed_[3];
  for (int i = 0; i < rank; ++i) {
    elem_count_[i] = elem_count[i];
    if (input_embed) {
      input_embed_[i] = input_embed[i];
    }
    if (output_embed) {
      output_embed_[i] = output_embed[i];
    }
  }
  parent_ = parent;
  fft_type_ = type;
  if (batch_count == 1 && input_embed == nullptr && output_embed == nullptr) {
    cufftResult_t ret;
    if (scratch_allocator == nullptr) {
      switch (rank) {
        case 1:
          // cufftPlan1d
          ret = cufftPlan1d(&plan_, elem_count_[0], CUDAFftType(type),
                            1 /* = batch */);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 1d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to create cuFFT 1d plan.");
          }
          return port::Status::OK();
        case 2:
          // cufftPlan2d
          ret = cufftPlan2d(&plan_, elem_count_[0], elem_count_[1],
                            CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 2d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to create cuFFT 2d plan.");
          }
          return port::Status::OK();
        case 3:
          // cufftPlan3d
          ret = cufftPlan3d(&plan_, elem_count_[0], elem_count_[1],
                            elem_count_[2], CUDAFftType(type));
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to create cuFFT 3d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to create cuFFT 3d plan.");
          }
          return port::Status::OK();
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status(port::error::INVALID_ARGUMENT,
                              "cufftPlan only takes rank 1, 2, or 3.");
      }
    } else {
      ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT plan:" << ret;
        return port::Status(port::error::INTERNAL,
                            "Failed to create cuFFT plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for cuFFT plan:" << ret;
        return port::Status(port::error::INTERNAL,
                            "Failed to set auto allocation for cuFFT plan.");
      }
      switch (rank) {
        case 1:
          ret = cufftMakePlan1d(plan_, elem_count_[0], CUDAFftType(type),
                                /*batch=*/1, &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 1d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to make cuFFT 1d plan.");
          }
          break;
        case 2:
          ret = cufftMakePlan2d(plan_, elem_count_[0], elem_count_[1],
                                CUDAFftType(type), &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 2d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to make cuFFT 2d plan.");
          }
          break;
        case 3:
          ret = cufftMakePlan3d(plan_, elem_count_[0], elem_count_[1],
                                elem_count_[2], CUDAFftType(type),
                                &scratch_size_bytes_);
          if (ret != CUFFT_SUCCESS) {
            LOG(ERROR) << "failed to make cuFFT 3d plan:" << ret;
            return port::Status(port::error::INTERNAL,
                                "Failed to make cuFFT 3d plan.");
          }
          break;
        default:
          LOG(ERROR) << "Invalid rank value for cufftPlan. "
                        "Requested 1, 2, or 3, given: "
                     << rank;
          return port::Status(port::error::INVALID_ARGUMENT,
                              "cufftPlan only takes rank 1, 2, or 3.");
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  } else {
    // For either multiple batches or rank higher than 3, use cufftPlanMany().
    if (scratch_allocator == nullptr) {
      auto ret = cufftPlanMany(
          &plan_, rank, elem_count_, input_embed ? input_embed_ : nullptr,
          input_stride, input_distance, output_embed ? output_embed_ : nullptr,
          output_stride, output_distance, CUDAFftType(type), batch_count);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT batched plan:" << ret;
        return port::Status(port::error::INTERNAL,
                            "Failed to create cuFFT batched plan.");
      }
    } else {
      auto ret = cufftCreate(&plan_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to create cuFFT batched plan:" << ret;
        return port::Status(port::error::INTERNAL,
                            "Failed to create cuFFT batched plan.");
      }
      ret = cufftSetAutoAllocation(plan_, 0);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to set auto allocation for cuFFT batched plan:"
                   << ret;
        return port::Status(
            port::error::INTERNAL,
            "Failed to set auto allocation for cuFFT batched plan.");
      }
      ret = cufftMakePlanMany(
          plan_, rank, elem_count_, input_embed ? input_embed_ : nullptr,
          input_stride, input_distance, output_embed ? output_embed_ : nullptr,
          output_stride, output_distance, CUDAFftType(type), batch_count,
          &scratch_size_bytes_);
      if (ret != CUFFT_SUCCESS) {
        LOG(ERROR) << "failed to make cuFFT batched plan:" << ret;
        return port::Status(port::error::INTERNAL,
                            "Failed to make cuFFT batched plan.");
      }
      return UpdateScratchAllocator(stream, scratch_allocator);
    }
  }
  return port::Status::OK();
}

port::Status CUDAFftPlan::Initialize(GpuExecutor *parent, Stream *stream,
                                     int rank, uint64_t *elem_count,
                                     fft::Type type,
                                     ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_3(mht_3_v, 416, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftPlan::Initialize");

  return Initialize(parent_, stream, rank, elem_count,
                    /*input_embed=*/nullptr, /*input_stride=*/0,
                    /*input_distance=*/0,
                    /*output_embed=*/nullptr, /*output_stride=*/0,
                    /*output_distance=*/0, type, 1, scratch_allocator);
}

port::Status CUDAFftPlan::UpdateScratchAllocator(
    Stream *stream, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_4(mht_4_v, 428, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftPlan::UpdateScratchAllocator");

  scratch_allocator_ = scratch_allocator;

  if (scratch_size_bytes_ != 0) {
    auto allocated = scratch_allocator->AllocateBytes(scratch_size_bytes_);
    if (!allocated.ok() || (scratch_ = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "failed to allocate work area.";
      return allocated.status();
    }
  }
  // Connect work area with allocated space.
  cuda::ScopedActivateExecutorContext sac(parent_);
  cufftResult_t ret = cufftSetWorkArea(plan_, scratch_.opaque());
  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to set work area for cuFFT plan:" << ret;
    return port::Status(port::error::INTERNAL,
                        "Failed to set work area for cuFFT plan.");
  }
  return port::Status::OK();
}

CUDAFftPlan::~CUDAFftPlan() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_5(mht_5_v, 452, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftPlan::~CUDAFftPlan");

  cuda::ScopedActivateExecutorContext sac(parent_);
  cufftDestroy(plan_);
}

int CUDAFftPlan::GetFftDirection() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_6(mht_6_v, 460, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFftPlan::GetFftDirection");

  if (!IsInitialized()) {
    LOG(FATAL) << "Try to get fft direction before initialization.";
  } else {
    switch (fft_type_) {
      case fft::Type::kC2CForward:
      case fft::Type::kZ2ZForward:
      case fft::Type::kR2C:
      case fft::Type::kD2Z:
        return CUFFT_FORWARD;
      case fft::Type::kC2CInverse:
      case fft::Type::kZ2ZInverse:
      case fft::Type::kC2R:
      case fft::Type::kZ2D:
        return CUFFT_INVERSE;
      default:
        LOG(FATAL) << "Invalid value of fft::Type.";
    }
  }
}

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlan(Stream *stream, uint64_t num_x,
                                                 fft::Type type,
                                                 bool in_place_fft) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_7(mht_7_v, 486, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create1dPlan");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  // TODO(yangzihao): In the future, send error msg back to TensorFlow
  // so it can fail gracefully,
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x;
    LOG(FATAL) << "failed to initialize cufft 1d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create1dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, fft::Type type, bool in_place_fft,
    ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_8(mht_8_v, 506, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create1dPlanWithScratchAllocator");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[1] = {num_x};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 1, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x;
    LOG(FATAL)
        << "failed to initialize cufft 1d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, fft::Type type,
                                                 bool in_place_fft) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_9(mht_9_v, 525, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create2dPlan");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 1, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y;
    LOG(FATAL) << "failed to initialize cufft 2d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create2dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_10(mht_10_v, 543, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create2dPlanWithScratchAllocator");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[2] = {num_x, num_y};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 2, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y;
    LOG(FATAL)
        << "failed to initialize cufft 2d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlan(Stream *stream, uint64_t num_x,
                                                 uint64_t num_y, uint64 num_z,
                                                 fft::Type type,
                                                 bool in_place_fft) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_11(mht_11_v, 563, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create3dPlan");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, 3, elem_count, type, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y
               << " num_z: " << num_z;
    LOG(FATAL) << "failed to initialize cufft 3d plan: "
               << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::Create3dPlanWithScratchAllocator(
    Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z, fft::Type type,
    bool in_place_fft, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_12(mht_12_v, 582, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::Create3dPlanWithScratchAllocator");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  uint64_t elem_count[3] = {num_x, num_y, num_z};
  port::Status status = fft_plan_ptr->Initialize(parent_, stream, 3, elem_count,
                                                 type, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Plan Parameters: num_x: " << num_x << " num_y: " << num_y
               << " num_z: " << num_z;
    LOG(FATAL)
        << "failed to initialize cufft 3d plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlan(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_13(mht_13_v, 604, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::CreateBatchedPlan");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, /*scratch_allocator=*/nullptr);
  if (!status.ok()) {
    LOG(ERROR) << "Initialize Params: rank: " << rank
               << " elem_count: " << *elem_count
               << " input_embed: " << *input_embed
               << " input_stride: " << input_stride
               << " input_distance: " << input_distance
               << " output_embed: " << *output_embed
               << " output_stride: " << output_stride
               << " output_distance: " << output_distance
               << " batch_count: " << batch_count;
    LOG(FATAL) << "failed to initialize batched cufft plan: "
               << status.error_message();
  }

  return std::move(fft_plan_ptr);
}

std::unique_ptr<fft::Plan> CUDAFft::CreateBatchedPlanWithScratchAllocator(
    Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
    uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
    uint64_t output_stride, uint64 output_distance, fft::Type type,
    bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_14(mht_14_v, 634, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::CreateBatchedPlanWithScratchAllocator");

  std::unique_ptr<CUDAFftPlan> fft_plan_ptr{new CUDAFftPlan()};
  port::Status status = fft_plan_ptr->Initialize(
      parent_, stream, rank, elem_count, input_embed, input_stride,
      input_distance, output_embed, output_stride, output_distance, type,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << "Initialize Params: rank: " << rank
               << " elem_count: " << *elem_count
               << " input_embed: " << *input_embed
               << " input_stride: " << input_stride
               << " input_distance: " << input_distance
               << " output_embed: " << *output_embed
               << " output_stride: " << output_stride
               << " output_distance: " << output_distance
               << " batch_count: " << batch_count;
    LOG(FATAL)
        << "failed to initialize batched cufft plan with customized allocator: "
        << status.error_message();
  }
  return std::move(fft_plan_ptr);
}

void CUDAFft::UpdatePlanWithScratchAllocator(
    Stream *stream, fft::Plan *plan, ScratchAllocator *scratch_allocator) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_15(mht_15_v, 661, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "CUDAFft::UpdatePlanWithScratchAllocator");

  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  port::Status status =
      cuda_fft_plan->UpdateScratchAllocator(stream, scratch_allocator);
  if (!status.ok()) {
    LOG(FATAL) << "failed to update custom allocator for cufft plan: "
               << status.error_message();
  }
}

template <typename FuncT, typename InputT, typename OutputT>
bool CUDAFft::DoFftInternal(Stream *stream, fft::Plan *plan, FuncT cufftExec,
                            const DeviceMemory<InputT> &input,
                            DeviceMemory<OutputT> *output) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);

  DeviceMemory<InputT> input_maybe_copy = input;

  if (cuda_fft_plan == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

#if CUDA_VERSION >= 10010
  // Workaround a cuFFT bug, which mutates the input buffer when it shouldn't.
  // See b/155276727 and go/nvbugs/2959622.
  // TODO(b/155276727): refine the bounding condition.
  if (input.opaque() != output->opaque() &&
      std::is_same<InputT, std::complex<float>>::value &&
      std::is_same<OutputT, float>::value && input.size() > 0) {
    auto *allocator = cuda_fft_plan->GetScratchAllocator();
    if (allocator) {
      auto allocated = allocator->AllocateBytes(input.size());
      if (allocated.ok()) {
        if (stream->ThenMemcpy(&allocated.ValueOrDie(), input, input.size())
                .ok()) {
          input_maybe_copy = DeviceMemory<InputT>(allocated.ValueOrDie());
        }
      }
      // Keep going even the workaround fails, since we don't have a good
      // bounding box. We don't want to give up on a potentially correct
      // execution just because the allocation for the incorrect case fails.
    }
  }
#endif

  cuda::ScopedActivateExecutorContext sac(parent_);
  auto ret =
      cufftExec(cuda_fft_plan->GetPlan(),
                GpuComplex(const_cast<InputT *>(GpuMemory(input_maybe_copy))),
                GpuComplex(GpuMemoryMutable(output)));

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine: " << ret;
    return false;
  }

  return true;
}

template <typename FuncT, typename InputT, typename OutputT>
bool CUDAFft::DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                         FuncT cufftExec,
                                         const DeviceMemory<InputT> &input,
                                         DeviceMemory<OutputT> *output) {
  CUDAFftPlan *cuda_fft_plan = dynamic_cast<CUDAFftPlan *>(plan);
  if (cuda_fft_plan == nullptr) {
    LOG(ERROR) << "the passed-in plan is not a CUDAFftPlan object.";
    return false;
  }

  if (!SetStream(parent_, cuda_fft_plan->GetPlan(), stream)) {
    return false;
  }

  cuda::ScopedActivateExecutorContext sac(parent_);
  auto ret = cufftExec(cuda_fft_plan->GetPlan(),
                       GpuComplex(const_cast<InputT *>(GpuMemory(input))),
                       GpuComplex(GpuMemoryMutable(output)),
                       cuda_fft_plan->GetFftDirection());

  if (ret != CUFFT_SUCCESS) {
    LOG(ERROR) << "failed to run cuFFT routine: " << ret;
    return false;
  }

  return true;
}

#define STREAM_EXECUTOR_CUDA_DEFINE_FFT(__type, __fft_type1, __fft_type2,      \
                                        __fft_type3)                           \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftWithDirectionInternal(stream, plan, cufftExec##__fft_type1,    \
                                      input, output);                          \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<__type> &input,                       \
                      DeviceMemory<std::complex<__type>> *output) {            \
    return DoFftInternal(stream, plan, cufftExec##__fft_type2, input, output); \
  }                                                                            \
  bool CUDAFft::DoFft(Stream *stream, fft::Plan *plan,                         \
                      const DeviceMemory<std::complex<__type>> &input,         \
                      DeviceMemory<__type> *output) {                          \
    return DoFftInternal(stream, plan, cufftExec##__fft_type3, input, output); \
  }

STREAM_EXECUTOR_CUDA_DEFINE_FFT(float, C2C, R2C, C2R)
STREAM_EXECUTOR_CUDA_DEFINE_FFT(double, Z2Z, D2Z, Z2D)

#undef STREAM_EXECUTOR_CUDA_DEFINE_FFT

}  // namespace gpu

void initialize_cufft() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTcc mht_16(mht_16_v, 783, "", "./tensorflow/stream_executor/cuda/cuda_fft.cc", "initialize_cufft");

  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::FftFactory>(
          cuda::kCudaPlatformId, gpu::kCuFftPlugin, "cuFFT",
          [](internal::StreamExecutorInterface *parent) -> fft::FftSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuFFT "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            return new gpu::CUDAFft(cuda_executor);
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuFFT factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kFft, gpu::kCuFftPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cufft,
                            { stream_executor::initialize_cufft(); });
