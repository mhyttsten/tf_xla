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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStriangular_solve_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStriangular_solve_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStriangular_solve_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

TriangularSolveThunk::TriangularSolveThunk(
    ThunkInfo thunk_info, const TriangularSolveOptions& options,
    se::GpuAsmOpts asm_opts,  //
    const BufferAllocation::Slice& a_buffer,
    const BufferAllocation::Slice& b_buffer,
    const BufferAllocation::Slice& temp_buffer,  //
    PrimitiveType type, int64_t batch_size, int64_t m, int64_t n,
    int64_t a_batch_stride, int64_t b_batch_stride)
    : Thunk(Kind::kTriangularSolve, thunk_info),
      asm_opts_(asm_opts),
      uplo_(options.lower() ? se::blas::UpperLower::kLower
                            : se::blas::UpperLower::kUpper),
      side_(options.left_side() ? se::blas::Side::kLeft
                                : se::blas::Side::kRight),
      unit_diagonal_(options.unit_diagonal() ? se::blas::Diagonal::kUnit
                                             : se::blas::Diagonal::kNonUnit),
      a_buffer_(a_buffer),
      b_buffer_(b_buffer),
      temp_buffer_(temp_buffer),
      type_(type),
      batch_size_(batch_size),
      m_(m),
      n_(n),
      a_batch_stride_(a_batch_stride),
      b_batch_stride_(b_batch_stride) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStriangular_solve_thunkDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.cc", "TriangularSolveThunk::TriangularSolveThunk");

  transpose_a_ = [&] {
    switch (options.transpose_a()) {
      case TriangularSolveOptions::NO_TRANSPOSE:
        return se::blas::Transpose::kNoTranspose;
      case TriangularSolveOptions::TRANSPOSE:
        return se::blas::Transpose::kTranspose;
      case TriangularSolveOptions::ADJOINT:
        return se::blas::Transpose::kConjugateTranspose;
      default:
        LOG(ERROR) << "Invalid triangular solve transpose value "
                   << options.transpose_a();
        return se::blas::Transpose::kNoTranspose;
    }
  }();
}

Status TriangularSolveThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStriangular_solve_thunkDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.cc", "TriangularSolveThunk::ExecuteOnStream");

  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(3) << "uplo=" << se::blas::UpperLowerString(uplo_)
          << " side=" << se::blas::SideString(side_)
          << " diagonal=" << se::blas::DiagonalString(unit_diagonal_)
          << " batch_size=" << batch_size_ << " m=" << m_ << " n=" << n_
          << " a_batch_stride=" << a_batch_stride_
          << " b_batch_stride=" << b_batch_stride_;

  const int lda = side_ == se::blas::Side::kLeft ? m_ : n_;
  const int ldb = m_;

  se::DeviceMemoryBase a_data = buffer_allocations.GetDeviceAddress(a_buffer_);
  se::DeviceMemoryBase b_data = buffer_allocations.GetDeviceAddress(b_buffer_);

  bool launch_ok;
  if (batch_size_ == 1) {
    switch (type_) {
      case F32: {
        se::DeviceMemory<float> b_data_typed(b_data);
        launch_ok = stream
                        .ThenBlasTrsm(side_, uplo_, transpose_a_,
                                      unit_diagonal_, m_, n_, /*alpha=*/1.0f,
                                      se::DeviceMemory<float>(a_data), lda,
                                      &b_data_typed, ldb)
                        .ok();
        break;
      }
      case F64: {
        se::DeviceMemory<double> b_data_typed(b_data);
        launch_ok = stream
                        .ThenBlasTrsm(side_, uplo_, transpose_a_,
                                      unit_diagonal_, m_, n_, /*alpha=*/1.0,
                                      se::DeviceMemory<double>(a_data), lda,
                                      &b_data_typed, ldb)
                        .ok();
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>> b_data_typed(b_data);
        launch_ok =
            stream
                .ThenBlasTrsm(side_, uplo_, transpose_a_, unit_diagonal_, m_,
                              n_, /*alpha=*/1.0f,
                              se::DeviceMemory<std::complex<float>>(a_data),
                              lda, &b_data_typed, ldb)
                .ok();
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>> b_data_typed(b_data);
        launch_ok =
            stream
                .ThenBlasTrsm(side_, uplo_, transpose_a_, unit_diagonal_, m_,
                              n_, /*alpha=*/1.0,
                              se::DeviceMemory<std::complex<double>>(a_data),
                              lda, &b_data_typed, ldb)
                .ok();
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
  } else {
    // cublas trsmBatched requires us to materialize out two arrays of
    // batch_size_ pointers, pointing to the individual `a` and `b` matrices of
    // our input.  batch_pointers_bytes is the size in bytes of one of these
    // arrays.
    int64_t batch_pointers_bytes = sizeof(void*) * batch_size_;
    TF_RET_CHECK(temp_buffer_.size() >= 2 * batch_pointers_bytes);
    void** temp_base = reinterpret_cast<void**>(
        buffer_allocations.GetDeviceAddress(temp_buffer_).opaque());
    se::DeviceMemoryBase a_pointers(temp_base, batch_pointers_bytes);
    se::DeviceMemoryBase b_pointers(temp_base + batch_size_,
                                    batch_pointers_bytes);

    TF_RETURN_IF_ERROR(MakeBatchPointers(
        &stream, asm_opts_, a_data, a_batch_stride_, batch_size_, a_pointers));
    TF_RETURN_IF_ERROR(MakeBatchPointers(
        &stream, asm_opts_, b_data, b_batch_stride_, batch_size_, b_pointers));

    switch (type_) {
      case F32: {
        se::DeviceMemory<float*> typed_b_pointers(b_pointers);
        launch_ok =
            stream
                .ThenBlasTrsmBatched(side_, uplo_, transpose_a_, unit_diagonal_,
                                     m_, n_, /*alpha=*/1.0f,
                                     se::DeviceMemory<float*>(a_pointers), lda,
                                     &typed_b_pointers, ldb, batch_size_)
                .ok();
        break;
      }
      case F64: {
        se::DeviceMemory<double*> typed_b_pointers(b_pointers);
        launch_ok =
            stream
                .ThenBlasTrsmBatched(side_, uplo_, transpose_a_, unit_diagonal_,
                                     m_, n_, /*alpha=*/1.0f,
                                     se::DeviceMemory<double*>(a_pointers), lda,
                                     &typed_b_pointers, ldb, batch_size_)
                .ok();
        break;
      }
      case C64: {
        se::DeviceMemory<std::complex<float>*> typed_b_pointers(b_pointers);
        launch_ok = stream
                        .ThenBlasTrsmBatched(
                            side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
                            /*alpha=*/1.0f,
                            se::DeviceMemory<std::complex<float>*>(a_pointers),
                            lda, &typed_b_pointers, ldb, batch_size_)
                        .ok();
        break;
      }
      case C128: {
        se::DeviceMemory<std::complex<double>*> typed_b_pointers(b_pointers);
        launch_ok = stream
                        .ThenBlasTrsmBatched(
                            side_, uplo_, transpose_a_, unit_diagonal_, m_, n_,
                            /*alpha=*/1.0f,
                            se::DeviceMemory<std::complex<double>*>(a_pointers),
                            lda, &typed_b_pointers, ldb, batch_size_)
                        .ok();
        break;
      }
      default:
        return InvalidArgument("Invalid type for triangular solve %d", type_);
    }
  }

  if (!launch_ok) {
    return InternalError("Unable to launch triangular solve for thunk %p",
                         this);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
