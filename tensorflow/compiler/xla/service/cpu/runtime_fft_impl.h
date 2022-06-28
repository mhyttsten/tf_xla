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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fft_implDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fft_implDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fft_implDTh() {
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


#include <array>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

namespace internal {

enum class FftType : int32_t {
  FFT = 0,    // Forward FFT; complex in, complex out.
  IFFT = 1,   // Inverse FFT; complex in, complex out.
  RFFT = 2,   // Forward real FFT; real in, fft_length / 2 + 1 complex out
  IRFFT = 3,  // Inverse real FFT; fft_length / 2 + 1 complex in,
              //                   fft_length real out
};
inline constexpr int FftTypeArraySize() { return 4; }

// Computes either a forward or reverse complex-to-complex FFT.
template <bool Forward, int FFTRank, typename EigenDevice, typename Complex>
void EigenFftC2C(const EigenDevice& device, Complex* out, Complex* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
  constexpr auto direction = Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;

  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> dims;
  dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    dims[i + 1] = fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, dims);
  Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, dims);
  output.device(device) = input.template fft<Eigen::BothParts, direction>(axes);
}

// Computes a forward real->complex FFT, slicing out redundant negative
// frequencies from the innermost dimension.
template <int FFTRank, typename EigenDevice, typename Real, typename Complex>
void EigenFftR2C(const EigenDevice& device, Complex* out, Real* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = fft_shape[i];
    out_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Real, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);

  // Compute the full FFT using a temporary tensor.
  Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor> full_fft(in_dims);

  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  full_fft.device(device) =
      input.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

  // Slice away the negative frequency components.
  output.device(device) = full_fft.slice(zero_start_indices, out_dims);
}

// Computes a reverse complex->real FFT, reconstructing redundant negative
// frequencies using reverse conjugate on innermost dimension after doing IFFT
// on outer dimensions.
template <int FFTRank, typename EigenDevice, typename Complex, typename Real>
void EigenFftC2R(const EigenDevice& device, Real* out, Complex* operand,
                 int64_t input_batch, int64_t fft_length0, int64_t fft_length1,
                 int64_t fft_length2) {
  const std::array<int64_t, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
    out_dims[i + 1] = fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<Real, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Calculate the shape of the temporary tensor for the full FFT and the
  // region we will slice from input given fft_shape. We slice input to
  // fft_shape on its inner-most dimensions, except the last (which we
  // slice to fft_shape[-1] / 2 + 1).
  Eigen::Tensor<Complex, FFTRank + 1, Eigen::RowMajor> full_fft(out_dims);

  // Calculate the starting point and range of the source of
  // negative frequency part.
  auto neg_sizes = in_dims;
  neg_sizes[FFTRank] = fft_shape[FFTRank - 1] - in_dims[FFTRank];
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_target_indices;
  neg_target_indices[FFTRank] = in_dims[FFTRank];

  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_start_indices;
  neg_start_indices[FFTRank] = 1;

  full_fft.slice(zero_start_indices, in_dims).device(device) = input;

  // First, conduct IFFTs on outer dimensions. We save computation (and
  // avoid touching uninitialized memory) by slicing full_fft to the
  // subregion we wrote input to.
  if (FFTRank > 1) {
    const auto outer_axes =
        Eigen::ArrayXi::LinSpaced(FFTRank - 1, 1, FFTRank - 1);
    full_fft.slice(zero_start_indices, in_dims).device(device) =
        full_fft.slice(zero_start_indices, in_dims)
            .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
  }

  // Reconstruct the full FFT by appending reversed and conjugated
  // spectrum as the negative frequency part.
  Eigen::array<bool, FFTRank + 1> reverse_last_axis;
  for (auto i = 0; i <= FFTRank; i++) {
    reverse_last_axis[i] = i == FFTRank;
  }

  if (neg_sizes[FFTRank] != 0) {
    full_fft.slice(neg_target_indices, neg_sizes).device(device) =
        full_fft.slice(neg_start_indices, neg_sizes)
            .reverse(reverse_last_axis)
            .conjugate();
  }

  auto inner_axis = Eigen::array<int, 1>{FFTRank};
  output.device(device) =
      full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(inner_axis);
}

template <int FFTRank, typename EigenDevice>
void EigenFftWithRank(const EigenDevice& device, void* out, void* operand,
                      FftType fft_type, bool double_precision,
                      int64_t input_batch, int64_t fft_length0,
                      int64_t fft_length1, int64_t fft_length2) {
  switch (fft_type) {
    case FftType::FFT:
      if (double_precision) {
        EigenFftC2C<true, FFTRank, EigenDevice, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2C<true, FFTRank, EigenDevice, complex64>(
            device, static_cast<complex64*>(out),
            static_cast<complex64*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      }
      break;
    case FftType::IFFT:
      if (double_precision) {
        EigenFftC2C<false, FFTRank, EigenDevice, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2C<false, FFTRank, EigenDevice, complex64>(
            device, static_cast<complex64*>(out),
            static_cast<complex64*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      }
      break;
    case FftType::RFFT:
      if (double_precision) {
        EigenFftR2C<FFTRank, EigenDevice, double, complex128>(
            device, static_cast<complex128*>(out),
            static_cast<double*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftR2C<FFTRank, EigenDevice, float, complex64>(
            device, static_cast<complex64*>(out), static_cast<float*>(operand),
            input_batch, fft_length0, fft_length1, fft_length2);
      }
      break;
    case FftType::IRFFT:
      if (double_precision) {
        EigenFftC2R<FFTRank, EigenDevice, complex128, double>(
            device, static_cast<double*>(out),
            static_cast<complex128*>(operand), input_batch, fft_length0,
            fft_length1, fft_length2);
      } else {
        EigenFftC2R<FFTRank, EigenDevice, complex64, float>(
            device, static_cast<float*>(out), static_cast<complex64*>(operand),
            input_batch, fft_length0, fft_length1, fft_length2);
      }
      break;
    default:
      // Unsupported FFT type
      abort();
  }
}

}  // namespace internal

template <typename EigenDevice>
void EigenFftImpl(const EigenDevice& device, void* out, void* operand,
                  internal::FftType fft_type, bool double_precision,
                  int32_t fft_rank, int64_t input_batch, int64_t fft_length0,
                  int64_t fft_length1, int64_t fft_length2) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSruntime_fft_implDTh mht_0(mht_0_v, 412, "", "./tensorflow/compiler/xla/service/cpu/runtime_fft_impl.h", "EigenFftImpl");

  switch (fft_rank) {
    case 1:
      internal::EigenFftWithRank<1, EigenDevice>(device, out, operand, fft_type,
                                                 double_precision, input_batch,
                                                 fft_length0, 0, 0);
      break;
    case 2:
      internal::EigenFftWithRank<2, EigenDevice>(device, out, operand, fft_type,
                                                 double_precision, input_batch,
                                                 fft_length0, fft_length1, 0);
      break;
    case 3:
      internal::EigenFftWithRank<3, EigenDevice>(
          device, out, operand, fft_type, double_precision, input_batch,
          fft_length0, fft_length1, fft_length2);
      break;
    default:
      // Unsupported FFT rank
      abort();
  }
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
