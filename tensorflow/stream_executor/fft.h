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

// Exposes the family of FFT routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsFft() for details.
//
// This abstraction makes it simple to entrain FFT operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<std::complex<float>> x =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  DeviceMemory<std::complex<float>> y =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  std::unique_ptr<Plan> plan =
//     stream_exec.AsFft()->Create1dPlan(&stream, 1024, Type::kC2CForward);
//  stream
//    .Init()
//    .ThenFft(plan.get(), x, &y);
//  SE_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned FFT
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOR_FFT_H_
#define TENSORFLOW_STREAM_EXECUTOR_FFT_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSfftDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSfftDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSfftDTh() {
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


#include <complex>
#include <memory>
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;
class ScratchAllocator;

namespace fft {

// Specifies FFT input and output types, and the direction.
// R, D, C, and Z stand for SP real, DP real, SP complex, and DP complex.
enum class Type {
  kInvalid,
  kC2CForward,
  kC2CInverse,
  kC2R,
  kR2C,
  kZ2ZForward,
  kZ2ZInverse,
  kZ2D,
  kD2Z
};

// FFT plan class. Each FFT implementation should define a plan class that is
// derived from this class. It does not provide any interface but serves
// as a common type that is used to execute the plan.
class Plan {
 public:
  virtual ~Plan() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSfftDTh mht_0(mht_0_v, 248, "", "./tensorflow/stream_executor/fft.h", "~Plan");
}
};

// FFT support interface -- this can be derived from a GPU executor when the
// underlying platform has an FFT library implementation available. See
// StreamExecutor::AsFft().
//
// This support interface is not generally thread-safe; it is only thread-safe
// for the CUDA platform (cuFFT) usage; host side FFT support is known
// thread-compatible, but not thread-safe.
class FftSupport {
 public:
  virtual ~FftSupport() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSfftDTh mht_1(mht_1_v, 263, "", "./tensorflow/stream_executor/fft.h", "~FftSupport");
}

  // Creates a 1d FFT plan.
  virtual std::unique_ptr<Plan> Create1dPlan(Stream *stream, uint64_t num_x,
                                             Type type, bool in_place_fft) = 0;

  // Creates a 2d FFT plan.
  virtual std::unique_ptr<Plan> Create2dPlan(Stream *stream, uint64_t num_x,
                                             uint64_t num_y, Type type,
                                             bool in_place_fft) = 0;

  // Creates a 3d FFT plan.
  virtual std::unique_ptr<Plan> Create3dPlan(Stream *stream, uint64_t num_x,
                                             uint64_t num_y, uint64 num_z,
                                             Type type, bool in_place_fft) = 0;

  // Creates a 1d FFT plan with scratch allocator.
  virtual std::unique_ptr<Plan> Create1dPlanWithScratchAllocator(
      Stream *stream, uint64_t num_x, Type type, bool in_place_fft,
      ScratchAllocator *scratch_allocator) = 0;

  // Creates a 2d FFT plan with scratch allocator.
  virtual std::unique_ptr<Plan> Create2dPlanWithScratchAllocator(
      Stream *stream, uint64_t num_x, uint64 num_y, Type type,
      bool in_place_fft, ScratchAllocator *scratch_allocator) = 0;

  // Creates a 3d FFT plan with scratch allocator.
  virtual std::unique_ptr<Plan> Create3dPlanWithScratchAllocator(
      Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z, Type type,
      bool in_place_fft, ScratchAllocator *scratch_allocator) = 0;

  // Creates a batched FFT plan.
  //
  // stream:          The GPU stream in which the FFT runs.
  // rank:            Dimensionality of the transform (1, 2, or 3).
  // elem_count:      Array of size rank, describing the size of each dimension.
  // input_embed, output_embed:
  //                  Pointer of size rank that indicates the storage dimensions
  //                  of the input/output data in memory. If set to null_ptr all
  //                  other advanced data layout parameters are ignored.
  // input_stride:    Indicates the distance (number of elements; same below)
  //                  between two successive input elements.
  // input_distance:  Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the input data.
  // output_stride:   Indicates the distance between two successive output
  //                  elements.
  // output_distance: Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the output data.
  virtual std::unique_ptr<Plan> CreateBatchedPlan(
      Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
      uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
      uint64_t output_stride, uint64 output_distance, Type type,
      bool in_place_fft, int batch_count) = 0;

  // Creates a batched FFT plan with scratch allocator.
  //
  // stream:          The GPU stream in which the FFT runs.
  // rank:            Dimensionality of the transform (1, 2, or 3).
  // elem_count:      Array of size rank, describing the size of each dimension.
  // input_embed, output_embed:
  //                  Pointer of size rank that indicates the storage dimensions
  //                  of the input/output data in memory. If set to null_ptr all
  //                  other advanced data layout parameters are ignored.
  // input_stride:    Indicates the distance (number of elements; same below)
  //                  between two successive input elements.
  // input_distance:  Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the input data.
  // output_stride:   Indicates the distance between two successive output
  //                  elements.
  // output_distance: Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the output data.
  virtual std::unique_ptr<Plan> CreateBatchedPlanWithScratchAllocator(
      Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,
      uint64_t input_stride, uint64 input_distance, uint64 *output_embed,
      uint64_t output_stride, uint64 output_distance, Type type,
      bool in_place_fft, int batch_count,
      ScratchAllocator *scratch_allocator) = 0;

  // Updates the plan's work area with space allocated by a new scratch
  // allocator. This facilitates plan reuse with scratch allocators.
  //
  // This requires that the plan was originally created using a scratch
  // allocator, as otherwise scratch space will have been allocated internally
  // by cuFFT.
  virtual void UpdatePlanWithScratchAllocator(
      Stream *stream, Plan *plan, ScratchAllocator *scratch_allocator) = 0;

  // Computes complex-to-complex FFT in the transform direction as specified
  // by direction parameter.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<float>> &input,
                     DeviceMemory<std::complex<float>> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<double>> &input,
                     DeviceMemory<std::complex<double>> *output) = 0;

  // Computes real-to-complex FFT in forward direction.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<float> &input,
                     DeviceMemory<std::complex<float>> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<double> &input,
                     DeviceMemory<std::complex<double>> *output) = 0;

  // Computes complex-to-real FFT in inverse direction.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<float>> &input,
                     DeviceMemory<float> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<double>> &input,
                     DeviceMemory<double> *output) = 0;

 protected:
  FftSupport() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSfftDTh mht_2(mht_2_v, 379, "", "./tensorflow/stream_executor/fft.h", "FftSupport");
}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(FftSupport);
};

// Macro used to quickly declare overrides for abstract virtuals in the
// fft::FftSupport base class. Assumes that it's emitted somewhere inside the
// ::stream_executor namespace.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES                   \
  std::unique_ptr<fft::Plan> Create1dPlan(Stream *stream, uint64_t num_x,      \
                                          fft::Type type, bool in_place_fft)   \
      override;                                                                \
  std::unique_ptr<fft::Plan> Create2dPlan(Stream *stream, uint64_t num_x,      \
                                          uint64_t num_y, fft::Type type,      \
                                          bool in_place_fft) override;         \
  std::unique_ptr<fft::Plan> Create3dPlan(                                     \
      Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z,              \
      fft::Type type, bool in_place_fft) override;                             \
  std::unique_ptr<fft::Plan> Create1dPlanWithScratchAllocator(                 \
      Stream *stream, uint64_t num_x, fft::Type type, bool in_place_fft,       \
      ScratchAllocator *scratch_allocator) override;                           \
  std::unique_ptr<fft::Plan> Create2dPlanWithScratchAllocator(                 \
      Stream *stream, uint64_t num_x, uint64 num_y, fft::Type type,            \
      bool in_place_fft, ScratchAllocator *scratch_allocator) override;        \
  std::unique_ptr<fft::Plan> Create3dPlanWithScratchAllocator(                 \
      Stream *stream, uint64_t num_x, uint64 num_y, uint64 num_z,              \
      fft::Type type, bool in_place_fft, ScratchAllocator *scratch_allocator)  \
      override;                                                                \
  std::unique_ptr<fft::Plan> CreateBatchedPlan(                                \
      Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,     \
      uint64_t input_stride, uint64 input_distance, uint64 *output_embed,      \
      uint64_t output_stride, uint64 output_distance, fft::Type type,          \
      bool in_place_fft, int batch_count) override;                            \
  std::unique_ptr<fft::Plan> CreateBatchedPlanWithScratchAllocator(            \
      Stream *stream, int rank, uint64_t *elem_count, uint64 *input_embed,     \
      uint64_t input_stride, uint64 input_distance, uint64 *output_embed,      \
      uint64_t output_stride, uint64 output_distance, fft::Type type,          \
      bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) \
      override;                                                                \
  void UpdatePlanWithScratchAllocator(Stream *stream, fft::Plan *plan,         \
                                      ScratchAllocator *scratch_allocator)     \
      override;                                                                \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<float>> &input,                   \
             DeviceMemory<std::complex<float>> *output) override;              \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<double>> &input,                  \
             DeviceMemory<std::complex<double>> *output) override;             \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<float> &input,                                 \
             DeviceMemory<std::complex<float>> *output) override;              \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<double> &input,                                \
             DeviceMemory<std::complex<double>> *output) override;             \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<float>> &input,                   \
             DeviceMemory<float> *output) override;                            \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<double>> &input,                  \
             DeviceMemory<double> *output) override;

}  // namespace fft
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_FFT_H_
