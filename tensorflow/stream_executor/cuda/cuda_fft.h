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

// CUDA-specific support for FFT functionality -- this wraps the cuFFT library
// capabilities, and is only included into CUDA implementation code -- it will
// not introduce cuda headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_
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
class MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh() {
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


#include "third_party/gpus/cuda/include/cufft.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace stream_executor {

class Stream;

namespace gpu {

class GpuExecutor;

// Opaque and unique indentifier for the cuFFT plugin.
extern const PluginId kCuFftPlugin;

// CUDAFftPlan uses deferred initialization. Only a single call of
// Initialize() is allowed to properly create cufft plan and set member
// variable is_initialized_ to true. Newly added interface that uses member
// variables should first check is_initialized_ to make sure that the values of
// member variables are valid.
class CUDAFftPlan : public fft::Plan {
 public:
  CUDAFftPlan()
      : parent_(nullptr),
        plan_(-1),
        fft_type_(fft::Type::kInvalid),
        scratch_(nullptr),
        scratch_size_bytes_(0),
        is_initialized_(false),
        scratch_allocator_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_0(mht_0_v, 223, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "CUDAFftPlan");
}
  ~CUDAFftPlan() override;

  // Get FFT direction in cuFFT based on FFT type.
  int GetFftDirection() const;
  cufftHandle GetPlan() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_1(mht_1_v, 231, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "GetPlan");

    if (IsInitialized()) {
      return plan_;
    } else {
      LOG(FATAL) << "Try to get cufftHandle value before initialization.";
    }
  }

  // Initialize function for batched plan
  port::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                          uint64_t* elem_count, uint64_t* input_embed,
                          uint64_t input_stride, uint64 input_distance,
                          uint64_t* output_embed, uint64_t output_stride,
                          uint64_t output_distance, fft::Type type,
                          int batch_count, ScratchAllocator* scratch_allocator);

  // Initialize function for 1d,2d, and 3d plan
  port::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                          uint64_t* elem_count, fft::Type type,
                          ScratchAllocator* scratch_allocator);

  port::Status UpdateScratchAllocator(Stream *stream,
                                      ScratchAllocator *scratch_allocator);

  ScratchAllocator* GetScratchAllocator() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_2(mht_2_v, 258, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "GetScratchAllocator");
 return scratch_allocator_; }

 protected:
  bool IsInitialized() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_3(mht_3_v, 264, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "IsInitialized");
 return is_initialized_; }

 private:
  GpuExecutor* parent_;
  cufftHandle plan_;
  fft::Type fft_type_;
  DeviceMemory<uint8> scratch_;
  size_t scratch_size_bytes_;
  bool is_initialized_;
  ScratchAllocator* scratch_allocator_;
};

// FFT support for CUDA platform via cuFFT library.
//
// This satisfies the platform-agnostic FftSupport interface.
//
// Note that the cuFFT handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the cuFFT handle when a
// CUDA context is active.
//
// Thread-safe. The CUDA context associated with all operations is the CUDA
// context of parent_, so all context is explicit.
class CUDAFft : public fft::FftSupport {
 public:
  explicit CUDAFft(GpuExecutor* parent) : parent_(parent) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_4(mht_4_v, 292, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "CUDAFft");
}
  ~CUDAFft() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPScudaPScuda_fftDTh mht_5(mht_5_v, 296, "", "./tensorflow/stream_executor/cuda/cuda_fft.h", "~CUDAFft");
}

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  GpuExecutor* parent_;

  // Two helper functions that execute dynload::cufftExec?2?.

  // This is for complex to complex FFT, when the direction is required.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                  FuncT cufft_exec,
                                  const DeviceMemory<InputT> &input,
                                  DeviceMemory<OutputT> *output);

  // This is for complex to real or real to complex FFT, when the direction
  // is implied.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftInternal(Stream *stream, fft::Plan *plan, FuncT cufft_exec,
                     const DeviceMemory<InputT> &input,
                     DeviceMemory<OutputT> *output);

  SE_DISALLOW_COPY_AND_ASSIGN(CUDAFft);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_
