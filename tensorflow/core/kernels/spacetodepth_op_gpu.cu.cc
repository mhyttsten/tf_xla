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
class MHTracer_DTPStensorflowPScorePSkernelsPSspacetodepth_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetodepth_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSspacetodepth_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/spacetodepth_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Space2Depth kernel for FORMAT_NHWC.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
__global__ void S2D_NHWC(const int32 nthreads,
                         const dtype* __restrict__ input_ptr,
                         const int block_size, const int batch_size,
                         const int input_height, const int input_width,
                         const int input_depth, const int output_height,
                         const int output_width, const int output_depth,
                         dtype* __restrict__ output_ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetodepth_op_gpuDTcuDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/spacetodepth_op_gpu.cu.cc", "S2D_NHWC");

  GPU_1D_KERNEL_LOOP(inp_idx, nthreads) {
    // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
    const int d = inp_idx % input_depth;
    const int inp_idx2 = inp_idx / input_depth;
    const int w = inp_idx2 % input_width;
    const int inp_idx3 = inp_idx2 / input_width;
    const int h = inp_idx3 % input_height;
    const int b = inp_idx3 / input_height;

    const int out_h = h / block_size;
    const int offset_h = h % block_size;
    const int out_w = w / block_size;
    const int offset_w = w % block_size;
    const int offset_d = (offset_h * block_size + offset_w) * input_depth;
    const int out_d = d + offset_d;
    const int out_idx =
        out_d +
        output_depth * (out_w + output_width * (out_h + output_height * b));
    *(output_ptr + out_idx) = ldg(input_ptr + inp_idx);
  }
}

// Space2Depth kernel for FORMAT_NCHW.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
__global__ void S2D_NCHW(const int32 nthreads,
                         const dtype* __restrict__ input_ptr,
                         const int block_size, const int output_width,
                         const int input_depth_by_output_height,
                         dtype* __restrict__ output_ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSspacetodepth_op_gpuDTcuDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/kernels/spacetodepth_op_gpu.cu.cc", "S2D_NCHW");

  GPU_1D_KERNEL_LOOP(input_idx, nthreads) {
    // We assume both the input and output are packed NCHW tensors.
    // input_idx represents an index within the flattened input tensor.
    // We can consider the block width and height as extra tensor dimensions,
    // then isolate the relevant components of input_idx and recombine them to
    // form output_idx. The layout transform performed is:
    // n, iC, oY, bY, oX, bX    (== input_idx)   to
    // n, bY, bX, iC, oY, oX    (== output_idx).

    const int n_iC_oY_bY_oX = input_idx / block_size;
    const int bX = input_idx - n_iC_oY_bY_oX * block_size;

    const int n_iC_oY_bY = n_iC_oY_bY_oX / output_width;
    const int oX = n_iC_oY_bY_oX - n_iC_oY_bY * output_width;

    const int n_iC_oY = n_iC_oY_bY / block_size;
    const int bY = n_iC_oY_bY - n_iC_oY * block_size;

    const int n = n_iC_oY / input_depth_by_output_height;
    const int iC_oY = n_iC_oY - n * input_depth_by_output_height;

    const int output_idx = oX + (((n * block_size + bY) * block_size + bX) *
                                     input_depth_by_output_height +
                                 iC_oY) *
                                    output_width;

    *(output_ptr + output_idx) = ldg(input_ptr + input_idx);
  }
}

// Space2Depth kernel for FORMAT_NCHW using a loop over block area.
// See 'spacetodepth_op.h' for functional specification.
template <typename dtype, int block_size>
__global__ void S2D_NCHW_LOOP(const int32 nthreads,
                              const dtype* __restrict__ input,
                              const int output_width, const int input_width,
                              const int input_depth_by_output_area,
                              const int output_depth_by_output_area,
                              dtype* __restrict__ output) {
  GPU_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // We will be converting the image from ordering:
    // n, iC, oY, bY, oX, bX   (== input index) to
    // n, bY, bX, iC, oY, oX   (== output index)

    // We assume thread_idx encodes n_iC_oY_oX, and use an unrolled loop over
    // bY and bX coordinates within the block. This kernel gets a small
    // performance improvement compared with S2D_NCHW due to a denser access
    // pattern on the input side. (Note: the equivalent D2S kernel gets a larger
    // improvement as a denser pattern on the output side makes more
    // difference).

    const int n_iC_oY = thread_idx / output_width;
    const int oX = thread_idx - n_iC_oY * output_width;
    const int n = thread_idx / input_depth_by_output_area;
    const int iC_oY_oX = thread_idx - n * input_depth_by_output_area;

    // Recombine the components and apply to the input and output pointers.
    auto input_ptr = input + (n_iC_oY * input_width + oX) * block_size;
    auto output_ptr = output + n * output_depth_by_output_area + iC_oY_oX;

#pragma unroll
    // Copy a patch of data to the output batch image.
    for (int bY = 0; bY < block_size; ++bY) {
#pragma unroll
      for (int bX = 0; bX < block_size; ++bX) {
        output_ptr[(bY * block_size + bX) * input_depth_by_output_area] =
            ldg(input_ptr + bY * input_width + bX);
      }
    }
  }
}

// Specialization of SpaceToDepthOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NHWC> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int input_depth = input.dimension(3);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int output_depth = output.dimension(3);

    const int total_count =
        batch_size * input_height * input_width * input_depth;
    if (total_count == 0) {
      return;
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        S2D_NHWC<T>, config.block_count, config.thread_per_block, 0, d.stream(),
        config.virtual_thread_count, input.data(), block_size, batch_size,
        input_height, input_width, input_depth, output_height, output_width,
        output_depth, output.data()));
  }
  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    LOG(FATAL) << "5-D tensors should not be used with NHWC format";
  }
};

template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NCHW> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_depth = input.dimension(1);
    const int output_depth = output.dimension(1);
    const int output_height = output.dimension(2);
    const int output_width = output.dimension(3);
    const int output_area = output_width * output_height;
    const int output_depth_by_output_area = output_depth * output_area;

    // We improve performance by generating instantiations of the loop kernel
    // for the most common block sizes.
    if (block_size <= 4) {
      const int input_width = input.dimension(3);
      const int input_depth_by_output_area = input_depth * output_area;
      const int total_count = batch_size * input_depth_by_output_area;
      if (total_count == 0) {
        return;
      }
      GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
      switch (block_size) {
        case 2:
          TF_CHECK_OK(GpuLaunchKernel(
              S2D_NCHW_LOOP<T, 2>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), output_width,
              input_width, input_depth_by_output_area,
              output_depth_by_output_area, output.data()));
          return;
        case 3:
          TF_CHECK_OK(GpuLaunchKernel(
              S2D_NCHW_LOOP<T, 3>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), output_width,
              input_width, input_depth_by_output_area,
              output_depth_by_output_area, output.data()));
          return;
        case 4:
          TF_CHECK_OK(GpuLaunchKernel(
              S2D_NCHW_LOOP<T, 4>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), output_width,
              input_width, input_depth_by_output_area,
              output_depth_by_output_area, output.data()));
          return;
      }
    }

    // Other block sizes are processed by the generic kernel.
    const int total_count = batch_size * output_depth_by_output_area;
    if (total_count == 0) {
      return;
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        S2D_NCHW<T>, config.block_count, config.thread_per_block, 0, d.stream(),
        config.virtual_thread_count, input.data(), block_size, output_width,
        input_depth * output_height, output.data()));
  }
  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    LOG(FATAL) << "5-D tensors should not be used with NCHW format";
  }
};
}  // end namespace functor

// Instantiate the GPU implementations for float.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NCHW>;
template struct functor::SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NHWC>;

// Instantiate the GPU implementations for Eigen::half.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, Eigen::half,
                                               FORMAT_NCHW>;
template struct functor::SpaceToDepthOpFunctor<GPUDevice, Eigen::half,
                                               FORMAT_NHWC>;

// Instantiate the GPU implementations for uint8.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, uint8, FORMAT_NCHW>;
template struct functor::SpaceToDepthOpFunctor<GPUDevice, uint8, FORMAT_NHWC>;

// NCHW_VECT_C with 4 x qint8 can be treated as NCHW int32.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, int32, FORMAT_NCHW>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
