/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_
#define TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSgpu_prim_helpersDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_prim_helpersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgpu_prim_helpersDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

namespace detail {

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const T size,
                                T* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_prim_helpersDTh mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/gpu_prim_helpers.h", "RangeInitKernel");

  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
template <typename T>
Status RangeInit(const Eigen::GpuDevice& d, const T start, const T delta,
                 const T size, T* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_prim_helpersDTh mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/gpu_prim_helpers.h", "RangeInit");

  if (size == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  return GpuLaunchKernel(RangeInitKernel<T>, config.block_count,
                         config.thread_per_block, 0, d.stream(), start, delta,
                         size, out);
}

}  // namespace detail

// Computes keys_out = sorted(keys_in), and indices_out = argsort(keys_in).
// If keys_out is not required, it can be set to nullptr.
// If indices_in is nullptr, the range of input indices [0, size) will be used.
template <typename Tkey, typename Tindex>
Status GpuRadixSort(OpKernelContext* context, int size, const Tkey* keys_in,
                    Tkey* keys_out,            // Optional
                    const Tindex* indices_in,  // Optional
                    Tindex* indices_out, int num_bits = sizeof(Tkey) * 8) {
  if (size == 0) return Status::OK();
  if (num_bits == 0) {
    // Workaround for CUB failing when begin_bit = end_bit = 0 (e.g., when all
    // keys are 0, so no sorting is needed).
    se::Stream* stream = context->op_device_context()->stream();
    if (keys_out) {
      // Copy keys_in to keys_out.
      size_t num_bytes = size * sizeof(Tkey);
      se::DeviceMemoryBase src(const_cast<Tkey*>(keys_in), num_bytes);
      se::DeviceMemoryBase dst(keys_out, num_bytes);
      if (!stream->ThenMemcpy(&dst, src, num_bytes).ok()) {
        return errors::Internal("Failed to copy keys_in to keys_out");
      }
    }
    if (indices_in) {
      // Copy indices_in to indices_out.
      size_t num_bytes = size * sizeof(Tindex);
      se::DeviceMemoryBase src(const_cast<Tindex*>(indices_in), num_bytes);
      se::DeviceMemoryBase dst(indices_out, num_bytes);
      if (!stream->ThenMemcpy(&dst, src, num_bytes).ok()) {
        return errors::Internal("Failed to copy indices_in to indices_out");
      }
    } else {
      // Set output indices to range.
      const Eigen::GpuDevice& device =
          context->eigen_device<Eigen::GpuDevice>();
      TF_RETURN_IF_ERROR(detail::RangeInit(device, Tindex(0), Tindex(1),
                                           Tindex(size), indices_out));
    }
    return Status::OK();
  }
  // Allocate temporary inputs/outputs if necessary.
  Tensor tmp_indices_in;
  if (!indices_in) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tindex>::value, TensorShape({size}), &tmp_indices_in));
    Tindex* mutable_indices_in = tmp_indices_in.flat<Tindex>().data();
    indices_in = mutable_indices_in;
    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    // Initialize indices_in to the input index range.
    TF_RETURN_IF_ERROR(detail::RangeInit(device, Tindex(0), Tindex(1),
                                         Tindex(size), mutable_indices_in));
  }
  Tensor tmp_keys_out;
  if (!keys_out) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Tkey>::value, TensorShape({size}), &tmp_keys_out));
    keys_out = tmp_keys_out.flat<Tkey>().data();
  }
  // Determine temporary device storage requirements.
  Tensor temp_storage;
  size_t temp_storage_bytes = 0;
  const auto& cu_stream = GetGpuStream(context);
  auto err = gpuprim::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, keys_in, keys_out, indices_in, indices_out,
      size, /*begin_bit=*/0, /*end_bit=*/num_bits, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  // Allocate temporary storage.
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  // Sort indices by keys.
  err = gpuprim::DeviceRadixSort::SortPairs(
      temp_storage.flat<int8>().data(), temp_storage_bytes, keys_in, keys_out,
      indices_in, indices_out, size, /*begin_bit=*/0, /*end_bit=*/num_bits,
      cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceRadixSort::SortPairs, "
        "temp_storage_bytes: ",
        temp_storage_bytes, "status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

template <typename InputIteratorT, typename OutputIteratorT>
Status GpuInclusivePrefixSum(OpKernelContext* context, int size,
                             InputIteratorT input, OutputIteratorT output) {
  static_assert(
      !std::is_same<typename std::remove_reference<decltype(*input)>::type,
                    bool>::value,
      "GpuInclusivePrefixSum does not work correct with booleans, please use "
      "TransformInputIterator to explicitly cast to an integer.");
  if (size == 0) return Status::OK();
  const auto& cu_stream = GetGpuStream(context);
  size_t temp_storage_bytes;
  auto err = gpuprim::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                               input, output, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::InclusiveSum to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceScan::InclusiveSum(temp_storage.flat<int8>().data(),
                                          temp_storage_bytes, input, output,
                                          size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceScan::InclusiveSum, "
        "temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

// Note that this behaves deterministically for repeat calls on the same device.
template <typename InputIteratorT, typename OutputIteratorT,
          typename OffsetIteratorT, typename ReduceOp, typename T>
Status GpuSegmentedReduce(
    OpKernelContext* context, int num_segments, ReduceOp reduce_op,
    const T& initial_value,
    InputIteratorT input,             // [any]
    OffsetIteratorT segment_offsets,  // [num_segments + 1]
    OutputIteratorT output) {         // [num_segments]
  if (num_segments == 0) return Status::OK();
  const auto& cu_stream = GetGpuStream(context);
  size_t temp_storage_bytes;
  auto err = gpuprim::DeviceSegmentedReduce::Reduce(
      nullptr, temp_storage_bytes, input, output, num_segments, segment_offsets,
      segment_offsets + 1, reduce_op, initial_value, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSegmentedReduce::Reduce to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceSegmentedReduce::Reduce(
      temp_storage.flat<int8>().data(), temp_storage_bytes, input, output,
      num_segments, segment_offsets, segment_offsets + 1, reduce_op,
      initial_value, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSegmentedReduce::Reduce"
        ", temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

template <typename InputIteratorT, typename FlagIteratorT,
          typename OutputIteratorT, typename NumSelectedT = int>
Status GpuSelectFlagged(OpKernelContext* context, int size,
                        InputIteratorT input, FlagIteratorT flags,
                        OutputIteratorT output,
                        NumSelectedT* out_num_selected = nullptr) {
  const auto& cu_stream = GetGpuStream(context);
  Tensor out_num_selected_t;
  if (!out_num_selected) {
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DataTypeToEnum<NumSelectedT>::value,
                               TensorShape({}), &out_num_selected_t));
    out_num_selected = out_num_selected_t.scalar<NumSelectedT>().data();
  }
  size_t temp_storage_bytes;
  auto err =
      gpuprim::DeviceSelect::Flagged(nullptr, temp_storage_bytes, input, flags,
                                     output, out_num_selected, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSelect::Flagged to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceSelect::Flagged(temp_storage.flat<int8>().data(),
                                       temp_storage_bytes, input, flags, output,
                                       out_num_selected, size, cu_stream);
  if (err != 0) {
    return errors::Internal(
        "Failed to launch gpuprim::DeviceSelect::Flagged, temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_HELPERS_H_
