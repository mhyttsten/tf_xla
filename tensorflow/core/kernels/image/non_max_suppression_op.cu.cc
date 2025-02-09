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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc() {
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
#include "tensorflow/core/kernels/image/non_max_suppression_op.h"

#include <limits>

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {
namespace {

struct
#if GOOGLE_CUDA
    __align__(16)
#endif
        Box {
  float x1, y1, x2, y2;
};
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

// This is the width of the bitmask for masking boxes for each thread.  This
// needs to be a multiple of 2 (a POD width usually) so that division and modulo
// can be implemented as bit operations during host selection.
constexpr int kNmsBoxesPerThread = 8 * sizeof(int);

// Helper to calculate modulo mask and shift bits.
//
// For kNmsBoxesPerThread=32 ModuloMask will be 31, i.e 0x1F, thus
// i % 32 == i & 31. Similarly ShiftBits will be 5 so that
// i / 32 == i >> 5. Using these bit operations should reduce the stall on host
// thread.
constexpr int NumBits(int n) { return (n == 0) ? 0 : NumBits(n >> 1) + 1; }
constexpr int kNmsBoxesPerThreadModuloMask = kNmsBoxesPerThread - 1;
constexpr int kNmsBoxesPerThreadShiftBits =
    NumBits(kNmsBoxesPerThreadModuloMask);

constexpr int kNmsBlockDim = 16;
constexpr int kNmsBlockDimMax = 128;
constexpr int kNmsChunkSize = 2000;

template <typename T>
__device__ EIGEN_STRONG_INLINE void Swap(T& a, T& b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_0(mht_0_v, 236, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Swap");

  T c(a);
  a = b;
  b = c;
}

// Check whether two boxes have an IoU greater than threshold.
template <typename T>
__device__ EIGEN_STRONG_INLINE bool OverThreshold(const Box* a, const Box* b,
                                                  const float a_area,
                                                  const T iou_threshold) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "OverThreshold");

  const float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
  if (a_area == 0.0f || b_area == 0.0f) return false;
  const float xx1 = fmaxf(a->x1, b->x1);
  const float yy1 = fmaxf(a->y1, b->y1);
  const float xx2 = fminf(a->x2, b->x2);
  const float yy2 = fminf(a->y2, b->y2);

  // fdimf computes the positive difference between xx2+1 and xx1.
  const float w = fdimf(xx2, xx1);
  const float h = fdimf(yy2, yy1);
  const float intersection = w * h;

  // Testing for aa/bb > t
  // eq with aa > bb*t (b is !=0)
  // avoiding divisions.
  const float aa = intersection;
  const float bb = a_area + b_area - intersection;
  const float bt = bb * iou_threshold;
  return aa > bt;
}

template <bool flip_box>
__device__ EIGEN_STRONG_INLINE void Flipped(Box& box);

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<false>(Box& box) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_2(mht_2_v, 278, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Flipped<false>");
}

template <>
__device__ EIGEN_STRONG_INLINE void Flipped<true>(Box& box) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_3(mht_3_v, 284, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Flipped<true>");

  if (box.x1 > box.x2) Swap(box.x1, box.x2);
  if (box.y1 > box.y2) Swap(box.y1, box.y2);
}
template <typename T>
__device__ EIGEN_STRONG_INLINE bool CheckBit(T* bit_mask, uint32 bit) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "CheckBit");

  constexpr uint32 kNumBits = 8 * sizeof(T);
  return (bit_mask[bit / kNumBits] >> (bit % kNumBits)) & 1;
}

// Produce a global bitmask (result_mask) of selected boxes from bitmask
// generated by NMSKernel. Abort early if max_boxes boxes are selected. Bitmask
// is num_boxes*bit_mask_len bits indicating whether to keep or remove a box.
__global__ void NMSReduce(const int* bitmask, const int bit_mask_len,
                          const int num_boxes, const int max_boxes,
                          char* result_mask) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("result_mask: \"" + (result_mask == nullptr ? std::string("nullptr") : std::string((char*)result_mask)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_5(mht_5_v, 306, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "NMSReduce");

  extern __shared__ int local[];
  // Set global mask to accept all boxes.
  for (int box : GpuGridRangeX(bit_mask_len)) {
    local[box] = 0xFFFFFFFF;
  }
  __syncthreads();

  int accepted_boxes = 0;
  for (int box = 0; box < num_boxes - 1; ++box) {
    // If current box is masked by an earlier box, skip it.
    if (!CheckBit(local, box)) {
      continue;
    }
    accepted_boxes += 1;
    int offset = box * bit_mask_len;
    // Update global mask with current box's mask.
    for (int b : GpuGridRangeX(bit_mask_len)) {
      local[b] &= ~bitmask[offset + b];
    }
    __syncthreads();
    if (accepted_boxes > max_boxes) break;
  }

  // Copy global mask to result_max char array, which we use in
  // cub::DeviceSelect later.  In theory we could skip this test and use the
  // bitmask in DeviceSelect directly, but in practice this part of the kernel
  // is very cheap anyway.
  for (int box : GpuGridRangeX(num_boxes)) {
    result_mask[box] = CheckBit(local, box);
  }
}

// For each box, compute a bitmask of boxes which has an overlap with given box
// above threshold.
//
// Starting from highest scoring box, mark any box which has IoU>threshold with
// given box. Each thread processes a kNmsBoxesPerThread boxes per stride, and
// each box has bitmask of overlaps of length bit_mask_len.
//
// If flip_box is true boxes may have x1>x2 and or y1>y2. If so change the
// coordinates such that for all boxes x1<x2 and y1<y2. Else boxes should have
// x1<x2 and y1<y2.
template <bool flip_box>
__launch_bounds__(kNmsBlockDim* kNmsBlockDim, 4) __global__
    void NMSKernel(const Box* d_desc_sorted_boxes, const int num_boxes,
                   const float iou_threshold, const int bit_mask_len,
                   int* d_delete_mask) {
  // Storing boxes used by this CUDA block in the shared memory.
  __shared__ Box shared_i_boxes[kNmsBlockDim];
  // Same thing with areas
  __shared__ float shared_i_areas[kNmsBlockDim];
  // The condition of the for loop is common to all threads in the block.
  // This is necessary to be able to call __syncthreads() inside of the loop.
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < num_boxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i = i_block_offset + threadIdx.x;
    if (i < num_boxes) {
      // One 1D line load the boxes for x-dimension.
      if (threadIdx.y == 0) {
        Box box = d_desc_sorted_boxes[i];
        Flipped<flip_box>(box);
        shared_i_boxes[threadIdx.x] = box;
        shared_i_areas[threadIdx.x] = (box.x2 - box.x1) * (box.y2 - box.y1);
      }
    }
    __syncthreads();
    for (int j_thread_offset =
             kNmsBoxesPerThread * (blockIdx.y * blockDim.y + threadIdx.y);
         j_thread_offset < num_boxes;
         j_thread_offset += kNmsBoxesPerThread * blockDim.y * gridDim.y) {
      // Note : We can do everything using multiplication,
      // and use fp16 - we are comparing against a low precision
      // threshold.
      int above_threshold = 0;
      // Make sure that threads are within valid domain.
      bool valid = false;
      // Loop over the next kNmsBoxesPerThread boxes and set corresponding bit
      // if it is overlapping with current box
      for (int ib = 0; ib < kNmsBoxesPerThread; ++ib) {
        // This thread will compare Box i and Box j.
        const int j = j_thread_offset + ib;
        if (i >= j || i >= num_boxes || j >= num_boxes) continue;
        valid = true;
        Box j_box = d_desc_sorted_boxes[j];
        const Box i_box = shared_i_boxes[threadIdx.x];
        Flipped<flip_box>(j_box);
        if (OverThreshold<float>(&i_box, &j_box, shared_i_areas[threadIdx.x],
                                 iou_threshold)) {
          // we have score[j] <= score[i].
          above_threshold |= (1U << ib);
        }
      }
      if (valid) {
        d_delete_mask[i * bit_mask_len + j_thread_offset / kNmsBoxesPerThread] =
            above_threshold;
      }
    }
    __syncthreads();  // making sure everyone is done reading shared memory.
  }
}
// Variadic template helpers for Index selecting multiple arrays at the same
// time
template <typename Index>
__device__ EIGEN_STRONG_INLINE void SelectHelper(const Index i_selected,
                                                 const Index i_original) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_6(mht_6_v, 414, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "SelectHelper");
}

template <typename Index, typename T, typename... Args>
__device__ EIGEN_STRONG_INLINE void SelectHelper(const Index i_selected,
                                                 const Index i_original,
                                                 const T* original, T* selected,
                                                 Args... args) {
  selected[i_selected] = original[i_original];
  SelectHelper(i_selected, i_original, args...);
}

// Helper template to select elements from original arrays using the index
// mapping and store into selected array. Each array sharing same mapping need
// to be passed as pairs of pointers to original and selected arrays. For
// selecting 2 arrays call would be
// IndexMultiSelect(num_elements, indices, original1 ,selected1, original2,
// selected2).
template <typename Index, typename T, typename... Args>
__global__ void IndexMultiSelect(const int num_elements, const Index* indices,
                                 const T* original, T* selected, Args... args) {
  for (const int idx : GpuGridRangeX(num_elements)) {
    SelectHelper(idx, indices[idx], original, selected, args...);
  }
}

template <typename T>
__global__ void Iota(const int num_elements, const T offset, T* to_fill) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_7(mht_7_v, 443, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Iota");

  for (int idx : GpuGridRangeX(num_elements)) {
    to_fill[idx] = static_cast<T>(idx) + offset;
  }
}

// TensorFlow with nvcc doesn't build with --extended-lambda, so we have to use
// an explicit functor instead of a device lambda.
struct GreaterThanCubOp {
  float threshold_;
  __host__ __device__ __forceinline__ GreaterThanCubOp(float threshold)
      : threshold_(threshold) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_8(mht_8_v, 457, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "GreaterThanCubOp");
}
  __host__ __device__ __forceinline__ bool operator()(const float& val) const {
    return (val > threshold_);
  }
};

// Uses DeviceSelect::If to count number of elements.
//
// (It might be better to use DeviceReduce::Sum with a custom iterator to do the
// count.  But in practice SelectIf is quite fast.)
template <typename Op>
StatusOr<int> CountIf(OpKernelContext* context, const float* dev_array,
                      const Op& op, int num_elements) {
  size_t workspace_size = 0;
  auto cuda_stream = tensorflow::GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  gpuprim::DeviceSelect::If(nullptr, workspace_size,
                            static_cast<float*>(nullptr),
                            static_cast<float*>(nullptr),
                            static_cast<int*>(nullptr), num_elements, op);

  Tensor scratch_output;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_elements}), &scratch_output));

  Tensor workspace;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)workspace_size}), &workspace));

  // num_selected is a host pinned tensor.  The GPU kernel can write to it
  // directly, instead of writing to GPU memory and then copying down to
  // num_selected, saving us a small D2H memcpy.  We've observed that even small
  // D2H copies on the compute stream can have an outsized effect on latency.
  Tensor num_selected;
  AllocatorAttributes pinned_alloc_attrs;
  pinned_alloc_attrs.set_on_host(true);
  pinned_alloc_attrs.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({1}), &num_selected, pinned_alloc_attrs));

  gpuEvent_t copy_done;
  TF_RETURN_IF_CUDA_ERROR(
      gpuEventCreateWithFlags(&copy_done, gpuEventDisableTiming));
  TF_RETURN_IF_CUDA_ERROR(gpuprim::DeviceSelect::If(
      workspace.flat<int8>().data(), workspace_size, dev_array,
      scratch_output.flat<float>().data(), num_selected.flat<int32>().data(),
      num_elements, op, cuda_stream));
  TF_RETURN_IF_CUDA_ERROR(gpuEventRecord(copy_done, device.stream()));
  TF_RETURN_IF_CUDA_ERROR(gpuEventSynchronize(copy_done));
  return *num_selected.flat<int32>().data();
}

Status DoNMS(OpKernelContext* context, const Tensor& boxes,
             const Tensor& scores, const int64_t max_output_size,
             const float iou_threshold_val, const float score_threshold,
             bool pad_to_max_output, int* num_saved_outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_9(mht_9_v, 515, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "DoNMS");

  int num_boxes = boxes.dim_size(0);
  size_t cub_sort_temp_storage_bytes = 0;
  auto cuda_stream = GetGpuStream(context);
  auto device = context->eigen_gpu_device();
  // Calling cub with nullptrs as inputs will make it return
  // workspace size needed for the operation instead of doing the operation.
  // In this specific instance, cub_sort_temp_storage_bytes will contain the
  // necessary workspace size for sorting after the call.
  if (num_boxes == 0) {
    Tensor* output_indices = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({0}), &output_indices));
    return Status::OK();
  }

  cudaError_t cuda_ret = gpuprim::DeviceRadixSort::SortPairsDescending(
      nullptr, cub_sort_temp_storage_bytes,
      static_cast<float*>(nullptr),  // scores
      static_cast<float*>(nullptr),  // sorted scores
      static_cast<int*>(nullptr),    // input indices
      static_cast<int*>(nullptr),    // sorted indices
      num_boxes,                     // num items
      0, 8 * sizeof(float),          // sort all bits
      cuda_stream);
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  Tensor d_cub_sort_buffer;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)cub_sort_temp_storage_bytes}),
      &d_cub_sort_buffer));
  Tensor d_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_indices));
  Tensor d_sorted_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_sorted_indices));
  Tensor d_selected_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_selected_indices));
  Tensor d_sorted_scores;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes}), &d_sorted_scores));
  Tensor d_sorted_boxes;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_FLOAT, TensorShape({num_boxes, 4}), &d_sorted_boxes));

  // this will return sorted scores and their indices
  auto config = GetGpuLaunchConfig(num_boxes, device);
  // initialize box and score indices
  TF_CHECK_OK(GpuLaunchKernel(Iota<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, 0,
                              d_indices.flat<int>().data()));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  cuda_ret = gpuprim::DeviceRadixSort::SortPairsDescending(
      d_cub_sort_buffer.flat<int8>().data(), cub_sort_temp_storage_bytes,
      scores.flat<float>().data(), d_sorted_scores.flat<float>().data(),
      d_indices.flat<int>().data(), d_sorted_indices.flat<int>().data(),
      num_boxes, 0,
      8 * sizeof(float),  // sort all bits
      cuda_stream);
  TF_RETURN_IF_CUDA_ERROR(cuda_ret);

  // get pointers for easy access
  const float4* original_boxes =
      reinterpret_cast<const float4*>(boxes.flat<float>().data());
  float4* sorted_boxes =
      reinterpret_cast<float4*>(d_sorted_boxes.flat<float>().data());
  const int* sorted_indices = d_sorted_indices.flat<int>().data();
  // sort boxes using indices
  TF_CHECK_OK(GpuLaunchKernel(IndexMultiSelect<int, float4>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, sorted_indices,
                              original_boxes, sorted_boxes));
  int limited_num_boxes = num_boxes;
  // filter boxes by scores if nms v3
  if (score_threshold > std::numeric_limits<float>::lowest()) {
    GreaterThanCubOp score_limit(score_threshold);
    TF_ASSIGN_OR_RETURN(limited_num_boxes,
                        CountIf(context, d_sorted_scores.flat<float>().data(),
                                score_limit, num_boxes));
    if (limited_num_boxes == 0) {
      Tensor* output_indices = nullptr;
      VLOG(1) << "Number of boxes above score threshold " << score_threshold
              << " is 0";
      int len_output = pad_to_max_output ? max_output_size : 0;
      *num_saved_outputs = 0;
      TF_RETURN_IF_ERROR(context->allocate_output(0, TensorShape({len_output}),
                                                  &output_indices));
      return Status::OK();
    } else {
      VLOG(2) << "Number of boxes above threshold=" << score_threshold << " is "
              << limited_num_boxes;
    }
  }
  int num_to_keep = 0;
  // There is no guarantee that boxes are given in the for x1<x2 and/or y1<y2,
  // flip boxes if necessary!
  const bool flip_boxes = true;
  auto status = NmsGpu(d_sorted_boxes.flat<float>().data(), limited_num_boxes,
                       iou_threshold_val, d_selected_indices.flat<int>().data(),
                       &num_to_keep, context, max_output_size, flip_boxes);
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  if (!status.ok()) {
    context->SetStatus(status);
    return status;
  }
  Tensor* output_indices = nullptr;
  int num_outputs = std::min(num_to_keep, (int)max_output_size);  // no padding!
  if (pad_to_max_output && num_outputs != max_output_size) {
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({max_output_size}), &output_indices));
    config = GetGpuLaunchConfig(max_output_size, device);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<int>, config.block_count,
                                config.thread_per_block, 0, device.stream(),
                                config.virtual_thread_count,
                                output_indices->flat<int>().data()));

  } else {
    TF_RETURN_IF_ERROR(context->allocate_output(0, TensorShape({num_outputs}),
                                                &output_indices));
  }
  if (num_outputs == 0) {
    *num_saved_outputs = num_outputs;
    return Status::OK();
  }
  config = GetGpuLaunchConfig(num_outputs, device);
  TF_CHECK_OK(GpuLaunchKernel(
      IndexMultiSelect<int, int>, config.block_count, config.thread_per_block,
      0, device.stream(), config.virtual_thread_count,
      d_selected_indices.flat<int>().data(), sorted_indices,
      (*output_indices).flat<int>().data()));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  *num_saved_outputs = num_outputs;
  return Status::OK();
}

Status CheckValidInputs(const Tensor& boxes, const Tensor& scores,
                        const Tensor& max_output_size,
                        const Tensor& iou_threshold) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_10(mht_10_v, 659, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "CheckValidInputs");

  if (!TensorShapeUtils::IsScalar(max_output_size.shape())) {
    return errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                   max_output_size.shape().DebugString(),
                                   " (Shape must be rank 0 but is ", "rank ",
                                   max_output_size.dims(), ")");
  }
  if (!TensorShapeUtils::IsScalar(iou_threshold.shape())) {
    return errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                   iou_threshold.shape().DebugString(),
                                   " (Shape must be rank 0 but is rank ",
                                   iou_threshold.dims(), ")");
  }
  const float iou_threshold_val = iou_threshold.scalar<float>()();
  if (iou_threshold_val < 0 || iou_threshold_val > 1) {
    return errors::InvalidArgument("iou_threshold must be in [0, 1]");
  }
  if (boxes.dims() != 2) {
    return errors::InvalidArgument(
        "boxes must be a rank 2 tensor! (Shape must "
        "be rank 2 but is rank ",
        boxes.dims(), ")");
  }
  int num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 4) {
    return errors::InvalidArgument(
        "boxes must be Nx4 (Dimension must be 4 but"
        " is ",
        boxes.dim_size(1), ")");
  }
  if (scores.dims() != 1) {
    return errors::InvalidArgument(
        "scores must be a vector! (Shape must be "
        "rank 1 but is rank ",
        scores.dims(), ")");
  }
  if (scores.dim_size(0) != num_boxes) {
    return errors::InvalidArgument(
        "scores has incompatible shape "        // message must be exactly this
        "(Dimensions must be equal, but are ",  // otherwise tests fail!
        num_boxes, " and ", scores.dim_size(0), ")");
  }
  return Status::OK();
}
class NonMaxSuppressionV2GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV2GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_11(mht_11_v, 709, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "NonMaxSuppressionV2GPUOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_12(mht_12_v, 714, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Compute");

    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    auto valid =
        CheckValidInputs(boxes, scores, max_output_size, iou_threshold);
    if (!valid.ok()) {
      context->SetStatus(valid);
      return;
    }
    int num_boxes = boxes.dim_size(0);
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    const int64_t output_size = max_output_size.scalar<int>()();

    OP_REQUIRES_OK(
        context,
        DoNMS(context, boxes, scores, output_size, iou_threshold_val,
              /*score_threshold is float lowest if score threshold is disabled*/
              std::numeric_limits<float>::lowest(),
              /*pad_to_max_output=*/false, &num_boxes));
  }
};

class NonMaxSuppressionV3GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV3GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_13(mht_13_v, 754, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "NonMaxSuppressionV3GPUOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_14(mht_14_v, 759, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Compute");

    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    auto valid =
        CheckValidInputs(boxes, scores, max_output_size, iou_threshold);
    if (!valid.ok()) {
      context->SetStatus(valid);
      return;
    }

    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();
    int num_boxes = boxes.dim_size(0);
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({0}),
                                                       &output_indices));
      return;
    }
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    const int64_t output_size = max_output_size.scalar<int>()();
    OP_REQUIRES_OK(context, DoNMS(context, boxes, scores, output_size,
                                  iou_threshold_val, score_threshold_val,
                                  /*pad_to_max_output=*/false, &num_boxes));
  }
};

class NonMaxSuppressionV4GPUOp : public OpKernel {
 public:
  explicit NonMaxSuppressionV4GPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_15(mht_15_v, 802, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "NonMaxSuppressionV4GPUOp");

    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_16(mht_16_v, 810, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "Compute");

    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    auto valid =
        CheckValidInputs(boxes, scores, max_output_size, iou_threshold);
    if (!valid.ok()) {
      context->SetStatus(valid);
      return;
    }

    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, tensorflow::TensorShape({}),
                                            &num_outputs_t));
    auto device = context->eigen_gpu_device();
    int num_boxes = boxes.dim_size(0);
    if (num_boxes == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                       &output_indices));
      device.memcpy(num_outputs_t->flat<int>().data(), &num_boxes, sizeof(int));
      return;
    }

    const float iou_threshold_val = iou_threshold.scalar<float>()();
    const int64_t output_size = max_output_size.scalar<int>()();
    int num_outputs = 0;
    OP_REQUIRES_OK(context, DoNMS(context, boxes, scores, output_size,
                                  iou_threshold_val, score_threshold_val,
                                  pad_to_max_output_size_, &num_outputs));
    device.memcpyHostToDevice(num_outputs_t->flat<int>().data(), &num_outputs,
                              sizeof(int));
    return;
  }

 private:
  bool pad_to_max_output_size_;
};

}  // anonymous namespace

Status NmsGpu(const float* d_sorted_boxes_float_ptr, const int num_boxes,
              const float iou_threshold, int* d_selected_indices, int* h_nkeep,
              OpKernelContext* context, const int max_boxes, bool flip_boxes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSnon_max_suppression_opDTcuDTcc mht_17(mht_17_v, 869, "", "./tensorflow/core/kernels/image/non_max_suppression_op.cu.cc", "NmsGpu");

  // Making sure we respect the __align(16)__
  // we promised to the compiler.
  auto iptr = reinterpret_cast<std::uintptr_t>(d_sorted_boxes_float_ptr);
  if ((iptr & 15) != 0) {
    return errors::InvalidArgument("Boxes should be aligned to 16 Bytes.");
  }
  // allocate bitmask arrays on host and on device
  Tensor h_num_selected, d_nms_mask;
  const int bit_mask_len =
      (num_boxes + kNmsBoxesPerThread - 1) / kNmsBoxesPerThread;

  int64 max_nms_mask_size = num_boxes * bit_mask_len;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({max_nms_mask_size}), &d_nms_mask));
  // reset data sensitive tensors
  auto device = context->eigen_gpu_device();
  auto config = GetGpuLaunchConfig(d_nms_mask.NumElements(), device);
  TF_CHECK_OK(GpuLaunchKernel(SetZero<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count,
                              d_nms_mask.flat<int32>().data()));

  // h_num_selected is a host pinned tensor.  The GPU kernel can write to it
  // directly, instead of writing to GPU memory and then copying down to
  // num_selected, saving us a small D2H memcpy.  We've observed that even small
  // D2H copies on the compute stream can have an outsized effect on latency.
  AllocatorAttributes pinned_alloc_attrs;
  pinned_alloc_attrs.set_on_host(true);
  pinned_alloc_attrs.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({1}), &h_num_selected,
                                            pinned_alloc_attrs));

  int* d_delete_mask = d_nms_mask.flat<int>().data();
  int* h_selected_count = h_num_selected.flat<int>().data();
  const Box* d_sorted_boxes =
      reinterpret_cast<const Box*>(d_sorted_boxes_float_ptr);
  dim3 block_dim, thread_block;
  int num_blocks = (num_boxes + kNmsBlockDim - 1) / kNmsBlockDim;
  num_blocks = std::max(std::min(num_blocks, kNmsBlockDimMax), 1);
  block_dim.x = num_blocks;
  block_dim.y = num_blocks;
  block_dim.z = 1;
  thread_block.x = kNmsBlockDim;
  thread_block.y = kNmsBlockDim;
  thread_block.z = 1;
  if (flip_boxes) {
    TF_CHECK_OK(GpuLaunchKernel(NMSKernel<true>, block_dim, thread_block, 0,
                                device.stream(), d_sorted_boxes, num_boxes,
                                iou_threshold, bit_mask_len, d_delete_mask));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(NMSKernel<false>, block_dim, thread_block, 0,
                                device.stream(), d_sorted_boxes, num_boxes,
                                iou_threshold, bit_mask_len, d_delete_mask));
  }
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  // Overlapping CPU computes and D2H memcpy
  // both take about the same time

  config = GetGpuLaunchConfig(num_boxes, device);
  Tensor selected_boxes;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({num_boxes}), &selected_boxes));
  Tensor d_indices;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT32, TensorShape({num_boxes}), &d_indices));
  TF_CHECK_OK(GpuLaunchKernel(Iota<int>, config.block_count,
                              config.thread_per_block, 0, device.stream(),
                              config.virtual_thread_count, 0,
                              d_indices.flat<int>().data()));

  char* selected = (char*)(selected_boxes.flat<int8>().data());
  TF_CHECK_OK(GpuLaunchKernel(NMSReduce, 1, 1024, bit_mask_len * sizeof(int),
                              device.stream(), d_delete_mask, bit_mask_len,
                              num_boxes, max_boxes, selected));
  TF_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  // do Cub::deviceSelect::flagged
  size_t flagged_buffer_size = 0;
  gpuprim::DeviceSelect::Flagged(static_cast<void*>(nullptr),  // temp_storage
                                 flagged_buffer_size,
                                 static_cast<int*>(nullptr),   // input
                                 static_cast<char*>(nullptr),  // selection flag
                                 static_cast<int*>(nullptr),   // selected items
                                 static_cast<int*>(nullptr),   // num_selected
                                 num_boxes, device.stream());
  Tensor cub_scratch;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DataType::DT_INT8, TensorShape({(int64)flagged_buffer_size}),
      &cub_scratch));
  Tensor d_num_selected;
  TF_RETURN_IF_ERROR(context->allocate_temp(DataType::DT_INT32,
                                            TensorShape({1}), &d_num_selected));

  gpuprim::DeviceSelect::Flagged(
      (void*)cub_scratch.flat<int8>().data(),  // temp_storage
      flagged_buffer_size,
      d_indices.flat<int>().data(),  // input
      selected,                      // selection flag
      d_selected_indices,            // selected items
      h_selected_count, num_boxes, device.stream());
  gpuEvent_t copy_done;
  TF_RETURN_IF_CUDA_ERROR(
      gpuEventCreateWithFlags(&copy_done, gpuEventDisableTiming));
  TF_RETURN_IF_CUDA_ERROR(gpuEventRecord(copy_done, device.stream()));
  TF_RETURN_IF_CUDA_ERROR(gpuEventSynchronize(copy_done));
  gpuEventDestroy(copy_done);

  *h_nkeep = *h_selected_count;
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size"),
                        NonMaxSuppressionV2GPUOp);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV3")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        NonMaxSuppressionV3GPUOp);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV4")
                            .TypeConstraint<float>("T")
                            .Device(DEVICE_GPU)
                            .HostMemory("iou_threshold")
                            .HostMemory("max_output_size")
                            .HostMemory("score_threshold"),
                        NonMaxSuppressionV4GPUOp);

}  // namespace tensorflow
#endif
