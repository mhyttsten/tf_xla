/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSscan_ops_gpuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_ops_gpuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscan_ops_gpuDTh() {
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

#if CUDA_VERSION >= 9000
#define CUB_USE_COOPERATIVE_GROUPS
#endif  // CUDA_VERSION >= 9000

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/scan_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/permutation_input_iterator.h"
#include "tensorflow/core/util/permutation_output_iterator.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::Index Index;

namespace functor {

// Map a contiguous range to the actual memory locations depending on which
// axis the scan is taking place over and whether or not reversed.
struct MapIndexToLocation {
  __host__ __device__ MapIndexToLocation(int dimx, int dimy, int dimz,
                                         bool reverse = false)
      : dimx_(dimx), dimy_(dimy), dimz_(dimz), reverse_(reverse) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_ops_gpuDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/scan_ops_gpu.h", "MapIndexToLocation");
}

  __host__ __device__ int operator()(int id) const {
    if (dimx_ == 1) {
      int row = id % dimy_;
      int col = id / dimy_;

      if (reverse_) return (dimy_ - row - 1) * dimz_ + col;

      return row * dimz_ + col;
    } else if (dimz_ == 1) {
      if (reverse_) {
        int row = id / dimy_;
        int col = id % dimy_;
        return row * dimy_ + (dimy_ - col - 1);
      }
      return id;
    } else {
      int col = id % dimy_;
      int tmp = id / dimy_;

      int row1 = id / (dimy_ * dimz_);
      int col1 = tmp % dimz_;

      if (reverse_)
        return row1 * dimy_ * dimz_ + (dimy_ - col - 1) * dimz_ + col1;

      return row1 * dimy_ * dimz_ + col * dimz_ + col1;
    }
  }

  int dimx_;
  int dimy_;
  int dimz_;
  bool reverse_;
};

template <typename T, typename Op>
struct BlockPrefixCallbackOp {
  // Running prefix
  T running_total_;
  Op op_;

  __device__ BlockPrefixCallbackOp(T running_total, Op op)
      : running_total_(running_total), op_(op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscan_ops_gpuDTh mht_1(mht_1_v, 264, "", "./tensorflow/core/kernels/scan_ops_gpu.h", "BlockPrefixCallbackOp");
}

  // Callback operator to be entered by the first warp of threads in the block.
  // tid 0 is responsible for returning a value for seeding the block-wide scan.
  __device__ T operator()(T block_aggregate) {
    T old_prefix = running_total_;
    running_total_ = op_(old_prefix, block_aggregate);
    return old_prefix;
  }
};

template <typename T>
struct Sum {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct Prod {
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

template <typename T, typename Op>
struct IsSum {
  constexpr static bool value =
      (std::is_same<Op, Sum<T>>::value ||
       std::is_same<Op, Eigen::internal::SumReducer<T>>::value);
};

template <typename T, typename Op>
struct IsProd {
  constexpr static bool value =
      (std::is_same<Op, Prod<T>>::value ||
       std::is_same<Op, Eigen::internal::ProdReducer<T>>::value);
};

template <typename T, typename Op>
struct IsLogSumExp {
  constexpr static bool value = (std::is_same<Op, LogSumExp<T>>::value ||
                                 std::is_same<Op, LogSumExpReducer<T>>::value);
};

template <typename T, typename Op>
struct IdentityValue {
  static_assert(IsSum<T, Op>::value || IsProd<T, Op>::value ||
                    IsLogSumExp<T, Op>::value,
                "IdentityValue not yet defined for this type.");

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U operator()(
      typename std::enable_if<IsSum<U, OpCopy>::value, U>::type t = U(0)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U operator()(
      typename std::enable_if<IsProd<U, OpCopy>::value, U>::type t = U(1)) {
    return t;
  }

  template <typename U = T, typename OpCopy = Op>
  __host__ __device__ U
  operator()(typename std::enable_if<IsLogSumExp<U, OpCopy>::value, U>::type t =
                 U(Eigen::NumTraits<U>::lowest())) {
    return t;
  }
};

// Each block is mapped to one sequence.  A contiguous range is mapped to the
// appropriate locations in memory by the permutation iterators.  This is
// ideal for 1-D and row based scans.  Column scans would be better if they
// did a block load and then locally transposed.  CUB's device wide scan is not
// used in the large 1D case, even though it would be more efficient, because
// it is not deterministic.
template <typename T, typename Op, int BlockDim = 128, int ItemsPerThread = 4>
__launch_bounds__(BlockDim) __global__
    void scan_kernel(const T* in, T* out, int dimx, int dimy, int dimz,
                     bool exclusive, bool reverse, Op op) {
  typedef gpuprim::BlockLoad<T, BlockDim, ItemsPerThread,
                             gpuprim::BLOCK_LOAD_TRANSPOSE>
      BlockLoad;
  typedef gpuprim::BlockStore<T, BlockDim, ItemsPerThread,
                              gpuprim::BLOCK_STORE_TRANSPOSE>
      BlockStore;
  typedef gpuprim::BlockScan<T, BlockDim> BlockScan;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  int problem_length = dimy;

  // Initialize running total
  BlockPrefixCallbackOp<T, Op> prefix_op(IdentityValue<T, Op>()(), op);

  MapIndexToLocation map_op(dimx, dimy, dimz, reverse);
  int block_start = problem_length * blockIdx.x;
  // Have the block iterate over segments of items
  for (int block_offset = block_start;
       block_offset < block_start + problem_length;
       block_offset += BlockDim * ItemsPerThread) {
    int valid_items = min(BlockDim * ItemsPerThread,
                          problem_length - (block_offset % problem_length));

    // first construct a counting iterator that has the desired start point
    typedef gpuprim::TransformInputIterator<int, MapIndexToLocation,
                                            gpuprim::CountingInputIterator<int>>
        MapIterType;

    gpuprim::CountingInputIterator<int> counting_iter(block_offset);

    // Next map the iterator to the actual locations in memory
    MapIterType map_iter(counting_iter, map_op);

    PermutationInputIterator<T, const T*, MapIterType> permutein_iter(in,
                                                                      map_iter);
    PermutationOutputIterator<T, T*, MapIterType> permuteout_iter(out,
                                                                  map_iter);

    // Load a segment of consecutive items that are blocked across threads
    T thread_data[ItemsPerThread];
    BlockLoad(temp_storage.load).Load(permutein_iter, thread_data, valid_items);
    __syncthreads();

    // Collectively compute the block-wide scan
    if (exclusive) {
      BlockScan(temp_storage.scan)
          .ExclusiveScan(thread_data, thread_data, op, prefix_op);
    } else {
      BlockScan(temp_storage.scan)
          .InclusiveScan(thread_data, thread_data, op, prefix_op);
    }
    __syncthreads();

    // Store scanned items to output segment
    BlockStore(temp_storage.store)
        .Store(permuteout_iter, thread_data, valid_items);
    __syncthreads();
  }
}

template <typename T, typename Op>
void LaunchScan(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                typename TTypes<T, 3>::Tensor out, Op op, const bool reverse,
                const bool exclusive) {
  const int items_per_thread = 4;

  int dimx = in.dimension(0);
  int dimy = in.dimension(1);
  int dimz = in.dimension(2);
  int num_blocks = dimx * dimz;

  int ideal_block_size = dimy / items_per_thread;
  const int rocm_threads_per_warp = 64;
  ideal_block_size = std::max(ideal_block_size, rocm_threads_per_warp);

  // There seems to be a bug when the type is not float and block_size 1024.
  // Launch on the smallest power of 2 block size that we can.
  if (ideal_block_size >= 1024 && std::is_same<T, float>::value) {
    const int block_size = 1024;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 512) {
    const int block_size = 512;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 256) {
    const int block_size = 256;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else if (ideal_block_size >= 128) {
    const int block_size = 128;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
#if TENSORFLOW_COMPILER_IS_HIP_CLANG
    // HIP-CLANG has some kind of problem here with 32 threads (possibly because
    // the warpsize is 64). Reenable when working properly
  } else if (true) {
#else
  } else if (ideal_block_size >= 64) {
#endif
    const int block_size = 64;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  } else {
    const int block_size = 32;
    TF_CHECK_OK(
        GpuLaunchKernel(scan_kernel<T, Op, block_size, items_per_thread>,
                        num_blocks, block_size, 0, d.stream(), in.data(),
                        out.data(), dimx, dimy, dimz, exclusive, reverse, op));
  }
}

template <typename T>
struct Scan<GPUDevice, Eigen::internal::SumReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::SumReducer<T>& reducer,
                  const bool reverse, const bool exclusive) {
    LaunchScan<T, Sum<T>>(d, in, out, Sum<T>(), reverse, exclusive);
  }
};

template <typename T>
struct Scan<GPUDevice, Eigen::internal::ProdReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const Eigen::internal::ProdReducer<T>& reducer,
                  const bool reverse, const bool exclusive) {
    LaunchScan<T, Prod<T>>(d, in, out, Prod<T>(), reverse, exclusive);
  }
};

template <typename T>
struct Scan<GPUDevice, LogSumExpReducer<T>, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out,
                  const LogSumExpReducer<T>& reducer, const bool reverse,
                  const bool exclusive) {
    LaunchScan<T, LogSumExp<T>>(d, in, out, LogSumExp<T>(), reverse, exclusive);
  }
};

}  // namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_SCAN_OPS_GPU_H_
