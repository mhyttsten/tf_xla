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
#ifndef TENSORFLOW_CORE_KERNELS_TOPK_OP_GPU_H_
#define TENSORFLOW_CORE_KERNELS_TOPK_OP_GPU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh() {
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

#include <cmath>
#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/topk_op.h"
#include "tensorflow/core/lib/gtl/top_n.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace impl {

enum class HeapType { kMinHeap, kMaxHeap };
enum class PreferIndices { kLower, kHigher };

template <typename T>
struct Entry {
  int index;
  T value;

  // Test-only.
  static bool greater(const Entry<T>& a, const Entry<T>& b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/kernels/topk_op_gpu.h", "greater");

    if (a.value == b.value) {
      return a.index < b.index;
    }
    return a.value > b.value;
  }
};

template <typename T>
struct LinearData {
  typedef impl::Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/topk_op_gpu.h", "lambda");
 return data[index]; }

  __device__ int get_index(int i) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_index");
 return data[i].index; }
  __device__ T get_value(int i) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_3(mht_3_v, 246, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_value");
 return data[i].value; }

  Entry* const data;
};

template <typename T>
struct IndirectLinearData {
  typedef impl::Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_4(mht_4_v, 258, "", "./tensorflow/core/kernels/topk_op_gpu.h", "lambda");
 return data[index]; }

  __device__ int get_index(int i) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_5(mht_5_v, 263, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_index");

    return backing_data[data[i].index].index;
  }
  __device__ T get_value(int i) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_6(mht_6_v, 269, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_value");
 return data[i].value; }

  Entry* const data;
  Entry* const backing_data;
};

template <typename T>
struct StridedData {
  typedef impl::Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_7(mht_7_v, 282, "", "./tensorflow/core/kernels/topk_op_gpu.h", "lambda");

    return data[index * blockDim.x + threadIdx.x];
  }

  __device__ int get_index(int i) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_8(mht_8_v, 289, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_index");
 return (*this)[i].index; }
  __device__ T get_value(int i) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_9(mht_9_v, 293, "", "./tensorflow/core/kernels/topk_op_gpu.h", "get_value");
 return (*this)[i].value; }

  Entry* const data;
};

// A heap of Entry<T> that can either work as a min-heap or as a max-heap.
template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
struct IndexedHeap {
  typedef typename Data<T>::Entry Entry;
  const Data<T> data;
  __device__ IndexedHeap(const Data<T>& d) : data(d) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_10(mht_10_v, 307, "", "./tensorflow/core/kernels/topk_op_gpu.h", "IndexedHeap");
}

  __device__ bool is_above(int left, int right) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_11(mht_11_v, 312, "", "./tensorflow/core/kernels/topk_op_gpu.h", "is_above");

    T left_value = data.get_value(left);
    T right_value = data.get_value(right);
    if (left_value == right_value) {
      if (preferIndices == PreferIndices::kLower) {
        return data.get_index(left) < data.get_index(right);
      } else {
        return data.get_index(left) > data.get_index(right);
      }
    }
    if (heapType == HeapType::kMinHeap) {
      return left_value < right_value;
    } else {
      return left_value > right_value;
    }
  }

  __device__ void assign(int i, const Entry& entry) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_12(mht_12_v, 332, "", "./tensorflow/core/kernels/topk_op_gpu.h", "assign");
 data[i] = entry; }

  __device__ void push_up(int i) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_13(mht_13_v, 337, "", "./tensorflow/core/kernels/topk_op_gpu.h", "push_up");

    int child = i;
    int parent;
    for (; child > 0; child = parent) {
      parent = (child - 1) / 2;
      if (!is_above(child, parent)) {
        // Heap property satisfied.
        break;
      }
      swap(child, parent);
    }
  }

  __device__ void swap(int a, int b) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_14(mht_14_v, 353, "", "./tensorflow/core/kernels/topk_op_gpu.h", "swap");

    auto tmp = data[b];
    data[b] = data[a];
    data[a] = tmp;
  }

  __device__ void push_root_down(int k) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_15(mht_15_v, 362, "", "./tensorflow/core/kernels/topk_op_gpu.h", "push_root_down");
 push_down(0, k); }

  // MAX-HEAPIFY in Cormen
  __device__ void push_down(int node, int k) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_16(mht_16_v, 368, "", "./tensorflow/core/kernels/topk_op_gpu.h", "push_down");

    while (true) {
      const int left = 2 * node + 1;
      const int right = left + 1;
      int smallest = node;
      if (left < k && is_above(left, smallest)) {
        smallest = left;
      }
      if (right < k && is_above(right, smallest)) {
        smallest = right;
      }
      if (smallest == node) {
        break;
      }
      swap(smallest, node);
      node = smallest;
    }
  }

  // BUILD-MAX-HEAPIFY in Cormen
  __device__ void build(int k) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_17(mht_17_v, 391, "", "./tensorflow/core/kernels/topk_op_gpu.h", "build");

    for (int node = (k - 1) / 2; node >= 0; node--) {
      push_down(node, k);
    }
  }

  // HEAP-EXTRACT-MAX in Cormen
  __device__ void remove_root(int k) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_18(mht_18_v, 401, "", "./tensorflow/core/kernels/topk_op_gpu.h", "remove_root");

    data[0] = data[k - 1];
    push_root_down(k - 1);
  }

  // in-place HEAPSORT in Cormen
  // This method destroys the heap property.
  __device__ void sort(int k) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_19(mht_19_v, 411, "", "./tensorflow/core/kernels/topk_op_gpu.h", "sort");

    for (int slot = k - 1; slot > 0; slot--) {
      // This is like remove_root but we insert the element at the end.
      swap(slot, 0);
      // Heap is now an element smaller.
      push_root_down(/*k=*/slot);
    }
  }

  __device__ void replace_root(const Entry& entry, int k) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_20(mht_20_v, 423, "", "./tensorflow/core/kernels/topk_op_gpu.h", "replace_root");

    data[0] = entry;
    push_root_down(k);
  }

  __device__ const Entry& root() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_21(mht_21_v, 431, "", "./tensorflow/core/kernels/topk_op_gpu.h", "root");
 return data[0]; }
};

template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T> make_indexed_heap(
    typename Data<T>::Entry* data) {
  return IndexedHeap<heapType, preferIndices, Data, T>{Data<T>{data}};
}

// heapTopK walks over [input, input+length) with `step_size` stride starting at
// `start_index`.
// It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
// access elements in `heap_entries`. If sorted=true, the elements will be
// sorted at the end.
template <typename T, template <typename> class Data = LinearData>
__device__ void heapTopK(const T* __restrict__ input, int length, int k,
                         Entry<T>* __restrict__ heap_entries,
                         bool sorted = false, int start_index = 0,
                         int step_size = 1) {
  assert(k <= length);

  auto heap =
      make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T>(
          heap_entries);

  int heap_end_index = start_index + k * step_size;
  if (heap_end_index > length) {
    heap_end_index = length;
  }
  // Initialize the min-heap.
  for (int index = start_index, slot = 0; index < heap_end_index;
       index += step_size, slot++) {
    heap.assign(slot, {index, input[index]});
  }

  heap.build(k);

  // Now iterate over the remaining items.
  // If an item is smaller than the min element, it is not amongst the top k.
  // Otherwise, replace the min element with it and push upwards.
  for (int index = heap_end_index; index < length; index += step_size) {
    // We prefer elements with lower indices. This is given here.
    // Later elements automatically have higher indices, so can be discarded.
    if (input[index] > heap.root().value) {
      // This element should replace the min.
      heap.replace_root({index, input[index]}, k);
    }
  }

  // Sort if wanted.
  if (sorted) {
    heap.sort(k);
  }
}

// mergeShards performs a top-k merge on `num_shards` many sorted streams that
// are sorted and stored in `entries` in a strided way:
// |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
// The overall top k elements are written to `top_k_values` and their indices
// to top_k_indices.
// `top_k_heap` is used as temporary storage for the merge heap.
template <typename T>
__device__ void mergeShards(int num_shards, int k,
                            Entry<T>* __restrict__ entries,
                            Entry<T>* __restrict__ top_k_heap, T* top_k_values,
                            int* top_k_indices) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_22(mht_22_v, 500, "", "./tensorflow/core/kernels/topk_op_gpu.h", "mergeShards");

  // If k < num_shards, we can use a min-heap with k elements to get the top k
  // of the sorted blocks.
  // If k > num_shards, we can initialize a min-heap with the top element from
  // each sorted block.
  const int heap_size = k < num_shards ? k : num_shards;

  // Min-heap part.
  {
    auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
                                IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
    // Initialize the heap as a min-heap.
    for (int slot = 0; slot < heap_size; slot++) {
      min_heap.assign(slot, {slot, entries[slot].value});
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards > heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      const auto entry = entries[shard];
      const auto root = min_heap.root();
      if (entry.value < root.value) {
        continue;
      }
      if (entry.value == root.value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root({shard, entry.value}, heap_size);
    }
  }

  // Max-part.
  {
    // Turn the min-heap into a max-heap in-place.
    auto max_heap = IndexedHeap<HeapType::kMaxHeap, PreferIndices::kLower,
                                IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
    // Heapify into a max heap.
    max_heap.build(heap_size);

    // Now extract the minimum k-1 times.
    // k is treated specially.
    const int last_k = k - 1;
    for (int rank = 0; rank < last_k; rank++) {
      const Entry<T>& max_element = max_heap.root();
      top_k_values[rank] = max_element.value;
      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      int next_shard_index = shard_index + num_shards;
      // For rank < k-1, each top k heap still contains at least 1 element,
      // so we can draw a replacement.
      max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
                            heap_size);
    }

    // rank == last_k.
    const Entry<T>& max_element = max_heap.root();
    top_k_values[last_k] = max_element.value;
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
  }
}

#if GOOGLE_CUDA
extern __shared__ char shared_memory[];
#endif  // GOOGLE_CUDA

template <typename T>
#if TENSORFLOW_USE_ROCM
__attribute__((amdgpu_flat_work_group_size(1, 256)))
#endif  // TENSORFLOW_USE_ROCM
__global__ void
TopKKernel(const T* __restrict__ input, int length, int k, bool sorted,
           T* __restrict__ output, int* __restrict__ indices) {
#if TENSORFLOW_USE_ROCM
  HIP_DYNAMIC_SHARED(char, shared_memory);
#endif  // TENSORFLOW_USE_ROCM

  const int batch_index = blockIdx.x;
  const T* batch_input = input + batch_index * length;

  const int thread_index = threadIdx.x;
  const int thread_count = blockDim.x;

  Entry<T>* shared_entries = (Entry<T>*)shared_memory;

  heapTopK<T, StridedData>(batch_input, length, k, shared_entries, true,
                           thread_index, thread_count);

  __syncthreads();
  if (thread_index == 0) {
    const int offset = batch_index * k;
    auto batch_output = output + offset;
    auto batch_indices = indices + offset;
    Entry<T>* top_k_heap = shared_entries + thread_count * k;

    // TODO(blackhc): Erich says: Performance can likely be improved
    // significantly by having the merge be done by multiple threads rather than
    // just one.  ModernGPU has some nice primitives that could help with this.
    mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
                batch_indices);
  }
}

template <typename T>
cudaError LaunchTopKKernel(const gpuStream_t& stream, int num_shards,
                           const T* input, int batch_size, int length, int k,
                           bool sorted, T* output, int* indices) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_23(mht_23_v, 613, "", "./tensorflow/core/kernels/topk_op_gpu.h", "LaunchTopKKernel");

  // This code assumes that k is small enough that the computation
  // fits inside shared memory (hard coded to 48KB).  In practice this
  // means k <= 3072 for T=float/int32 and k <= 2048 for T=double/int64.
  // The calculation is:
  //   shared_memory_size / (2 * (sizeof(int) + sizeof(T))) < k.

  // Use as many shards as possible.
  if (num_shards <= 0) {
    constexpr auto shared_memory_size = 48 << 10;  // 48 KB
    const auto heap_size = k * sizeof(Entry<T>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    if (num_shards <= 0) {
      num_shards = 1;
    }
    auto shard_size = length / num_shards;
    auto min_shard_size = 2 * k;
    if (shard_size < min_shard_size) {
      num_shards = length / min_shard_size;
    }
    if (num_shards <= 0) {
      num_shards = 1;
#if GOOGLE_CUDA
    } else if (num_shards > 1024) {
      num_shards = 1024;
    }
#elif TENSORFLOW_USE_ROCM
      // ROCm can't execute with 1024 and requires an explicit
      // amdgpu_flat_work_group_size attribute with >256
    } else if (num_shards > 256) {
      num_shards = 256;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }
  // We are limited by the amount of shared memory we have per block.
  auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T>);

  TF_CHECK_OK(GpuLaunchKernel(TopKKernel<T>, batch_size, num_shards,
                              shared_memory_size, stream, input, length, k,
                              sorted, output, indices));
  return cudaGetLastError();
}

struct SegmentOffsetCreator {
  EIGEN_DEVICE_FUNC
  SegmentOffsetCreator(int num_cols) : num_cols_(num_cols) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_24(mht_24_v, 662, "", "./tensorflow/core/kernels/topk_op_gpu.h", "SegmentOffsetCreator");
}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

struct ColumnIndexCreator {
  ColumnIndexCreator(int num_cols) : num_cols_(num_cols) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_25(mht_25_v, 675, "", "./tensorflow/core/kernels/topk_op_gpu.h", "ColumnIndexCreator");
}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }

  int num_cols_;
};

template <typename T>
Status LaunchSortKernel(OpKernelContext* ctx, const T* input, int num_rows,
                        int num_cols, int k,
                        typename TTypes<T, 2>::Tensor values,
                        TTypes<int, 2>::Tensor indices) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_26(mht_26_v, 692, "", "./tensorflow/core/kernels/topk_op_gpu.h", "LaunchSortKernel");

  const GPUDevice& d = ctx->eigen_device<GPUDevice>();
  const auto& cu_stream = GetGpuStream(ctx);
  size_t temp_storage_bytes = -1;

  // TODO(ebrevdo): Once gpuprim supports iterators for ValueT replace that
  // tensor with an iterator that directly returns the correct value.
  Tensor input_indices;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT32, TensorShape({num_rows, num_cols}), &input_indices));
  auto input_indices_t = To32Bit(input_indices.flat<int32>());
  input_indices_t.device(d) =
      input_indices_t.generate(ColumnIndexCreator(num_cols));

  gpuprim::CountingInputIterator<int> counting_iter(0);
  gpuprim::TransformInputIterator<int, SegmentOffsetCreator,
                                  gpuprim::CountingInputIterator<int>>
      segment_offsets_t(counting_iter, SegmentOffsetCreator(num_cols));

  Tensor temp_values;
  Tensor temp_indices;
  T* sorted_values_ptr;
  int* sorted_indices_ptr;
  if (k == num_cols) {
    // Doing a full sort, no intermediate values needed.
    sorted_values_ptr = values.data();
    sorted_indices_ptr = indices.data();
  } else {
    // Need to create intermediate values for sorting.
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT32, TensorShape({num_rows, num_cols}), &temp_indices));
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value,
                                          TensorShape({num_rows, num_cols}),
                                          &temp_values));
    sorted_indices_ptr = temp_indices.flat<int32>().data();
    sorted_values_ptr = temp_values.flat<T>().data();
  }

  auto err = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ nullptr,
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_keys_in */ input,
      /* d_keys_out */ sorted_values_ptr,
      /* d_values_in */ input_indices_t.data(),
      /* d_values_out */ sorted_indices_ptr,
      /* num_items */ num_cols * num_rows,
      /* num_segments */ num_rows,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(T) * 8,
      /* stream */ cu_stream);
  if (err != cudaSuccess) {
    return errors::Internal(
        "TopKOp: Could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to calculate "
        "temp_storage_bytes, status: ",
        cudaGetErrorString(err));
  }
  Tensor temp_storage;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
      &temp_storage));
  err = gpuprim::DeviceSegmentedRadixSort::SortPairsDescending(
      /* d_temp_storage */ temp_storage.flat<int8>().data(),
      /* temp_storage_bytes */ temp_storage_bytes,
      /* d_keys_in */ input,
      /* d_keys_out */ sorted_values_ptr,
      /* d_values_in */ input_indices_t.data(),
      /* d_values_out */ sorted_indices_ptr,
      /* num_items */ num_cols * num_rows,
      /* num_segments */ num_rows,
      /* d_begin_offsets */ segment_offsets_t,
      /* d_end_offsets */ segment_offsets_t + 1,
      /* begin_bit */ 0,
      /* end_bit */ sizeof(T) * 8,
      /* stream */ cu_stream);
  if (err != cudaSuccess) {
    return errors::Internal(
        "TopKOp: Could not launch "
        "gpuprim::DeviceSegmentedRadixSort::SortPairsDescending to sort input, "
        "temp_storage_bytes: ",
        temp_storage_bytes, ", status: ", cudaGetErrorString(err));
  }
  if (k < num_cols) {
    // Need to copy subsets of sorted_indices and sorted_outputs to
    // indices and outputs.
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, 0};
    const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, k};
    To32Bit(indices).device(d) =
        To32Bit(temp_indices.matrix<int32>()).slice(slice_indices, slice_sizes);
    To32Bit(values).device(d) =
        To32Bit(temp_values.matrix<T>()).slice(slice_indices, slice_sizes);
  }
  return Status::OK();
}

}  // end namespace impl

namespace functor {

template <typename T>
struct TopKFunctor<GPUDevice, T> {
  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* context, bool sorted, int k,
          const typename TTypes<T, 2>::ConstTensor& input, const int64 num_rows,
          const int64 num_cols, typename TTypes<T, 2>::Tensor values,
          typename TTypes<int, 2>::Tensor indices) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStopk_op_gpuDTh mht_27(mht_27_v, 802, "", "./tensorflow/core/kernels/topk_op_gpu.h", "Compute");

    // For small k, use the heap implementation.  For larger k, use
    // the in-place gpuprim sort.  For k == num_cols, always use the
    // in-place gpuprim sort.  The thresholds for n and k were determined
    // empirically.
    if (num_cols <= 1000 || k == num_cols || k >= 100) {
      return impl::LaunchSortKernel(context, input.data(), num_rows, num_cols,
                                    k, values, indices);
    } else {
      const auto& cu_stream = GetGpuStream(context);
      auto err = impl::LaunchTopKKernel(cu_stream, /* num_shards */ 0,
                                        input.data(), num_rows, num_cols, k,
                                        sorted, values.data(), indices.data());
      if (err != cudaSuccess) {
        return errors::Internal(
            "Could not launch TopKKernel: ", cudaGetErrorString(err), ".");
      } else {
        return Status::OK();
      }
    }
  }
};

}  // end namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_TOPK_OP_GPU_H_
