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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace tflite {
namespace internal {
namespace sparsity {

namespace {
uint64_t GetFlattenedIndex(const std::vector<int>& indices,
                           const std::vector<int>& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "GetFlattenedIndex");

  uint64_t index = 0;
  int sub_elements = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    index += indices[i] * sub_elements;
    sub_elements *= shape[i];
  }
  return index;
}

std::vector<int> TfLiteIntArrayToVector(const TfLiteIntArray* int_array) {
  std::vector<int> values;
  if (!int_array) {
    return values;
  }

  values.resize(int_array->size);
  for (size_t i = 0; i < int_array->size; i++) {
    values[i] = int_array->data[i];
  }

  return values;
}

}  // namespace

template <typename T>
FormatConverter<T>::FormatConverter(
    const std::vector<int>& shape, const std::vector<int>& traversal_order,
    const std::vector<TfLiteDimensionType>& format,
    const std::vector<int>& block_size, const std::vector<int>& block_map)
    : dense_shape_(shape),
      traversal_order_(traversal_order),
      block_size_(block_size),
      block_map_(block_map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_1(mht_1_v, 233, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::FormatConverter");

  dense_size_ = 1;
  int block_dim = 0;
  blocked_shape_.resize(shape.size());
  format_.resize(shape.size() + block_map.size());
  for (int i = 0; i < shape.size(); i++) {
    format_[i] = format[traversal_order[i]];
    dense_size_ *= shape[i];
    if (block_dim < block_map.size() && block_map[block_dim] == i) {
      blocked_shape_[i] = shape[i] / block_size[block_dim];
      block_dim++;
    } else {
      blocked_shape_[i] = shape[i];
    }
  }

  // Only dense blocks are supported.
  for (int i = 0; i < block_map.size(); i++) {
    format_[i + shape.size()] = kTfLiteDimDense;
  }
}

template <typename T>
TfLiteStatus FormatConverter<T>::DenseToSparse(const T* src_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_2(mht_2_v, 259, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::DenseToSparse");

  int num_original_dims = dense_shape_.size();
  int num_block_dims = block_map_.size();
  int num_expanded_dims = num_original_dims + num_block_dims;
  std::vector<int> expanded_shape(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; i++) {
    if (i < num_original_dims) {
      expanded_shape[i] = blocked_shape_[i];
    } else {
      expanded_shape[i] = block_size_[i - num_original_dims];
    }
  }

  std::vector<int> shape_offset(num_original_dims);
  shape_offset[shape_offset.size() - 1] = 1;
  for (int i = num_original_dims - 1; i > 0; --i) {
    shape_offset[i - 1] = shape_offset[i] * dense_shape_[i];
  }

  std::vector<int> expanded_shape_offset(num_expanded_dims);
  for (int i = 0; i < num_original_dims; ++i) {
    expanded_shape_offset[i] = shape_offset[i];
  }
  for (int i = 0; i < num_block_dims; ++i) {
    int mapped_dim = block_map_[i];
    expanded_shape_offset[num_original_dims + i] = shape_offset[mapped_dim];
    expanded_shape_offset[mapped_dim] *= block_size_[i];
  }

  std::vector<int> dst_ordered_offset(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; ++i) {
    dst_ordered_offset[i] = expanded_shape_offset[traversal_order_[i]];
  }

  std::vector<bool> dst_dim_has_nonzeroes(num_expanded_dims);
  std::fill(dst_dim_has_nonzeroes.begin(), dst_dim_has_nonzeroes.end(), false);
  std::vector<int> inner_compressed_dim(num_expanded_dims);
  int most_recent_compressed_dim = -1;
  std::vector<int> num_segments_of_next_compressed_dim(num_expanded_dims);
  int segment_count = 1;
  for (int i = num_expanded_dims - 1; i >= 0; --i) {
    inner_compressed_dim[i] = most_recent_compressed_dim;
    if (format_[i] == kTfLiteDimSparseCSR) {
      most_recent_compressed_dim = i;
      num_segments_of_next_compressed_dim[i] = segment_count;
      segment_count = 1;
    } else {
      num_segments_of_next_compressed_dim[i] = -1;
      segment_count *= expanded_shape[traversal_order_[i]];
    }
  }

  dim_metadata_.resize(num_expanded_dims * 2);
  std::vector<int> dst_sparse_dims;
  dst_sparse_dims.reserve(num_expanded_dims);
  for (int i = 0; i < num_expanded_dims; ++i) {
    dim_metadata_[i * 2].clear();
    dim_metadata_[i * 2 + 1].clear();
    if (format_[i] == kTfLiteDimDense) {
      // If dimension is dense, just store the shape.
      dim_metadata_[i * 2].push_back(expanded_shape[traversal_order_[i]]);
    } else {
      dim_metadata_[i * 2].push_back(0);  // Segment array always begins with 0.
      dst_sparse_dims.push_back(i);       // Add dimension to the sparse list.
    }
  }

  // This algorithm assumes that the block size is small enough for all the
  // elements to fit in cache, so the strided accesses from different traversal
  // order and the write-first-erase-later strategy shouldn't be too slow
  int dst_dim_idx = num_expanded_dims;
  std::vector<int> coordinate(num_expanded_dims, 0);
  int dense_tensor_idx = 0;
  while (dst_dim_idx >= 0) {
    if (dst_dim_idx == num_expanded_dims) {
      // We have a complete coordinate. Add the element to the value array if it
      // is not zero, or if the last dimension is dense.
      if (!IsZero(src_data[dense_tensor_idx])) {
        data_.push_back(src_data[dense_tensor_idx]);
        // Mark all sparse dimensions that their current indices have nonzeroes.
        for (auto dst_dim : dst_sparse_dims) {
          if (!dst_dim_has_nonzeroes[dst_dim]) {
            // Only add the index to the indices array if the current nonzero
            // is the first nonzero of the block.
            dim_metadata_[2 * dst_dim + 1].push_back(coordinate[dst_dim]);
            dst_dim_has_nonzeroes[dst_dim] = true;
          }
        }
      } else if (format_[num_expanded_dims - 1] == kTfLiteDimDense) {
        data_.push_back(src_data[dense_tensor_idx]);
      }
      --dst_dim_idx;
    } else {
      int original_dim_idx = traversal_order_[dst_dim_idx];
      int dim_size = expanded_shape[original_dim_idx];
      if (dst_dim_has_nonzeroes[dst_dim_idx]) {
        // If the previous block has nonzeroes, reset the flag to false since
        // we have just moved to a new block.
        dst_dim_has_nonzeroes[dst_dim_idx] = false;
      } else if (format_[dst_dim_idx] == kTfLiteDimSparseCSR) {
        // This block is empty. Delete unnecessary values if compressed.
        int next_compressed_dim = inner_compressed_dim[dst_dim_idx];
        int erase_offset = dim_metadata_[2 * dst_dim_idx + 1].size() *
                           num_segments_of_next_compressed_dim[dst_dim_idx];
        if (next_compressed_dim >= 0) {
          auto& segments = dim_metadata_[2 * inner_compressed_dim[dst_dim_idx]];
          segments.erase(segments.begin() + 1 + erase_offset, segments.end());
        } else {
          data_.erase(data_.begin() + erase_offset, data_.end());
        }
      }
      if (++coordinate[dst_dim_idx] < dim_size) {
        // The current dst_dim_idx is valid (not out of bound).
        dense_tensor_idx += dst_ordered_offset[dst_dim_idx];
        ++dst_dim_idx;
      } else {
        // dst_dim_idx has reached its dim size. Update segment array and go
        // back to incrementing the previous dimension (dst_dim_idx - 1).
        if (format_[dst_dim_idx] == kTfLiteDimSparseCSR) {
          dim_metadata_[2 * dst_dim_idx].push_back(
              dim_metadata_[2 * dst_dim_idx + 1].size());
        }
        coordinate[dst_dim_idx] = -1;
        dense_tensor_idx -= dst_ordered_offset[dst_dim_idx] * dim_size;
        --dst_dim_idx;
      }
    }
  }

  return kTfLiteOk;
}

template <typename T>
FormatConverter<T>::FormatConverter(
    const std::vector<int>& shape, const std::vector<int>& traversal_order,
    const std::vector<TfLiteDimensionType>& format,
    const std::vector<int>& dense_size,
    const std::vector<std::vector<int>>& segments,
    const std::vector<std::vector<int>>& indices,
    const std::vector<int>& block_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_3(mht_3_v, 401, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::FormatConverter");

  InitSparseToDenseConverter(shape, traversal_order, format, dense_size,
                             segments, indices, block_map);
}

template <typename T>
FormatConverter<T>::FormatConverter(const std::vector<int>& shape,
                                    const TfLiteSparsity& sparsity) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_4(mht_4_v, 411, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::FormatConverter");

  auto traversal_order = TfLiteIntArrayToVector(sparsity.traversal_order);
  auto block_map = TfLiteIntArrayToVector(sparsity.block_map);

  std::vector<TfLiteDimensionType> format(sparsity.dim_metadata_size);
  std::vector<int> dense_size(sparsity.dim_metadata_size);
  std::vector<std::vector<int>> segments(sparsity.dim_metadata_size);
  std::vector<std::vector<int>> indices(sparsity.dim_metadata_size);
  for (int i = 0; i < sparsity.dim_metadata_size; i++) {
    format[i] = sparsity.dim_metadata[i].format;
    dense_size[i] = sparsity.dim_metadata[i].dense_size;
    segments[i] =
        TfLiteIntArrayToVector(sparsity.dim_metadata[i].array_segments);
    indices[i] = TfLiteIntArrayToVector(sparsity.dim_metadata[i].array_indices);
  }

  InitSparseToDenseConverter(shape, std::move(traversal_order),
                             std::move(format), std::move(dense_size),
                             std::move(segments), std::move(indices),
                             std::move(block_map));
}

template <typename T>
void FormatConverter<T>::InitSparseToDenseConverter(
    std::vector<int> shape, std::vector<int> traversal_order,
    std::vector<TfLiteDimensionType> format, std::vector<int> dense_size,
    std::vector<std::vector<int>> segments,
    std::vector<std::vector<int>> indices, std::vector<int> block_map) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_5(mht_5_v, 441, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::InitSparseToDenseConverter");

  dense_shape_ = std::move(shape);
  traversal_order_ = std::move(traversal_order);
  block_map_ = std::move(block_map);
  format_ = std::move(format);

  dense_size_ = 1;
  for (int i = 0; i < dense_shape_.size(); i++) {
    dense_size_ *= dense_shape_[i];
  }

  dim_metadata_.resize(2 * format_.size());
  for (int i = 0; i < format_.size(); i++) {
    if (format_[i] == kTfLiteDimDense) {
      dim_metadata_[2 * i] = {dense_size[i]};
    } else {
      dim_metadata_[2 * i] = std::move(segments[i]);
      dim_metadata_[2 * i + 1] = std::move(indices[i]);
    }
  }

  int original_rank = dense_shape_.size();
  int block_dim = 0;

  blocked_shape_.resize(original_rank);
  block_size_.resize(block_map_.size());
  for (int i = 0; i < original_rank; i++) {
    if (block_dim < block_map_.size() && block_map_[block_dim] == i) {
      if (original_rank + block_dim < traversal_order_.size()) {
        int orig_dim = traversal_order_[original_rank + block_dim];
        block_size_[block_dim] = dense_size[orig_dim];
        blocked_shape_[i] = dense_shape_[i] / dense_size[orig_dim];
        block_dim++;
      }
    } else {
      blocked_shape_[i] = dense_shape_[i];
    }
  }
}

template <typename T>
void FormatConverter<T>::Populate(const T* src_data, std::vector<int> indices,
                                  int level, int prev_idx, int* src_data_ptr,
                                  T* dest_data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_6(mht_6_v, 487, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::Populate");

  if (level == indices.size()) {
    int orig_rank = dense_shape_.size();
    std::vector<int> orig_idx;
    orig_idx.resize(orig_rank);
    int i = 0;
    for (; i < orig_idx.size(); i++) {
      int orig_dim = traversal_order_[i];
      orig_idx[orig_dim] = indices[i];
    }

    for (; i < indices.size(); i++) {
      const int block_idx = traversal_order_[i] - orig_rank;
      const int orig_dim = block_map_[block_idx];
      orig_idx[orig_dim] =
          orig_idx[orig_dim] * block_size_[block_idx] + indices[i];
    }

    dest_data[GetFlattenedIndex(orig_idx, dense_shape_)] =
        src_data[*src_data_ptr];

    *src_data_ptr = *src_data_ptr + 1;
    return;
  }

  const int metadata_idx = 2 * level;
  const int shape_of_level = dim_metadata_[metadata_idx][0];
  if (format_[level] == kTfLiteDimDense) {
    for (int i = 0; i < shape_of_level; i++) {
      indices[level] = i;
      Populate(src_data, indices, level + 1, prev_idx * shape_of_level + i,
               src_data_ptr, dest_data);
    }
  } else if (prev_idx + 1 < dim_metadata_[metadata_idx].size()) {
    const auto& array_segments = dim_metadata_[metadata_idx];
    const auto& array_indices = dim_metadata_[metadata_idx + 1];
    for (int i = array_segments[prev_idx]; i < array_segments[prev_idx + 1];
         i++) {
      if (i < array_indices.size() && level < indices.size()) {
        indices[level] = array_indices[i];
        Populate(src_data, indices, level + 1, i, src_data_ptr, dest_data);
      }
    }
  }
}

template <typename T>
TfLiteStatus FormatConverter<T>::SparseToDense(const T* src_data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_7(mht_7_v, 537, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::SparseToDense");

  data_.resize(dense_size_);
  std::fill(data_.begin(), data_.end(), T(0));

  int total_rank = traversal_order_.size();
  int src_data_ptr = 0;
  std::vector<int> indices(total_rank);
  Populate(src_data, indices, 0, 0, &src_data_ptr, data_.data());

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus FormatConverter<T>::SparseToDense(const T* src_data,
                                               const size_t dest_size,
                                               T* dest_data,
                                               TfLiteContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_8(mht_8_v, 556, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::SparseToDense");

  if (dest_size != dense_size_) {
    TF_LITE_MAYBE_KERNEL_LOG(
        context, "unexpected buffer size for densified data, expected %lld.\n",
        dense_size_);
    return kTfLiteError;
  }

  // For types like Eigen::half, we cannot do a simple memset() with 0 values.
  for (auto i = 0; i < dest_size; i++) {
    dest_data[i] = T(0);
  }

  const int total_rank = traversal_order_.size();
  int src_data_ptr = 0;
  std::vector<int> indices(total_rank);
  Populate(src_data, indices, 0, 0, &src_data_ptr, dest_data);

  return kTfLiteOk;
}

template <typename T>
bool FormatConverter<T>::IsZero(const T val) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTcc mht_9(mht_9_v, 581, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc", "FormatConverter<T>::IsZero");

  return (val == static_cast<T>(0));
}

template class FormatConverter<int32_t>;
template class FormatConverter<int8_t>;
template class FormatConverter<float>;
template class FormatConverter<Eigen::half>;

}  // namespace sparsity
}  // namespace internal
}  // namespace tflite
