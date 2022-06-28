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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTh() {
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


#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace internal {
namespace sparsity {

// A converter that keeps an internal representation of sparse tensor parameters
// and converts tensors between dense and sparse formats.
template <typename T>
class FormatConverter {
 public:
  /*
   * Creates a dense to sparse converter.
   * @param shape             Shape of the dense tensor.
   * @param traversal_order   In what order to traverse all dimensions,
   *                          including block dimensions.
   * @param format            Whether each dimension in the dense tensor is
   *                          dense or sparse (not in the traversal order).
   * @param block_size        Size of each block dimension.
   * @param block_map         Map from block dimension to original tensor
   *                          dimension.
   */
  FormatConverter(const std::vector<int>& shape,
                  const std::vector<int>& traversal_order,
                  const std::vector<TfLiteDimensionType>& format,
                  const std::vector<int>& block_size = {},
                  const std::vector<int>& block_map = {});

  /*
   * Creates a sparse to dense converter.
   * @param shape             Shape of the target dense tensor.
   * @param traversal_order   In what order to traverse all dimensions,
   *                          including block dimensions.
   * @param format            Whether each dimension in the dense tensor is
   *                          dense or sparse (not in the traversal order).
   * @param dense_size        Size of each dense dimension in the sparse tensor.
   *                          Should be 0 for sparse dimensions.
   * @param segments          Segments of each dimension in the sparse tensor.
   *                          Should be empty for dense dimensions.
   * @param indices           Indices in the dense tensor for each dimension.
   *                          Should be empty for dense dimensions.
   * @param block_map         Map from block dimension to original tensor
   *                          dimension.
   */
  FormatConverter(const std::vector<int>& shape,
                  const std::vector<int>& traversal_order,
                  const std::vector<TfLiteDimensionType>& format,
                  const std::vector<int>& dense_size,
                  const std::vector<std::vector<int>>& segments,
                  const std::vector<std::vector<int>>& indices,
                  const std::vector<int>& block_map = {});

  /* Creates a sparse to dense converter.
   * @param shape      Shape of the target dense tensor.
   * @param sparsity   Sparsity parameter of the sparse TfLiteTensor.
   */
  FormatConverter(const std::vector<int>& shape,
                  const TfLiteSparsity& sparsity);

  const std::vector<T>& GetData() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTh mht_0(mht_0_v, 249, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h", "GetData");
 return data_; }
  const std::vector<std::vector<int>>& GetDimMetadata() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSutilsPSsparsity_format_converterDTh mht_1(mht_1_v, 253, "", "./tensorflow/lite/kernels/internal/utils/sparsity_format_converter.h", "GetDimMetadata");

    return dim_metadata_;
  }

  // Method for dense to sparse conversion. Need to call GetData() method to get
  // the compressed data.
  TfLiteStatus DenseToSparse(const T* src_data);

  // Method for sparse to dense conversion. Need to call GetData() method to get
  // the decompressed data.
  TfLiteStatus SparseToDense(const T* src_data);
  // Method for sparse to dense conversion with caller provided buffer. No need
  // to call GetData() with this method.
  TfLiteStatus SparseToDense(const T* src_data, const size_t dest_size,
                             T* dest_data, TfLiteContext* context = nullptr);

 private:
  // Helper function for initializing this converter for sparse to dense
  // conversion.
  void InitSparseToDenseConverter(std::vector<int> shape,
                                  std::vector<int> traversal_order,
                                  std::vector<TfLiteDimensionType> format,
                                  std::vector<int> dense_size,
                                  std::vector<std::vector<int>> segments,
                                  std::vector<std::vector<int>> indices,
                                  std::vector<int> block_map);

  // A recursive function to fetch data from the compressed src_data buffer and
  // populate the dense buffer.
  void Populate(const T* src_data, std::vector<int> indices, int level,
                int prev_idx, int* src_data_ptr, T* dest_data);

  // Check if val is equal to zero.
  bool IsZero(const T val);

  // Shape of the conceptual dense tensor.
  std::vector<int> dense_shape_;
  // Shape of the dense tensor with inner blocks reduced. For example, a (4, 4)
  // tensor with (2, 2) block has blocked_shape (2, 2).
  std::vector<int> blocked_shape_;
  // Total number of elements in the dense tensor.
  size_t dense_size_;
  // Has n(original dimension)+k(block_dimension) elements.
  std::vector<int> traversal_order_;
  // Format of each dimension in the traversal order.
  std::vector<TfLiteDimensionType> format_;
  // Size of each block dimension, in the same order as block map.
  std::vector<int> block_size_;
  // Map from block dimension to the original tensor dimension.
  std::vector<int> block_map_;
  // Metadata of each dimension in the traversal order.
  // Each dimension needs two vectors. For dense dimensions, the first vector
  // stores the size of that dimension, and the second vector is empty. For
  // sparse dimensions, the first vector stores the segments and the second one
  // stores the indices.
  std::vector<std::vector<int>> dim_metadata_;
  // Actual buffer holding data after conversion. Could be sparse buffer or
  // dense buffer.
  std::vector<T> data_;
};

extern template class FormatConverter<int32_t>;
extern template class FormatConverter<int8_t>;
extern template class FormatConverter<float>;
extern template class FormatConverter<Eigen::half>;
}  // namespace sparsity
}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_
