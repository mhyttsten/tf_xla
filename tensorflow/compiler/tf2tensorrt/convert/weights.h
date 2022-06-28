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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh() {
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


#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

// Class to convert TF compile-time constants (e.g. Const nodes) to TRT weight.
class TRT_ShapedWeights {
 public:
  explicit TRT_ShapedWeights(
      nvinfer1::DataType type = nvinfer1::DataType::kFLOAT);

  // Constructs a weights from another weights.
  //
  // NOTE: this does not copy the underlying buffer but only increase its
  // reference count.
  TRT_ShapedWeights(const TRT_ShapedWeights& rhs) = default;

  nvinfer1::Weights GetTrtWeights() const;

  const Tensor& GetTensor() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_0(mht_0_v, 216, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "GetTensor");
 return tensor_; }

  // Returns a pointer of type const T to the underlying buffer of the tensor.
  template <typename T>
  const T* GetPointer() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_1(mht_1_v, 223, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "GetPointer");

    int64 num_elem =
        (tensor_.NumElements() * DataTypeSize(tensor_.dtype())) / sizeof(T);
    return tensor_.bit_casted_shaped<T, 1>({num_elem}).data();
  }

  // Returns a pointer of type T to the underlying buffer of the tensor.
  template <typename T>
  T* GetPointer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_2(mht_2_v, 234, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "GetPointer");

    int64 num_elem =
        (tensor_.NumElements() * DataTypeSize(tensor_.dtype())) / sizeof(T);
    return tensor_.bit_casted_shaped<T, 1>({num_elem}).data();
  }

  // Fills all the weight values with value.
  template <typename T>
  Status SetValues(T value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_3(mht_3_v, 245, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "SetValues");

    switch (type_) {
      case nvinfer1::DataType::kFLOAT: {
        float* ptr = tensor_.flat<float>().data();
        std::fill(ptr, ptr + volume_, value);
        break;
      }
      case nvinfer1::DataType::kHALF: {
        Eigen::half* ptr = tensor_.flat<Eigen::half>().data();
        std::fill(ptr, ptr + volume_, Eigen::half(value));
        break;
      }
      case nvinfer1::DataType::kINT32: {
        int32* ptr = tensor_.flat<int32>().data();
        std::fill(ptr, ptr + volume_, value);
        break;
      }
      default:
        return errors::InvalidArgument(
            "Unsupported data type ", tensorflow::tensorrt::DebugString(type_));
    }
    return Status::OK();
  }

  Status SetShape(DimsAdapter dims);
  void SetShapeUnsafe(DimsAdapter dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_4(mht_4_v, 273, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "SetShapeUnsafe");
 shape_ = std::move(dims); }

  // Returns total number of elements. Returning 0 means either some dim is 0
  // or the number of dims is 0. Note that a TF scalar constant is marked as
  // Dims{0, {1}}, and has a count() == 1.
  int64_t count() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_5(mht_5_v, 281, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "count");
 return volume_; }

  size_t size_bytes() const;

  string DebugString() const;

  template <typename T>
  absl::Span<const T> GetSpan() const {
    return absl::Span<const T>(tensor_.flat<T>().data(), volume_);
  }

  template <typename T>
  std::vector<T> ToVector() const {
    auto span = GetSpan<T>();
    return std::vector<T>(span.data(), span.data() + span.size());
  }

  nvinfer1::DataType TrtDType() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_6(mht_6_v, 301, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "TrtDType");
 return type_; }

  const DimsAdapter& Shape() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_7(mht_7_v, 306, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "Shape");
 return shape_; }
  DimsAdapter& Shape() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_8(mht_8_v, 310, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "Shape");
 return shape_; }

 private:
  // The shape of the weights. Defaults to the empty shape.
  DimsAdapter shape_;

  // This creation method is only used by TrtWeightStore, which creates the
  // underlying buffer.
  static StatusOr<TRT_ShapedWeights> CreateWithTensor(nvinfer1::DataType type,
                                                      DimsAdapter dims,
                                                      Tensor tensor);

  nvinfer1::DataType type_;

  // All weights should be stored inside TrtWeightStore to make sure lifetime of
  // all the underlying tensors are available until the engine is built. For
  // this reason, tensor_ should never be reassigned to a different value that
  // is not already present in the TrtWeightStore.
  Tensor tensor_;
  // Contains the volume of the weight's shape.
  int64_t volume_;

  friend class TrtWeightStore;
};

// Container for TRT_ShapedWeights. We need this container because TRT does not
// manage the lifetime of the weights buffer, it only keeps a pointer to it and
// requires that the data referenced by the pointer be available until the
// building of engine is complete. For more information see
// https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_weights.html
//
// TODO(laigd): consider adding garbage collection to the unused weights.
class TrtWeightStore {
 public:
  // Gets a TRT_ShapedWeights with 'type' and 'dims'.
  StatusOr<TRT_ShapedWeights> GetTempWeights(nvinfer1::DataType trt_type,
                                             const DimsAdapter& dims);

  // Gets a TRT_ShapedWeights with the same data type and dimensions as
  // 'weights'.
  StatusOr<TRT_ShapedWeights> GetTempWeights(const TRT_ShapedWeights& weights) {
    return GetTempWeights(weights.TrtDType(), weights.Shape());
  }

 private:
  // The backend storage of the TRT_ShapedWeights.
  std::vector<Tensor> store_;
};

// Represents a TRT-style input to a TF node, it can be either a
// ITensorProxyPtr (representing nvinfer1::ITensor* or SimpleITensor),
// or TRT_ShapedWeights which is compile-time constant.
//
// TODO(laigd): maybe rename it to TrtArgument, or mimic XlaCompiler::Argument.
class TRT_TensorOrWeights {
 public:
  TRT_TensorOrWeights() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_9(mht_9_v, 369, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "TRT_TensorOrWeights");
}
  TRT_TensorOrWeights(ITensorProxyPtr);
  TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size);

  // Constructs a wrapper for the given ITensor.
  // This is used by Converter when building the TRT network, where the ITensor
  // is owned by the TRT network being built. See comment for 'trt_tensor_'
  // in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::ITensor* tensor, int batch_size = -1);

  // Creates a SimpleITensor for trt_dtype and trt_dims and takes ownership of
  // the object. Constructs a wrapper for the SimpleITensor. This is used by
  // TrtNodeValidator to encapsulate the type and shape information for
  // validation of graph nodes, and the created ITensor is fake and temporary,
  // and should not be used to build any TRT network. See comment for
  // 'simple_tensor_' in trt_proxy_tensor.h.
  explicit TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                               const nvinfer1::Dims& trt_dims, int batch_size);

  // Constructs a wrapper for the given weights.
  explicit TRT_TensorOrWeights(const TRT_ShapedWeights& weights);

  TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs);

  void operator=(const TRT_TensorOrWeights& rhs);

  bool is_tensor() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_10(mht_10_v, 398, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "is_tensor");
 return initialized_ && is_tensor_; }
  bool is_weights() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_11(mht_11_v, 402, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "is_weights");
 return initialized_ && !is_tensor_; }

  ITensorProxyPtr tensor() const;

  TRT_ShapedWeights& weights() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_12(mht_12_v, 409, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "weights");

    DCHECK(is_weights());
    return weights_;
  }

  const TRT_ShapedWeights& weights() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_13(mht_13_v, 417, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "weights");

    DCHECK(is_weights());
    return weights_;
  }

  nvinfer1::Dims GetTrtDims() const;

  Status GetTfType(DataType* tf_type) const;

  int batch_size() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_14(mht_14_v, 429, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "batch_size");
 return batch_size_; }

  string DebugString() const;

  nvinfer1::DataType TrtDType() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_15(mht_15_v, 436, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "TrtDType");

    return is_tensor_ ? tensor_proxy_ptr_->getType() : weights_.TrtDType();
  }

 private:
  void set_batch_size(int batch_size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTh mht_16(mht_16_v, 444, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.h", "set_batch_size");
 batch_size_ = batch_size; }

  // First dimension of the TF tensor (NOT tensor_) that is represented by
  // tensor_ is treated as the "batch dimension" by TRT, and tensor_'s
  // dimensions (obtained via tensor_->getDimensions()) do not contain the batch
  // dimension. For example, when a TF tensor with shape (A,B,C) is represented
  // in TRT, tensor_->getDimensions() will be (B,C) and batch_size_ will be A.
  //
  // This requires that all tensors in the subgraph that is converted to a TRT
  // engine have the same batch size are represented by the first dimension of
  // their shape, and Converter will verify this during conversion. The drawback
  // is that currently it cannot convert a graph that doesn't have the batch
  // size represented in the shapes or the batch sizes are different. See
  // b/118387490 for more details.
  //
  // If use_implicit_batch is false, batch_size_ is unused and
  // tensor_->getDimensions() will contain the entire shape (A,B,C).
  ITensorProxyPtr tensor_proxy_ptr_ = nullptr;
  int batch_size_ = -1;

  TRT_ShapedWeights weights_;
  bool initialized_ = false;
  bool is_tensor_ = false;

  friend class Converter;
};
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_TRT_WEIGHTS_H_
