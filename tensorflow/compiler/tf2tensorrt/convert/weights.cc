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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc() {
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
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"

#include <functional>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

namespace convert {

TRT_ShapedWeights::TRT_ShapedWeights(nvinfer1::DataType type)
    : shape_(0, DimsAdapter::StorageType{}), type_(type), volume_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::TRT_ShapedWeights");
}

StatusOr<TRT_ShapedWeights> TRT_ShapedWeights::CreateWithTensor(
    nvinfer1::DataType type, DimsAdapter dims, Tensor tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::CreateWithTensor");

  TRT_ShapedWeights weights(type);
  weights.shape_ = dims;
  weights.tensor_ = std::forward<Tensor>(tensor);
  weights.volume_ = weights.shape_.Volume();
  if (weights.shape_.NumDims() == 0) {
    DCHECK(weights.shape_.IsEmpty() || weights.shape_.IsScalar());
  }
  return weights;
}

nvinfer1::Weights TRT_ShapedWeights::GetTrtWeights() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::GetTrtWeights");

  return nvinfer1::Weights{type_, GetPointer<int8>(), volume_};
}

Status TRT_ShapedWeights::SetShape(DimsAdapter dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_3(mht_3_v, 227, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::SetShape");

  if (volume_ != dims.Volume()) {
    VLOG(2) << "Changing shape from " << shape_.DebugString() << ", to "
            << dims.DebugString();
    return errors::Internal("SetShape would change number of elements");
  }
  shape_ = std::move(dims);
  return Status::OK();
}

size_t TRT_ShapedWeights::size_bytes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_4(mht_4_v, 240, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::size_bytes");

  size_t data_type_size = -1;
  switch (type_) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      data_type_size = 4;
      break;
    case nvinfer1::DataType::kHALF:
      data_type_size = 2;
      break;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
      data_type_size = 1;
      break;
  }
  return volume_ * data_type_size;
}

string TRT_ShapedWeights::DebugString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_5(mht_5_v, 261, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_ShapedWeights::DebugString");

  return absl::StrCat(
      "TRT_ShapedWeights(shape=", shape_.DebugString(),
      ", type=", tensorflow::tensorrt::DebugString(type_),
      ", values=", reinterpret_cast<uintptr_t>(GetPointer<int8>()), ")");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor)
    : tensor_proxy_ptr_(tensor), initialized_(true), is_tensor_(true) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_6(mht_6_v, 272, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(ITensorProxyPtr tensor, int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_7(mht_7_v, 281, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::ITensor* tensor,
                                         int batch_size)
    : tensor_proxy_ptr_(tensor),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_8(mht_8_v, 291, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(nvinfer1::DataType trt_dtype,
                                         const nvinfer1::Dims& trt_dims,
                                         int batch_size)
    : tensor_proxy_ptr_(new SimpleITensor(trt_dtype, trt_dims)),
      batch_size_(batch_size),
      initialized_(true),
      is_tensor_(true) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_9(mht_9_v, 302, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_ShapedWeights& weights)
    : weights_(weights), initialized_(true), is_tensor_(false) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_10(mht_10_v, 308, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

TRT_TensorOrWeights::TRT_TensorOrWeights(const TRT_TensorOrWeights& rhs)
    : tensor_proxy_ptr_(rhs.tensor_proxy_ptr_),
      batch_size_(rhs.batch_size_),
      weights_(rhs.weights_),
      initialized_(rhs.initialized_),
      is_tensor_(rhs.is_tensor_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_11(mht_11_v, 318, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::TRT_TensorOrWeights");
}

void TRT_TensorOrWeights::operator=(const TRT_TensorOrWeights& rhs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_12(mht_12_v, 323, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "=");

  tensor_proxy_ptr_ = rhs.tensor_proxy_ptr_;
  batch_size_ = rhs.batch_size_;
  weights_ = rhs.weights_;
  initialized_ = rhs.initialized_;
  is_tensor_ = rhs.is_tensor_;
}

ITensorProxyPtr TRT_TensorOrWeights::tensor() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_13(mht_13_v, 334, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::tensor");

  DCHECK(is_tensor());
  return tensor_proxy_ptr_;
}

nvinfer1::Dims TRT_TensorOrWeights::GetTrtDims() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_14(mht_14_v, 342, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::GetTrtDims");

  if (is_tensor()) {
    return tensor()->getDimensions();
  }
  return weights().Shape().AsTrtDims();
}

Status TRT_TensorOrWeights::GetTfType(DataType* tf_type) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_15(mht_15_v, 352, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::GetTfType");

  if (is_tensor()) {
    nvinfer1::DataType trt_type = tensor()->getType();
    return TrtTypeToTfType(trt_type, tf_type);
  }
  if (is_weights()) {
    *tf_type = weights().GetTensor().dtype();
    return Status::OK();
  }
  return errors::Internal("The object is probably not initialized");
}

string TRT_TensorOrWeights::DebugString() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_16(mht_16_v, 367, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TRT_TensorOrWeights::DebugString");

  string output = "TRT_TensorOrWeights(type=";
  if (is_tensor()) {
    absl::StrAppend(&output,
                    "tensor=", tensorflow::tensorrt::DebugString(tensor()),
                    ", batch_size=", batch_size_);
  } else {
    absl::StrAppend(&output, "weights=", weights_.DebugString());
  }
  absl::StrAppend(&output, ")");
  return output;
}

StatusOr<TRT_ShapedWeights> TrtWeightStore::GetTempWeights(
    nvinfer1::DataType trt_dtype, const DimsAdapter& dims) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSweightsDTcc mht_17(mht_17_v, 384, "", "./tensorflow/compiler/tf2tensorrt/convert/weights.cc", "TrtWeightStore::GetTempWeights");

  DataType tf_dtype;
  TF_RETURN_IF_ERROR(TrtTypeToTfType(trt_dtype, &tf_dtype));
  TensorShape shape;
  TF_RETURN_IF_ERROR(dims.TensorShape(&shape));
  // TODO(jie): check weights size_bytes. 0 means type error
  Tensor tensor(tf_dtype, shape);
  StatusOr<TRT_ShapedWeights> weights =
      TRT_ShapedWeights::CreateWithTensor(trt_dtype, dims, tensor);
  TRT_ENSURE_OK(weights);
  store_.emplace_back(std::move(tensor));
  return weights;
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
