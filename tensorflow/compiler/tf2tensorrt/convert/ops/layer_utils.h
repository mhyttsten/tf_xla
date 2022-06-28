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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh() {
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

#include <type_traits>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/statusor.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {

namespace convert {

// Facilitates the creation of TensorRT layers inside a network. The user
// provides a INetworkDefinition pointer during construction. They can then add
// operations to the network through the provided functions. Each function
// returns a struct which contains the symbolic result of the operation (ITensor
// pointer) as well as a pointer to the last TensorRT ILayer created. Some
// operations may create multiple layers in order to accomplish the desired
// result (e.g. Sign).
class TRTNetworkBuilder {
 public:
  static StatusOr<TRTNetworkBuilder> Create(
      nvinfer1::INetworkDefinition* network, TrtWeightStore* weight_store) {
    TRT_ENSURE(network);
    TRT_ENSURE(weight_store);
    return TRTNetworkBuilder(network, weight_store);
  }

 private:
  TRTNetworkBuilder(nvinfer1::INetworkDefinition* network,
                    TrtWeightStore* weight_store)
      : network_(network), weight_store_(weight_store) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_0(mht_0_v, 222, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "TRTNetworkBuilder");
}

 public:
  // Adds an Add operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Add(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kSUM);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an elementwise min(lhs, rhs) operation to the network. The output has
  // the same data type as the input.
  StatusOr<nvinfer1::IElementWiseLayer*> Min(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kMIN);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an elementwise max(lhs, rhs) operation to the network. The output has
  // the same datatype as the input.
  StatusOr<nvinfer1::IElementWiseLayer*> Max(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kMAX);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an absolute value operation to the network. Note that this unary
  // operation will do an implict float conversion. For int32 tensors, use
  // "AbsInt".
  StatusOr<nvinfer1::IUnaryLayer*> AbsFloat(nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);
    TRT_ENSURE(input->getType() != nvinfer1::DataType::kFLOAT &&
               input->getType() != nvinfer1::DataType::kHALF);
    nvinfer1::IUnaryLayer* layer =
        network_->addUnary(*input, nvinfer1::UnaryOperation::kABS);
    TRT_ENSURE(layer);
    return layer;
  }

  // Performs Abs without implict float conversion. The input should be of type
  // kInt32. For float datatypes, use "Abs".
  StatusOr<nvinfer1::IElementWiseLayer*> AbsInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);
    TRT_ENSURE(input->getType() == nvinfer1::DataType::kINT32);
    StatusOr<nvinfer1::IElementWiseLayer*> sign = this->SignInt(input);
    return this->Mul(input, (*sign)->getOutput(0));
  }

  // Returns elementwise sign(x) for int32 input tensors where sign(x) is
  // defined as 1 where x > 0, -1 where x < 0 and 0 where x == 0.
  StatusOr<nvinfer1::IElementWiseLayer*> SignInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);

    // Create constants +1 and -1.
    StatusOr<nvinfer1::IConstantLayer*> one =
        this->Constant<int32>(1, input->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(one);

    StatusOr<nvinfer1::IConstantLayer*> neg_one =
        this->Constant<int32>(-1, input->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(neg_one);

    // Turn all negaitve elements into -1, positive and zero elements
    // unaffected.
    StatusOr<nvinfer1::IElementWiseLayer*> max =
        this->Max(input, (*neg_one)->getOutput(0));
    TRT_ENSURE_PTR_OK(max);

    // Turn all positive elements into +1, negative and zero elements
    // unaffected.
    StatusOr<nvinfer1::IElementWiseLayer*> min =
        this->Min((*max)->getOutput(0), (*one)->getOutput(0));
    TRT_ENSURE_PTR_OK(min);
    return min;
  }

  // Adds a Sub operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Sub(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kSUB);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds an Greater operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Greater(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kGREATER);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds an Equal operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Equal(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kEQUAL);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a FloorDiv operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> FloorDiv(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
    TRT_ENSURE(layer);
    return layer;
  }

  // Returns the equivalent of ceil_divide(abs(x)/abs(y))) operation. The inputs
  // "lhs" and "rhs" should be int32 tensors.
  StatusOr<nvinfer1::IElementWiseLayer*> AbsCeilDivInt(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    TRT_ENSURE(lhs->getType() == nvinfer1::DataType::kINT32);
    TRT_ENSURE(rhs->getType() == nvinfer1::DataType::kINT32);

    StatusOr<nvinfer1::IElementWiseLayer*> rhs_abs = this->AbsInt(rhs);
    TRT_ENSURE_PTR_OK(rhs_abs);
    StatusOr<nvinfer1::IElementWiseLayer*> lhs_abs = this->AbsInt(lhs);
    TRT_ENSURE_PTR_OK(lhs_abs);
    StatusOr<nvinfer1::IElementWiseLayer*> add1 =
        this->Add((*lhs_abs)->getOutput(0), (*rhs_abs)->getOutput(0));
    TRT_ENSURE_PTR_OK(add1);
    StatusOr<nvinfer1::IConstantLayer*> one_const =
        this->Constant<int32>(1, rhs->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(one_const);
    StatusOr<nvinfer1::IElementWiseLayer*> numerator =
        this->Sub((*add1)->getOutput(0), (*one_const)->getOutput(0));
    TRT_ENSURE_PTR_OK(numerator);
    return FloorDiv((*numerator)->getOutput(0), (*rhs_abs)->getOutput(0));
  }

  // Adds an elementwise multiplication operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Mul(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kPROD);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a sequence of elementwise multiplication operations to the network.
  // The returned layer's output contains the cumulative elementwise product of
  // all tensors in the input.
  StatusOr<nvinfer1::ILayer*> CumulativeProd(
      absl::Span<nvinfer1::ITensor*> inputs) noexcept {
    TRT_ENSURE(!absl::c_any_of(
        inputs, [](nvinfer1::ITensor* x) { return x == nullptr; }));
    nvinfer1::ILayer* out = nullptr;
    if (inputs.size() == 1) {
      out = network_->addIdentity(*inputs[0]);
      TRT_ENSURE(out != nullptr);
      return out;
    }
    nvinfer1::ITensor* last = inputs[0];
    for (int i = 1; i < inputs.size(); i++) {
      StatusOr<nvinfer1::IElementWiseLayer*> mul = this->Mul(last, inputs[i]);
      TRT_ENSURE_PTR_OK(mul);
      out = *mul;
      last = (*mul)->getOutput(0);
    }
    return out;
  }

  // Adds a Constant layer whose output is a TensorRT shape tensor. The shape
  // tensor's size and values correspond to dim's nbDims and d[], respectively.
  StatusOr<nvinfer1::IConstantLayer*> ConstantShape(
      const DimsAdapter& shape_data) noexcept {
    TRT_ENSURE(shape_data.NumDims() > 0);
    nvinfer1::Dims shape_dims;
    shape_dims.nbDims = 1;
    shape_dims.d[0] = shape_data.NumDims();
    StatusOr<TRT_ShapedWeights> const_weights =
        weight_store_->GetTempWeights(nvinfer1::DataType::kINT32, shape_dims);
    TRT_ENSURE_OK(const_weights);
    absl::c_copy(shape_data, const_weights->GetPointer<int32>());
    StatusOr<nvinfer1::Dims> trt_dims = const_weights->Shape().AsTrtDims();
    TRT_ENSURE_OK(trt_dims);
    nvinfer1::IConstantLayer* const_layer =
        network_->addConstant(*trt_dims, const_weights->GetTrtWeights());
    TRT_ENSURE(const_layer);
    nvinfer1::ITensor* output = const_layer->getOutput(0);
    TRT_ENSURE(output);
    TRT_ENSURE(output->getType() == nvinfer1::DataType::kINT32);
    return const_layer;
  }

  // Adds a Constant layer whose output is a TensorRT shape tensor. The shape
  // tensor's size and values correspond to dim's nbDims and d[], respectively.
  StatusOr<nvinfer1::IConstantLayer*> Constant(
      const std::vector<int>& data) noexcept {
    nvinfer1::Dims shape_dims;
    shape_dims.nbDims = 1;
    shape_dims.d[0] = data.size();
    StatusOr<TRT_ShapedWeights> const_weights =
        weight_store_->GetTempWeights(nvinfer1::DataType::kINT32, shape_dims);
    TRT_ENSURE_OK(const_weights);
    int32* values = const_weights->GetPointer<int32>();
    for (int i = 0; i < data.size(); i++) {
      values[i] = static_cast<int32>(data[i]);
    }
    StatusOr<nvinfer1::Dims> trt_dims = const_weights->Shape().AsTrtDims();
    TRT_ENSURE_OK(trt_dims);
    nvinfer1::IConstantLayer* const_layer =
        network_->addConstant(*trt_dims, const_weights->GetTrtWeights());
    TRT_ENSURE(const_layer);
    nvinfer1::ITensor* output = const_layer->getOutput(0);
    TRT_ENSURE(output);
    TRT_ENSURE(output->getType() == nvinfer1::DataType::kINT32);
    TRT_ENSURE(const_layer);
    return const_layer;
  }

  // Adds a Constant layer that produces a tensor of shape "shape",
  // type "data_type" and filled with value "scalar".
  template <typename T>
  StatusOr<nvinfer1::IConstantLayer*> Constant(
      const T value, nvinfer1::Dims shape,
      nvinfer1::DataType data_type) noexcept {
    StatusOr<TRT_ShapedWeights> const_weights =
        weight_store_->GetTempWeights(data_type, shape);
    TRT_ENSURE_OK(const_weights);
    TRT_ENSURE(const_weights->SetValues(value).ok());
    nvinfer1::IConstantLayer* const_layer =
        network_->addConstant(shape, const_weights->GetTrtWeights());
    TRT_ENSURE(const_layer);
    return const_layer;
  }

  // Adds a Constant layer that produces a tensor with a single value "scalar".
  // The tensor has "nb_dims" dimensions and each dimension has only one
  // element. The data type of the tensor is determined by the data type of
  // "scalar".
  template <typename T,
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  StatusOr<nvinfer1::IConstantLayer*> Constant(const T scalar,
                                               const int nb_dims) noexcept {
    TRT_ENSURE(nb_dims <= nvinfer1::Dims::MAX_DIMS);
    auto data_type = nvinfer1::DataType::kINT32;
    if (std::is_floating_point<T>::value) {
      data_type = nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::Dims zero_shape;
    zero_shape.nbDims = nb_dims;
    std::fill_n(zero_shape.d, nb_dims, 1);
    return Constant<T>(scalar, zero_shape, data_type);
  }

  // Adds a Constant layer from a TRT_ShapedWeights object.
  StatusOr<nvinfer1::IConstantLayer*> WeightsToConstant(
      const nvinfer1::Weights& weights, const DimsAdapter& dims) noexcept {
    StatusOr<int64_t> vol = dims.Volume();
    TRT_ENSURE_OK(vol);
    TRT_ENSURE(*vol == weights.count);
    StatusOr<nvinfer1::Dims> trt_dims = dims.AsTrtDims();
    TRT_ENSURE_OK(trt_dims);
    nvinfer1::IConstantLayer* const_layer =
        network_->addConstant(*trt_dims, weights);
    TRT_ENSURE(const_layer);
    return const_layer;
  }

  // Creates a nvinfer1::Weights object containing a single scalar.
  template <typename T,
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  StatusOr<nvinfer1::Weights> ScalarWeights(const T scalar,
                                            const int nb_dims) noexcept {
    TRT_ENSURE(nb_dims <= nvinfer1::Dims::MAX_DIMS);
    auto data_type = nvinfer1::DataType::kINT32;
    if (std::is_floating_point<T>::value) {
      data_type = nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::Dims weights_shape;
    weights_shape.nbDims = nb_dims;
    std::fill_n(weights_shape.d, nb_dims, 1);
    StatusOr<TRT_ShapedWeights> const_weights =
        weight_store_->GetTempWeights(data_type, weights_shape);
    TRT_ENSURE_OK(const_weights);
    const_weights->GetPointer<T>()[0] = scalar;
    return const_weights->GetTrtWeights();
  }

  // Adds a TensorRT Slice operation to the network.
  StatusOr<nvinfer1::ISliceLayer*> Slice(
      nvinfer1::ITensor* input, const nvinfer1::Dims& begin,
      const nvinfer1::Dims& size, const nvinfer1::Dims& stride) noexcept {
    nvinfer1::ISliceLayer* layer =
        network_->addSlice(*input, begin, size, stride);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a TensorRT Concatenate operation to the network.
  StatusOr<nvinfer1::IConcatenationLayer*> Concat(
      absl::Span<nvinfer1::ITensor* const> inputs, const int axis) {
    for (nvinfer1::ITensor* input : inputs) {
      TRT_ENSURE(input);
    }
    nvinfer1::IConcatenationLayer* layer = network_->addConcatenation(
        inputs.data(), static_cast<int32_t>(inputs.size()));
    TRT_ENSURE(layer);
    layer->setAxis(axis);
    return layer;
  }

  // Adds a TensorRT Concatenate operation to the network.
  StatusOr<nvinfer1::IConcatenationLayer*> Concat(
      const std::vector<nvinfer1::ITensor*>& inputs, const int axis) {
    return this->Concat(absl::MakeSpan(inputs), axis);
  }

  // Adds a TensorRT Shape operation, which determines the runtime shape of the
  // input tensor, to the network.
  StatusOr<nvinfer1::IShapeLayer*> Shape(nvinfer1::ITensor* input) {
    TRT_ENSURE(input);
    nvinfer1::IShapeLayer* layer = network_->addShape(*input);
    TRT_ENSURE(layer);
    return layer;
  }

  // Creates a Gather operation on the shape of the input tensor. The output of
  // the gather operation is a 1D shape tensor where output[i] = (!sub_one ?
  // input_shape[i] : input_shape[i] -1) if i is in "indices", otherwise zero.
  StatusOr<nvinfer1::IGatherLayer*> GetPartialShapeOf(
      nvinfer1::ITensor* input, absl::InlinedVector<int64, 4> indices,
      bool sub_one = false) {
    TRT_ENSURE(input);
    TRT_ENSURE(indices.size() <= nvinfer1::Dims::MAX_DIMS);

    // Get the runtime shape of input;
    StatusOr<nvinfer1::IShapeLayer*> shape_layer = this->Shape(input);
    TRT_ENSURE_PTR_OK(shape_layer);
    nvinfer1::ITensor* runtime_shape = (*shape_layer)->getOutput(0);

    if (sub_one) {
      StatusOr<nvinfer1::IConstantLayer*> ones = this->Constant<int32>(1, 1);
      TRT_ENSURE_PTR_OK(ones);
      StatusOr<nvinfer1::IElementWiseLayer*> sub =
          this->Sub(runtime_shape, (*ones)->getOutput(0));
      TRT_ENSURE_PTR_OK(sub);
      runtime_shape = (*sub)->getOutput(0);
    }

    // Create a constant tensor containing the gather indices.
    // For any dim not in "indices", we mark it size to gather a zero.
    const int input_nb_dims = input->getDimensions().nbDims;
    std::vector<int> indices_all(input_nb_dims, input_nb_dims);
    for (auto idx : indices) {
      TRT_ENSURE(idx < input_nb_dims);
      indices_all[idx] = idx;
    }

    StatusOr<nvinfer1::IConstantLayer*> indices_result =
        this->Constant(indices_all);
    TRT_ENSURE_PTR_OK(indices_result);
    nvinfer1::ITensor* gather_indices = (*indices_result)->getOutput(0);
    TRT_ENSURE(gather_indices->getDimensions().nbDims == 1);
    TRT_ENSURE(gather_indices->getType() == nvinfer1::DataType::kINT32);

    // Append a zero to the shape tensor.
    StatusOr<nvinfer1::IConstantLayer*> zero_result =
        this->Constant(std::vector<int>{0});
    TRT_ENSURE_PTR_OK(zero_result);
    std::array<nvinfer1::ITensor*, 2> cat_inputs = {
        runtime_shape, (*zero_result)->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat_layer =
        network_->addConcatenation(cat_inputs.data(), cat_inputs.size());
    TRT_ENSURE(cat_layer);
    nvinfer1::ITensor* gather_input = cat_layer->getOutput(0);
    TRT_ENSURE(gather_input);

    // Finally, gather the indices from the input.
    nvinfer1::IGatherLayer* gather =
        network_->addGather(*gather_input, *gather_indices, 0);
    TRT_ENSURE(gather);
    return gather;
  }

  // Adds a scale layer that uniformly scales the input tensor by the specified
  // amount.
  StatusOr<nvinfer1::IScaleLayer*> AddUniformScale(nvinfer1::ITensor* input,
                                                   float scale,
                                                   const std::string& name) {
    TRT_ENSURE(input);
    TRT_ENSURE(!name.empty());
    StatusOr<nvinfer1::Weights> weight = this->ScalarWeights<float>(scale, 1);
    TRT_ENSURE_OK(weight);
    const nvinfer1::Weights empty_weights =
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
    nvinfer1::IScaleLayer* scale_layer =
        network_->addScale(*input, nvinfer1::ScaleMode::kUNIFORM, empty_weights,
                           (*weight), empty_weights);
    TRT_ENSURE(scale_layer != nullptr);
    scale_layer->setName(name.c_str());
    TRT_ENSURE((*scale_layer).getPower().count == 0);
    TRT_ENSURE((*scale_layer).getShift().count == 0);
    TRT_ENSURE((*scale_layer).getScale().count == 1);
    return scale_layer;
  }

  // Adds a quantization layer that uniformly scales the input tensor
  // by the given multiplicative "scaling_factor", then rounds
  // (round-to-nearest-ties-to-even) to the nearest integer and clamps in the
  // range of [-128, 127].
  StatusOr<nvinfer1::ILayer*> Quantize(nvinfer1::ITensor* input,
                                       const float scaling_factor,
                                       const std::string& name) {
    TRT_ENSURE(input);
    TRT_ENSURE(!name.empty());
    // Preprocessor usage here is unavoidable because TRT8 API is new.
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    // The TensorRT IQuantizeLayer divides by the scale factor rather than
    // multiplies. To be consistent, in this function we expect a multiplicative
    // scale factor, so we take the reciprical.
    StatusOr<nvinfer1::IConstantLayer*> scaling_const =
        this->Constant<float>(1.0f / scaling_factor, 1);
    TRT_ENSURE_PTR_OK(scaling_const);
    (*scaling_const)->setDimensions(nvinfer1::Dims{0, {}});
    nvinfer1::IQuantizeLayer* quant_layer =
        network_->addQuantize(*input, *(*scaling_const)->getOutput(0));
    TRT_ENSURE(quant_layer);
    quant_layer->setAxis(1);
    return quant_layer;
#else
    StatusOr<nvinfer1::IScaleLayer*> result =
        this->AddUniformScale(input, scaling_factor, name);
    TRT_ENSURE_PTR_OK(result);
    (*result)->setOutputType(0, nvinfer1::DataType::kINT8);
    (*result)->setPrecision(nvinfer1::DataType::kFLOAT);
    return result;
#endif
  }

  // Adds a dequantize layer that casts the input tensor to TensorRT float type
  // and scales it uniformly by the given multiplicative "scaling_factor".
  StatusOr<nvinfer1::ILayer*> Dequantize(nvinfer1::ITensor* input,
                                         const float scaling_factor,
                                         const std::string& name) {
    TRT_ENSURE(input);
    TRT_ENSURE(!name.empty());
#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    StatusOr<nvinfer1::IConstantLayer*> scaling_const =
        this->Constant<float>(scaling_factor, 1);
    TRT_ENSURE_PTR_OK(scaling_const);
    (*scaling_const)->setDimensions(nvinfer1::Dims{0, {}});
    nvinfer1::IDequantizeLayer* dequant_layer =
        network_->addDequantize(*input, *(*scaling_const)->getOutput(0));
    dequant_layer->setAxis(1);
    TRT_ENSURE(dequant_layer);
    return dequant_layer;
#else
    StatusOr<nvinfer1::IScaleLayer*> result =
        this->AddUniformScale(input, scaling_factor, name);
    TRT_ENSURE_PTR_OK(result);
    (*result)->setOutputType(0, nvinfer1::DataType::kFLOAT);
    (*result)->setPrecision(nvinfer1::DataType::kINT8);
    return result;
#endif
  }

  // Adds TensorRT Q/DQ operations. This is for explicit precision mode.
  StatusOr<nvinfer1::ILayer*> UniformQuantizeDequantizeExplicit(
      nvinfer1::ITensor* input, float quantize_scale, float dequantize_scale,
      const std::string& name) {
    TRT_ENSURE(input);
    if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {
      TRT_ENSURE(network_->hasExplicitPrecision());
    }
    TRT_ENSURE(IS_TRT_VERSION_GE(7, 1, 0, 0));

    static int count = 0;
    TRT_ENSURE(input->getType() == nvinfer1::DataType::kFLOAT);
    std::string quant_name = absl::StrCat(input->getName(), "_quant_", count);

    StatusOr<nvinfer1::ILayer*> quant =
        this->Quantize(input, quantize_scale, quant_name);
    TRT_ENSURE_PTR_OK(quant);

    std::string dequant_name =
        absl::StrCat(input->getName(), "_dequant_", count);
    StatusOr<nvinfer1::ILayer*> dequant = this->Dequantize(
        (*quant)->getOutput(0), dequantize_scale, dequant_name);
    TRT_ENSURE_PTR_OK(dequant);

    count++;
    return dequant;
  }

  StatusOr<nvinfer1::IShuffleLayer*> Reshape(nvinfer1::ITensor* input,
                                             const nvinfer1::Dims& new_shape) {
    TRT_ENSURE(input);
    nvinfer1::IShuffleLayer* layer = network_->addShuffle(*input);
    TRT_ENSURE(layer);
    layer->setReshapeDimensions(new_shape);
    return layer;
  }

  StatusOr<nvinfer1::ILayer*> FindProducerOf(const nvinfer1::ITensor* tensor) {
    const char* name = tensor->getName();
    const int num_layers = network_->getNbLayers();
    for (int i = 0; i < num_layers; i++) {
      nvinfer1::ILayer* layer = network_->getLayer(i);
      const int num_outputs = layer->getNbOutputs();
      for (int j = 0; j < num_outputs; j++) {
        nvinfer1::ITensor* t = layer->getOutput(j);
        if (std::string(t->getName()) == name) {
          return layer;
        }
      }
    }
    return errors::NotFound("could not find producing layer of ", name);
  }

  StatusOr<nvinfer1::ILayer*> UniqueParentOf(const nvinfer1::ILayer* layer,
                                             int input_idx = 0) {
    return FindProducerOf(layer->getInput(input_idx));
  }

  nvinfer1::INetworkDefinition* Network() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_1(mht_1_v, 772, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "Network");
 return network_; }

 private:
  nvinfer1::INetworkDefinition* const network_;
  TrtWeightStore* const weight_store_;
};

class ShuffleBuilder {
 private:
  explicit ShuffleBuilder(TRTNetworkBuilder* builder, nvinfer1::ITensor* input)
      : builder_(builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_2(mht_2_v, 785, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "ShuffleBuilder");

    layer_ = builder->Network()->addShuffle(*input);
  }

 public:
  static StatusOr<ShuffleBuilder> Create(TRTNetworkBuilder* builder,
                                         nvinfer1::ITensor* input) {
    TRT_ENSURE(builder != nullptr);
    TRT_ENSURE(input != nullptr);
    return ShuffleBuilder(builder, input);
  }

  ShuffleBuilder& SetReshape(const nvinfer1::Dims& dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_3(mht_3_v, 800, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "SetReshape");

    layer_->setReshapeDimensions(dims);
    return *this;
  }

  ShuffleBuilder& SetReshape(nvinfer1::ITensor* shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_4(mht_4_v, 808, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "SetReshape");

    layer_->setInput(1, *shape);
    return *this;
  }

  ShuffleBuilder& SetFirstTranspose(const nvinfer1::Permutation& perm) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_5(mht_5_v, 816, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "SetFirstTranspose");

    layer_->setFirstTranspose(perm);
    return *this;
  }

  ShuffleBuilder& SetSecondTranspose(const nvinfer1::Permutation& perm) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSopsPSlayer_utilsDTh mht_6(mht_6_v, 824, "", "./tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h", "SetSecondTranspose");

    layer_->setSecondTranspose(perm);
    return *this;
  }

  StatusOr<nvinfer1::ITensor*> Output() {
    TRT_ENSURE(layer_ != nullptr);
    TRT_ENSURE(layer_->getOutput(0) != nullptr);
    return layer_->getOutput(0);
  }

 private:
  TRTNetworkBuilder* builder_;
  nvinfer1::IShuffleLayer* layer_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
