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
class MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/coreml/builders/convolution_op_builder.h"

#include "google/protobuf/repeated_field.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/activation_layer_builder.h"
#include "tensorflow/lite/delegates/coreml/builders/op_factory.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
const std::string& ConvolutionOpBuilder::DebugName() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::DebugName");

  if (debug_name_.empty()) SetDebugName("ConvolutionOpBuilder", node_id_);
  return debug_name_;
}

void ConvolutionOpBuilder::SetWeights(TfLiteTensor* weights) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::SetWeights");

  weights_ = weights;
}

void ConvolutionOpBuilder::SetBias(TfLiteTensor* bias) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_2(mht_2_v, 212, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::SetBias");
 bias_ = bias; }

void ConvolutionOpBuilder::SetOutputShape(TfLiteTensor* output_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_3(mht_3_v, 217, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::SetOutputShape");

  output_shape_ = output_shape;
}

CoreML::Specification::NeuralNetworkLayer* ConvolutionOpBuilder::Build() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_4(mht_4_v, 224, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::Build");

  if (layer_ == nullptr) {
    layer_.reset(new CoreML::Specification::NeuralNetworkLayer);
  }
  layer_->set_name(DebugName());

  int stride_height = 0;
  int stride_width = 0;
  int dilation_height = 0;
  int dilation_width = 0;
  TfLitePadding padding;

  switch (conv_type_) {
    case ConvolutionType::kConv: {
      const auto* conv_params =
          reinterpret_cast<const TfLiteConvParams*>(builtin_data_);
      stride_height = conv_params->stride_height;
      stride_width = conv_params->stride_width;
      dilation_height = conv_params->dilation_height_factor;
      dilation_width = conv_params->dilation_width_factor;
      padding = conv_params->padding;

      layer_->mutable_convolution()->set_ngroups(1);
      break;
    }
    case ConvolutionType::kDepthwiseConv: {
      const auto* depthwise_conv_params =
          reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
      stride_height = depthwise_conv_params->stride_height;
      stride_width = depthwise_conv_params->stride_width;
      dilation_height = depthwise_conv_params->dilation_height_factor;
      dilation_width = depthwise_conv_params->dilation_width_factor;
      padding = depthwise_conv_params->padding;

      // n_groups = kernel_channel / depth_multiplier
      layer_->mutable_convolution()->set_ngroups(
          weights_->dims->data[3] / depthwise_conv_params->depth_multiplier);
      break;
    }
    case ConvolutionType::kTransposeConv: {
      const auto* transpose_conv_params =
          reinterpret_cast<const TfLiteTransposeConvParams*>(builtin_data_);
      const int height_index = 1;
      const int width_index = 2;

      stride_height = transpose_conv_params->stride_height;
      stride_width = transpose_conv_params->stride_width;
      padding = transpose_conv_params->padding;

      layer_->mutable_convolution()->mutable_outputshape()->Add(
          GetTensorData<int32_t>(output_shape_)[height_index]);
      layer_->mutable_convolution()->mutable_outputshape()->Add(
          GetTensorData<int32_t>(output_shape_)[width_index]);
      break;
    }
  }

  // If not set, it will default to (1,1)
  if (stride_height) {
    layer_->mutable_convolution()->add_stride(stride_height);
    layer_->mutable_convolution()->add_stride(stride_width);
  }

  if (dilation_height) {
    layer_->mutable_convolution()->add_dilationfactor(dilation_height);
    layer_->mutable_convolution()->add_dilationfactor(dilation_width);
  }

  switch (padding) {
    case kTfLitePaddingSame:
      layer_->mutable_convolution()->mutable_same();
      break;
    case kTfLitePaddingValid:
      layer_->mutable_convolution()->mutable_valid();
      break;
    case kTfLitePaddingUnknown:
      fprintf(stderr, "Padding is unknown.\n");
      break;
  }

  FillCoreMLWeights();
  FillCoreMLBias();

  return layer_.release();
}

void ConvolutionOpBuilder::FillCoreMLWeights() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_5(mht_5_v, 313, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::FillCoreMLWeights");

  if (conv_type_ == ConvolutionType::kDepthwiseConv) {
    layer_->mutable_convolution()->set_kernelchannels(1);
    layer_->mutable_convolution()->set_outputchannels(weights_->dims->data[3]);
  } else {
    layer_->mutable_convolution()->set_kernelchannels(weights_->dims->data[3]);
    layer_->mutable_convolution()->set_outputchannels(weights_->dims->data[0]);
  }
  layer_->mutable_convolution()->add_kernelsize(weights_->dims->data[1]);
  layer_->mutable_convolution()->add_kernelsize(weights_->dims->data[2]);

  TransposeKernelWeights();  // Should be called after CoreML shape is set.
}

void ConvolutionOpBuilder::TransposeKernelWeights() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_6(mht_6_v, 330, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::TransposeKernelWeights");

  RuntimeShape tfl_shape(4, weights_->dims->data);
  // CoreML kernel has shape of (C_out, C_in, H, W)
  RuntimeShape coreml_shape(
      {static_cast<int>(layer_->convolution().outputchannels()),
       static_cast<int>(layer_->convolution().kernelchannels()),
       static_cast<int>(layer_->convolution().kernelsize()[0]),
       static_cast<int>(layer_->convolution().kernelsize()[1])});

  TransposeParams params;

  if (conv_type_ == ConvolutionType::kDepthwiseConv) {
    // DepthwiseConv2D: TFL kernel has shape of (1, H, W, C_out),
    // and CoreML kernel has shape of (C_out, 1, H, W)
    params = {/*perm_count=*/4, /*perm=*/{3, 0, 1, 2}};
  } else {
    // Conv2D and TransposeConv: TFL kernel has shape of (C_out, H, W, C_in),
    // and CoreML kernel has shape of (C_out, C_in, H, W)
    params = {/*perm_count=*/4, /*perm=*/{0, 3, 1, 2}};
  }

  if (conv_type_ == ConvolutionType::kTransposeConv) {
    layer_->mutable_convolution()->set_isdeconvolution(true);
  }

  if (weights_->type == kTfLiteFloat32) {
    auto* coreml_weights =
        layer_->mutable_convolution()->mutable_weights()->mutable_floatvalue();
    coreml_weights->Resize(NumElements(weights_), 0);

    optimized_ops::Transpose<float>(params, tfl_shape, weights_->data.f,
                                    coreml_shape,
                                    coreml_weights->mutable_data());
  } else if (weights_->type == kTfLiteFloat16) {
    auto* coreml_weights = layer_->mutable_convolution()
                               ->mutable_weights()
                               ->mutable_float16value();
    // float16value has type of bytes (std::string)
    coreml_weights->resize(weights_->bytes, 0);

    optimized_ops::Transpose<uint16_t>(
        params, tfl_shape, reinterpret_cast<uint16_t*>(weights_->data.raw),
        coreml_shape, reinterpret_cast<uint16_t*>(&coreml_weights->front()));
  }
}

void ConvolutionOpBuilder::FillCoreMLBias() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_7(mht_7_v, 379, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::FillCoreMLBias");

  if (bias_ != nullptr) {
    layer_->mutable_convolution()->set_hasbias(true);
    if (bias_->type == kTfLiteFloat32) {
      std::copy(bias_->data.f, bias_->data.f + NumElements(bias_->dims),
                google::protobuf::RepeatedFieldBackInserter(layer_->mutable_convolution()
                                                      ->mutable_bias()
                                                      ->mutable_floatvalue()));
    } else if (bias_->type == kTfLiteFloat16) {
      // float16value has type of bytes (std::string)
      layer_->mutable_convolution()
          ->mutable_bias()
          ->mutable_float16value()
          ->assign(bias_->data.raw, bias_->bytes);
    }
  }
}

TfLiteStatus ConvolutionOpBuilder::PopulateSubgraph(TfLiteContext* context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_8(mht_8_v, 400, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::PopulateSubgraph");

  TfLiteFusedActivation activation;
  switch (conv_type_) {
    case ConvolutionType::kConv: {
      const auto* conv_params =
          reinterpret_cast<const TfLiteConvParams*>(builtin_data_);
      activation = conv_params->activation;
      break;
    }
    case ConvolutionType::kDepthwiseConv: {
      const auto* depthwise_conv_params =
          reinterpret_cast<const TfLiteDepthwiseConvParams*>(builtin_data_);
      activation = depthwise_conv_params->activation;
      break;
    }
    case ConvolutionType::kTransposeConv: {
      activation = kTfLiteActNone;
      break;
    }
  }

  if (activation == kTfLiteActNone) {
    builder_output_ = AddOutput();
  } else {
    ActivationLayerBuilder* activation_builder =
        reinterpret_cast<ActivationLayerBuilder*>(
            graph_builder_->AddBuilder(CreateActivationLayerBuilder, nullptr));
    activation_builder->SetActivation(activation);
    activation_builder->AddInput(AddOutput());
    activation_builder->PopulateSubgraph(context);
    builder_output_ = activation_builder->GetOutput(context);
  }
  return kTfLiteOk;
}

TfLiteStatus ConvolutionOpBuilder::RegisterInputs(const TfLiteIntArray* inputs,
                                                  TfLiteContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_9(mht_9_v, 439, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::RegisterInputs");

  if (conv_type_ == ConvolutionType::kTransposeConv) {
    if (inputs->size != 3) {
      TF_LITE_KERNEL_LOG(context,
                         "Transpose Conv should have 3 inputs, %d given.",
                         inputs->size);
      return kTfLiteError;
    }
    AddInput(inputs->data[2]);
    SetOutputShape(&context->tensors[inputs->data[0]]);
  } else {
    if (inputs->size != 2 && inputs->size != 3) {
      TF_LITE_KERNEL_LOG(context,
                         "Convolution and depthwise convolution should have 2 "
                         "or 3 inputs, %d given.",
                         inputs->size);
      return kTfLiteError;
    }
    AddInput(inputs->data[0]);
    if (inputs->size > 2) {
      SetBias(&context->tensors[inputs->data[2]]);
    }
  }
  SetWeights(&context->tensors[inputs->data[1]]);
  return kTfLiteOk;
}

TfLiteStatus ConvolutionOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_10(mht_10_v, 470, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "ConvolutionOpBuilder::RegisterOutputs");

  if (outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Wrong # of outputs!.");
    return kTfLiteError;
  }
  TensorID output_tensor = GetOutput(context);
  if (output_tensor.NodeID() == -1) {
    TF_LITE_KERNEL_LOG(context, "Failed to build output tensor.");
    return kTfLiteError;
  }
  graph_builder_->AddTensorWithID(outputs->data[0], output_tensor);
  return kTfLiteOk;
}

OpBuilder* CreateConvolutionOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_11(mht_11_v, 487, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "CreateConvolutionOpBuilder");

  return new ConvolutionOpBuilder(graph_builder, ConvolutionType::kConv);
}

OpBuilder* CreateDepthwiseConvolutionOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_12(mht_12_v, 494, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "CreateDepthwiseConvolutionOpBuilder");

  return new ConvolutionOpBuilder(graph_builder,
                                  ConvolutionType::kDepthwiseConv);
}

OpBuilder* CreateTransposeConvolutionOpBuilder(GraphBuilder* graph_builder) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_13(mht_13_v, 502, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "CreateTransposeConvolutionOpBuilder");

  return new ConvolutionOpBuilder(graph_builder,
                                  ConvolutionType::kTransposeConv);
}

bool IsConvolutionOpSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_14(mht_14_v, 511, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "IsConvolutionOpSupported");

  if (node->builtin_data == nullptr) return false;

  TfLiteFusedActivation activation;

  if (registration->builtin_code == kTfLiteBuiltinConv2d) {
    const auto* conv_params =
        reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
    activation = conv_params->activation;
  } else if (registration->builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
    const auto* depthwise_conv_params =
        reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
    activation = depthwise_conv_params->activation;
  } else if (registration->builtin_code == kTfLiteBuiltinTransposeConv) {
    activation = kTfLiteActNone;
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "Invalid op: op must be Conv2D, DepthwiseConv2D or TransposeConv.");
    return false;
  }

  if (activation == kTfLiteActSignBit) {
    return false;
  }

  const int kOutputShapeTensor = 0;  // Only used for TransposeConv
  const int kWeightTensor = 1;
  const int kBiasTensor = 2;  // Only used for non-TransposeConv
  const TfLiteTensor* weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightTensor, &weights));
  const int max_kernel_size = 16384;
  if (!IsConstantTensor(weights)) {
    return false;
  }
  if (weights->dims->data[1] > max_kernel_size ||
      weights->dims->data[2] > max_kernel_size) {
    return false;
  }
  if (registration->builtin_code == kTfLiteBuiltinTransposeConv) {
    if (!IsConstantTensor(GetInput(context, node, kOutputShapeTensor))) {
      return false;
    }
  } else {
    if (node->inputs->size >= kBiasTensor &&
        !IsConstantTensor(GetInput(context, node, kBiasTensor))) {
      return false;
    }
  }

  return true;
}

bool IsDepthwiseConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_15(mht_15_v, 570, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "IsDepthwiseConvolutionOpSupported");

  return IsConvolutionOpSupported(registration, node, context);
}

bool IsTransposeConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPScoremlPSbuildersPSconvolution_op_builderDTcc mht_16(mht_16_v, 579, "", "./tensorflow/lite/delegates/coreml/builders/convolution_op_builder.cc", "IsTransposeConvolutionOpSupported");

  return IsConvolutionOpSupported(registration, node, context);
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
