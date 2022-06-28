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
class MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc() {
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
#include "tensorflow/lite/tools/versioning/op_signature.h"

#include <cstdlib>

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace {

// A BuiltinDataAllocator which just uses malloc()/free().
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/tools/versioning/op_signature.cc", "Allocate");

    return malloc(size);
  }
  void Deallocate(void* data) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc mht_1(mht_1_v, 206, "", "./tensorflow/lite/tools/versioning/op_signature.cc", "Deallocate");
 free(data); }
};

// Get the number of dimensions of a tensor with idx of an operator op.
inline int GetNumDims(const SubGraph* subgraph, const Operator* op, int idx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc mht_2(mht_2_v, 213, "", "./tensorflow/lite/tools/versioning/op_signature.cc", "GetNumDims");

  return subgraph->tensors()->Get(op->inputs()->Get(idx))->shape()->size();
}

std::vector<OpSignatureTensorSpec> GetOpSignatureTensorSpecs(
    const flatbuffers::Vector<int32_t>* tensors, const SubGraph* subgraph,
    const Model* model) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  StderrReporter error_reporter;

  for (int32_t i = 0; i < tensors->Length(); ++i) {
    int32_t tensor_no = tensors->Get(i);

    OpSignatureTensorSpec tensor_spec = {kTfLiteNoType};
    if (tensor_no >= 0) {
      if (subgraph->tensors() && tensor_no < subgraph->tensors()->Length()) {
        auto* fb_tensor = subgraph->tensors()->Get(tensor_no);
        ConvertTensorType(fb_tensor->type(), &tensor_spec.type,
                          &error_reporter);
        auto buffer_idx = fb_tensor->buffer();
        // Check if the tensor is a constant tensor.
        if (buffer_idx != 0 && buffer_idx < model->buffers()->Length()) {
          auto* buffer = model->buffers()->Get(buffer_idx);
          if (buffer->data() && buffer->data()->size() != 0) {
            tensor_spec.is_const = true;
          }
        }
        const flatbuffers::Vector<int32_t>* shape_vec =
            subgraph->tensors()->Get(tensor_no)->shape();
        if (shape_vec) {
          for (int32_t j = 0; j < shape_vec->Length(); ++j) {
            tensor_spec.dims.push_back(shape_vec->Get(j));
          }
        }
      }
    }
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

std::vector<OpSignatureTensorSpec> GetOpSignatureTensorSpecs(
    TfLiteIntArray* tensors, const TfLiteContext* context,
    const TfLiteNode* tflite_node) {
  std::vector<OpSignatureTensorSpec> tensor_specs;

  for (int32_t i = 0; i < tensors->size; ++i) {
    int32_t tensor_no = tensors->data[i];

    OpSignatureTensorSpec tensor_spec = {kTfLiteNoType};
    if (tensor_no >= 0) {
      const TfLiteTensor* tfl_tensor;
      if (context->tensors != nullptr) {
        tfl_tensor = &context->tensors[tensor_no];
      } else {
        tfl_tensor = context->GetTensor(context, tensor_no);
      }
      if (tfl_tensor != nullptr) {
        tensor_spec.type = tfl_tensor->type;
        tensor_spec.is_const = (tfl_tensor->allocation_type == kTfLiteMmapRo);
        if (tfl_tensor->dims) {
          for (int32_t j = 0; j < tfl_tensor->dims->size; ++j) {
            tensor_spec.dims.push_back(tfl_tensor->dims->data[j]);
          }
        }
      }
    }
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

}  // namespace

OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph, const Model* model) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc mht_3(mht_3_v, 291, "", "./tensorflow/lite/tools/versioning/op_signature.cc", "GetOpSignature");

  auto builtin_code = GetBuiltinCode(op_code);
  OpSignature op_sig = {builtin_code};
  std::memset(&op_sig.ext_options, 0, sizeof(op_sig.ext_options));

  if (builtin_code != BuiltinOperator_CUSTOM) {
    StderrReporter error_reporter;
    MallocDataAllocator allocator;
    ParseOpData(op, builtin_code, &error_reporter, &allocator,
                &op_sig.builtin_data);
  } else {
    op_sig.custom_name = op_code->custom_code()->str();
  }

  switch (builtin_code) {
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_channels = filter_tensor->shape()->Get(3);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_channels) {
        op_sig.ext_options.depthwise_conv_2d.is_per_channel_quantized = true;
      }
    } break;

    case BuiltinOperator_FULLY_CONNECTED: {
      const Tensor* weight_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      op_sig.ext_options.fully_connected.sparse_weight =
          (weight_tensor->sparsity() != nullptr);
    } break;

    case BuiltinOperator_MUL: {
      if (op->inputs()->Length() < 2 || op->outputs()->Length() < 1) {
        break;
      }
      const Tensor* input1_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const Tensor* input2_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const Tensor* output_tensor =
          subgraph->tensors()->Get(op->outputs()->Get(0));
      const QuantizationParameters* input1_quant =
          input1_tensor->quantization();
      const QuantizationParameters* input2_qunt = input2_tensor->quantization();
      const QuantizationParameters* output_quant =
          output_tensor->quantization();
      if (input1_quant && input1_quant->scale() &&
          input1_quant->scale()->Length() && input2_qunt &&
          input2_qunt->scale() && input2_qunt->scale()->Length() &&
          output_quant && output_quant->scale() &&
          output_quant->scale()->Length()) {
        op_sig.ext_options.mul.input1_scale = input1_quant->scale()->Get(0);
        op_sig.ext_options.mul.input2_scale = input2_qunt->scale()->Get(0);
        op_sig.ext_options.mul.output_scale = output_quant->scale()->Get(0);
      }
    } break;

    case BuiltinOperator_CONV_2D: {
      const Tensor* input_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_filters = filter_tensor->shape()->Get(0);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_filters) {
        op_sig.ext_options.conv_2d.is_per_channel_quantized = true;
      }
      if (input_tensor->shape()->size()) {
        int num_input_channels = input_tensor->shape()->Get(3);
        int num_filter_input_channels = filter_tensor->shape()->Get(3);
        op_sig.ext_options.conv_2d.is_grouped_convolution =
            num_input_channels != num_filter_input_channels;
      } else {
        op_sig.ext_options.conv_2d.is_grouped_convolution = false;
      }
    } break;

    case BuiltinOperator_STRIDED_SLICE: {
      op_sig.ext_options.strided_slice.num_dims = GetNumDims(subgraph, op, 0);
    } break;

    case BuiltinOperator_ABS: {
      if (subgraph->tensors()->Get(op->inputs()->Get(0))->quantization()) {
        op_sig.ext_options.abs.input_quantized = true;
      }
    } break;

    case BuiltinOperator_DEQUANTIZE: {
      const Tensor* input_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const QuantizationParameters* input_quant = input_tensor->quantization();
      if (input_quant && input_quant->scale() &&
          input_quant->scale()->Length() > 1 &&
          input_quant->scale()->Length() ==
              input_tensor->shape()->Get(input_quant->quantized_dimension())) {
        op_sig.ext_options.dequantize.is_per_channel_quantized = true;
      }
    } break;

    case BuiltinOperator_QUANTIZE: {
      const Tensor* output_tensor =
          subgraph->tensors()->Get(op->outputs()->Get(0));
      const QuantizationParameters* output_quant =
          output_tensor->quantization();
      if (output_quant && output_quant->scale() &&
          output_quant->scale()->Length() > 1 &&
          output_quant->scale()->Length() ==
              output_tensor->shape()->Get(
                  output_quant->quantized_dimension())) {
        op_sig.ext_options.quantize.is_per_channel_quantized = true;
      }
    } break;

    default:
      break;
  }

  op_sig.inputs = GetOpSignatureTensorSpecs(op->inputs(), subgraph, model);
  op_sig.outputs = GetOpSignatureTensorSpecs(op->outputs(), subgraph, model);
  return op_sig;
}

OpSignature GetOpSignature(const TfLiteContext* context, const TfLiteNode* node,
                           const TfLiteRegistration* registration) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_signatureDTcc mht_4(mht_4_v, 424, "", "./tensorflow/lite/tools/versioning/op_signature.cc", "GetOpSignature");

  OpSignature op_sig = {
      static_cast<BuiltinOperator>(registration->builtin_code)};
  op_sig.builtin_data = node->builtin_data;
  if (op_sig.op == BuiltinOperator_CUSTOM) {
    op_sig.custom_name = registration->custom_name;
    op_sig.custom_initial_data = node->custom_initial_data;
  }
  std::memset(&op_sig.ext_options, 0, sizeof(op_sig.ext_options));

  op_sig.inputs = GetOpSignatureTensorSpecs(node->inputs, context, node);
  op_sig.outputs = GetOpSignatureTensorSpecs(node->outputs, context, node);
  return op_sig;
}

}  // namespace tflite
