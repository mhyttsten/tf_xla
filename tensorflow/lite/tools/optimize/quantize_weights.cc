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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc() {
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
#include "tensorflow/lite/tools/optimize/quantize_weights.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

namespace tflite {
namespace optimize {

namespace {

struct ConsumerOpInfo {
  OperatorT* op;
  // The index of the op in the operators vector.
  int32_t op_idx;
  // The index of the tensor to quantize in subgraph->tensors.
  int32_t op_input_idx;
};

struct TensorPerChannel {
  TensorT* t;
  bool is_per_channel;
  int channel_dim;
};

// The default minimum number of elements a weights array must have to be
// quantized by this transformation.
const int kWeightsMinNumElementsDefault = 1024;

// Convert the MLIR CustomOpMap from the TFlite CustomOpMap as their member
// variables differ.
void ConstructMLIRCustomOpMap(mlir::lite::CustomOpMap& mlir_map,
                              const CustomOpMap& tflite_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_0(mht_0_v, 230, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "ConstructMLIRCustomOpMap");

  for (const auto& entry : tflite_map) {
    mlir_map[entry.first].quantizable_input_indices =
        entry.second.quantizable_input_indices;
    mlir_map[entry.first].is_weight_only = !entry.second.is_hybrid;
    mlir_map[entry.first].no_side_effect = true;
  }
}

// Gets the operators that consume tensor_idx.
std::vector<ConsumerOpInfo> GetTensorConsumers(const ModelT* model,
                                               const SubGraphT* subgraph,
                                               int32_t tensor_idx) {
  // TODO(suharshs): If this proves to be too slow, avoid calling it per tensor,
  // instead doing one sweep for the entire model.
  std::vector<ConsumerOpInfo> consumer_ops;
  for (size_t op_idx = 0; op_idx < subgraph->operators.size(); ++op_idx) {
    OperatorT* op = subgraph->operators[op_idx].get();
    if (op == nullptr) {
      continue;
    }
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == tensor_idx) {
        consumer_ops.push_back(
            {op, static_cast<int32_t>(op_idx), static_cast<int32_t>(i)});
      }
    }
  }
  return consumer_ops;
}

// Gets the list of op->inputs indices of the weights inputs to be quantized for
// the provided op.
std::vector<int32_t> GetWeightInputIndices(const OperatorCodeT* op_code,
                                           const CustomOpMap& custom_op_map) {
  const BuiltinOperator builtin_op_code = GetBuiltinCode(op_code);
  if (builtin_op_code == BuiltinOperator_CUSTOM) {
    const std::string custom_code = op_code->custom_code;
    const auto& custom_op_info = custom_op_map.find(custom_code);
    if (custom_op_info != custom_op_map.end()) {
      return custom_op_info->second.quantizable_input_indices;
    }
  } else if (builtin_op_code == BuiltinOperator_CONV_2D ||
             builtin_op_code == BuiltinOperator_DEPTHWISE_CONV_2D ||
             builtin_op_code == BuiltinOperator_FULLY_CONNECTED ||
             builtin_op_code == BuiltinOperator_BATCH_MATMUL ||
             builtin_op_code == BuiltinOperator_EMBEDDING_LOOKUP ||
             builtin_op_code == BuiltinOperator_TRANSPOSE_CONV) {
    return {1};
  } else if (builtin_op_code == BuiltinOperator_SVDF) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/svdf.cc
    return {1, 2};
  } else if (builtin_op_code == BuiltinOperator_LSTM ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/lstm.cc
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/unidirectional_sequence_lstm.cc
    return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
  } else if (builtin_op_code == BuiltinOperator_RNN ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/basic_rnn.cc
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/unidirectional_sequence_rnn.cc
    return {1, 2};
  } else if (builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/bidirectional_sequence_lstm.cc
    return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 33, 40, 41, 42, 43, 44, 45, 46, 47};
  } else if (builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc
    return {1, 2, 4, 5, 6, 8, 9, 10, 11};
  } else if (builtin_op_code == BuiltinOperator_GATHER) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/gather.cc
    return {0};
  }
  return {};
}

// Checks that a specific input can be quantized.
bool IsQuantizedInput(const OperatorCodeT* op_code,
                      const CustomOpMap& custom_op_map, int op_input_idx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_1(mht_1_v, 311, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "IsQuantizedInput");

  const auto quantized_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  return std::find(std::begin(quantized_input_indices),
                   std::end(quantized_input_indices),
                   op_input_idx) != std::end(quantized_input_indices);
}

// Returns true if the operator supports hybrid evaluation.
bool IsHybridEvaluationOp(const OperatorT* op, const OperatorCodeT* op_code,
                          const CustomOpMap& custom_op_map,
                          bool use_updated_hybrid_scheme) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_2(mht_2_v, 325, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "IsHybridEvaluationOp");

  const BuiltinOperator builtin_op_code = GetBuiltinCode(op_code);
  // Operations that support hybrid evaluation.
  bool eval_hybrid = false;
  if (builtin_op_code == BuiltinOperator_CUSTOM) {
    const std::string custom_code = op_code->custom_code;
    const auto custom_op_info = custom_op_map.find(custom_code);
    if (custom_op_info == custom_op_map.end()) {
      return {};
    } else {
      return custom_op_info->second.is_hybrid;
    }
  } else if (builtin_op_code == BuiltinOperator_FULLY_CONNECTED ||
             builtin_op_code == BuiltinOperator_BATCH_MATMUL ||
             builtin_op_code == BuiltinOperator_CONV_2D ||
             builtin_op_code == BuiltinOperator_SVDF ||
             builtin_op_code == BuiltinOperator_RNN ||
             builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM ||
             builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    eval_hybrid = true;
  } else if (builtin_op_code == BuiltinOperator_LSTM) {
    const LSTMOptionsT* options = op->builtin_options.AsLSTMOptions();
    // Only lstm kernel_type full supports hybrid evaluation.
    if (options->kernel_type == LSTMKernelType_FULL) {
      eval_hybrid = true;
    }
  } else if (builtin_op_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
    eval_hybrid = use_updated_hybrid_scheme;
  }
  return eval_hybrid;
}

// Returns true if all of the op's inputs are quantized.
bool CheckAllOpInputsQuantized(const SubGraphT* subgraph, const OperatorT* op,
                               const OperatorCodeT* op_code,
                               const CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_3(mht_3_v, 365, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "CheckAllOpInputsQuantized");

  std::vector<int32_t> op_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  for (const int32_t op_input_idx : op_input_indices) {
    int32_t tensor_idx = op->inputs[op_input_idx];

    if (tensor_idx == -1) {
      // Optional tensor.
      continue;
    }

    TensorT* tensor = subgraph->tensors[tensor_idx].get();

    if (tensor->type != TensorType_INT8) {
      return false;
    }
  }
  return true;
}

// Inserts Tensors for each input tensor of op that should be
// quantized into tensor_map.
TfLiteStatus InsertQuantizableInputTensorsFromOperator(
    const ModelT* model, OperatorT* op, uint64_t weights_min_num_elements,
    const CustomOpMap& custom_op_map,
    absl::flat_hash_map<int32_t, TensorPerChannel>* tensor_map,
    int subgraph_index, bool use_updated_hybrid_scheme) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_4(mht_4_v, 394, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "InsertQuantizableInputTensorsFromOperator");

  SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();
  const OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
  auto builtin_code = GetBuiltinCode(op_code);

  std::vector<int32_t> op_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  for (const int32_t op_input_idx : op_input_indices) {
    int32_t tensor_idx = op->inputs[op_input_idx];
    if (tensor_idx == -1) {
      LOG(INFO) << "Skipping optional tensor input " << op_input_idx
                << " of operation " << EnumNameBuiltinOperator(builtin_code);
      continue;
    }

    TensorT* tensor = subgraph->tensors[tensor_idx].get();
    if (tensor->type != TensorType_FLOAT32) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " that is not type float.";
      continue;
    }

    uint64_t num_elements;
    TF_LITE_ENSURE_STATUS(utils::NumElements(*tensor, &num_elements));
    if (num_elements < weights_min_num_elements) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has fewer than " << weights_min_num_elements
                << " elements (" << num_elements << ").";
      continue;
    }

    // Some tensors may have a null buffer vector, indicating an intermediate
    // array.
    if (model->buffers[tensor->buffer]->data.data() == nullptr) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has no allocated buffer.";
      continue;
    }

    if (builtin_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
      tensor_map->insert({tensor_idx,
                          {tensor, /*is_per_channel=*/use_updated_hybrid_scheme,
                           /*dim=*/3}});
    } else if (builtin_code == BuiltinOperator_CONV_2D) {
      tensor_map->insert({tensor_idx,
                          {tensor, /*is_per_channel=*/use_updated_hybrid_scheme,
                           /*dim=*/0}});
    } else {
      switch (builtin_code) {
        case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
          op->builtin_options.AsBidirectionalSequenceLSTMOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
          op->builtin_options.AsBidirectionalSequenceRNNOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_FULLY_CONNECTED:
          op->builtin_options.AsFullyConnectedOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_BATCH_MATMUL:
          op->builtin_options.AsBatchMatMulOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_LSTM:
          op->builtin_options.AsLSTMOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_RNN:
          op->builtin_options.AsRNNOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_SVDF:
          op->builtin_options.AsSVDFOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
          op->builtin_options.AsUnidirectionalSequenceLSTMOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
          op->builtin_options.AsSequenceRNNOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        default:
          break;
      }
      tensor_map->insert({tensor_idx, {tensor, /*is_per_channel=*/false}});
    }
  }

  return kTfLiteOk;
}

// Updates operator code versions for the operators with INT8 inputs.
void UpdateInt8OperatorVersions(ModelT* model, bool use_updated_hybrid_scheme) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_5(mht_5_v, 493, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "UpdateInt8OperatorVersions");

  for (int i = 0, end = model->operator_codes.size(); i < end; ++i) {
    const BuiltinOperator& op_code =
        GetBuiltinCode(model->operator_codes[i].get());
    if (op_code == BuiltinOperator_RNN ||
        op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN ||
        op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM ||
        op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 3 : 2;
    } else if (op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM ||
               op_code == BuiltinOperator_EMBEDDING_LOOKUP) {
      model->operator_codes[i]->version = 3;
    } else if (op_code == BuiltinOperator_LSTM) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 3;
    } else if (op_code == BuiltinOperator_CONV_2D) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 5 : 2;
    } else if (op_code == BuiltinOperator_FULLY_CONNECTED) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 9 : 3;
    } else if (op_code == BuiltinOperator_BATCH_MATMUL) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 1;
    } else if (op_code == BuiltinOperator_SVDF) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 2;
    } else if (op_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
      model->operator_codes[i]->version = 6;
    }
  }
}

// Returns true if the op in consumer_op_infos can pass through quantization.
bool IsQuantizationPassThroughOps(
    const ModelT* model, const std::vector<ConsumerOpInfo>& consumer_op_infos) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_6(mht_6_v, 526, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "IsQuantizationPassThroughOps");

  if (consumer_op_infos.size() != 1) {
    return false;
  }
  const OperatorT* consumer_op = consumer_op_infos.front().op;
  const BuiltinOperator op_code =
      GetBuiltinCode(model->operator_codes[consumer_op->opcode_index].get());
  return op_code == BuiltinOperator_GATHER ||
         op_code == BuiltinOperator_EMBEDDING_LOOKUP;
}

// Copies quantization parameters from input to output and returns consumers of
// the output tensor as a tuple with values:
// - index of the output tensor
// - pointer to the output tensor
// - vector of consumers ops.
std::tuple<int32_t, TensorT*, std::vector<ConsumerOpInfo>>
PassQuantizationAndGetConsumers(
    const ModelT* model, const SubGraphT* subgraph,
    const std::vector<ConsumerOpInfo>& consumer_op_infos,
    const CustomOpMap& custom_op_map) {
  const OperatorT* op = consumer_op_infos.front().op;
  const OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
  if (op->outputs.size() != 1) {
    LOG(ERROR)
        << "An op that passes quantization has more than one quantized output";
    return std::make_tuple(-1, nullptr, std::vector<ConsumerOpInfo>());
  }
  const int32_t output_tensor_idx = op->outputs.front();
  const auto input_idx = GetWeightInputIndices(op_code, custom_op_map);
  if (input_idx.size() != 1) {
    LOG(ERROR)
        << "An op that passes quantization has more than one quantized input";
    return std::make_tuple(-1, nullptr, std::vector<ConsumerOpInfo>());
  }
  const int32_t input_tensor_idx = op->inputs[input_idx.front()];

  // Propagate quantization params.
  const TensorT* input_tensor = subgraph->tensors[input_tensor_idx].get();
  TensorT* output_tensor = subgraph->tensors[output_tensor_idx].get();
  if (!output_tensor->quantization) {
    output_tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  *output_tensor->quantization = *input_tensor->quantization;
  output_tensor->type = TensorType_INT8;
  return std::make_tuple(
      output_tensor_idx, output_tensor,
      GetTensorConsumers(model, subgraph, output_tensor_idx));
}

inline bool IsOpDenylisted(const flat_hash_set<BuiltinOperator>& op_denylist,
                           const BuiltinOperator op_code) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_7(mht_7_v, 580, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "IsOpDenylisted");

  return op_denylist.find(op_code) != op_denylist.end();
}

TfLiteStatus QuantizeWeightsInt8(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    bool use_hybrid_evaluation, uint64_t weights_min_num_elements,
    const CustomOpMap& custom_op_map, bool use_updated_hybrid_scheme,
    const flat_hash_set<BuiltinOperator>& op_denylist = {}) {
  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());

  for (int subgraph_index = 0, end = model->subgraphs.size();
       subgraph_index < end; ++subgraph_index) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();

    absl::flat_hash_map<int32_t, TensorPerChannel> tensor_map;
    for (int i = 0; i < subgraph->operators.size(); ++i) {
      OperatorT* op = subgraph->operators[i].get();
      TF_LITE_ENSURE_STATUS(InsertQuantizableInputTensorsFromOperator(
          model.get(), op, weights_min_num_elements, custom_op_map, &tensor_map,
          subgraph_index, use_updated_hybrid_scheme));
    }

    for (std::pair<int32_t, TensorPerChannel> tensor_pair : tensor_map) {
      // Quantize the tensor.
      if (tensor_pair.second.is_per_channel) {
        TF_LITE_ENSURE_STATUS(utils::SymmetricQuantizeTensorPerChannel(
            model.get(), tensor_pair.second.t, tensor_pair.second.channel_dim,
            nullptr));
      } else {
        TF_LITE_ENSURE_STATUS(
            utils::SymmetricQuantizeTensor(model.get(), tensor_pair.second.t));
      }
    }

    // Examine the tensor consumers to determine which require dequantize ops.
    for (const auto& tensor_pair : tensor_map) {
      int32_t tensor_idx = tensor_pair.first;
      TensorT* tensor = tensor_pair.second.t;
      std::vector<ConsumerOpInfo> consumer_op_infos =
          GetTensorConsumers(model.get(), subgraph, tensor_idx);
      if (IsQuantizationPassThroughOps(model.get(), consumer_op_infos)) {
        std::tie(tensor_idx, tensor, consumer_op_infos) =
            PassQuantizationAndGetConsumers(model.get(), subgraph,
                                            consumer_op_infos, custom_op_map);
        if (tensor_idx < 0) {
          // Error message is already logged by PassQuantizationAndGetConsumers.
          return kTfLiteError;
        }
      }

      std::vector<ConsumerOpInfo> dequant_op_infos;  // Ops that need dequants.
      for (ConsumerOpInfo& consumer_op_info : consumer_op_infos) {
        OperatorT* consumer_op = consumer_op_info.op;
        const OperatorCodeT* consumer_op_code =
            model->operator_codes[consumer_op->opcode_index].get();
        // If the op is a hybrid op and all the required tensors are quantized,
        // we have no further work to do, but for all ops that require
        // dequantization we need to add a Dequantize op.
        bool eval_hybrid =
            use_hybrid_evaluation &&
            !IsOpDenylisted(op_denylist, GetBuiltinCode(consumer_op_code)) &&
            IsHybridEvaluationOp(consumer_op, consumer_op_code, custom_op_map,
                                 use_updated_hybrid_scheme) &&
            CheckAllOpInputsQuantized(subgraph, consumer_op, consumer_op_code,
                                      custom_op_map) &&
            IsQuantizedInput(consumer_op_code, custom_op_map,
                             consumer_op_info.op_input_idx);
        if (!eval_hybrid) {
          dequant_op_infos.push_back(consumer_op_info);
        }
      }

      // Check if this tensor is an output tensor.
      int32_t output_index = -1;
      for (int32_t i = 0; i < subgraph->outputs.size(); ++i) {
        if (subgraph->outputs[i] == tensor_idx) {
          output_index = i;
          break;
        }
      }

      // If no ops require dequant and it is not output, we are done for this
      // tensor.
      if (dequant_op_infos.empty() && output_index < 0) {
        continue;
      }

      // Create a new tensor to be the output of the dequantize op.
      std::unique_ptr<TensorT> dequantize_output;
      const string dequant_name = tensor->name + "_dequantize";
      utils::MakeTensor(dequant_name, tensor->shape, tensor->shape_signature,
                        TensorType_FLOAT32, &dequantize_output);
      const int32_t dequantize_output_idx = subgraph->tensors.size();
      subgraph->tensors.push_back(std::move(dequantize_output));

      // Create the Dequantize operation.
      std::unique_ptr<OperatorT> dequantize_op;
      utils::MakeDequantizeOperator(model.get(), &dequantize_op, tensor_idx,
                                    dequantize_output_idx);

      // Update the op_input of all the ops that need the created dequantize
      // operation.
      int32_t min_op_idx = subgraph->operators.size();
      for (ConsumerOpInfo& dequant_op_info : dequant_op_infos) {
        dequant_op_info.op->inputs[dequant_op_info.op_input_idx] =
            dequantize_output_idx;
        min_op_idx = std::min(dequant_op_info.op_idx, min_op_idx);
      }
      // Update output name.
      if (output_index >= 0) {
        subgraph->outputs[output_index] = dequantize_output_idx;
      }

      // Insert the newly created Dequantize operation before the earliest
      // consumer, since TFLite requires operators to be topo-sorted.
      subgraph->operators.insert(subgraph->operators.begin() + min_op_idx,
                                 std::move(dequantize_op));
    }
  }

  // Update the modified operator code versions.
  UpdateInt8OperatorVersions(model.get(), use_updated_hybrid_scheme);

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

TfLiteStatus QuantizeWeightsFloat16(flatbuffers::FlatBufferBuilder* builder,
                                    const Model* input_model) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_8(mht_8_v, 716, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeightsFloat16");

  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());

  for (int subgraph_index = 0, end = model->subgraphs.size();
       subgraph_index < end; ++subgraph_index) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();

    absl::flat_hash_map<int32_t, TensorT*> tensor_map;
    for (int i = 0, sub_end = subgraph->operators.size(); i < sub_end; ++i) {
      OperatorT* op = subgraph->operators[i].get();
      for (auto tensor_idx : op->inputs) {
        // Skip optional tensors.
        if (tensor_idx == kTfLiteOptionalTensor) {
          continue;
        }
        TensorT* tensor = subgraph->tensors[tensor_idx].get();
        BufferT* buffer = model->buffers[tensor->buffer].get();
        if (buffer == nullptr) {
          return kTfLiteError;
        }
        // Quantize tensors that have data to quantize.
        bool is_constant = !model->buffers[tensor->buffer].get()->data.empty();
        if (tensor->type == TensorType_FLOAT32 && is_constant) {
          tensor_map.insert({tensor_idx, tensor});
        }
      }
    }

    // The hash map ensures that we quantize each tensor exactly once.
    for (std::pair<int32_t, TensorT*> tensor_pair : tensor_map) {
      // Quantize the tensor.
      TF_LITE_ENSURE_STATUS(
          utils::QuantizeTensorFloat16(model.get(), tensor_pair.second));

      int32_t tensor_idx = tensor_pair.first;
      TensorT* tensor = tensor_pair.second;
      std::vector<ConsumerOpInfo> dequant_op_infos =
          GetTensorConsumers(model.get(), subgraph, tensor_idx);

      // Create a new tensor to be the output of the dequantize op.
      std::unique_ptr<TensorT> dequantize_output;
      const string dequant_name = tensor->name + "_dequantize";
      utils::MakeTensor(dequant_name, tensor->shape, tensor->shape_signature,
                        TensorType_FLOAT32, &dequantize_output);
      const int32_t dequantize_output_idx = subgraph->tensors.size();
      subgraph->tensors.push_back(std::move(dequantize_output));

      // Create the Dequantize operation.
      std::unique_ptr<OperatorT> dequantize_op;
      utils::MakeDequantizeOperator(model.get(), &dequantize_op, tensor_idx,
                                    dequantize_output_idx);

      // Update the op_input of all the ops that need the created dequantize
      // operation.
      int32_t min_op_idx = subgraph->operators.size();
      for (ConsumerOpInfo& dequant_op_info : dequant_op_infos) {
        dequant_op_info.op->inputs[dequant_op_info.op_input_idx] =
            dequantize_output_idx;
        min_op_idx = std::min(dequant_op_info.op_idx, min_op_idx);
      }

      // Insert the newly created Dequantize operation before the earliest
      // consumer, since TFLite requires operators to be topo-sorted.
      subgraph->operators.insert(subgraph->operators.begin() + min_op_idx,
                                 std::move(dequantize_op));
    }
  }

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);
  return kTfLiteOk;
}
}  // namespace

namespace internal {
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             bool use_hybrid_evaluation,
                             QuantizerType quantizer_type) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_9(mht_9_v, 800, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeights");

  // By default we require that only weights with more than
  // kWeightsMinSizeDefault elements are quantized.
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    return mlir::lite::QuantizeWeights(
        builder, input_model, weights_min_num_elements, use_hybrid_evaluation);
  }
  CustomOpMap custom_op_map;
  return QuantizeWeightsInt8(builder, input_model, use_hybrid_evaluation,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}
}  // namespace internal

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             QuantizerType quantizer_type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_10(mht_10_v, 820, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeights");

  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    return mlir::lite::QuantizeWeights(builder, input_model,
                                       weights_min_num_elements);
  }
  CustomOpMap custom_op_map;
  return QuantizeWeightsInt8(builder, input_model, true,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model, BufferType quant_type,
                             bool use_updated_hybrid_scheme,
                             QuantizerType quantizer_type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_11(mht_11_v, 837, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeights");

  // By default we require that only weights with more than
  // kWeightsMinSizeDefault elements are quantized.
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    return mlir::lite::QuantizeWeights(builder, input_model,
                                       (mlir::lite::BufferType)quant_type,
                                       use_updated_hybrid_scheme);
  }
  switch (quant_type) {
    case BufferType::QUANTIZED_INT8: {
      CustomOpMap custom_op_map;
      return QuantizeWeightsInt8(builder, input_model, true,
                                 kWeightsMinNumElementsDefault, custom_op_map,
                                 use_updated_hybrid_scheme);
    }
    case BufferType::QUANTIZED_FLOAT16:
      return QuantizeWeightsFloat16(builder, input_model);
  }
}

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             QuantizerType quantizer_type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_12(mht_12_v, 864, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeights");

  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    mlir::lite::CustomOpMap mlir_custom_op_map;
    ConstructMLIRCustomOpMap(mlir_custom_op_map, custom_op_map);
    return mlir::lite::QuantizeWeights(
        builder, input_model, weights_min_num_elements, mlir_custom_op_map);
  }
  return QuantizeWeightsInt8(builder, input_model, true,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme,
                             const flat_hash_set<BuiltinOperator>& op_denylist,
                             QuantizerType quantizer_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePSquantize_weightsDTcc mht_13(mht_13_v, 885, "", "./tensorflow/lite/tools/optimize/quantize_weights.cc", "QuantizeWeights");

  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    mlir::lite::CustomOpMap mlir_custom_op_map;
    ConstructMLIRCustomOpMap(mlir_custom_op_map, custom_op_map);
    return mlir::lite::QuantizeWeights(
        builder, input_model, weights_min_num_elements, mlir_custom_op_map,
        use_updated_hybrid_scheme, op_denylist);
  }
  return QuantizeWeightsInt8(builder, input_model,
                             /*use_hybrid_evaluation=*/true,
                             weights_min_num_elements, custom_op_map,
                             use_updated_hybrid_scheme, op_denylist);
}

}  // namespace optimize
}  // namespace tflite
