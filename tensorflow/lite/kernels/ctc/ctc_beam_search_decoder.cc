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
class MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc() {
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
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/ctc/ctc_beam_search.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace custom {
namespace ctc_beam_search_decoder {

constexpr int kInputsTensor = 0;
constexpr int kSequenceLengthTensor = 1;

typedef struct {
  int beam_width;
  int top_paths;
  bool merge_repeated;
} CTCBeamSearchDecoderParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Init");

  TFLITE_CHECK(buffer != nullptr);
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  CTCBeamSearchDecoderParams* option = new CTCBeamSearchDecoderParams;
  option->beam_width = m["beam_width"].AsInt32();
  option->top_paths = m["top_paths"].AsInt32();
  option->merge_repeated = m["merge_repeated"].AsBool();

  return option;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Free");

  delete reinterpret_cast<CTCBeamSearchDecoderParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Prepare");

  const CTCBeamSearchDecoderParams* option =
      reinterpret_cast<CTCBeamSearchDecoderParams*>(node->user_data);
  const int top_paths = option->top_paths;
  TF_LITE_ENSURE(context, option->beam_width >= top_paths);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  // The outputs should be top_paths * 3 + 1.
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3 * top_paths + 1);

  const TfLiteTensor* inputs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &inputs));
  TF_LITE_ENSURE_EQ(context, NumDimensions(inputs), 3);
  // TensorFlow only supports float.
  TF_LITE_ENSURE_EQ(context, inputs->type, kTfLiteFloat32);
  const int batch_size = SizeOfDimension(inputs, 1);

  const TfLiteTensor* sequence_length;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSequenceLengthTensor,
                                          &sequence_length));
  TF_LITE_ENSURE_EQ(context, NumDimensions(sequence_length), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(sequence_length), batch_size);
  // TensorFlow only supports int32.
  TF_LITE_ENSURE_EQ(context, sequence_length->type, kTfLiteInt32);

  // Resize decoded outputs.
  // Do not resize indices & values cause we don't know the values yet.
  for (int i = 0; i < top_paths; ++i) {
    TfLiteTensor* indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &indices));
    SetTensorToDynamic(indices);
    TfLiteTensor* values;
    TF_LITE_ENSURE_OK(context,
                      GetOutputSafe(context, node, i + top_paths, &values));
    SetTensorToDynamic(values);
    TfLiteTensor* output_shape;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i + 2 * top_paths,
                                             &output_shape));
    SetTensorToDynamic(output_shape);
  }

  // Resize log probability outputs.
  TfLiteTensor* log_probability_output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, top_paths * 3,
                                           &log_probability_output));
  TfLiteIntArray* log_probability_output_shape_array = TfLiteIntArrayCreate(2);
  log_probability_output_shape_array->data[0] = batch_size;
  log_probability_output_shape_array->data[1] = top_paths;
  return context->ResizeTensor(context, log_probability_output,
                               log_probability_output_shape_array);
}

TfLiteStatus Resize(TfLiteContext* context,
                    std::initializer_list<int32_t> output_shape,
                    TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_3(mht_3_v, 289, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Resize");

  const int dimensions = output_shape.size();
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(dimensions);
  int i = 0;
  for (const int v : output_shape) {
    output_shape_array->data[i++] = v;
  }
  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus StoreAllDecodedSequences(
    TfLiteContext* context,
    const std::vector<std::vector<std::vector<int>>>& sequences,
    TfLiteNode* node, int top_paths) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_4(mht_4_v, 305, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "StoreAllDecodedSequences");

  const int32_t batch_size = sequences.size();
  std::vector<int32_t> num_entries(top_paths, 0);

  // Calculate num_entries per path
  for (const auto& batch_s : sequences) {
    TF_LITE_ENSURE_EQ(context, batch_s.size(), top_paths);
    for (int p = 0; p < top_paths; ++p) {
      num_entries[p] += batch_s[p].size();
    }
  }

  for (int p = 0; p < top_paths; ++p) {
    const int32_t p_num = num_entries[p];

    // Resize the decoded outputs.
    TfLiteTensor* indices;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, p, &indices));
    TF_LITE_ENSURE_OK(context, Resize(context, {p_num, 2}, indices));

    TfLiteTensor* values;
    TF_LITE_ENSURE_OK(context,
                      GetOutputSafe(context, node, p + top_paths, &values));
    TF_LITE_ENSURE_OK(context, Resize(context, {p_num}, values));

    TfLiteTensor* decoded_shape;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, p + 2 * top_paths,
                                             &decoded_shape));
    TF_LITE_ENSURE_OK(context, Resize(context, {2}, decoded_shape));

    int32_t max_decoded = 0;
    int32_t offset = 0;

    int32_t* indices_data = GetTensorData<int32_t>(indices);
    int32_t* values_data = GetTensorData<int32_t>(values);
    int32_t* decoded_shape_data = GetTensorData<int32_t>(decoded_shape);
    for (int b = 0; b < batch_size; ++b) {
      auto& p_batch = sequences[b][p];
      int32_t num_decoded = p_batch.size();
      max_decoded = std::max(max_decoded, num_decoded);

      std::copy_n(p_batch.begin(), num_decoded, values_data + offset);
      for (int32_t t = 0; t < num_decoded; ++t, ++offset) {
        indices_data[offset * 2] = b;
        indices_data[offset * 2 + 1] = t;
      }
    }

    decoded_shape_data[0] = batch_size;
    decoded_shape_data[1] = max_decoded;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_5(mht_5_v, 362, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Eval");

  const TfLiteTensor* inputs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputsTensor, &inputs));
  const TfLiteTensor* sequence_length;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kSequenceLengthTensor,
                                          &sequence_length));
  const CTCBeamSearchDecoderParams* option =
      reinterpret_cast<CTCBeamSearchDecoderParams*>(node->user_data);

  const int max_time = SizeOfDimension(inputs, 0);
  const int batch_size = SizeOfDimension(inputs, 1);
  const int num_classes = SizeOfDimension(inputs, 2);

  const int beam_width = option->beam_width;
  const int top_paths = option->top_paths;
  const bool merge_repeated = option->merge_repeated;

  // Validate sequence length is less or equal than max time.
  for (int i = 0; i < batch_size; ++i) {
    TF_LITE_ENSURE(context,
                   max_time >= GetTensorData<int32_t>(sequence_length)[i]);
  }

  // The following logic is implemented like
  // tensorflow/core/kernels/ctc_decoder_ops.cc
  std::vector<optimized_ops::TTypes<float>::UnalignedConstMatrix> input_list_t;

  for (std::size_t t = 0; t < max_time; ++t) {
    input_list_t.emplace_back(
        GetTensorData<float>(inputs) + t * batch_size * num_classes, batch_size,
        num_classes);
  }

  ::tflite::custom::ctc::CTCBeamSearchDecoder<>::DefaultBeamScorer beam_scorer;
  ::tflite::custom::ctc::CTCBeamSearchDecoder<> beam_search(
      num_classes, beam_width, &beam_scorer, 1 /* batch_size */,
      merge_repeated);

  // Allocate temporary memory for holding chip operation data.
  float* input_chip_t_data =
      static_cast<float*>(malloc(num_classes * sizeof(float)));
  Eigen::array<Eigen::DenseIndex, 1> dims;
  dims[0] = num_classes;
  optimized_ops::TTypes<float>::Flat input_chip_t(input_chip_t_data, dims);

  std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
  std::vector<float> log_probs;

  TfLiteTensor* log_probabilities;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, 3 * top_paths, &log_probabilities));
  float* log_probabilities_output = GetTensorData<float>(log_probabilities);

  // Assumption: the blank index is num_classes - 1
  for (int b = 0; b < batch_size; ++b) {
    auto& best_paths_b = best_paths[b];
    best_paths_b.resize(top_paths);
    for (int t = 0; t < GetTensorData<int32_t>(sequence_length)[b]; ++t) {
      input_chip_t = input_list_t[t].chip(b, 0);
      auto input_bi =
          Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
      beam_search.Step(input_bi);
    }
    TF_LITE_ENSURE(context, beam_search.TopPaths(top_paths, &best_paths_b,
                                                 &log_probs, merge_repeated));
    beam_search.Reset();

    // Fill in log_probabilities output.
    for (int bp = 0; bp < top_paths; ++bp) {
      log_probabilities_output[b * top_paths + bp] = log_probs[bp];
    }
  }

  free(input_chip_t_data);
  return StoreAllDecodedSequences(context, best_paths, node, top_paths);
}

}  // namespace ctc_beam_search_decoder

TfLiteRegistration* Register_CTC_BEAM_SEARCH_DECODER() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSctcPSctc_beam_search_decoderDTcc mht_6(mht_6_v, 445, "", "./tensorflow/lite/kernels/ctc/ctc_beam_search_decoder.cc", "Register_CTC_BEAM_SEARCH_DECODER");

  static TfLiteRegistration r = {
      ctc_beam_search_decoder::Init, ctc_beam_search_decoder::Free,
      ctc_beam_search_decoder::Prepare, ctc_beam_search_decoder::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
