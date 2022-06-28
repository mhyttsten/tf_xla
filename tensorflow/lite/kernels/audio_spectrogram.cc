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
class MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc() {
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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/spectrogram.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace audio_spectrogram {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
};

typedef struct {
  int window_size;
  int stride;
  bool magnitude_squared;
  int output_height;
  internal::Spectrogram* spectrogram;
} TfLiteAudioSpectrogramParams;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc mht_0(mht_0_v, 221, "", "./tensorflow/lite/kernels/audio_spectrogram.cc", "Init");

  auto* data = new TfLiteAudioSpectrogramParams;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->window_size = m["window_size"].AsInt64();
  data->stride = m["stride"].AsInt64();
  data->magnitude_squared = m["magnitude_squared"].AsBool();

  data->spectrogram = new internal::Spectrogram;

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/audio_spectrogram.cc", "Free");

  auto* params = reinterpret_cast<TfLiteAudioSpectrogramParams*>(buffer);
  delete params->spectrogram;
  delete params;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/kernels/audio_spectrogram.cc", "Prepare");

  auto* params =
      reinterpret_cast<TfLiteAudioSpectrogramParams*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  TF_LITE_ENSURE(context, params->spectrogram->Initialize(params->window_size,
                                                          params->stride));
  const int64_t sample_count = input->dims->data[0];
  const int64_t length_minus_window = (sample_count - params->window_size);
  if (length_minus_window < 0) {
    params->output_height = 0;
  } else {
    params->output_height = 1 + (length_minus_window / params->stride);
  }
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
  output_size->data[0] = input->dims->data[1];
  output_size->data[1] = params->output_height;
  output_size->data[2] = params->spectrogram->output_frequency_channels();

  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc mht_3(mht_3_v, 287, "", "./tensorflow/lite/kernels/audio_spectrogram.cc", "Eval");

  auto* params =
      reinterpret_cast<TfLiteAudioSpectrogramParams*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE(context, params->spectrogram->Initialize(params->window_size,
                                                          params->stride));

  const float* input_data = GetTensorData<float>(input);

  const int64_t sample_count = input->dims->data[0];
  const int64_t channel_count = input->dims->data[1];

  const int64_t output_width = params->spectrogram->output_frequency_channels();

  float* output_flat = GetTensorData<float>(output);

  std::vector<float> input_for_channel(sample_count);
  for (int64_t channel = 0; channel < channel_count; ++channel) {
    float* output_slice =
        output_flat + (channel * params->output_height * output_width);
    for (int i = 0; i < sample_count; ++i) {
      input_for_channel[i] = input_data[i * channel_count + channel];
    }
    std::vector<std::vector<float>> spectrogram_output;
    TF_LITE_ENSURE(context,
                   params->spectrogram->ComputeSquaredMagnitudeSpectrogram(
                       input_for_channel, &spectrogram_output));
    TF_LITE_ENSURE_EQ(context, spectrogram_output.size(),
                      params->output_height);
    TF_LITE_ENSURE(context, spectrogram_output.empty() ||
                                (spectrogram_output[0].size() == output_width));
    for (int row_index = 0; row_index < params->output_height; ++row_index) {
      const std::vector<float>& spectrogram_row = spectrogram_output[row_index];
      TF_LITE_ENSURE_EQ(context, spectrogram_row.size(), output_width);
      float* output_row = output_slice + (row_index * output_width);
      if (params->magnitude_squared) {
        for (int i = 0; i < output_width; ++i) {
          output_row[i] = spectrogram_row[i];
        }
      } else {
        for (int i = 0; i < output_width; ++i) {
          output_row[i] = sqrtf(spectrogram_row[i]);
        }
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace audio_spectrogram

TfLiteRegistration* Register_AUDIO_SPECTROGRAM() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSaudio_spectrogramDTcc mht_4(mht_4_v, 347, "", "./tensorflow/lite/kernels/audio_spectrogram.cc", "Register_AUDIO_SPECTROGRAM");

  static TfLiteRegistration r = {
      audio_spectrogram::Init, audio_spectrogram::Free,
      audio_spectrogram::Prepare,
      audio_spectrogram::Eval<audio_spectrogram::kReference>};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
