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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc() {
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
#include <memory>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_register.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg.cc", "Init");

  if (!buffer) {
    return nullptr;
  }
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  // TODO(b/172544567): Add error handling for incorrect/missing attributes.
  OpData* op_data = new OpData();
  op_data->height = m["height"].AsInt32();
  op_data->width = m["width"].AsInt32();
  op_data->num_images = m["num_images"].AsInt32();
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc mht_2(mht_2_v, 226, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg.cc", "Prepare");

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data);
  TF_LITE_ENSURE(context, op_data->height > 0);
  TF_LITE_ENSURE(context, op_data->width > 0);
  TF_LITE_ENSURE(context, op_data->num_images > 0);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input_buffer;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, /*index=*/0, &input_buffer));

  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output_tensor));

  TF_LITE_ENSURE_TYPES_EQ(context, input_buffer->type, kTfLiteString);
  TF_LITE_ENSURE_TYPES_EQ(context, output_tensor->type, kTfLiteUInt8);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input_buffer), 1);
  TF_LITE_ENSURE_EQ(context, input_buffer->dims->data[0], op_data->num_images);

  // Resize output.
  // Output shape is determined as {num_images, height, width, channels}.
  TfLiteIntArray* new_dims = TfLiteIntArrayCreate(4);
  new_dims->data[0] = op_data->num_images;
  new_dims->data[1] = op_data->height;
  new_dims->data[2] = op_data->width;
  // TODO(b/172544567): Support grayscale images.
  new_dims->data[3] = 3;  // Channels.
  output_tensor->type = kTfLiteUInt8;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output_tensor, new_dims));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc mht_3(mht_3_v, 267, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg.cc", "Eval");

  // Decodes a batch of JPEG images.

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input_buffer;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, /*index=*/0, &input_buffer));
  TF_LITE_ENSURE(context, input_buffer);
  TF_LITE_ENSURE(context, input_buffer->data.raw);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, /*index=*/0, &output_tensor));
  // kTfliteUInt8 corresponds to unsigned char as shown in
  // "tensorflow/lite/portable_type_to_tflitetype.h".
  unsigned char* output_arr = GetTensorData<unsigned char>(output_tensor);
  Status decoder_status;
  std::unique_ptr<LibjpegDecoder> decoder =
      LibjpegDecoder::Create(decoder_status);
  if (decoder_status.code != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, decoder_status.error_message.c_str());
    return kTfLiteError;
  }

  const int kImageSize = op_data->width * op_data->height * 3;
  int output_array_offset = 0;
  for (int img = 0; img < op_data->num_images; ++img) {
    tflite::StringRef inputref =
        tflite::GetString(input_buffer, /*string_index=*/img);

    Status decode_status = decoder->DecodeImage(
        inputref, {op_data->height, op_data->width, /*channels=*/3},
        output_arr + output_array_offset, kImageSize);

    output_array_offset += kImageSize;

    if (decode_status.code != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context, decode_status.error_message.c_str());
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteRegistration* Register_DECODE_JPEG() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSdecode_jpegDTcc mht_4(mht_4_v, 314, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg.cc", "Register_DECODE_JPEG");

  static TfLiteRegistration r = {
      decode_jpeg_kernel::Init, decode_jpeg_kernel::Free,
      decode_jpeg_kernel::Prepare, decode_jpeg_kernel::Eval};
  return &r;
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
