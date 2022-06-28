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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc() {
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
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <streambuf>
#include <string>

#include "absl/base/casts.h"
#include "absl/strings/ascii.h"
#include "tensorflow/core/lib/jpeg/jpeg_handle.h"
#include "tensorflow/core/lib/jpeg/jpeg_mem.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/preprocessing_steps.pb.h"

namespace tflite {
namespace evaluation {
namespace {

// We assume 3-channel RGB images.
const int kNumChannels = 3;

// Returns the offset for the element in the raw image array based on the image
// height/weight & coordinates of a pixel (h, w, c).
inline int ImageArrayOffset(int height, int width, int h, int w, int c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ImageArrayOffset");

  return (h * width + w) * kNumChannels + c;
}

// Stores data and size information of an image.
struct ImageData {
  uint32_t width;
  uint32_t height;
  std::unique_ptr<std::vector<float>> data;

  // GetData performs no checks.
  float GetData(int h, int w, int c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "GetData");

    return data->at(ImageArrayOffset(height, width, h, w, c));
  }
};

// Loads the raw image.
inline void LoadImageRaw(std::string* filename, ImageData* image_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_2(mht_2_v, 238, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "LoadImageRaw");

  std::ifstream stream(filename->c_str(), std::ios::in | std::ios::binary);
  std::vector<uint8_t> raw_data((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
  std::vector<float>* orig_image = new std::vector<float>();
  orig_image->reserve(raw_data.size());
  for (int i = 0; i < raw_data.size(); ++i) {
    orig_image->push_back(static_cast<float>(raw_data[i]));
  }
  image_data->data.reset(orig_image);
}

// Loads the jpeg image.
inline void LoadImageJpeg(std::string* filename, ImageData* image_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_3(mht_3_v, 254, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "LoadImageJpeg");

  // Reads image.
  std::ifstream t(*filename);
  std::string image_str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
  const int fsize = image_str.size();
  auto temp = absl::bit_cast<const uint8_t*>(image_str.data());
  std::unique_ptr<uint8_t[]> original_image;
  int original_width, original_height, original_channels;
  tensorflow::jpeg::UncompressFlags flags;
  // JDCT_ISLOW performs slower but more accurate pre-processing.
  // This isn't always obvious in unit tests, but makes a difference during
  // accuracy testing with ILSVRC dataset.
  flags.dct_method = JDCT_ISLOW;
  // We necessarily require a 3-channel image as the output.
  flags.components = kNumChannels;
  original_image.reset(Uncompress(temp, fsize, flags, &original_width,
                                  &original_height, &original_channels,
                                  nullptr));
  // Copies the image data.
  image_data->width = original_width;
  image_data->height = original_height;
  int original_size = original_height * original_width * original_channels;
  std::vector<float>* float_image = new std::vector<float>();
  float_image->reserve(original_size);
  for (int i = 0; i < original_size; ++i) {
    float_image->push_back(static_cast<float>(original_image[i]));
  }
  image_data->data.reset(float_image);
}

// Central-cropping.
inline void Crop(ImageData* image_data, const CroppingParams& crop_params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_4(mht_4_v, 289, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "Crop");

  int crop_height, crop_width;
  int input_width = image_data->width;
  int input_height = image_data->height;
  if (crop_params.has_cropping_fraction()) {
    crop_height =
        static_cast<int>(round(crop_params.cropping_fraction() * input_height));
    crop_width =
        static_cast<int>(round(crop_params.cropping_fraction() * input_width));
  } else if (crop_params.has_target_size()) {
    crop_height = crop_params.target_size().height();
    crop_width = crop_params.target_size().width();
  }
  if (crop_params.has_cropping_fraction() && crop_params.square_cropping()) {
    crop_height = std::min(crop_height, crop_width);
    crop_width = crop_height;
  }
  int start_w = static_cast<int>(round((input_width - crop_width) / 2.0));
  int start_h = static_cast<int>(round((input_height - crop_height) / 2.0));
  std::vector<float>* cropped_image = new std::vector<float>();
  cropped_image->reserve(crop_height * crop_width * kNumChannels);
  for (int in_h = start_h; in_h < start_h + crop_height; ++in_h) {
    for (int in_w = start_w; in_w < start_w + crop_width; ++in_w) {
      for (int c = 0; c < kNumChannels; ++c) {
        cropped_image->push_back(image_data->GetData(in_h, in_w, c));
      }
    }
  }
  image_data->height = crop_height;
  image_data->width = crop_width;
  image_data->data.reset(cropped_image);
}

// Performs billinear interpolation for 3-channel RGB image.
// See: https://en.wikipedia.org/wiki/Bilinear_interpolation
inline void ResizeBilinear(ImageData* image_data,
                           const ResizingParams& params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_5(mht_5_v, 328, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ResizeBilinear");

  tflite::ResizeBilinearParams resize_params;
  resize_params.align_corners = false;
  // TODO(b/143292772): Set this to true for more accurate behavior?
  resize_params.half_pixel_centers = false;
  tflite::RuntimeShape input_shape({1, static_cast<int>(image_data->height),
                                    static_cast<int>(image_data->width),
                                    kNumChannels});
  // Calculates output size.
  int output_height, output_width;
  if (params.aspect_preserving()) {
    float ratio_w =
        params.target_size().width() / static_cast<float>(image_data->width);
    float ratio_h =
        params.target_size().height() / static_cast<float>(image_data->height);
    if (ratio_w >= ratio_h) {
      output_width = params.target_size().width();
      output_height = static_cast<int>(round(image_data->height * ratio_w));
    } else {
      output_width = static_cast<int>(round(image_data->width * ratio_h));
      output_height = params.target_size().height();
    }
  } else {
    output_height = params.target_size().height();
    output_width = params.target_size().width();
  }
  tflite::RuntimeShape output_size_dims({1, 1, 1, 2});
  std::vector<int32_t> output_size_data = {output_height, output_width};
  tflite::RuntimeShape output_shape(
      {1, output_height, output_width, kNumChannels});
  int output_size = output_width * output_height * kNumChannels;
  std::vector<float>* output_data = new std::vector<float>(output_size, 0);
  tflite::reference_ops::ResizeBilinear(
      resize_params, input_shape, image_data->data->data(), output_size_dims,
      output_size_data.data(), output_shape, output_data->data());
  image_data->height = output_height;
  image_data->width = output_width;
  image_data->data.reset(output_data);
}

// Pads the image to a pre-defined size.
inline void Pad(ImageData* image_data, const PaddingParams& params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_6(mht_6_v, 372, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "Pad");

  int output_width = params.target_size().width();
  int output_height = params.target_size().height();
  int pad_value = params.padding_value();
  tflite::PadParams pad_params;
  pad_params.left_padding_count = 4;
  std::uninitialized_fill_n(pad_params.left_padding, 4, 0);
  pad_params.left_padding[1] =
      static_cast<int>(round((output_height - image_data->height) / 2.0));
  pad_params.left_padding[2] =
      static_cast<int>(round((output_width - image_data->width) / 2.0));
  pad_params.right_padding_count = 4;
  std::uninitialized_fill_n(pad_params.right_padding, 4, 0);
  pad_params.right_padding[1] =
      output_height - pad_params.left_padding[1] - image_data->height;
  pad_params.right_padding[2] =
      output_width - pad_params.left_padding[2] - image_data->width;
  tflite::RuntimeShape input_shape({1, static_cast<int>(image_data->height),
                                    static_cast<int>(image_data->width),
                                    kNumChannels});
  tflite::RuntimeShape output_shape(
      {1, output_height, output_width, kNumChannels});
  int output_size = output_width * output_height * kNumChannels;
  std::vector<float>* output_data = new std::vector<float>(output_size, 0);
  tflite::reference_ops::Pad(pad_params, input_shape, image_data->data->data(),
                             &pad_value, output_shape, output_data->data());
  image_data->height = output_height;
  image_data->width = output_width;
  image_data->data.reset(output_data);
}

// Normalizes the image data to a specific range with mean and scale.
inline void Normalize(ImageData* image_data,
                      const NormalizationParams& params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_7(mht_7_v, 408, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "Normalize");

  float scale = params.scale();
  float* data_end = image_data->data->data() + image_data->data->size();
  if (params.has_channelwise_mean()) {
    float mean = params.channelwise_mean();
    for (float* data = image_data->data->data(); data < data_end; ++data) {
      *data = (*data - mean) * scale;
    }
  } else {
    float r_mean = params.means().r_mean();
    float g_mean = params.means().g_mean();
    float b_mean = params.means().b_mean();
    for (float* data = image_data->data->data(); data < data_end;) {
      *data = (*data - r_mean) * scale;
      ++data;
      *data = (*data - g_mean) * scale;
      ++data;
      *data = (*data - b_mean) * scale;
      ++data;
    }
  }
}
}  // namespace

TfLiteStatus ImagePreprocessingStage::Init() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_8(mht_8_v, 435, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ImagePreprocessingStage::Init");

  if (!config_.has_specification() ||
      !config_.specification().has_image_preprocessing_params()) {
    LOG(ERROR) << "No preprocessing params";
    return kTfLiteError;
  }
  const ImagePreprocessingParams& params =
      config_.specification().image_preprocessing_params();
  // Validates the cropping fraction.
  for (const ImagePreprocessingStepParams& param : params.steps()) {
    if (param.has_cropping_params()) {
      const CroppingParams& crop_params = param.cropping_params();
      if (crop_params.has_cropping_fraction() &&
          (crop_params.cropping_fraction() <= 0 ||
           crop_params.cropping_fraction() > 1.0)) {
        LOG(ERROR) << "Invalid cropping fraction";
        return kTfLiteError;
      }
    }
  }
  output_type_ = static_cast<TfLiteType>(params.output_type());
  return kTfLiteOk;
}

TfLiteStatus ImagePreprocessingStage::Run() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_9(mht_9_v, 462, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ImagePreprocessingStage::Run");

  if (!image_path_) {
    LOG(ERROR) << "Image path not set";
    return kTfLiteError;
  }

  ImageData image_data;
  const ImagePreprocessingParams& params =
      config_.specification().image_preprocessing_params();
  int64_t start_us = profiling::time::NowMicros();
  // Loads the image from file.
  string image_ext = image_path_->substr(image_path_->find_last_of("."));
  absl::AsciiStrToLower(&image_ext);
  bool is_raw_image = (image_ext == ".rgb8");
  if (image_ext == ".rgb8") {
    LoadImageRaw(image_path_, &image_data);
  } else if (image_ext == ".jpg" || image_ext == ".jpeg") {
    LoadImageJpeg(image_path_, &image_data);
  } else {
    LOG(ERROR) << "Extension " << image_ext << " is not supported";
    return kTfLiteError;
  }

  // Cropping, padding and resizing are not supported with raw images since raw
  // images do not contain image size information. Those steps are assumed to
  // be done before raw images are generated.
  for (const ImagePreprocessingStepParams& param : params.steps()) {
    if (param.has_cropping_params()) {
      if (is_raw_image) {
        LOG(WARNING) << "Image cropping will not be performed on raw images";
        continue;
      }
      Crop(&image_data, param.cropping_params());
    } else if (param.has_resizing_params()) {
      if (is_raw_image) {
        LOG(WARNING) << "Image resizing will not be performed on raw images";
        continue;
      }
      ResizeBilinear(&image_data, param.resizing_params());
    } else if (param.has_padding_params()) {
      if (is_raw_image) {
        LOG(WARNING) << "Image padding will not be performed on raw images";
        continue;
      }
      Pad(&image_data, param.padding_params());
    } else if (param.has_normalization_params()) {
      Normalize(&image_data, param.normalization_params());
    }
  }

  // Converts data to output type.
  if (output_type_ == kTfLiteUInt8) {
    uint8_preprocessed_image_.clear();
    uint8_preprocessed_image_.reserve(image_data.data->size());
    for (int i = 0; i < image_data.data->size(); ++i) {
      uint8_preprocessed_image_.push_back(
          static_cast<uint8_t>(image_data.data->at(i)));
    }
  } else if (output_type_ == kTfLiteInt8) {
    int8_preprocessed_image_.clear();
    int8_preprocessed_image_.reserve(image_data.data->size());
    for (int i = 0; i < image_data.data->size(); ++i) {
      int8_preprocessed_image_.push_back(
          static_cast<int8_t>(image_data.data->at(i)));
    }
  } else if (output_type_ == kTfLiteFloat32) {
    float_preprocessed_image_ = *image_data.data;
  }

  latency_stats_.UpdateStat(profiling::time::NowMicros() - start_us);
  return kTfLiteOk;
}

void* ImagePreprocessingStage::GetPreprocessedImageData() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_10(mht_10_v, 538, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ImagePreprocessingStage::GetPreprocessedImageData");

  if (latency_stats_.count() == 0) return nullptr;

  if (output_type_ == kTfLiteUInt8) {
    return uint8_preprocessed_image_.data();
  } else if (output_type_ == kTfLiteInt8) {
    return int8_preprocessed_image_.data();
  } else if (output_type_ == kTfLiteFloat32) {
    return float_preprocessed_image_.data();
  }
  return nullptr;
}

EvaluationStageMetrics ImagePreprocessingStage::LatestMetrics() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSstagesPSimage_preprocessing_stageDTcc mht_11(mht_11_v, 554, "", "./tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc", "ImagePreprocessingStage::LatestMetrics");

  EvaluationStageMetrics metrics;
  auto* latency_metrics =
      metrics.mutable_process_metrics()->mutable_total_latency();
  latency_metrics->set_last_us(latency_stats_.newest());
  latency_metrics->set_max_us(latency_stats_.max());
  latency_metrics->set_min_us(latency_stats_.min());
  latency_metrics->set_sum_us(latency_stats_.sum());
  latency_metrics->set_avg_us(latency_stats_.avg());
  metrics.set_num_runs(static_cast<int>(latency_stats_.count()));
  return metrics;
}

}  // namespace evaluation
}  // namespace tflite
