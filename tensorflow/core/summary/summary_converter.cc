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
class MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/summary/summary_converter.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/lib/wav/wav_io.h"

namespace tensorflow {
namespace {

template <typename T>
Status TensorValueAt(Tensor t, int64_t i, T* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/summary/summary_converter.cc", "TensorValueAt");

#define CASE(I)                            \
  case DataTypeToEnum<I>::value:           \
    *out = static_cast<T>(t.flat<I>()(i)); \
    break;
#define COMPLEX_CASE(I)                           \
  case DataTypeToEnum<I>::value:                  \
    *out = static_cast<T>(t.flat<I>()(i).real()); \
    break;
  // clang-format off
  switch (t.dtype()) {
    TF_CALL_bool(CASE)
    TF_CALL_half(CASE)
    TF_CALL_float(CASE)
    TF_CALL_double(CASE)
    TF_CALL_int8(CASE)
    TF_CALL_int16(CASE)
    TF_CALL_int32(CASE)
    TF_CALL_int64(CASE)
    TF_CALL_uint8(CASE)
    TF_CALL_uint16(CASE)
    TF_CALL_uint32(CASE)
    TF_CALL_uint64(CASE)
    TF_CALL_complex64(COMPLEX_CASE)
    TF_CALL_complex128(COMPLEX_CASE)
    default:
        return errors::Unimplemented("SummaryFileWriter ",
                                     DataTypeString(t.dtype()),
                                     " not supported.");
  }
  // clang-format on
  return Status::OK();
#undef CASE
#undef COMPLEX_CASE
}

typedef Eigen::Tensor<uint8, 2, Eigen::RowMajor> Uint8Image;

// Add the sequence of images specified by ith_image to the summary.
//
// Factoring this loop out into a helper function lets ith_image behave
// differently in the float and uint8 cases: the float case needs a temporary
// buffer which can be shared across calls to ith_image, but the uint8 case
// does not.
Status AddImages(const string& tag, int max_images, int batch_size, int w,
                 int h, int depth,
                 const std::function<Uint8Image(int)>& ith_image, Summary* s) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/summary/summary_converter.cc", "AddImages");

  const int N = std::min<int>(max_images, batch_size);
  for (int i = 0; i < N; ++i) {
    Summary::Value* v = s->add_value();
    // The tag depends on the number of requested images (not the number
    // produced.)
    //
    // Note that later on avisu uses "/" to figure out a consistent naming
    // convention for display, so we append "/image" to guarantee that the
    // image(s) won't be displayed in the global scope with no name.
    if (max_images > 1) {
      v->set_tag(strings::StrCat(tag, "/image/", i));
    } else {
      v->set_tag(strings::StrCat(tag, "/image"));
    }

    const auto image = ith_image(i);
    Summary::Image* si = v->mutable_image();
    si->set_height(h);
    si->set_width(w);
    si->set_colorspace(depth);
    const int channel_bits = 8;
    const int compression = -1;  // Use zlib default
    if (!png::WriteImageToBuffer(image.data(), w, h, w * depth, depth,
                                 channel_bits, compression,
                                 si->mutable_encoded_image_string(), nullptr)) {
      return errors::Internal("PNG encoding failed");
    }
  }
  return Status::OK();
}

template <class T>
void NormalizeFloatImage(int hw, int depth,
                         typename TTypes<T>::ConstMatrix values,
                         typename TTypes<uint8>::ConstVec bad_color,
                         Uint8Image* image) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_2(mht_2_v, 288, "", "./tensorflow/core/summary/summary_converter.cc", "NormalizeFloatImage");

  if (!image->size()) return;  // Nothing to do for empty images

  // Rescale the image to uint8 range.
  //
  // We are trying to generate an RGB image from a float/half tensor.  We do
  // not have any info about the expected range of values in the tensor
  // but the generated image needs to have all RGB values within [0, 255].
  //
  // We use two different algorithms to generate these values.  If the
  // tensor has only positive values we scale them all by 255/max(values).
  // If the tensor has both negative and positive values we scale them by
  // the max of their absolute values and center them around 127.
  //
  // This works for most cases, but does not respect the relative dynamic
  // range across different instances of the tensor.

  // Compute min and max ignoring nonfinite pixels
  float image_min = std::numeric_limits<float>::infinity();
  float image_max = -image_min;
  for (int i = 0; i < hw; i++) {
    bool finite = true;
    for (int j = 0; j < depth; j++) {
      if (!Eigen::numext::isfinite(values(i, j))) {
        finite = false;
        break;
      }
    }
    if (finite) {
      for (int j = 0; j < depth; j++) {
        float value(values(i, j));
        image_min = std::min(image_min, value);
        image_max = std::max(image_max, value);
      }
    }
  }

  // Pick an affine transform into uint8
  const float kZeroThreshold = 1e-6;
  T scale, offset;
  if (image_min < 0) {
    const float max_val = std::max(std::abs(image_min), std::abs(image_max));
    scale = T(max_val < kZeroThreshold ? 0.0f : 127.0f / max_val);
    offset = T(128.0f);
  } else {
    scale = T(image_max < kZeroThreshold ? 0.0f : 255.0f / image_max);
    offset = T(0.0f);
  }

  // Transform image, turning nonfinite values to bad_color
  for (int i = 0; i < hw; i++) {
    bool finite = true;
    for (int j = 0; j < depth; j++) {
      if (!Eigen::numext::isfinite(values(i, j))) {
        finite = false;
        break;
      }
    }
    if (finite) {
      image->chip<0>(i) =
          (values.template chip<0>(i) * scale + offset).template cast<uint8>();
    } else {
      image->chip<0>(i) = bad_color;
    }
  }
}

template <class T>
Status NormalizeAndAddImages(const Tensor& tensor, int max_images, int h, int w,
                             int hw, int depth, int batch_size,
                             const string& base_tag, Tensor bad_color_tensor,
                             Summary* s) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("base_tag: \"" + base_tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_3(mht_3_v, 363, "", "./tensorflow/core/summary/summary_converter.cc", "NormalizeAndAddImages");

  // For float and half images, nans and infs are replaced with bad_color.
  if (bad_color_tensor.dim_size(0) < depth) {
    return errors::InvalidArgument(
        "expected depth <= bad_color.size, got depth = ", depth,
        ", bad_color.size = ", bad_color_tensor.dim_size(0));
  }
  auto bad_color_full = bad_color_tensor.vec<uint8>();
  typename TTypes<uint8>::ConstVec bad_color(bad_color_full.data(), depth);

  // Float images must be scaled and translated.
  Uint8Image image(hw, depth);
  auto ith_image = [&tensor, &image, bad_color, batch_size, hw, depth](int i) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_4(mht_4_v, 378, "", "./tensorflow/core/summary/summary_converter.cc", "lambda");

    auto tensor_eigen = tensor.template shaped<T, 3>({batch_size, hw, depth});
    typename TTypes<T>::ConstMatrix values(
        &tensor_eigen(i, 0, 0), Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
    NormalizeFloatImage<T>(hw, depth, values, bad_color, &image);
    return image;
  };
  return AddImages(base_tag, max_images, batch_size, w, h, depth, ith_image, s);
}

}  // namespace

Status AddTensorAsScalarToSummary(const Tensor& t, const string& tag,
                                  Summary* s) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_5(mht_5_v, 395, "", "./tensorflow/core/summary/summary_converter.cc", "AddTensorAsScalarToSummary");

  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  float value;
  TF_RETURN_IF_ERROR(TensorValueAt<float>(t, 0, &value));
  v->set_simple_value(value);
  return Status::OK();
}

Status AddTensorAsHistogramToSummary(const Tensor& t, const string& tag,
                                     Summary* s) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_6(mht_6_v, 409, "", "./tensorflow/core/summary/summary_converter.cc", "AddTensorAsHistogramToSummary");

  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  histogram::Histogram histo;
  for (int64_t i = 0; i < t.NumElements(); i++) {
    double double_val;
    TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &double_val));
    if (Eigen::numext::isnan(double_val)) {
      return errors::InvalidArgument("Nan in summary histogram for: ", tag);
    } else if (Eigen::numext::isinf(double_val)) {
      return errors::InvalidArgument("Infinity in summary histogram for: ",
                                     tag);
    }
    histo.Add(double_val);
  }
  histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);
  return Status::OK();
}

Status AddTensorAsImageToSummary(const Tensor& tensor, const string& tag,
                                 int max_images, const Tensor& bad_color,
                                 Summary* s) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_7(mht_7_v, 434, "", "./tensorflow/core/summary/summary_converter.cc", "AddTensorAsImageToSummary");

  if (!(tensor.dims() == 4 &&
        (tensor.dim_size(3) == 1 || tensor.dim_size(3) == 3 ||
         tensor.dim_size(3) == 4))) {
    return errors::InvalidArgument(
        "Tensor must be 4-D with last dim 1, 3, or 4, not ",
        tensor.shape().DebugString());
  }
  if (!(tensor.dim_size(0) < (1LL << 31) && tensor.dim_size(1) < (1LL << 31) &&
        tensor.dim_size(2) < (1LL << 31) &&
        (tensor.dim_size(1) * tensor.dim_size(2)) < (1LL << 29))) {
    return errors::InvalidArgument("Tensor too large for summary ",
                                   tensor.shape().DebugString());
  }
  // The casts and h * w cannot overflow because of the limits above.
  const int batch_size = static_cast<int>(tensor.dim_size(0));
  const int h = static_cast<int>(tensor.dim_size(1));
  const int w = static_cast<int>(tensor.dim_size(2));
  const int hw = h * w;  // Compact these two dims for simplicity
  const int depth = static_cast<int>(tensor.dim_size(3));
  if (tensor.dtype() == DT_UINT8) {
    // For uint8 input, no normalization is necessary
    auto ith_image = [&tensor, batch_size, hw, depth](int i) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_8(mht_8_v, 459, "", "./tensorflow/core/summary/summary_converter.cc", "lambda");

      auto values = tensor.shaped<uint8, 3>({batch_size, hw, depth});
      return typename TTypes<uint8>::ConstMatrix(
          &values(i, 0, 0), Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
    };
    TF_RETURN_IF_ERROR(
        AddImages(tag, max_images, batch_size, w, h, depth, ith_image, s));
  } else if (tensor.dtype() == DT_HALF) {
    TF_RETURN_IF_ERROR(NormalizeAndAddImages<Eigen::half>(
        tensor, max_images, h, w, hw, depth, batch_size, tag, bad_color, s));
  } else if (tensor.dtype() == DT_FLOAT) {
    TF_RETURN_IF_ERROR(NormalizeAndAddImages<float>(
        tensor, max_images, h, w, hw, depth, batch_size, tag, bad_color, s));
  } else if (tensor.dtype() == DT_DOUBLE) {
    TF_RETURN_IF_ERROR(NormalizeAndAddImages<double>(
        tensor, max_images, h, w, hw, depth, batch_size, tag, bad_color, s));
  } else {
    return errors::InvalidArgument(
        "Only DT_INT8, DT_HALF, DT_DOUBLE, and DT_FLOAT images are supported. "
        "Got ",
        DataTypeString(tensor.dtype()));
  }
  return Status::OK();
}

Status AddTensorAsAudioToSummary(const Tensor& tensor, const string& tag,
                                 int max_outputs, float sample_rate,
                                 Summary* s) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_converterDTcc mht_9(mht_9_v, 490, "", "./tensorflow/core/summary/summary_converter.cc", "AddTensorAsAudioToSummary");

  if (sample_rate <= 0.0f) {
    return errors::InvalidArgument("sample_rate must be > 0");
  }
  const int batch_size = tensor.dim_size(0);
  const int64_t length_frames = tensor.dim_size(1);
  const int64_t num_channels =
      tensor.dims() == 2 ? 1 : tensor.dim_size(tensor.dims() - 1);
  const int N = std::min<int>(max_outputs, batch_size);
  for (int i = 0; i < N; ++i) {
    Summary::Value* v = s->add_value();
    if (max_outputs > 1) {
      v->set_tag(strings::StrCat(tag, "/audio/", i));
    } else {
      v->set_tag(strings::StrCat(tag, "/audio"));
    }

    Summary::Audio* sa = v->mutable_audio();
    sa->set_sample_rate(sample_rate);
    sa->set_num_channels(num_channels);
    sa->set_length_frames(length_frames);
    sa->set_content_type("audio/wav");

    auto values =
        tensor.shaped<float, 3>({batch_size, length_frames, num_channels});
    auto channels_by_frames = typename TTypes<float>::ConstMatrix(
        &values(i, 0, 0),
        Eigen::DSizes<Eigen::DenseIndex, 2>(length_frames, num_channels));
    size_t sample_rate_truncated = lrintf(sample_rate);
    if (sample_rate_truncated == 0) {
      sample_rate_truncated = 1;
    }
    TF_RETURN_IF_ERROR(wav::EncodeAudioAsS16LEWav(
        channels_by_frames.data(), sample_rate_truncated, num_channels,
        length_frames, sa->mutable_encoded_audio_string()));
  }
  return Status::OK();
}

}  // namespace tensorflow
