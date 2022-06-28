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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/convert.h"

#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "fp16.h"  // from @FP16
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {

constexpr int kPhwc4ChannelsInPlane = 4;
constexpr int kPhwo4i4ChannelsInPlane = 4;
constexpr int kPiohw4ChannelsInPlane = 4;

// Layout is Po,H,W,OI4x4.
absl::Status ConvertToPHWO4I4(absl::Span<const float> in, const OHWI& shape,
                              absl::Span<float> out, bool reverse_space) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_0(mht_0_v, 213, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertToPHWO4I4");

  if (in.size() != shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWO4I4: Input data size does not match expected size: ",
        in.size(), " != ", shape.DimensionsProduct()));
  }
  if (out.size() != GetElementsSizeForPHWO4I4(shape)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWO4I4: Output data size does not match expected size: ",
        out.size(), " != ", GetElementsSizeForPHWO4I4(shape)));
  }

  float* output = out.data();
  for (int p = 0; p < DivideRoundUp(shape.o, kPhwo4i4ChannelsInPlane); ++p) {
    for (int h = 0; h < shape.h; ++h) {
      for (int w = 0; w < shape.w; ++w) {
        for (int c = 0; c < DivideRoundUp(shape.i, kPhwo4i4ChannelsInPlane);
             ++c) {
          for (int co = 0; co < kPhwo4i4ChannelsInPlane; ++co) {
            for (int ci = 0; ci < kPhwo4i4ChannelsInPlane; ++ci) {
              float value = 0;
              if (c * kPhwo4i4ChannelsInPlane + ci < shape.i &&
                  p * kPhwo4i4ChannelsInPlane + co < shape.o) {
                // tensor is in OHWI
                int tensor_o = p * kPhwo4i4ChannelsInPlane + co;
                int tensor_i = c * kPhwo4i4ChannelsInPlane + ci;
                const int in_h = reverse_space ? shape.h - 1 - h : h;
                const int in_w = reverse_space ? shape.w - 1 - w : w;
                value = in[shape.LinearIndex({tensor_o, in_h, in_w, tensor_i})];
              }
              (*output++) = value;
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

uint32_t GetElementsSizeForPHWO4I4(const OHWI& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_1(mht_1_v, 258, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "GetElementsSizeForPHWO4I4");

  return AlignByN(shape.i, kPhwo4i4ChannelsInPlane) *
         AlignByN(shape.o, kPhwo4i4ChannelsInPlane) * shape.h * shape.w;
}

uint32_t GetElementsSizeForPHWO4I4(const IHWO& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_2(mht_2_v, 266, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "GetElementsSizeForPHWO4I4");

  return AlignByN(shape.i, kPhwo4i4ChannelsInPlane) *
         AlignByN(shape.o, kPhwo4i4ChannelsInPlane) * shape.h * shape.w;
}

std::vector<float> ConvertToPHWO4I4(
    const Tensor<OHWI, DataType::FLOAT32>& tensor) {
  std::vector<float> transposed(GetElementsSizeForPHWO4I4(tensor.shape));
  ConvertToPHWO4I4(tensor.data, tensor.shape,
                   absl::MakeSpan(transposed.data(), transposed.size()),
                   /*reverse_space=*/false)
      .IgnoreError();
  return transposed;
}

std::vector<float> ConvertToPHWO4I4Transposed(
    const Tensor<OHWI, DataType::FLOAT32>& tensor) {
  std::vector<float> transposed(GetElementsSizeForPHWO4I4(tensor.shape));
  ConvertToPHWO4I4(tensor.data, tensor.shape,
                   absl::MakeSpan(transposed.data(), transposed.size()),
                   /*reverse_space=*/true)
      .IgnoreError();
  return transposed;
}

uint3 Get3DSizeForPHWO4I4(const OHWI& shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_3(mht_3_v, 294, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "Get3DSizeForPHWO4I4");

  return uint3(AlignByN(shape.i, 4), shape.h * shape.w,
               DivideRoundUp(shape.o, 4));
}

// Layout is Po,H,W,OI4x4.
absl::Status ConvertToPHWO4I4(absl::Span<const float> in, const IHWO& shape,
                              absl::Span<float> out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_4(mht_4_v, 304, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertToPHWO4I4");

  if (in.size() != shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWO4I4: Input data size does not match expected size: ",
        in.size(), " != ", shape.DimensionsProduct()));
  }
  if (out.size() != GetElementsSizeForPHWO4I4(shape)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWO4I4: Output data size does not match expected size: ",
        out.size(), " != ", GetElementsSizeForPHWO4I4(shape)));
  }

  const int dst_depth = DivideRoundUp(shape.o, 4);
  const int src_depth = DivideRoundUp(shape.i, 4);

  float* output = out.data();
  for (int f = 0; f < dst_depth; ++f) {
    for (int y = 0; y < shape.h; ++y) {
      for (int x = 0; x < shape.w; ++x) {
        for (int ch = 0; ch < src_depth; ++ch) {
          for (int co = 0; co < 4; ++co) {
            for (int ci = 0; ci < 4; ++ci) {
              const int src_channel = ch * 4 + ci;
              const int dst_channel = f * 4 + co;
              float value = 0;
              if (src_channel < shape.i && dst_channel < shape.o) {
                // tensor is in IHWO
                value = in[shape.LinearIndex({src_channel, y, x, dst_channel})];
              }
              (*output++) = value;
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

std::vector<float> ConvertToPHWO4I4(
    const Tensor<IHWO, DataType::FLOAT32>& tensor) {
  std::vector<float> transposed(GetElementsSizeForPHWO4I4(tensor.shape));
  ConvertToPHWO4I4(tensor.data, tensor.shape,
                   absl::MakeSpan(transposed.data(), transposed.size()))
      .IgnoreError();
  return transposed;
}

uint32_t GetElementsSizeForPIOHW4(const OHWI& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_5(mht_5_v, 355, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "GetElementsSizeForPIOHW4");

  return AlignByN(shape.o * shape.i, kPiohw4ChannelsInPlane) * shape.h *
         shape.w;
}

absl::Status ConvertToPIOHW4(absl::Span<const float> in, const OHWI& shape,
                             absl::Span<float> out) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_6(mht_6_v, 364, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertToPIOHW4");

  if (in.size() != shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPIOHW4: Input data size does not match expected size: ",
        in.size(), " != ", shape.DimensionsProduct()));
  }
  if (out.size() != GetElementsSizeForPIOHW4(shape)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPIOHW4: Output data size does not match expected size: ",
        out.size(), " != ", GetElementsSizeForPIOHW4(shape)));
  }

  int32_t output_channels = shape.o * shape.i;
  int32_t num_planes = DivideRoundUp(output_channels, kPiohw4ChannelsInPlane);
  float* output = out.data();
  for (int p = 0; p < num_planes; ++p) {
    for (int h = 0; h < shape.h; ++h) {
      for (int w = 0; w < shape.w; ++w) {
        for (int c = 0; c < kPiohw4ChannelsInPlane; ++c) {
          int output_c = p * kPiohw4ChannelsInPlane + c;
          (*output++) = output_c >= output_channels
                            ? 0
                            : in[shape.LinearIndex({output_c % shape.o, h, w,
                                                    output_c / shape.o})];
        }
      }
    }
  }
  return absl::OkStatus();
}

std::vector<float> ConvertToPIOHW4(
    const Tensor<OHWI, DataType::FLOAT32>& tensor) {
  std::vector<float> transposed(GetElementsSizeForPIOHW4(tensor.shape));
  ConvertToPIOHW4(tensor.data, tensor.shape,
                  absl::MakeSpan(transposed.data(), transposed.size()))
      .IgnoreError();
  return transposed;
}

template <typename T>
absl::Status ValidateConvertToPHWC4(absl::Span<const float> in,
                                    const BHWC& shape, absl::Span<T> out) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_7(mht_7_v, 409, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ValidateConvertToPHWC4");

  if (in.size() != shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWC4: Input data size does not match expected size: ",
        in.size(), " != ", shape.DimensionsProduct()));
  }
  if (out.size() != GetElementsSizeForPHWC4(shape)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertToPHWC4: Output data size does not match expected size: ",
        out.size(), " != ", GetElementsSizeForPHWC4(shape)));
  }
  return absl::OkStatus();
}

// Layout is Pc,H,W,C4 where P - is a plane based on channels.
absl::Status ConvertToPHWC4(absl::Span<const float> in, const BHWC& shape,
                            absl::Span<float> out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_8(mht_8_v, 428, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertToPHWC4");

  RETURN_IF_ERROR(ValidateConvertToPHWC4(in, shape, out));
  if (shape.c == 4) {
    std::memcpy(out.data(), in.data(),
                shape.DimensionsProduct() * sizeof(float));
    return absl::OkStatus();
  }
  // Layout is Pc,H,W,C4 where P - is a plane based on channels.
  int num_planes = DivideRoundUp(shape.c, kPhwc4ChannelsInPlane);
  const int num_pixels = shape.h * shape.w;
  // A layer is a set of kPhwc4ChannelsInPlane channels images.
  const int num_full_planes = shape.c / kPhwc4ChannelsInPlane;
  for (int b = 0; b < shape.b; b++) {
    float* dest =
        out.data() + b * num_pixels * num_planes * kPhwc4ChannelsInPlane;
    for (int p = 0; p < num_full_planes; p++) {
      const float* src =
          in.data() + shape.LinearIndex({b, 0, 0, p * kPhwc4ChannelsInPlane});
      for (int i = 0; i < num_pixels; i++) {
        std::memcpy(dest, src, kPhwc4ChannelsInPlane * sizeof(float));
        src += shape.c;
        dest += kPhwc4ChannelsInPlane;
      }
    }
  }

  // Padding last kPhwc4ChannelsInPlane-channel layer to multiple of
  // kPhwc4ChannelsInPlane.
  const int padded_size = num_pixels * num_planes * kPhwc4ChannelsInPlane;
  const int remaining_channels =
      shape.c - num_full_planes * kPhwc4ChannelsInPlane;
  if (remaining_channels == 0) {
    return absl::OkStatus();
  }
  for (int b = 0; b < shape.b; b++) {
    const float* src =
        in.data() +
        shape.LinearIndex({b, 0, 0, num_full_planes * kPhwc4ChannelsInPlane});
    float* dest = out.data() + b * padded_size +
                  num_pixels * num_full_planes * kPhwc4ChannelsInPlane;
    for (int p = 0; p < num_pixels; p++) {
      std::memcpy(dest, src, remaining_channels * sizeof(float));
      std::memset(dest + remaining_channels, 0,
                  (4 - remaining_channels) * sizeof(float));
      src += shape.c;
      dest += kPhwc4ChannelsInPlane;
    }
  }
  return absl::OkStatus();
}

// Layout is Pc,H,W,C4 where P - is a plane based on channels.
absl::Status ConvertToPHWC4Half(absl::Span<const float> in, const BHWC& shape,
                                absl::Span<HalfBits> out) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_9(mht_9_v, 484, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertToPHWC4Half");

  RETURN_IF_ERROR(ValidateConvertToPHWC4(in, shape, out));

  // Layout is Pc,H,W,C4 where P - is a plane based on channels.
  int num_planes = DivideRoundUp(shape.c, kPhwc4ChannelsInPlane);
  const int num_pixels = shape.h * shape.w;
  // A layer is a set of kPhwc4ChannelsInPlane channels images.
  const int num_full_planes = shape.c / kPhwc4ChannelsInPlane;
  for (int b = 0; b < shape.b; b++) {
    HalfBits* dest =
        out.data() + b * num_pixels * num_planes * kPhwc4ChannelsInPlane;
    for (int p = 0; p < num_full_planes; p++) {
      const float* src =
          in.data() + shape.LinearIndex({b, 0, 0, p * kPhwc4ChannelsInPlane});
      for (int i = 0; i < num_pixels; i++) {
        dest[0] = fp16_ieee_from_fp32_value(src[0]);
        dest[1] = fp16_ieee_from_fp32_value(src[1]);
        dest[2] = fp16_ieee_from_fp32_value(src[2]);
        dest[3] = fp16_ieee_from_fp32_value(src[3]);
        src += shape.c;
        dest += kPhwc4ChannelsInPlane;
      }
    }
  }

  // Padding last kPhwc4ChannelsInPlane-channel layer to multiple of
  // kPhwc4ChannelsInPlane.
  const int padded_size = num_pixels * num_planes * kPhwc4ChannelsInPlane;
  const int remaining_channels =
      shape.c - num_full_planes * kPhwc4ChannelsInPlane;
  if (remaining_channels == 0) {
    return absl::OkStatus();
  }

  for (int b = 0; b < shape.b; b++) {
    const float* src =
        in.data() +
        shape.LinearIndex({b, 0, 0, num_full_planes * kPhwc4ChannelsInPlane});
    HalfBits* dest = out.data() + b * padded_size +
                     num_pixels * num_full_planes * kPhwc4ChannelsInPlane;
    switch (remaining_channels) {
      case 1:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_from_fp32_value(src[0]);
          dest[1] = 0;
          dest[2] = 0;
          dest[3] = 0;
          src += shape.c;
          dest += kPhwc4ChannelsInPlane;
        }
        break;
      case 2:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_from_fp32_value(src[0]);
          dest[1] = fp16_ieee_from_fp32_value(src[1]);
          dest[2] = 0;
          dest[3] = 0;
          src += shape.c;
          dest += kPhwc4ChannelsInPlane;
        }
        break;
      case 3:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_from_fp32_value(src[0]);
          dest[1] = fp16_ieee_from_fp32_value(src[1]);
          dest[2] = fp16_ieee_from_fp32_value(src[2]);
          dest[3] = 0;
          src += shape.c;
          dest += kPhwc4ChannelsInPlane;
        }
        break;
      default:
        return absl::UnimplementedError(
            "ConvertToPHWC4Half: Unsupported channels per planes count.");
    }
  }
  return absl::OkStatus();
}

std::vector<float> ConvertToPHWC4(
    const Tensor<BHWC, DataType::FLOAT32>& tensor) {
  std::vector<float> transposed(GetElementsSizeForPHWC4(tensor.shape));
  ConvertToPHWC4(tensor.data, tensor.shape,
                 absl::MakeSpan(transposed.data(), transposed.size()))
      .IgnoreError();
  // TODO(akulik): Maybe safer to return Status.
  return transposed;
}

std::vector<float> ConvertToPHWC4(
    const Tensor<HWC, DataType::FLOAT32>& tensor) {
  const BHWC batched_shape =
      BHWC(1, tensor.shape.h, tensor.shape.w, tensor.shape.c);
  std::vector<float> transposed(GetElementsSizeForPHWC4(batched_shape));
  ConvertToPHWC4(tensor.data, batched_shape,
                 absl::MakeSpan(transposed.data(), transposed.size()))
      .IgnoreError();
  // TODO(akulik): Maybe safer to return Status.
  return transposed;
}

uint32_t GetElementsSizeForPHWC4(const BHWC& shape) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_10(mht_10_v, 588, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "GetElementsSizeForPHWC4");

  return shape.b * shape.h * shape.w * AlignByN(shape.c, kPhwc4ChannelsInPlane);
}

template <typename T>
absl::Status ValidateConvertFromPHWC4(absl::Span<const T> in, const BHWC& shape,
                                      absl::Span<float> out) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_11(mht_11_v, 597, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ValidateConvertFromPHWC4");

  if (in.size() != GetElementsSizeForPHWC4(shape)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertFromPHWC4: Input data size does not match expected size: ",
        in.size(), " != ", GetElementsSizeForPHWC4(shape)));
  }
  if (out.size() != shape.DimensionsProduct()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ConvertFromPHWC4: Output data size does not match expected size: ",
        out.size(), " != ", shape.DimensionsProduct()));
  }
  return absl::OkStatus();
}

absl::Status ConvertFromPHWC4(absl::Span<const float> in, const BHWC& shape,
                              absl::Span<float> out) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_12(mht_12_v, 615, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertFromPHWC4");

  RETURN_IF_ERROR(ValidateConvertFromPHWC4(in, shape, out));
  if (shape.c == 4) {
    std::memcpy(out.data(), in.data(),
                shape.DimensionsProduct() * sizeof(float));
    return absl::OkStatus();
  }

  int num_planes = DivideRoundUp(shape.c, kPhwc4ChannelsInPlane);
  const int num_pixels = shape.h * shape.w;
  const int padded_size = num_pixels * num_planes * kPhwc4ChannelsInPlane;
  // A layer is a set of kPhwc4ChannelsInPlane channels images.
  const int num_full_planes = shape.c / kPhwc4ChannelsInPlane;
  for (int b = 0; b < shape.b; b++) {
    const float* src = in.data() + b * padded_size;
    for (int p = 0; p < num_full_planes; p++) {
      float* dest =
          out.data() + shape.LinearIndex({b, 0, 0, p * kPhwc4ChannelsInPlane});
      for (int i = 0; i < num_pixels; i++) {
        std::memcpy(dest, src, kPhwc4ChannelsInPlane * sizeof(float));
        src += kPhwc4ChannelsInPlane;
        dest += shape.c;
      }
    }
  }

  // Unpadding last kPhwc4ChannelsInPlane-channel plane
  const int remaining_channels =
      shape.c - num_full_planes * kPhwc4ChannelsInPlane;
  if (remaining_channels == 0) {
    return absl::OkStatus();
  }
  for (int b = 0; b < shape.b; b++) {
    const float* src = in.data() + b * padded_size +
                       num_pixels * num_full_planes * kPhwc4ChannelsInPlane;
    float* dest =
        out.data() +
        shape.LinearIndex({b, 0, 0, num_full_planes * kPhwc4ChannelsInPlane});
    for (int p = 0; p < num_pixels; p++) {
      std::memcpy(dest, src, remaining_channels * sizeof(float));
      src += kPhwc4ChannelsInPlane;
      dest += shape.c;
    }
  }
  return absl::OkStatus();
}

absl::Status ConvertFromPHWC4Half(absl::Span<const HalfBits> in,
                                  const BHWC& shape, absl::Span<float> out) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSconvertDTcc mht_13(mht_13_v, 666, "", "./tensorflow/lite/delegates/gpu/common/convert.cc", "ConvertFromPHWC4Half");

  RETURN_IF_ERROR(ValidateConvertFromPHWC4(in, shape, out));
  int num_planes = DivideRoundUp(shape.c, kPhwc4ChannelsInPlane);
  const int num_pixels = shape.h * shape.w;
  const int padded_size = num_pixels * num_planes * kPhwc4ChannelsInPlane;
  // A layer is a set of kPhwc4ChannelsInPlane channels images.
  const int num_full_planes = shape.c / kPhwc4ChannelsInPlane;
  for (int b = 0; b < shape.b; b++) {
    const HalfBits* src = in.data() + b * padded_size;
    for (int p = 0; p < num_full_planes; p++) {
      float* dest =
          out.data() + shape.LinearIndex({b, 0, 0, p * kPhwc4ChannelsInPlane});
      for (int i = 0; i < num_pixels; i++) {
        dest[0] = fp16_ieee_to_fp32_value(src[0]);
        dest[1] = fp16_ieee_to_fp32_value(src[1]);
        dest[2] = fp16_ieee_to_fp32_value(src[2]);
        dest[3] = fp16_ieee_to_fp32_value(src[3]);
        src += kPhwc4ChannelsInPlane;
        dest += shape.c;
      }
    }
  }

  // Unpadding last kPhwc4ChannelsInPlane-channel plane
  const int remaining_channels =
      shape.c - num_full_planes * kPhwc4ChannelsInPlane;
  if (remaining_channels == 0) {
    return absl::OkStatus();
  }
  for (int b = 0; b < shape.b; b++) {
    const HalfBits* src = in.data() + b * padded_size +
                          num_pixels * num_full_planes * kPhwc4ChannelsInPlane;
    float* dest =
        out.data() +
        shape.LinearIndex({b, 0, 0, num_full_planes * kPhwc4ChannelsInPlane});
    switch (remaining_channels) {
      case 1:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_to_fp32_value(src[0]);
          src += kPhwc4ChannelsInPlane;
          dest += shape.c;
        }
        break;
      case 2:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_to_fp32_value(src[0]);
          dest[1] = fp16_ieee_to_fp32_value(src[1]);
          src += kPhwc4ChannelsInPlane;
          dest += shape.c;
        }
        break;
      case 3:
        for (int p = 0; p < num_pixels; p++) {
          dest[0] = fp16_ieee_to_fp32_value(src[0]);
          dest[1] = fp16_ieee_to_fp32_value(src[1]);
          dest[2] = fp16_ieee_to_fp32_value(src[2]);
          src += kPhwc4ChannelsInPlane;
          dest += shape.c;
        }
        break;
      default:
        return absl::UnimplementedError(
            "ConvertToPHWC4Half: Unsupported channels per planes count.");
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
