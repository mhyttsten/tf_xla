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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

/*

JPEG file overall file structure

SOI Marker                 FFD8
Marker XX size=SSSS        FFXX	SSSS	DDDD......
Marker YY size=TTTT        FFYY	TTTT	DDDD......
SOFn marker with the info we want
SOS Marker size=UUUU       FFDA	UUUU	DDDD....
Image stream               I I I I....
EOI Marker                 FFD9

The first marker is either APP0 (JFIF format) or APP1 (EXIF format)

We support only JFIF images
*/

namespace {

using MarkerId = uint16_t;

void AsWord(int value, char* msb, char* lsb) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("msb: \"" + (msb == nullptr ? std::string("nullptr") : std::string((char*)msb)) + "\"");
   mht_0_v.push_back("lsb: \"" + (lsb == nullptr ? std::string("nullptr") : std::string((char*)lsb)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "AsWord");

  *msb = static_cast<char>(value >> 8);
  *lsb = static_cast<char>(value);
}

// JFIF spec at
// https://www.ecma-international.org/publications-and-standards/technical-reports/ecma-tr-98/
// Marker definition summary at
// http://lad.dsc.ufcg.edu.br/multimidia/jpegmarker.pdf
// Overall JPEG File structure with discussion of the supported number of
// channels per format
// https://docs.oracle.com/javase/8/docs/api/javax/imageio/metadata/doc-files/jpeg_metadata.html
//

class JfifHeaderParser {
 public:
  explicit JfifHeaderParser(const tflite::StringRef& jpeg_image_data)
      : jpeg_image_data_(jpeg_image_data), offset_(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_1(mht_1_v, 243, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "JfifHeaderParser");

    if (!IsJpegImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Not a valid JPEG image.";
    } else if (!IsJfifImage(jpeg_image_data_)) {
      is_valid_image_buffer_ = false;
      validation_error_message_ = "Image is not in JFIF format.";
      return;
    } else {
      is_valid_image_buffer_ = true;
    }
  }

#define ENSURE_READ_STATUS(a)                           \
  do {                                                  \
    const TfLiteStatus s = (a);                         \
    if (s != kTfLiteOk) {                               \
      return {s, "Error trying to parse JPEG header."}; \
    }                                                   \
  } while (0)

  Status ReadJpegHeader(JpegHeader* result) {
    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }

    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }

    ENSURE_READ_STATUS(SkipBytes(2));  // skipping marker length
    char precision;
    ENSURE_READ_STATUS(ReadByte(&precision));
    uint16_t height;
    ENSURE_READ_STATUS(ReadWord(&height));
    uint16_t width;
    ENSURE_READ_STATUS(ReadWord(&width));
    char num_of_components;
    ENSURE_READ_STATUS(ReadByte(&num_of_components));

    if (num_of_components != 1 && num_of_components != 3) {
      return {kTfLiteError,
              "A JFIF image without App14 marker doesn't support a number of "
              "components = " +
                  std::to_string(static_cast<int>(num_of_components))};
    }

    result->width = width;
    result->height = height;
    result->channels = num_of_components;
    result->bits_per_sample = precision;

    return {kTfLiteOk, ""};
  }

  Status ApplyHeaderToImage(const JpegHeader& new_header,
                            std::string& write_to) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_2(mht_2_v, 303, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "ApplyHeaderToImage");

    if (!is_valid_image_buffer_) {
      return {kTfLiteError, validation_error_message_};
    }

    Status move_to_sof_status = MoveToStartOfFrameMarker();
    if (move_to_sof_status.code != kTfLiteOk) {
      return move_to_sof_status;
    }
    ENSURE_READ_STATUS(SkipBytes(2));  // skipping marker length

    if (!HasData(6)) {
      return {kTfLiteError,
              "Invalid SOF marker, image buffer ends before end of marker"};
    }

    char header[6];
    header[0] = static_cast<char>(new_header.bits_per_sample);
    AsWord(new_header.height, header + 1, header + 2);
    AsWord(new_header.width, header + 3, header + 4);
    header[5] = static_cast<char>(new_header.channels);

    write_to.clear();
    write_to.append(jpeg_image_data_.str, offset_);
    write_to.append(header, 6);

    ENSURE_READ_STATUS(SkipBytes(6));
    if (HasData()) {
      write_to.append(jpeg_image_data_.str + offset_,
                      jpeg_image_data_.len - offset_);
    }

    return {kTfLiteOk, ""};
  }

 private:
  const tflite::StringRef jpeg_image_data_;
  // Using int for consistency with the size in StringRef
  int offset_;
  bool is_valid_image_buffer_;
  std::string validation_error_message_;

  // Moves to the begin of the first SOF marker
  Status MoveToStartOfFrameMarker() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_3(mht_3_v, 349, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "MoveToStartOfFrameMarker");

    const MarkerId kStartOfStreamMarkerId = 0xFFDA;  // Start of image data

    offset_ = 0;
    ENSURE_READ_STATUS(SkipBytes(4));  // skipping SOI and APP0 marker IDs
    ENSURE_READ_STATUS(SkipCurrentMarker());  // skipping APP0
    MarkerId curr_marker_id;
    // We need at least 2 bytes for the marker ID and 2 for the length
    while (HasData(/*min_data_size=*/4)) {
      ENSURE_READ_STATUS(ReadWord(&curr_marker_id));
      // We are breaking at the first met SOF marker. This won't generate
      // results inconsistent with LibJPEG because only
      // image with a single SOF marker are successfully parsed by it.
      // LibJPEG fails if more than one marker is found in the header (see
      // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jerror.h#L121
      // and
      // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jdmarker.c#L264-L265
      if (IsStartOfFrameMarkerId(curr_marker_id)) {
        break;
      }
      if (curr_marker_id == kStartOfStreamMarkerId) {
        return {kTfLiteError, "Error trying to parse JPEG header."};
      }
      ENSURE_READ_STATUS(SkipCurrentMarker());
    }

    return {kTfLiteOk, ""};
  }

#undef ENSURE_READ_STATUS

  bool HasData(int min_data_size = 1) {
    return offset_ <= jpeg_image_data_.len - min_data_size;
  }

  TfLiteStatus SkipBytes(int bytes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_4(mht_4_v, 387, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "SkipBytes");

    if (!HasData(bytes)) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Trying to move out of image boundaries from offset %d, "
                 "skipping %d bytes",
                 offset_, bytes);
      return kTfLiteError;
    }

    offset_ += bytes;

    return kTfLiteOk;
  }

  TfLiteStatus ReadByte(char* result) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("result: \"" + (result == nullptr ? std::string("nullptr") : std::string((char*)result)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_5(mht_5_v, 405, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "ReadByte");

    if (!HasData()) {
      return kTfLiteError;
    }

    *result = jpeg_image_data_.str[offset_];

    return SkipBytes(1);
  }

  TfLiteStatus ReadWord(uint16_t* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_6(mht_6_v, 418, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "ReadWord");

    TF_LITE_ENSURE_STATUS(ReadWordAt(jpeg_image_data_, offset_, result));
    return SkipBytes(2);
  }

  TfLiteStatus SkipCurrentMarker() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_7(mht_7_v, 426, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "SkipCurrentMarker");

    // We just read the marker ID so we are on top of the marker len
    uint16_t full_marker_len;
    TF_LITE_ENSURE_STATUS(ReadWord(&full_marker_len));
    if (full_marker_len <= 2) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid marker length %d read at offset %X", full_marker_len,
                 offset_);
      return kTfLiteError;
    }

    // The marker len includes the 2 bytes of marker length
    return SkipBytes(full_marker_len - 2);
  }

  static TfLiteStatus ReadWordAt(const tflite::StringRef& jpeg_image_data,
                                 int read_offset, uint16_t* result) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_8(mht_8_v, 445, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "ReadWordAt");

    if (read_offset < 0 || read_offset > jpeg_image_data.len - 2) {
      return kTfLiteError;
    }
    // Cast to unsigned since char can be signed.
    const unsigned char* buf =
        reinterpret_cast<const unsigned char*>(jpeg_image_data.str);

    *result = (buf[read_offset] << 8) + buf[read_offset + 1];

    return kTfLiteOk;
  }

  static bool IsJpegImage(const tflite::StringRef& jpeg_image_data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_9(mht_9_v, 461, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "IsJpegImage");

    const MarkerId kStartOfImageMarkerId = 0xFFD8;
    const MarkerId kEndOfImageMarkerId = 0xFFD9;

    MarkerId soi_marker_id;
    MarkerId eoi_marker_id;
    if (ReadWordAt(jpeg_image_data, 0, &soi_marker_id) != kTfLiteOk) {
      return false;
    }
    if (ReadWordAt(jpeg_image_data, jpeg_image_data.len - 2, &eoi_marker_id) !=
        kTfLiteOk) {
      return false;
    }

    return (soi_marker_id == kStartOfImageMarkerId) &&
           (eoi_marker_id == kEndOfImageMarkerId);
  }

  static bool IsJfifImage(const tflite::StringRef& jpeg_image_data) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_10(mht_10_v, 482, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "IsJfifImage");

    const MarkerId kApp0MarkerId = 0xFFE0;  // First marker in JIFF image

    MarkerId app_marker_id;
    if ((ReadWordAt(jpeg_image_data, 2, &app_marker_id) != kTfLiteOk) ||
        (app_marker_id != kApp0MarkerId)) {
      return false;
    }

    // Checking Jfif identifier string "JFIF\0" in APP0 Marker
    const std::string kJfifIdString{"JFIF\0", 5};

    // The ID starts after SOI (2 bytes), APP0 marker IDs (2 bytes) and 2 other
    // bytes with APP0 marker length
    const int KJfifIdStringStartOffset = 6;

    if (KJfifIdStringStartOffset + kJfifIdString.size() >=
        jpeg_image_data.len) {
      TFLITE_LOG(TFLITE_LOG_WARNING,
                 "Invalid image, reached end of data at offset while "
                 "parsing APP0 header");
      return false;
    }

    const std::string actualImgId(
        jpeg_image_data.str + KJfifIdStringStartOffset, kJfifIdString.size());
    if (kJfifIdString != actualImgId) {
      TFLITE_LOG(TFLITE_LOG_WARNING, "Invalid image, invalid APP0 header");

      return false;
    }

    return true;
  }

  static bool IsStartOfFrameMarkerId(MarkerId marker_id) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_11(mht_11_v, 520, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "IsStartOfFrameMarkerId");

    return 0xFFC0 <= marker_id && marker_id < 0xFFCF;
  }
};

}  // namespace
Status ReadJpegHeader(const tflite::StringRef& jpeg_image_data,
                      JpegHeader* header) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_12(mht_12_v, 530, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "ReadJpegHeader");

  JfifHeaderParser parser(jpeg_image_data);

  return parser.ReadJpegHeader(header);
}

Status BuildImageWithNewHeader(const tflite::StringRef& orig_jpeg_image_data,
                               const JpegHeader& new_header,
                               std::string& new_image_data) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSjpeg_header_parserDTcc mht_13(mht_13_v, 541, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.cc", "BuildImageWithNewHeader");

  JfifHeaderParser parser(orig_jpeg_image_data);

  return parser.ApplyHeaderToImage(new_header, new_image_data);
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
