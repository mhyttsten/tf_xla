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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc() {
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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.h"

#include <setjmp.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_decompress_buffered_struct.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/jpeg_header_parser.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_handle.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// Limiting max image size to 10,000x10,000x3
// This size would fit on 32 bit systems.
// static
const size_t LibjpegDecoder::kMaxImageHeight = 10000;
// static
const size_t LibjpegDecoder::kMaxImageWidth = 10000;

constexpr char kSizeMismatchError[] =
    "JPEG parameter struct mismatch: library thinks size is ";

LibjpegDecoder::Impl::Impl(size_t decompress_struct_size,
                           const LibjpegHandle* handle)
    : decompress_struct_size_(decompress_struct_size),
      handle_(handle),
      cinfo_(decompress_struct_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_0(mht_0_v, 227, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "LibjpegDecoder::Impl::Impl");

  cinfo_.get()->err = handle->jpeg_std_error_(&jerr_);
  jerr_.error_exit = ErrorExit;
  cinfo_.get()->client_data = this;
}

void LibjpegDecoder::Impl::ErrorExit(j_common_ptr cinfo) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "LibjpegDecoder::Impl::ErrorExit");

  Impl* const impl = reinterpret_cast<Impl*>(cinfo->client_data);
  char message[JMSG_LENGTH_MAX];
  cinfo->err->format_message(cinfo, message);
  impl->status_.code = kTfLiteError;
  impl->status_.error_message = message;
  // Libjpeg aborts the program in case of any errors by using longjmp and then
  // calling exit(). The only way to avoid this, is to transfer the control flow
  // to the caller by using setjmp/longjmp.
  // Note: Ensure that function containing the corresponding setjmp() is
  // guaranteed not to have completed execution.
  // https://wiki.sei.cmu.edu/confluence/display/c/MSC22-C.+Use+the+setjmp%28%29%2C+longjmp%28%29+facility+securely
  longjmp(impl->env_, 1);
}

Status ExtractSizeFromErrorMessage(const std::string& error_message,
                                   size_t& expected_size) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_2(mht_2_v, 256, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "ExtractSizeFromErrorMessage");

  Status status;
  // Special error handling for struct mismatch issues.
  // If there's a mismatch, set `expected_size` with the expected
  // size. Error messages are like this: "JPEG parameter struct
  // mismatch: library thinks size is 480, caller expects 464".
  static const int kExpLengthStart = strlen(kSizeMismatchError);
  int end = kExpLengthStart;
  while (end < error_message.length() && std::isdigit(error_message[end])) {
    end++;
  }
  if (end > kExpLengthStart) {
    expected_size = std::stoi(error_message.substr(kExpLengthStart, end));
  } else {
    status.code = kTfLiteError;
    status.error_message =
        "Couldn't parse the size from message: \'" + error_message + "\'";
  }
  return status;
}

std::unique_ptr<LibjpegDecoder> LibjpegDecoder::Create(Status& status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_3(mht_3_v, 280, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "LibjpegDecoder::Create");

  std::unique_ptr<LibjpegDecoder> decoder(
      new LibjpegDecoder(LibCHandle::Create(status)));
  if (status.code != kTfLiteOk) {
    return nullptr;
  }
  decoder->libjpeg_handle_ = LibjpegHandle::Create(status);
  if (decoder->libjpeg_handle_ == nullptr) {
    return nullptr;
  }

  // Tries to probe the libjpeg library to get the expected size of
  // `jpeg_decompress_struct`.
  Impl impl(sizeof(jpeg_decompress_struct), decoder->libjpeg_handle_.get());
  impl.jpeg_CreateDecompress(LibjpegHandle::kLibjpegVersion,
                             sizeof(jpeg_decompress_struct));
  status = impl.status();
  if (status.code == kTfLiteOk) {
    decoder->expected_size_for_decompress_struct_ =
        sizeof(jpeg_decompress_struct);
    return decoder;
  }
  if (!absl::StrContains(status.error_message, kSizeMismatchError)) {
    return nullptr;
  }
  status = ExtractSizeFromErrorMessage(
      status.error_message, decoder->expected_size_for_decompress_struct_);
  if (status.code != kTfLiteOk) {
    return nullptr;
  }
  return decoder;
}

namespace {

std::string JpegHeaderToString(const JpegHeader& header) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_4(mht_4_v, 318, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "JpegHeaderToString");

  return "(" + std::to_string(header.height) + ", " +
         std::to_string(header.width) + ", " + std::to_string(header.channels) +
         ", " + std::to_string(header.bits_per_sample) + ")";
}

}  // namespace

Status LibjpegDecoder::DecodeImage(const tflite::StringRef& encoded,
                                   const JpegHeader& expected_image_dimensions,
                                   unsigned char* decoded,
                                   const size_t& decoded_size) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("decoded: \"" + (decoded == nullptr ? std::string("nullptr") : std::string((char*)decoded)) + "\"");
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSlibjpeg_decoderDTcc mht_5(mht_5_v, 333, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/libjpeg_decoder.cc", "LibjpegDecoder::DecodeImage");

  if (expected_image_dimensions.bits_per_sample != 8) {
    return {kTfLiteError, "Supporting only images with 8 bits per sample"};
  }
  if (expected_image_dimensions.channels != 1 &&
      expected_image_dimensions.channels != 3) {
    return {kTfLiteError, "Supporting only images with 1 or 3 channels"};
  }
  if (expected_image_dimensions.width > kMaxImageWidth ||
      expected_image_dimensions.height > kMaxImageHeight) {
    return {kTfLiteError, "Image is too big, dimensions (" +
                              std::to_string(expected_image_dimensions.width) +
                              "," +
                              std::to_string(expected_image_dimensions.width) +
                              ") larger than the maximum allowed (" +
                              std::to_string(kMaxImageWidth) + ", " +
                              std::to_string(kMaxImageHeight) + ")"};
  }
  // We match the buffer size and the expected size of the decoded image from
  // the header to prevent buffer overflows.
  JpegHeader header;
  Status read_header_status = ReadJpegHeader(encoded, &header);
  if (read_header_status.code != kTfLiteOk) {
    return read_header_status;
  }

  if (expected_image_dimensions.channels != header.channels ||
      expected_image_dimensions.width != header.width ||
      expected_image_dimensions.height != header.height ||
      expected_image_dimensions.bits_per_sample != header.bits_per_sample) {
    return {kTfLiteError, "Decoded image size " + JpegHeaderToString(header) +
                              " is different from provided image size " +
                              JpegHeaderToString(expected_image_dimensions)};
  }

  size_t header_image_size = static_cast<size_t>(header.width) *
                             static_cast<size_t>(header.height) *
                             static_cast<size_t>(header.channels);

  if (header_image_size != decoded_size) {
    return {kTfLiteError, "Size of buffer(" + std::to_string(decoded_size) +
                              ") for storing decoded image must be equal to "
                              "the size of decoded image(" +
                              std::to_string(header_image_size) + ")."};
  }

  // Dropping constness as fmemopen requires non-const buffers.
  char* image_buffer = const_cast<char*>(encoded.str);
  size_t image_size = encoded.len;
  std::unique_ptr<FILE, std::function<void(FILE*)>> file(
      libc_handle_.fmemopen(image_buffer, image_size, "r"),
      [](FILE* f) { fclose(f); });
  if (file == nullptr) {
    return {kTfLiteError, "Fmemopen failed."};
  }

  Impl impl(expected_size_for_decompress_struct_, libjpeg_handle_.get());
  if (impl.jpeg_CreateDecompress(LibjpegHandle::kLibjpegVersion,
                                 expected_size_for_decompress_struct_)) {
    return impl.status();
  }
  if (impl.jpeg_stdio_src(file.get())) {
    return impl.status();
  }
  // jpeg_read_header() must be called before calling jpeg_start_decompress().
  // It initialises decompression parameters to default values.
  // jpeg_read_header() should not be relied upon for getting image information
  // (width, height) etc. Fields populated by jpeg_read_header() such as
  // `image_width` and `image_height` come after the `jpeg_common_fields` and
  // these may have been shifted in the struct on some builds of libjpeg.
  // See go/libjpeg-android.
  int read_header_result = 0;
  if (impl.jpeg_read_header(read_header_result, true) != kTfLiteOk) {
    return impl.status();
  }
  if (read_header_result != JPEG_HEADER_OK) {
    return {kTfLiteError, "Failed call jpeg_read_header"};
  }
  boolean start_decompress_result = false;
  if (impl.jpeg_start_decompress(start_decompress_result) != kTfLiteOk) {
    return impl.status();
  }
  if (!start_decompress_result) {
    return {kTfLiteError, "Failed call jpeg_start_decompress_"};
  }

  size_t height = header.height;
  size_t row_stride = header.width * header.channels;

  // Decoding the data in a buffer as large as the largest allowed JPEG
  // image row to avoid overflows in case  we are reading a wrong value for the
  // image size in the header or we are receiving an image with an header
  // deliberately incorrect to cause a buffer overflow.
  // Using 4 channels because the decode color process can handle 3 or 4
  // channels:
  // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jdcolor.c#L383
  const size_t kMaxImageSize = JPEG_MAX_DIMENSION * 4;
  // Initializing the buffer in case we are trying to read more data than
  // actually available to avoid having access to uninitialized memory.
  // Libjpeg turbo actually fills the unread bytes with zeros
  // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/jdhuff.c#L360
  // but we don't know what the library on the target system would do.
  // Output tensor data stored as RGBRGBRGB..., row-wise for all images.
  std::vector<unsigned char> decode_buffer(kMaxImageSize);
  // Do not rely on fields such as `output_scanline` from
  // `jpeg_decompress_struct` as these would have been shifted. See
  // go/libjpeg-android.
  unsigned char* buffer_array[1];
  buffer_array[0] = decode_buffer.data();
  size_t decoded_offset = 0;
  while (height--) {
    // According to the documentation, jpeg_read_scanlines returns the number
    // of lines read
    // https://android.googlesource.com/platform/external/jpeg/+/c6859b743e7248b9f401264aac939a5af0d63799/libjpeg.doc#655
    // In case of premature ending of the image, the implementation of
    // jpeg_read_scanlines in the version of JPEG Turbo we are using to test
    // emits a warning ("Corrupt JPEG data: premature end of data segment")
    // but doesn't fail and consider the line as successfully read.
    // See test
    // LibjpegDecoderTest::DoesNotFailDecodingAnImageWithLessDataThanDeclaredInJpegHeader
    unsigned int num_of_scanlines_read = 0;
    if (impl.jpeg_read_scanlines(num_of_scanlines_read, buffer_array, 1) !=
        kTfLiteOk) {
      return impl.status();
    }

    if (num_of_scanlines_read != 1) {
      return {kTfLiteError, "Expected " + std::to_string(header.height) +
                                " lines but found only " +
                                std::to_string(header.height - height) +
                                " read scanlines is " +
                                std::to_string(num_of_scanlines_read)};
    }

    std::copy_n(buffer_array[0], row_stride, decoded + decoded_offset);

    decoded_offset += row_stride;
  }
  boolean finish_decompress_result = false;
  if (impl.jpeg_finish_decompress(finish_decompress_result) != kTfLiteOk) {
    return impl.status();
  }
  if (!finish_decompress_result) {
    return {kTfLiteError, "Failed call jpeg_finish_decompress_"};
  }
  if (impl.jpeg_destroy_decompress() != kTfLiteOk) {
    return impl.status();
  }
  return impl.status();
}
}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
