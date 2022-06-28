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
class MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/base64.h"

#include <cstring>
#include <memory>
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {
// This array must have signed type.
// clang-format off
constexpr int8 kBase64Bytes[128] = {
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  0x3E,  -1,   -1,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D,  -1,   -1,
     -1,   -1,   -1,   -1,   -1,  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,  -1,   -1,   -1,   -1,  0x3F,
     -1,  0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24,
    0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
    0x31, 0x32, 0x33,  -1,   -1,   -1,   -1,   -1};
// clang-format on

constexpr char kBase64UrlSafeChars[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

constexpr char kPadChar = '=';

// Converts a char (8 bits) into a 6-bit value for decoding. If the input char
// is invalid for base64 encoding, the return value has at least its upper 25
// bits set.
inline uint32 Convert(char x) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("x: '" + std::string(1, x) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/platform/base64.cc", "Convert");

  // If x < 128, then we look up x in the table. If x is valid, then the table
  // will have a value <= 0x3F, otherwise the table will have -1. If x >= 128,
  // we still do some table lookup, but the value is ignored since we explicitly
  // set the high bit of y to 1. Either way, y is negative (high bit set) in
  // case of error.
  const int8_t y = kBase64Bytes[x & 0x7F] | (x & 0x80);
  // Casting from int8 to int32 preserves sign by sign extension. If y was
  // negative, at least its 25 high bits of the return value are set.
  const int32_t z = static_cast<int32>(y);
  return static_cast<uint32>(z);
}

Status DecodeThreeChars(const char* codes, char* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("codes: \"" + (codes == nullptr ? std::string("nullptr") : std::string((char*)codes)) + "\"");
   mht_1_v.push_back("result: \"" + (result == nullptr ? std::string("nullptr") : std::string((char*)result)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc mht_1(mht_1_v, 236, "", "./tensorflow/core/platform/base64.cc", "DecodeThreeChars");

  const uint32 packed = (Convert(codes[0]) << 18) | (Convert(codes[1]) << 12) |
                        (Convert(codes[2]) << 6) | (Convert(codes[3]));
  // Convert() return value has upper 25 bits set if input is invalid.
  // Therefore `packed` has high bits set iff at least one of code is invalid.
  if (TF_PREDICT_FALSE((packed & 0xFF000000) != 0)) {
    return errors::InvalidArgument("Invalid character found in base64.");
  }
  result[0] = static_cast<char>(packed >> 16);
  result[1] = static_cast<char>(packed >> 8);
  result[2] = static_cast<char>(packed);
  return Status::OK();
}
}  // namespace

template <typename T>
Status Base64Decode(StringPiece data, T* decoded) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/platform/base64.cc", "Base64Decode");

  if (decoded == nullptr) {
    return errors::Internal("'decoded' cannot be nullptr.");
  }

  if (data.empty()) {
    decoded->clear();
    return Status::OK();
  }

  // This decoding procedure will write 3 * ceil(data.size() / 4) bytes to be
  // output buffer, then truncate if necessary. Therefore we must overestimate
  // and allocate sufficient amount. Currently max_decoded_size may overestimate
  // by up to 3 bytes.
  const size_t max_decoded_size = 3 * (data.size() / 4) + 3;
  std::unique_ptr<char[]> buffer(new char[max_decoded_size]);
  char* current = buffer.get();
  if (current == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate buffer for decoded string.");
  }

  const char* b64 = data.data();
  const char* end = data.data() + data.size();

  while (end - b64 > 4) {
    TF_RETURN_IF_ERROR(DecodeThreeChars(b64, current));
    b64 += 4;
    current += 3;
  }

  if (end - b64 == 4) {
    // The data length is a multiple of 4. Check for padding.
    // Base64 cannot have more than 2 paddings.
    if (b64[2] == kPadChar && b64[3] == kPadChar) {
      end -= 2;
    }
    if (b64[2] != kPadChar && b64[3] == kPadChar) {
      end -= 1;
    }
  }

  const int remain = static_cast<int>(end - b64);
  if (TF_PREDICT_FALSE(remain == 1)) {
    // We may check this condition early by checking data.size() % 4 == 1.
    return errors::InvalidArgument(
        "Base64 string length cannot be 1 modulo 4.");
  }

  // A valid base64 character will replace paddings, if any.
  char tail[4] = {kBase64UrlSafeChars[0], kBase64UrlSafeChars[0],
                  kBase64UrlSafeChars[0], kBase64UrlSafeChars[0]};
  // Copy tail of the input into the array, then decode.
  std::memcpy(tail, b64, remain * sizeof(*b64));
  TF_RETURN_IF_ERROR(DecodeThreeChars(tail, current));
  // We know how many parsed characters are valid.
  current += remain - 1;

  decoded->assign(buffer.get(), current - buffer.get());
  return Status::OK();
}

template <typename T>
Status Base64Encode(StringPiece source, T* encoded) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/platform/base64.cc", "Base64Encode");

  return Base64Encode(source, false, encoded);
}

template <typename T>
Status Base64Encode(StringPiece source, bool with_padding, T* encoded) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSbase64DTcc mht_4(mht_4_v, 329, "", "./tensorflow/core/platform/base64.cc", "Base64Encode");

  const char* const base64_chars = kBase64UrlSafeChars;
  if (encoded == nullptr) {
    return errors::Internal("'encoded' cannot be nullptr.");
  }

  // max_encoded_size may overestimate by up to 4 bytes.
  const size_t max_encoded_size = 4 * (source.size() / 3) + 4;
  std::unique_ptr<char[]> buffer(new char[max_encoded_size]);
  char* current = buffer.get();
  if (current == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate buffer for encoded string.");
  }

  const char* data = source.data();
  const char* const end = source.data() + source.size();

  // Encode each block.
  while (end - data >= 3) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[0] & 0x03) << 4) | ((data[1] >> 4) & 0x0F)];
    *current++ =
        base64_chars[((data[1] & 0x0F) << 2) | ((data[2] >> 6) & 0x03)];
    *current++ = base64_chars[data[2] & 0x3F];

    data += 3;
  }

  // Take care of the tail.
  if (end - data == 2) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ =
        base64_chars[((data[0] & 0x03) << 4) | ((data[1] >> 4) & 0x0F)];
    *current++ = base64_chars[(data[1] & 0x0F) << 2];
    if (with_padding) {
      *current++ = kPadChar;
    }
  } else if (end - data == 1) {
    *current++ = base64_chars[(data[0] >> 2) & 0x3F];
    *current++ = base64_chars[(data[0] & 0x03) << 4];
    if (with_padding) {
      *current++ = kPadChar;
      *current++ = kPadChar;
    }
  }

  encoded->assign(buffer.get(), current - buffer.get());
  return Status::OK();
}

template Status Base64Decode<std::string>(StringPiece data,
                                          std::string* decoded);
template Status Base64Encode<std::string>(StringPiece source,
                                          std::string* encoded);
template Status Base64Encode<std::string>(StringPiece source, bool with_padding,
                                          std::string* encoded);

template Status Base64Decode<tstring>(StringPiece data, tstring* decoded);
template Status Base64Encode<tstring>(StringPiece source, tstring* encoded);
template Status Base64Encode<tstring>(StringPiece source, bool with_padding,
                                      tstring* encoded);

}  // namespace tensorflow
