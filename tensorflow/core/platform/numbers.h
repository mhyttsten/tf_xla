/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_NUMBERS_H_
#define TENSORFLOW_CORE_PLATFORM_NUMBERS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh() {
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


#include <string>

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace strings {

// ----------------------------------------------------------------------
// FastIntToBufferLeft()
//    These are intended for speed.
//
//    All functions take the output buffer as an arg.  FastInt() uses
//    at most 22 bytes, FastTime() uses exactly 30 bytes.  They all
//    return a pointer to the beginning of the output, which is the same as
//    the beginning of the input buffer.
//
//    NOTE: In 64-bit land, sizeof(time_t) is 8, so it is possible
//    to pass to FastTimeToBuffer() a time whose year cannot be
//    represented in 4 digits. In this case, the output buffer
//    will contain the string "Invalid:<value>"
// ----------------------------------------------------------------------

// Previously documented minimums -- the buffers provided must be at least this
// long, though these numbers are subject to change:
//     Int32, UInt32:                   12 bytes
//     Int64, UInt64, Int, Uint:        22 bytes
//     Time:                            30 bytes
// Use kFastToBufferSize rather than hardcoding constants.
static const int kFastToBufferSize = 32;

// ----------------------------------------------------------------------
// FastInt32ToBufferLeft()
// FastUInt32ToBufferLeft()
// FastInt64ToBufferLeft()
// FastUInt64ToBufferLeft()
//
// These functions convert their numeric argument to an ASCII
// representation of the numeric value in base 10, with the
// representation being left-aligned in the buffer.  The caller is
// responsible for ensuring that the buffer has enough space to hold
// the output.  The buffer should typically be at least kFastToBufferSize
// bytes.
//
// Returns the number of characters written.
// ----------------------------------------------------------------------

size_t FastInt32ToBufferLeft(int32_t i, char* buffer);  // at least 12 bytes
size_t FastUInt32ToBufferLeft(uint32 i, char* buffer);  // at least 12 bytes
size_t FastInt64ToBufferLeft(int64_t i, char* buffer);  // at least 22 bytes
size_t FastUInt64ToBufferLeft(uint64 i, char* buffer);  // at least 22 bytes

// Required buffer size for DoubleToBuffer is kFastToBufferSize.
// Required buffer size for FloatToBuffer is kFastToBufferSize.
size_t DoubleToBuffer(double value, char* buffer);
size_t FloatToBuffer(float value, char* buffer);

// Convert a 64-bit fingerprint value to an ASCII representation.
std::string FpToString(Fprint fp);

// Attempt to parse a fingerprint in the form encoded by FpToString.  If
// successful, stores the fingerprint in *fp and returns true.  Otherwise,
// returns false.
bool StringToFp(const std::string& s, Fprint* fp);

// Convert a 64-bit fingerprint value to an ASCII representation that
// is terminated by a '\0'.
// Buf must point to an array of at least kFastToBufferSize characters
StringPiece Uint64ToHexString(uint64 v, char* buf);

// Attempt to parse a uint64 in the form encoded by FastUint64ToHexString.  If
// successful, stores the value in *v and returns true.  Otherwise,
// returns false.
bool HexStringToUint64(const StringPiece& s, uint64* v);

// Convert strings to 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strto32(StringPiece str, int32* value);

// Convert strings to unsigned 32bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strtou32(StringPiece str, uint32* value);

// Convert strings to 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strto64(StringPiece str, int64_t* value);

// Convert strings to unsigned 64bit integer values.
// Leading and trailing spaces are allowed.
// Return false with overflow or invalid input.
bool safe_strtou64(StringPiece str, uint64* value);

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
// Returns false on invalid input or if `strlen(value) >= kFastToBufferSize`.
bool safe_strtof(StringPiece str, float* value);

// Convert strings to double precision floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
// Returns false on invalid input or if `strlen(value) >= kFastToBufferSize`.
bool safe_strtod(StringPiece str, double* value);

inline bool ProtoParseNumeric(StringPiece s, int32* value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_0(mht_0_v, 295, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strto32(s, value);
}

inline bool ProtoParseNumeric(StringPiece s, uint32* value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_1(mht_1_v, 302, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strtou32(s, value);
}

inline bool ProtoParseNumeric(StringPiece s, int64_t* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_2(mht_2_v, 309, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strto64(s, value);
}

inline bool ProtoParseNumeric(StringPiece s, uint64* value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_3(mht_3_v, 316, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strtou64(s, value);
}

inline bool ProtoParseNumeric(StringPiece s, float* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_4(mht_4_v, 323, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strtof(s, value);
}

inline bool ProtoParseNumeric(StringPiece s, double* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_5(mht_5_v, 330, "", "./tensorflow/core/platform/numbers.h", "ProtoParseNumeric");

  return safe_strtod(s, value);
}

// Convert strings to number of type T.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
template <typename T>
bool SafeStringToNumeric(StringPiece s, T* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSnumbersDTh mht_6(mht_6_v, 341, "", "./tensorflow/core/platform/numbers.h", "SafeStringToNumeric");

  return ProtoParseNumeric(s, value);
}

// Converts from an int64 to a human readable string representing the
// same number, using decimal powers.  e.g. 1200000 -> "1.20M".
std::string HumanReadableNum(int64_t value);

// Converts from an int64 representing a number of bytes to a
// human readable string representing the same number.
// e.g. 12345678 -> "11.77MiB".
std::string HumanReadableNumBytes(int64_t num_bytes);

// Converts a time interval as double to a human readable
// string. For example:
//   0.001       -> "1 ms"
//   10.0        -> "10 s"
//   933120.0    -> "10.8 days"
//   39420000.0  -> "1.25 years"
//   -10         -> "-10 s"
std::string HumanReadableElapsedTime(double seconds);

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NUMBERS_H_
