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
#ifndef TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh() {
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


#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Enumeration for unicode encodings.  Used by ops such as
// tf.strings.unicode_encode and tf.strings.unicode_decode.
enum class UnicodeEncoding { UTF8, UTF16BE, UTF32BE };

// Enumeration for character units.  Used by string such as
// tf.strings.length and tf.substr.
// TODO(edloper): Add support for: UTF32_CHAR, etc.
enum class CharUnit { BYTE, UTF8_CHAR };

// Whether or not the given byte is the trailing byte of a UTF-8/16/32 char.
inline bool IsTrailByte(char x) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("x: '" + std::string(1, x) + "'");
   MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/string_util.h", "IsTrailByte");
 return static_cast<signed char>(x) < -0x40; }

// Sets `encoding` based on `str`.
Status ParseUnicodeEncoding(const string& str, UnicodeEncoding* encoding);

// Sets `unit` value based on `str`.
Status ParseCharUnit(const string& str, CharUnit* unit);

// Returns the number of Unicode characters in a UTF-8 string.
// Result may be incorrect if the input string is not valid UTF-8.
int32 UTF8StrLen(const string& str);

// Get the next UTF8 character position starting at the given position and
// skipping the given number of characters. Position is a byte offset, and
// should never be `null`. The function return true if successful. However, if
// the end of the string is reached before the requested characters, then the
// position will point to the end of string and this function will return false.
template <typename T>
bool ForwardNUTF8CharPositions(const StringPiece in,
                               const T num_utf8_chars_to_shift, T* pos) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/string_util.h", "ForwardNUTF8CharPositions");

  const size_t size = in.size();
  T utf8_chars_counted = 0;
  while (utf8_chars_counted < num_utf8_chars_to_shift && *pos < size) {
    // move forward one utf-8 character
    do {
      ++*pos;
    } while (IsTrailByte(in[*pos]) && *pos < size);
    ++utf8_chars_counted;
  }
  return utf8_chars_counted == num_utf8_chars_to_shift;
}

// Get the previous UTF8 character position starting at the given position and
// skipping the given number of characters. Position is a byte offset with a
// positive value, relative to the beginning of the string, and should never be
// `null`. The function return true if successful. However, if the beginning of
// the string is reached before the requested character, then the position will
// point to the beginning of the string and this function will return false.
template <typename T>
bool BackNUTF8CharPositions(const StringPiece in,
                            const T num_utf8_chars_to_shift, T* pos) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstring_utilDTh mht_2(mht_2_v, 248, "", "./tensorflow/core/kernels/string_util.h", "BackNUTF8CharPositions");

  const size_t start = 0;
  T utf8_chars_counted = 0;
  while (utf8_chars_counted < num_utf8_chars_to_shift && (*pos > start)) {
    // move back one utf-8 character
    do {
      --*pos;
    } while (IsTrailByte(in[*pos]) && *pos > start);
    ++utf8_chars_counted;
  }
  return utf8_chars_counted == num_utf8_chars_to_shift;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
