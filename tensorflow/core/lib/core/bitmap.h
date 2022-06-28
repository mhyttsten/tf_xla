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

#ifndef TENSORFLOW_CORE_LIB_CORE_BITMAP_H_
#define TENSORFLOW_CORE_LIB_CORE_BITMAP_H_
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
class MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh() {
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


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace core {

class Bitmap {
 public:
  // Create a bitmap that holds 0 bits.
  Bitmap();

  // Create a bitmap that holds n bits, all initially zero.
  explicit Bitmap(size_t n);

  ~Bitmap();

  Bitmap(const Bitmap&) = delete;
  Bitmap& operator=(const Bitmap&) = delete;

  // Return the number of bits that the bitmap can hold.
  size_t bits() const;

  // Replace contents of *this with a bitmap of n bits, all set to zero.
  void Reset(size_t n);

  // Return the contents of the ith bit.
  // REQUIRES: i < bits()
  bool get(size_t i) const;

  // Set the contents of the ith bit to true.
  // REQUIRES: i < bits()
  void set(size_t i);

  // Set the contents of the ith bit to false.
  // REQUIRES: i < bits()
  void clear(size_t i);

  // Return the smallest i such that i >= start and !get(i).
  // Returns bits() if no such i exists.
  size_t FirstUnset(size_t start) const;

  // Returns the bitmap as an ascii string of '0' and '1' characters, bits()
  // characters in length.
  string ToString() const;

 private:
  typedef uint32 Word;
  static constexpr size_t kBits = 32;

  // Return the number of words needed to store n bits.
  static size_t NumWords(size_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_0(mht_0_v, 238, "", "./tensorflow/core/lib/core/bitmap.h", "NumWords");
 return (n + kBits - 1) / kBits; }

  // Return the mask to use for the ith bit in a word.
  static Word Mask(size_t i) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_1(mht_1_v, 244, "", "./tensorflow/core/lib/core/bitmap.h", "Mask");
 return 1ull << i; }

  size_t nbits_;  // Length of bitmap in bits.
  Word* word_;
};

// Implementation details follow.  Clients should ignore.

inline Bitmap::Bitmap() : nbits_(0), word_(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::Bitmap");
}

inline Bitmap::Bitmap(size_t n) : Bitmap() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_3(mht_3_v, 260, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::Bitmap");
 Reset(n); }

inline Bitmap::~Bitmap() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_4(mht_4_v, 265, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::~Bitmap");
 delete[] word_; }

inline size_t Bitmap::bits() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_5(mht_5_v, 270, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::bits");
 return nbits_; }

inline bool Bitmap::get(size_t i) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_6(mht_6_v, 275, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::get");

  DCHECK_LT(i, nbits_);
  return word_[i / kBits] & Mask(i % kBits);
}

inline void Bitmap::set(size_t i) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_7(mht_7_v, 283, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::set");

  DCHECK_LT(i, nbits_);
  word_[i / kBits] |= Mask(i % kBits);
}

inline void Bitmap::clear(size_t i) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTh mht_8(mht_8_v, 291, "", "./tensorflow/core/lib/core/bitmap.h", "Bitmap::clear");

  DCHECK_LT(i, nbits_);
  word_[i / kBits] &= ~Mask(i % kBits);
}

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_BITMAP_H_
