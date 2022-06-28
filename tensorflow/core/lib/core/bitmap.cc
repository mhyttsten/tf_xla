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
class MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc() {
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

#include "tensorflow/core/lib/core/bitmap.h"

#include <string.h>

namespace tensorflow {
namespace core {

void Bitmap::Reset(size_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/lib/core/bitmap.cc", "Bitmap::Reset");

  const size_t num_words = NumWords(n);
  if (num_words != NumWords(nbits_)) {
    // Reallocate.
    Word* w = new Word[num_words];
    delete[] word_;
    word_ = w;
  }
  memset(word_, 0, sizeof(word_[0]) * num_words);
  nbits_ = n;
}

// Return 1+index of the first set bit in w; return 0 if w == 0.
static size_t FindFirstSet(uint32 w) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/lib/core/bitmap.cc", "FindFirstSet");

  // TODO(jeff,sanjay): If this becomes a performance issue, we could
  // use the __builtin_ffs(w) routine on GCC, or the ffs(w) routine on
  // some other platforms.

  // clang-format off
  static uint8 kLowestBitSet[256] = {
    /*  0*/ 0,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 16*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 32*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 48*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 64*/ 7,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 80*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /* 96*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*112*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*128*/ 8,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*144*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*160*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*176*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*192*/ 7,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*208*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*224*/ 6,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
    /*240*/ 5,  1,  2,  1,  3,  1,  2,  1,  4,  1,  2,  1,  3,  1,  2,  1,
  };
  // clang-format on
  if (w & 0xff) {
    return kLowestBitSet[w & 0xff];
  } else if ((w >> 8) & 0xff) {
    return kLowestBitSet[(w >> 8) & 0xff] + 8;
  } else if ((w >> 16) & 0xff) {
    return kLowestBitSet[(w >> 16) & 0xff] + 16;
  } else if ((w >> 24) & 0xff) {
    return kLowestBitSet[(w >> 24) & 0xff] + 24;
  } else {
    return 0;
  }
}

size_t Bitmap::FirstUnset(size_t start) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/lib/core/bitmap.cc", "Bitmap::FirstUnset");

  if (start >= nbits_) {
    return nbits_;
  }

  // Mask to or-into first word to account for bits to skip in that word.
  size_t mask = (1ull << (start % kBits)) - 1;
  const size_t nwords = NumWords(nbits_);
  for (size_t i = start / kBits; i < nwords; i++) {
    Word word = word_[i] | mask;
    mask = 0;  // Only ignore bits in the first word we process.
    size_t r = FindFirstSet(~word);

    if (r) {
      size_t result = i * kBits + (r - 1);
      if (result > nbits_) result = nbits_;
      return result;
    }
  }

  return nbits_;
}

string Bitmap::ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmapDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/lib/core/bitmap.cc", "Bitmap::ToString");

  string result;
  result.resize(bits());
  for (size_t i = 0; i < nbits_; i++) {
    result[i] = get(i) ? '1' : '0';
  }
  return result;
}

}  // namespace core
}  // namespace tensorflow
