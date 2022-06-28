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
class MHTracer_DTPStensorflowPScorePSlibPScorePSbitmap_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmap_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSbitmap_testDTcc() {
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

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

// Return next size to test after n.
size_t NextSize(size_t n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmap_testDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/lib/core/bitmap_test.cc", "NextSize");
 return n + ((n < 75) ? 1 : 25); }

static void MakeRandomBitmap(random::SimplePhilox* rnd, Bitmap* bitmap) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitmap_testDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/lib/core/bitmap_test.cc", "MakeRandomBitmap");

  size_t n = rnd->Uniform(200);
  bitmap->Reset(n);
  for (size_t i = 0; i < n; i++) {
    if (rnd->OneIn(2)) bitmap->set(i);
  }
}

TEST(BitmapTest, Basic) {
  for (size_t n = 0; n < 200; n = NextSize(n)) {
    Bitmap bits(n);
    for (size_t i = 0; i < n; i++) {
      EXPECT_FALSE(bits.get(i)) << n << " " << i << " " << bits.ToString();
      bits.set(i);
      EXPECT_TRUE(bits.get(i)) << n << " " << i << " " << bits.ToString();
      bits.clear(i);
      EXPECT_FALSE(bits.get(i)) << n << " " << i << " " << bits.ToString();
    }
  }
}

TEST(BitmapTest, ToString) {
  Bitmap bits(10);
  bits.set(1);
  bits.set(3);
  EXPECT_EQ(bits.ToString(), "0101000000");
}

TEST(BitmapTest, FirstUnset) {
  for (size_t n = 0; n < 200; n = NextSize(n)) {
    for (size_t p = 0; p <= 100; p++) {
      for (size_t q = 0; q <= 100; q++) {
        // Generate a bitmap of length n with long runs of ones.
        Bitmap bitmap(n);
        // Set first p bits to 1.
        int one_count = 0;
        size_t i = 0;
        while (i < p && i < n) {
          one_count++;
          bitmap.set(i);
          i++;
        }
        // Fill rest with a pattern of 0 followed by q 1s.
        while (i < n) {
          i++;
          for (size_t j = 0; j < q && i < n; j++, i++) {
            one_count++;
            bitmap.set(i);
          }
        }

        // Now use FirstUnset to iterate over unset bits and verify
        // that all encountered bits are clear.
        int seen = 0;
        size_t pos = 0;
        while (true) {
          pos = bitmap.FirstUnset(pos);
          if (pos == n) break;
          ASSERT_FALSE(bitmap.get(pos)) << pos << " " << bitmap.ToString();
          seen++;
          pos++;
        }
        EXPECT_EQ(seen, n - one_count) << " " << bitmap.ToString();
      }
    }
  }
}

TEST(BitmapTest, FirstUnsetRandom) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int iter = 0; iter < 10000; iter++) {
    Bitmap bitmap;
    MakeRandomBitmap(&rnd, &bitmap);

    // Count number of unset bits in bitmap.
    size_t zero_bits = 0;
    for (size_t i = 0; i < bitmap.bits(); i++) {
      if (!bitmap.get(i)) zero_bits++;
    }

    // Now use FirstUnset to iterate over unset bits and verify
    // that all encountered bits are clear.
    int seen = 0;
    size_t pos = 0;
    while (true) {
      pos = bitmap.FirstUnset(pos);
      if (pos == bitmap.bits()) break;
      ASSERT_FALSE(bitmap.get(pos)) << pos << " " << bitmap.ToString();
      seen++;
      pos++;
    }

    EXPECT_EQ(seen, zero_bits) << " " << bitmap.ToString();
  }
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
