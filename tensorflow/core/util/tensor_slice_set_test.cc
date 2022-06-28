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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_set_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_set_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_set_testDTcc() {
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

#include "tensorflow/core/util/tensor_slice_set.h"

#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace checkpoint {

namespace {

// A simple test: we have a 2-d tensor of shape 4 X 5 that looks like this:
//
//   0   1   2   3   4
//   5   6   7   8   9
//  10  11  12  13  14
//  15  16  17  18  19
//
// We assume this is a row-major matrix.
//
// Testing the meta version of the tensor slice set.
TEST(TensorSliceSetTest, QueryMetaTwoD) {
  TensorShape shape({4, 5});

  TensorSliceSet tss(shape, DT_INT32);
  // We store a few slices.

  // Slice #1 is the top two rows:
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  TensorSlice slice_1 = TensorSlice::ParseOrDie("0,2:-");
  TF_CHECK_OK(tss.Register(slice_1, "slice_1"));

  // Slice #2 is the bottom left corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //  10  11  12   .   .
  //  15  16  17   .   .
  TensorSlice slice_2 = TensorSlice::ParseOrDie("2,2:0,3");
  TF_CHECK_OK(tss.Register(slice_2, "slice_2"));

  // Slice #3 is the bottom right corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .  18  19
  TensorSlice slice_3 = TensorSlice::ParseOrDie("3,1:3,2");
  TF_CHECK_OK(tss.Register(slice_3, "slice_3"));

  // Notice that we leave a hole in the tensor
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   . (13) (14)
  //   .   .   .   .   .

  // Now we query some of the slices

  // Slice #1 is an exact match
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  // We just need slice_1 for this
  {
    TensorSlice s = TensorSlice::ParseOrDie("0,2:-");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(1, results.size());
    EXPECT_EQ("0,2:-", results[0].first.DebugString());
    EXPECT_EQ("slice_1", results[0].second);
  }

  // Slice #2 is a subset match
  //   .   .   .   .   .
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  // We just need slice_1 for this
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,1:-");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(1, results.size());
    EXPECT_EQ("0,2:-", results[0].first.DebugString());
    EXPECT_EQ("slice_1", results[0].second);
  }

  // Slice #3 is a more complicated match: it needs the combination of a couple
  // of slices
  //   .   .   .   .   .
  //   5   6   7   .   .
  //  10  11  12   .   .
  //   .   .   .   .   .
  // We need both slice_1 and slice_2 for this.
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:0,3");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(2, results.size());
    // Allow results to be returned in either order
    if (results[0].second == "slice_2") {
      EXPECT_EQ("2,2:0,3", results[0].first.DebugString());
      EXPECT_EQ("slice_2", results[0].second);
      EXPECT_EQ("0,2:-", results[1].first.DebugString());
      EXPECT_EQ("slice_1", results[1].second);
    } else {
      EXPECT_EQ("0,2:-", results[0].first.DebugString());
      EXPECT_EQ("slice_1", results[0].second);
      EXPECT_EQ("2,2:0,3", results[1].first.DebugString());
      EXPECT_EQ("slice_2", results[1].second);
    }
  }

  // Slice #4 includes the hole and so there is no match
  //   .   .   .   .   .
  //   .   .   7   8   9
  //   .   .  12  13  14
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:2,3");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_FALSE(tss.QueryMeta(s, &results));
    EXPECT_EQ(0, results.size());
  }
}

static void BM_RegisterOneByOne(::testing::benchmark::State& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_set_testDTcc mht_0(mht_0_v, 316, "", "./tensorflow/core/util/tensor_slice_set_test.cc", "BM_RegisterOneByOne");

  TensorShape shape({static_cast<int>(state.max_iterations), 41});
  TensorSliceSet slice_set(shape, DT_INT32);
  int i = 0;
  for (auto s : state) {
    TensorSlice part({{i, 1}, {0, -1}});
    TF_CHECK_OK(slice_set.Register(part, part.DebugString()));
    ++i;
  }
}

BENCHMARK(BM_RegisterOneByOne);

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
