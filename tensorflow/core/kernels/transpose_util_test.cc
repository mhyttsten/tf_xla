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
class MHTracer_DTPStensorflowPScorePSkernelsPStranspose_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStranspose_util_testDTcc() {
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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TransposeUtilTest : public ::testing::Test {
 protected:
  void TestDimensionReduction(const TensorShape& shape,
                              const gtl::ArraySlice<int32> perm,
                              const gtl::ArraySlice<int32> expected_perm,
                              const gtl::ArraySlice<int64_t> expected_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_util_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/kernels/transpose_util_test.cc", "TestDimensionReduction");

    internal::TransposePermsVec new_perm;
    internal::TransposeDimsVec new_dims;
    internal::ReduceTransposeDimensions(shape, perm, &new_perm, &new_dims);

    gtl::ArraySlice<int32> computed_perm(new_perm);
    gtl::ArraySlice<int64_t> computed_dims(new_dims);
    EXPECT_EQ(computed_perm, expected_perm);
    EXPECT_EQ(computed_dims, expected_dims);
  }
};

TEST_F(TransposeUtilTest, NormalDimensionReduction) {
  TestDimensionReduction({2, 3, 4}, {0, 2, 1}, {0, 2, 1}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4}, {1, 0, 2}, {1, 0, 2}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4}, {2, 1, 0}, {2, 1, 0}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4, 5}, {0, 2, 3, 1}, {0, 2, 1}, {2, 3, 20});

  TestDimensionReduction({2, 3, 4, 5}, {0, 3, 1, 2}, {0, 2, 1}, {2, 12, 5});

  TestDimensionReduction({2, 3, 4, 5}, {3, 1, 2, 0}, {2, 1, 0}, {2, 12, 5});

  TestDimensionReduction({2, 3, 4, 5}, {2, 3, 1, 0}, {2, 1, 0}, {2, 3, 20});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 2, 3, 4, 1}, {0, 2, 1},
                         {2, 3, 120});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 4, 1, 2, 3}, {0, 2, 1},
                         {2, 60, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {4, 1, 2, 3, 0}, {2, 1, 0},
                         {2, 60, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {3, 4, 1, 2, 0}, {2, 1, 0},
                         {2, 12, 30});

  TestDimensionReduction({2, 3}, {1, 0}, {1, 0}, {2, 3});

  TestDimensionReduction({2, 3, 4}, {2, 0, 1}, {1, 0}, {6, 4});

  TestDimensionReduction({2, 3, 4}, {1, 2, 0}, {1, 0}, {2, 12});

  TestDimensionReduction({2, 3, 4, 5}, {2, 3, 0, 1}, {1, 0}, {6, 20});

  TestDimensionReduction({2, 3, 4, 5}, {1, 2, 3, 0}, {1, 0}, {2, 60});

  TestDimensionReduction({2, 3, 4, 5, 6}, {2, 3, 4, 0, 1}, {1, 0}, {6, 120});

  TestDimensionReduction({2, 3, 4, 5, 6}, {4, 0, 1, 2, 3}, {1, 0}, {120, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, {0}, {720});

  TestDimensionReduction({2, 3, 4, 5}, {0, 1, 2, 3}, {0}, {120});

  TestDimensionReduction({2, 3, 4}, {0, 1, 2}, {0}, {24});

  TestDimensionReduction({2, 3}, {0, 1}, {0}, {6});
}

TEST_F(TransposeUtilTest, LargeDimensionReduction) {
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 2, 3, 4, 5, 6, 7, 8, 9, 1}, {0, 2, 1},
                         {2, 3, 12096000});
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 1, 2, 3, 4, 5, 6, 7, 9, 8}, {0, 2, 1},
                         {362880, 10, 20});
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0}, {72576000});
}

TEST_F(TransposeUtilTest, NonSingletonDimensionAlignment) {
  // Non-singleton dims 0, 2
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {1, 0, 2}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {0, 2, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {2, 0, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {2, 1, 0}));

  // Non-singleton dims 0, 2, 4
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {0, 2, 4, 1, 3}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {0, 2, 1, 4, 3}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {1, 3, 0, 2, 4}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {3, 0, 1, 2, 4}));
  EXPECT_FALSE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {3, 2, 0, 1, 4}));

  // Non-singleton dims 2, 4, 5
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {3, 2, 1, 4, 0, 5}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {3, 1, 0, 2, 4, 5}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {2, 4, 5, 0, 3, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                     {0, 1, 5, 2, 4, 3}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                     {0, 1, 2, 5, 4, 3}));
}

}  // namespace tensorflow
