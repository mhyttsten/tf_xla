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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScase_format_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScase_format_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScase_format_testDTcc() {
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
#include "tensorflow/c/experimental/ops/gen/common/case_format.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace generator {

namespace {

// For each test case, we manually construct the 4 variations in string case and
// test all 16 conversions: from and to each of the 4 string case variations.
struct Variations {
  string lower_camel;
  string lower_snake;
  string upper_camel;
  string upper_snake;
};

void TestSingleVariation(const string &str, Variations expected,
                         char delimiter = '_') {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScase_format_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/c/experimental/ops/gen/common/case_format_test.cc", "TestSingleVariation");

  EXPECT_EQ(expected.lower_camel, toLowerCamel(str, delimiter));
  EXPECT_EQ(expected.lower_snake, toLowerSnake(str, delimiter));
  EXPECT_EQ(expected.upper_camel, toUpperCamel(str, delimiter));
  EXPECT_EQ(expected.upper_snake, toUpperSnake(str, delimiter));
}

void TestAllVariations(Variations variations, char delimiter = '_') {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScase_format_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/c/experimental/ops/gen/common/case_format_test.cc", "TestAllVariations");

  TestSingleVariation(variations.lower_camel, variations, delimiter);
  TestSingleVariation(variations.lower_snake, variations, delimiter);
  TestSingleVariation(variations.upper_camel, variations, delimiter);
  TestSingleVariation(variations.upper_snake, variations, delimiter);
}

TEST(CppOpGenCaseFormat, test_single_word) {
  TestAllVariations(Variations{
      "three",
      "three",
      "Three",
      "THREE",
  });
}

TEST(CppOpGenCaseFormat, test_complex_string) {
  TestAllVariations(Variations{
      "threeNTest33Words",
      "three_n_test33_words",
      "ThreeNTest33Words",
      "THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_hyphen_delimiter) {
  TestAllVariations(
      Variations{
          "threeNTest33Words",
          "three-n-test33-words",
          "ThreeNTest33Words",
          "THREE-N-TEST33-WORDS",
      },
      '-');
}

TEST(CppOpGenCaseFormat, test_trailing_underscore) {
  TestAllVariations(Variations{
      "threeNTest33Words_",
      "three_n_test33_words_",
      "ThreeNTest33Words_",
      "THREE_N_TEST33_WORDS_",
  });
}

TEST(CppOpGenCaseFormat, test_double_trailing_underscores) {
  TestAllVariations(Variations{
      "xxY__",
      "xx_y__",
      "XxY__",
      "XX_Y__",
  });
}

TEST(CppOpGenCaseFormat, test_leading_underscore) {
  TestAllVariations(Variations{
      "_threeNTest33Words",
      "_three_n_test33_words",
      "_ThreeNTest33Words",
      "_THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_double_leading_underscores) {
  TestAllVariations(Variations{
      "__threeNTest33Words",
      "__three_n_test33_words",
      "__ThreeNTest33Words",
      "__THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_leading_and_trailing_underscores) {
  TestAllVariations(Variations{
      "__threeNTest33Words____",
      "__three_n_test33_words____",
      "__ThreeNTest33Words____",
      "__THREE_N_TEST33_WORDS____",
  });
}

}  // namespace

}  // namespace generator
}  // namespace tensorflow
