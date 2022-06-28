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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class AccuInlineRewrite : public InlineRewrite {
 public:
  explicit AccuInlineRewrite(std::vector<std::string>* blocks)
      : blocks_(blocks) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor_test.cc", "AccuInlineRewrite");
}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor_test.cc", "Rewrite");

    blocks_->push_back(std::string(input.data(), input.size()));
    output->append("r:");
    output->append(input.data(), input.size());
    return RewriteStatus::SUCCESS;
  }

  std::vector<std::string>* blocks_;
};

std::vector<std::string> ParseInlines(const std::string& text) {
  std::vector<std::string> blocks;
  TextPreprocessor preprocessor('$', false);
  AccuInlineRewrite rewrite(&blocks);
  preprocessor.AddRewrite(&rewrite);
  std::string discard;
  preprocessor.Rewrite(text, &discard).IgnoreError();
  return blocks;
}

TEST(Preprocessor, CornerCases) {
  EXPECT_THAT(ParseInlines(""), testing::ElementsAre());
  EXPECT_THAT(ParseInlines("text text"), testing::ElementsAre());
  EXPECT_THAT(ParseInlines("$$"), testing::ElementsAre(""));
}

TEST(Preprocessor, One) {
  EXPECT_THAT(ParseInlines("$text$"), testing::ElementsAre("text"));
  EXPECT_THAT(ParseInlines(" $text$ "), testing::ElementsAre("text"));
}

TEST(Preprocessor, More) {
  EXPECT_THAT(ParseInlines("Test $inline1$\n$inline2$ test $inline3$ "),
              testing::ElementsAre("inline1", "inline2", "inline3"));
}

std::string RewriteInlines(const std::string& text) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor_test.cc", "RewriteInlines");

  std::vector<std::string> blocks;
  TextPreprocessor preprocessor('$', false);
  AccuInlineRewrite rewrite(&blocks);
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  preprocessor.Rewrite(text, &out).IgnoreError();
  return out;
}

TEST(Preprocessor, RewriteCornerCases) {
  EXPECT_EQ(RewriteInlines(""), "");
  EXPECT_EQ(RewriteInlines("text text"), "text text");
  EXPECT_EQ(RewriteInlines("$$"), "r:");
}

TEST(Preprocessor, RewriteOne) {
  EXPECT_EQ(RewriteInlines("$text$"), "r:text");
  EXPECT_EQ(RewriteInlines(" $text$ "), " r:text ");
}

TEST(Preprocessor, RewriteMore) {
  EXPECT_EQ(RewriteInlines("Test $inline1$\n$inline2$ test $inline3$ "),
            "Test r:inline1\nr:inline2 test r:inline3 ");
}

class SingleRewrite : public InlineRewrite {
 public:
  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("input: \"" + std::string(input.data(), input.size()) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSpreprocessor_testDTcc mht_3(mht_3_v, 279, "", "./tensorflow/lite/delegates/gpu/gl/compiler/preprocessor_test.cc", "Rewrite");

    if (input == "foo") {
      output->append("bla");
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

  std::vector<std::string>* blocks_;
};

TEST(Preprocessor, KeepUnknownRewrites) {
  TextPreprocessor preprocessor('$', true);
  SingleRewrite rewrite;
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  ASSERT_TRUE(preprocessor.Rewrite("Good morning, $name$! $foo$", &out).ok());
  EXPECT_EQ("Good morning, $name$! bla", out);
}

TEST(Preprocessor, KeepUnknownRewrites_Fail) {
  TextPreprocessor preprocessor('$', false);
  SingleRewrite rewrite;
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  EXPECT_FALSE(preprocessor.Rewrite("Good morning, $name$! $foo$", &out).ok());
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
