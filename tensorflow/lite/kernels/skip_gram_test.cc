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
class MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gram_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gram_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gram_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

static char kSentence[] = "The quick\t brown fox\n jumps over\n the lazy dog!";

class SkipGramOp : public SingleOpModel {
 public:
  SkipGramOp(int ngram_size, int max_skip_size, bool include_all_ngrams) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gram_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/skip_gram_test.cc", "SkipGramOp");

    input_ = AddInput(TensorType_STRING);
    output_ = AddOutput(TensorType_STRING);

    SetBuiltinOp(BuiltinOperator_SKIP_GRAM, BuiltinOptions_SkipGramOptions,
                 CreateSkipGramOptions(builder_, ngram_size, max_skip_size,
                                       include_all_ngrams)
                     .Union());
    BuildInterpreter({{1}});
  }
  void SetInput(const string& content) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSskip_gram_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/kernels/skip_gram_test.cc", "SetInput");

    PopulateStringTensor(input_, {content});
  }

  std::vector<string> GetOutput() {
    std::vector<string> ans;
    TfLiteTensor* tensor = interpreter_->tensor(output_);

    int num = GetStringCount(tensor);
    for (int i = 0; i < num; i++) {
      StringRef strref = GetString(tensor, i);
      ans.push_back(string(strref.str, strref.len));
    }
    return ans;
  }

 private:
  int input_;
  int output_;
};

TEST(SkipGramTest, TestUnigram) {
  SkipGramOp m(1, 0, false);

  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), testing::UnorderedElementsAreArray(
                                 {"The", "quick", "brown", "fox", "jumps",
                                  "over", "the", "lazy", "dog!"}));
}

TEST(SkipGramTest, TestBigram) {
  SkipGramOp m(2, 0, false);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!"}));
}

TEST(SkipGramTest, TestAllBigram) {
  SkipGramOp m(2, 0, true);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {// Unigram
                   "The", "quick", "brown", "fox", "jumps", "over", "the",
                   "lazy", "dog!",
                   //  Bigram
                   "The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!"}));
}

TEST(SkipGramTest, TestAllTrigram) {
  SkipGramOp m(3, 0, true);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {// Unigram
                   "The", "quick", "brown", "fox", "jumps", "over", "the",
                   "lazy", "dog!",
                   // Bigram
                   "The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!",
                   // Trigram
                   "The quick brown", "quick brown fox", "brown fox jumps",
                   "fox jumps over", "jumps over the", "over the lazy",
                   "the lazy dog!"}));
}

TEST(SkipGramTest, TestSkip1Bigram) {
  SkipGramOp m(2, 1, false);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      testing::UnorderedElementsAreArray(
          {"The quick", "The brown", "quick brown", "quick fox", "brown fox",
           "brown jumps", "fox jumps", "fox over", "jumps over", "jumps the",
           "over the", "over lazy", "the lazy", "the dog!", "lazy dog!"}));
}

TEST(SkipGramTest, TestSkip2Bigram) {
  SkipGramOp m(2, 2, false);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick",  "The brown",   "The fox",    "quick brown",
                   "quick fox",  "quick jumps", "brown fox",  "brown jumps",
                   "brown over", "fox jumps",   "fox over",   "fox the",
                   "jumps over", "jumps the",   "jumps lazy", "over the",
                   "over lazy",  "over dog!",   "the lazy",   "the dog!",
                   "lazy dog!"}));
}

TEST(SkipGramTest, TestSkip1Trigram) {
  SkipGramOp m(3, 1, false);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick brown", "The quick fox",    "The brown fox",
                   "The brown jumps", "quick brown fox",  "quick brown jumps",
                   "quick fox jumps", "quick fox over",   "brown fox jumps",
                   "brown fox over",  "brown jumps over", "brown jumps the",
                   "fox jumps over",  "fox jumps the",    "fox over the",
                   "fox over lazy",   "jumps over the",   "jumps over lazy",
                   "jumps the lazy",  "jumps the dog!",   "over the lazy",
                   "over the dog!",   "over lazy dog!",   "the lazy dog!"}));
}

TEST(SkipGramTest, TestSkip2Trigram) {
  SkipGramOp m(3, 2, false);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick brown",  "The quick fox",     "The quick jumps",
                   "The brown fox",    "The brown jumps",   "The brown over",
                   "The fox jumps",    "The fox over",      "The fox the",
                   "quick brown fox",  "quick brown jumps", "quick brown over",
                   "quick fox jumps",  "quick fox over",    "quick fox the",
                   "quick jumps over", "quick jumps the",   "quick jumps lazy",
                   "brown fox jumps",  "brown fox over",    "brown fox the",
                   "brown jumps over", "brown jumps the",   "brown jumps lazy",
                   "brown over the",   "brown over lazy",   "brown over dog!",
                   "fox jumps over",   "fox jumps the",     "fox jumps lazy",
                   "fox over the",     "fox over lazy",     "fox over dog!",
                   "fox the lazy",     "fox the dog!",      "jumps over the",
                   "jumps over lazy",  "jumps over dog!",   "jumps the lazy",
                   "jumps the dog!",   "jumps lazy dog!",   "over the lazy",
                   "over the dog!",    "over lazy dog!",    "the lazy dog!"}));
}

TEST(SkipGramTest, TestAllSkip2Trigram) {
  SkipGramOp m(3, 2, true);
  m.SetInput(kSentence);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(
      m.GetOutput(),
      testing::UnorderedElementsAreArray(
          {// Unigram
           "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
           "dog!",
           // Bigram
           "The quick", "The brown", "The fox", "quick brown", "quick fox",
           "quick jumps", "brown fox", "brown jumps", "brown over", "fox jumps",
           "fox over", "fox the", "jumps over", "jumps the", "jumps lazy",
           "over the", "over lazy", "over dog!", "the lazy", "the dog!",
           "lazy dog!",
           // Trigram
           "The quick brown", "The quick fox", "The quick jumps",
           "The brown fox", "The brown jumps", "The brown over",
           "The fox jumps", "The fox over", "The fox the", "quick brown fox",
           "quick brown jumps", "quick brown over", "quick fox jumps",
           "quick fox over", "quick fox the", "quick jumps over",
           "quick jumps the", "quick jumps lazy", "brown fox jumps",
           "brown fox over", "brown fox the", "brown jumps over",
           "brown jumps the", "brown jumps lazy", "brown over the",
           "brown over lazy", "brown over dog!", "fox jumps over",
           "fox jumps the", "fox jumps lazy", "fox over the", "fox over lazy",
           "fox over dog!", "fox the lazy", "fox the dog!", "jumps over the",
           "jumps over lazy", "jumps over dog!", "jumps the lazy",
           "jumps the dog!", "jumps lazy dog!", "over the lazy",
           "over the dog!", "over lazy dog!", "the lazy dog!"}));
}

TEST(SkipGramTest, TestSingleWord) {
  SkipGramOp m(1, 1, false);
  m.SetInput("Hi");
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre("Hi"));
}

TEST(SkipGramTest, TestWordsLessThanGram) {
  SkipGramOp m(3, 1, false);
  m.SetInput("Hi hi");
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), std::vector<string>());
}

TEST(SkipGramTest, TestEmptyInput) {
  SkipGramOp m(1, 1, false);
  m.SetInput("");
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre());
}

TEST(SkipGramTest, TestWhitespaceInput) {
  SkipGramOp m(1, 1, false);
  m.SetInput("    ");
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre());
}

TEST(SkipGramTest, TestInputWithExtraSpace) {
  SkipGramOp m(1, 1, false);
  m.SetInput("   Hello   world    !  ");
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre("Hello", "world", "!"));
}

}  // namespace
}  // namespace tflite
