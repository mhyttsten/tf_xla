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
class MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc() {
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

#include "tensorflow/lite/core/api/op_resolver.h"

#include <cstring>

#include <gtest/gtest.h>
#include "tensorflow/lite/schema/schema_conversion_utils.h"

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "MockInit");

  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "MockFree");

  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "MockPrepare");

  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_3(mht_3_v, 217, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "MockInvoke");

  return kTfLiteOk;
}

class MockOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_4(mht_4_v, 227, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "FindOp");

    if (op == BuiltinOperator_CONV_2D) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_5(mht_5_v, 240, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "FindOp");

    if (strcmp(op, "mock_custom") == 0) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
};

class MockErrorReporter : public ErrorReporter {
 public:
  MockErrorReporter() : buffer_size_(0) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_6(mht_6_v, 256, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "MockErrorReporter");
}
  int Report(const char* format, va_list args) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_7(mht_7_v, 261, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "Report");

    buffer_size_ = vsnprintf(buffer_, kBufferSize, format, args);
    return buffer_size_;
  }
  char* GetBuffer() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_8(mht_8_v, 268, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "GetBuffer");
 return buffer_; }
  int GetBufferSize() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePScorePSapiPSop_resolver_testDTcc mht_9(mht_9_v, 272, "", "./tensorflow/lite/core/api/op_resolver_test.cc", "GetBufferSize");
 return buffer_size_; }

 private:
  static constexpr int kBufferSize = 256;
  char buffer_[kBufferSize];
  int buffer_size_;
};

}  // namespace

TEST(OpResolver, TestResolver) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;

  const TfLiteRegistration* registration =
      resolver->FindOp(BuiltinOperator_CONV_2D, 0);
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp(BuiltinOperator_CAST, 0);
  EXPECT_EQ(nullptr, registration);

  registration = resolver->FindOp("mock_custom", 0);
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));

  registration = resolver->FindOp("nonexistent_custom", 0);
  EXPECT_EQ(nullptr, registration);
}

TEST(OpResolver, TestGetRegistrationFromOpCodeConv) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CONV_2D, nullptr, 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteOk, GetRegistrationFromOpCode(conv_code, *resolver, reporter,
                                                 &registration));
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeCast) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CAST, nullptr, 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteError, GetRegistrationFromOpCode(conv_code, *resolver,
                                                    reporter, &registration));
  EXPECT_EQ(nullptr, registration);
  EXPECT_NE(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeCustom) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset = CreateOperatorCodeDirect(
      builder, BuiltinOperator_CUSTOM, "mock_custom", 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteOk, GetRegistrationFromOpCode(conv_code, *resolver, reporter,
                                                 &registration));
  EXPECT_NE(nullptr, registration);
  EXPECT_EQ(nullptr, registration->init(nullptr, nullptr, 0));
  EXPECT_EQ(kTfLiteOk, registration->prepare(nullptr, nullptr));
  EXPECT_EQ(kTfLiteOk, registration->invoke(nullptr, nullptr));
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

TEST(OpResolver, TestGetRegistrationFromOpCodeNonexistentCustom) {
  MockOpResolver mock_resolver;
  OpResolver* resolver = &mock_resolver;
  MockErrorReporter mock_reporter;
  ErrorReporter* reporter = &mock_reporter;

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> conv_offset = CreateOperatorCodeDirect(
      builder, BuiltinOperator_CUSTOM, "nonexistent_custom", 0);
  builder.Finish(conv_offset);
  void* conv_pointer = builder.GetBufferPointer();
  const OperatorCode* conv_code =
      flatbuffers::GetRoot<OperatorCode>(conv_pointer);
  const TfLiteRegistration* registration = nullptr;
  EXPECT_EQ(kTfLiteError, GetRegistrationFromOpCode(conv_code, *resolver,
                                                    reporter, &registration));
  EXPECT_EQ(nullptr, registration);
  // There is no error, since unresolved custom ops are checked while preparing
  // nodes.
  EXPECT_EQ(0, mock_reporter.GetBufferSize());
}

}  // namespace tflite
