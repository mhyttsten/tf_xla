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
class MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc() {
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

#include "tensorflow/lite/mutable_op_resolver.h"

#include <stddef.h>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

// We need some dummy functions to identify the registrations.
TfLiteStatus DummyInvoke(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "DummyInvoke");

  return kTfLiteOk;
}

TfLiteRegistration* GetDummyRegistration() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "GetDummyRegistration");

  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare = nullptr,
      .invoke = DummyInvoke,
  };
  return &registration;
}

TfLiteStatus Dummy2Invoke(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_2(mht_2_v, 218, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "Dummy2Invoke");

  return kTfLiteOk;
}

TfLiteStatus Dummy2Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_3(mht_3_v, 225, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "Dummy2Prepare");

  return kTfLiteOk;
}

void* Dummy2Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "Dummy2Init");

  return nullptr;
}

void Dummy2free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_5(mht_5_v, 240, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "Dummy2free");
}

TfLiteRegistration* GetDummy2Registration() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSmutable_op_resolver_testDTcc mht_6(mht_6_v, 245, "", "./tensorflow/lite/mutable_op_resolver_test.cc", "GetDummy2Registration");

  static TfLiteRegistration registration = {
      .init = Dummy2Init,
      .free = Dummy2free,
      .prepare = Dummy2Prepare,
      .invoke = Dummy2Invoke,
  };
  return &registration;
}

TEST(MutableOpResolverTest, FinOp) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_ADD);
  EXPECT_EQ(found_registration->version, 1);
}

TEST(MutableOpResolverTest, FindMissingOp) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp(BuiltinOperator_CONV_2D, 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, RegisterOpWithSingleVersion) {
  MutableOpResolver resolver;
  // The kernel supports version 2 only
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  ASSERT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, RegisterOpWithMultipleVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 2);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 2);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 3);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 3);
}

TEST(MutableOpResolverTest, FindOpWithUnsupportedVersions) {
  MutableOpResolver resolver;
  // The kernel supports version 2 and 3
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration(), 2, 3);

  const TfLiteRegistration* found_registration;

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 1);
  EXPECT_EQ(found_registration, nullptr);

  found_registration = resolver.FindOp(BuiltinOperator_ADD, 4);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, FindCustomOp) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("AWESOME", 1);
  ASSERT_NE(found_registration, nullptr);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_TRUE(found_registration->invoke == DummyInvoke);
  EXPECT_EQ(found_registration->version, 1);
}

TEST(MutableOpResolverTest, FindCustomName) {
  MutableOpResolver resolver;
  TfLiteRegistration* reg = GetDummyRegistration();

  reg->custom_name = "UPDATED";
  resolver.AddCustom(reg->custom_name, reg);
  const TfLiteRegistration* found_registration =
      resolver.FindOp(reg->custom_name, 1);

  ASSERT_NE(found_registration, nullptr);
  EXPECT_EQ(found_registration->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_EQ(found_registration->invoke, GetDummyRegistration()->invoke);
  EXPECT_EQ(found_registration->version, 1);
  EXPECT_EQ(found_registration->custom_name, "UPDATED");
}

TEST(MutableOpResolverTest, FindBuiltinName) {
  MutableOpResolver resolver1;
  TfLiteRegistration* reg = GetDummy2Registration();

  reg->custom_name = "UPDATED";
  resolver1.AddBuiltin(BuiltinOperator_ADD, reg);

  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->prepare,
            GetDummy2Registration()->prepare);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->init,
            GetDummy2Registration()->init);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->free,
            GetDummy2Registration()->free);
  // custom_name for builtin ops will be nullptr
  EXPECT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->custom_name, nullptr);
}

TEST(MutableOpResolverTest, FindMissingCustomOp) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration =
      resolver.FindOp("EXCELLENT", 1);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, FindCustomOpWithUnsupportedVersion) {
  MutableOpResolver resolver;
  resolver.AddCustom("AWESOME", GetDummyRegistration());

  const TfLiteRegistration* found_registration = resolver.FindOp("AWESOME", 2);
  EXPECT_EQ(found_registration, nullptr);
}

TEST(MutableOpResolverTest, AddAll) {
  MutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver1.AddBuiltin(BuiltinOperator_MUL, GetDummy2Registration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_SUB, GetDummyRegistration());
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  resolver1.AddAll(resolver2);

  // resolver2's ADD op should replace resolver1's ADD op, while augmenting
  // non-overlapping ops.
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_MUL, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummyRegistration()->invoke);
}

class ChainingMutableOpResolver : public MutableOpResolver {
 public:
  using MutableOpResolver::ChainOpResolver;
};

TEST(MutableOpResolverTest, ChainOpResolver) {
  ChainingMutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver1.AddBuiltin(BuiltinOperator_MUL, GetDummy2Registration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_SUB, GetDummyRegistration());
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  resolver1.ChainOpResolver(&resolver2);

  // resolver2's ADD op should NOT replace resolver1's ADD op;
  // but resolver2's non-overlapping ops should augment resolver1's.
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_MUL, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver1.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, CopyConstructChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3(resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, AssignChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3;
  resolver3 = resolver2;

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, AddAllChainedOpResolver) {
  MutableOpResolver resolver;
  resolver.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());
  resolver.AddBuiltin(BuiltinOperator_SUB, GetDummy2Registration());
  resolver.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver);

  MutableOpResolver resolver3;
  resolver3.AddAll(resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_SUB, 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_MUL, 1), nullptr);
  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
  ASSERT_EQ(resolver3.FindOp("NotMyCustom", 1), nullptr);
}

TEST(MutableOpResolverTest, ChainOpResolverCustomOpPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  MutableOpResolver resolver2;
  resolver2.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver3;
  resolver3.ChainOpResolver(&resolver1);
  resolver3.ChainOpResolver(&resolver2);

  ASSERT_EQ(resolver3.FindOp("MyCustom", 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, ChainOpResolverBuiltinOpPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddBuiltin(BuiltinOperator_ADD, GetDummyRegistration());

  MutableOpResolver resolver2;
  resolver2.AddBuiltin(BuiltinOperator_ADD, GetDummy2Registration());

  ChainingMutableOpResolver resolver3;
  resolver3.ChainOpResolver(&resolver1);
  resolver3.ChainOpResolver(&resolver2);

  ASSERT_EQ(resolver3.FindOp(BuiltinOperator_ADD, 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, ChainOpResolverAddVersusChainPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver1);

  MutableOpResolver resolver3;
  resolver3.AddCustom("MyCustom", GetDummy2Registration());

  ChainingMutableOpResolver resolver4;
  resolver4.ChainOpResolver(&resolver2);
  resolver4.ChainOpResolver(&resolver3);

  ASSERT_EQ(resolver4.FindOp("MyCustom", 1)->invoke,
            GetDummyRegistration()->invoke);
}

TEST(MutableOpResolverTest, AddAllAddVersusChainPrecedence) {
  MutableOpResolver resolver1;
  resolver1.AddCustom("MyCustom", GetDummyRegistration());

  ChainingMutableOpResolver resolver2;
  resolver2.ChainOpResolver(&resolver1);

  MutableOpResolver resolver3;
  resolver3.AddCustom("MyCustom", GetDummy2Registration());

  MutableOpResolver resolver4;
  resolver4.AddAll(resolver2);
  resolver4.AddAll(resolver3);

  ASSERT_EQ(resolver4.FindOp("MyCustom", 1)->invoke,
            GetDummy2Registration()->invoke);
}

}  // namespace
}  // namespace tflite
