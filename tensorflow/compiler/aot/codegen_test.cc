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
class MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc() {
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

#include "tensorflow/compiler/aot/codegen.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/TargetSelect.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace {

using ::xla::cpu_function_runtime::BufferInfo;

void ExpectErrorContains(const Status& status, absl::string_view str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/aot/codegen_test.cc", "ExpectErrorContains");

  EXPECT_NE(Status::OK(), status);
  EXPECT_TRUE(absl::StrContains(status.error_message(), str))
      << "expected error: " << status.error_message() << " to contain: " << str;
}

TEST(ValidateCppIdent, Simple) {
  TF_EXPECT_OK(ValidateCppIdent("a", ""));
  TF_EXPECT_OK(ValidateCppIdent("abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc123", ""));
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  TF_EXPECT_OK(ValidateCppIdent(ident, ""));

  ExpectErrorContains(ValidateCppIdent("", ""), "empty identifier");
  ExpectErrorContains(ValidateCppIdent(" ", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("0", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(".", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(":", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("a.", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
}

class ParseCppClassTest : public ::testing::Test {
 protected:
  void ExpectOK(const string& cpp_class, const string& want_class_name,
                const std::vector<string>& want_namespaces) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("cpp_class: \"" + cpp_class + "\"");
   mht_1_v.push_back("want_class_name: \"" + want_class_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc mht_1(mht_1_v, 255, "", "./tensorflow/compiler/aot/codegen_test.cc", "ExpectOK");

    string class_name;
    std::vector<string> namespaces;
    TF_EXPECT_OK(ParseCppClass(cpp_class, &class_name, &namespaces));
    EXPECT_EQ(class_name, want_class_name);
    EXPECT_EQ(namespaces, want_namespaces);
  }

  void ExpectFail(const string& cpp_class) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("cpp_class: \"" + cpp_class + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc mht_2(mht_2_v, 267, "", "./tensorflow/compiler/aot/codegen_test.cc", "ExpectFail");

    string class_name;
    std::vector<string> namespaces;
    EXPECT_NE(ParseCppClass(cpp_class, &class_name, &namespaces), Status::OK())
        << cpp_class;
  }
};

TEST_F(ParseCppClassTest, ParseOK) {
  ExpectOK("MyClass", "MyClass", {});
  ExpectOK("_MyClass", "_MyClass", {});
  ExpectOK("a::MyClass", "MyClass", {"a"});
  ExpectOK("a::foo::MyClass", "MyClass", {"a", "foo"});
  ExpectOK("a::foo::b::MyClass", "MyClass", {"a", "foo", "b"});
  ExpectOK("a::foo::b::bar::MyClass", "MyClass", {"a", "foo", "b", "bar"});
  ExpectOK("foo::MyClass", "MyClass", {"foo"});
  ExpectOK("_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("_foo::_MyClass", "_MyClass", {"_foo"});
  ExpectOK("::foo::bar::MyClass", "MyClass", {"foo", "bar"});
  ExpectOK("::_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("::_foo::_MyClass", "_MyClass", {"_foo"});
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  ExpectOK(ident, ident, {});
  ExpectOK(ident + "::" + ident, ident, {ident});
  ExpectOK(ident + "::" + ident + "::" + ident, ident, {ident, ident});
}

TEST_F(ParseCppClassTest, ParseFail) {
  ExpectFail("");
  ExpectFail("::");
  ExpectFail("0");
  ExpectFail("a.b");
  ExpectFail("a:b");
  ExpectFail(":foo::bar");
  ExpectFail("good::.bad");
  ExpectFail("good:::bad");
  ExpectFail("good::bad::");
  ExpectFail("good::::bad");
  ExpectFail("::::bad");
  ExpectFail("good:: bad");
  ExpectFail("good::0bad");
}

static void CompareWithGoldenFile(
    const string& tensorflow_relative_golden_file_name,
    const string& expected_contents, bool ignore_cr) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tensorflow_relative_golden_file_name: \"" + tensorflow_relative_golden_file_name + "\"");
   mht_3_v.push_back("expected_contents: \"" + expected_contents + "\"");
   MHTracer_DTPStensorflowPScompilerPSaotPScodegen_testDTcc mht_3(mht_3_v, 328, "", "./tensorflow/compiler/aot/codegen_test.cc", "CompareWithGoldenFile");

  // Get rid of all CR characters, we may be running under windows.
  string sanitized_expected_contents(expected_contents);
  if (ignore_cr) {
    sanitized_expected_contents.erase(
        std::remove(sanitized_expected_contents.begin(),
                    sanitized_expected_contents.end(), '\r'),
        sanitized_expected_contents.end());
  }

  // To update the golden file, flip update_golden to true and run the
  // following:
  // bazel test --test_strategy=local \
  //   "third_party/tensorflow/compiler/aot:codegen_test"
  const bool update_golden = false;
  string golden_file_name =
      GetDataDependencyFilepath(tensorflow_relative_golden_file_name);

  if (update_golden) {
    TF_EXPECT_OK(
        WriteStringToFile(Env::Default(), golden_file_name, expected_contents));
  }

  string golden_file_contents;
  TF_ASSERT_OK(ReadFileToString(Env::Default(), golden_file_name,
                                &golden_file_contents));
  if (ignore_cr) {
    golden_file_contents.erase(std::remove(golden_file_contents.begin(),
                                           golden_file_contents.end(), '\r'),
                               golden_file_contents.end());
  }
  EXPECT_EQ(golden_file_contents, expected_contents);
}

TEST(CodegenTest, Golden) {
  // Normally CpuCompiler::CpuCompiler does this, but in this test we've
  // bypassed the Cpu compiler so we have to do this manually.
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();

  CodegenOpts opts;
  opts.class_name = "MyClass";
  opts.target_triple = "x86_64-pc-linux";
  opts.namespaces = {"foo", "bar"};
  opts.gen_name_to_index = true;
  opts.gen_program_shape = true;
  tf2xla::Config config;
  tf2xla::Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("feed0");
  feed->set_name("myfeed");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("feed1");
  tf2xla::Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("fetch0");
  fetch->set_name("myfetch");
  tf2xla::Variable* variable = config.add_variable();
  variable->set_node_name("myvar_readonly");
  variable->mutable_shape()->add_dim()->set_size(1);
  variable->set_type(DT_FLOAT);
  variable->set_readonly(true);
  tf2xla::Variable* variable2 = config.add_variable();
  variable2->set_node_name("myvar");
  variable2->mutable_shape()->add_dim()->set_size(1);
  variable2->set_type(DT_FLOAT);
  tf2xla::Variable* variable3 = config.add_variable();
  variable3->set_node_name("my/var");
  variable3->set_name("myvar2");
  variable3->mutable_shape()->add_dim()->set_size(5);
  variable3->set_type(DT_INT32);
  CompileResult compile_result;
  compile_result.aot.reset(new xla::cpu::CpuAotCompilationResult(
      {},
      {BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/8, /*param_number=*/0),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/1),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/2),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/3),
       BufferInfo::MakeTempBuffer(1),
       BufferInfo::MakeEntryParameter(/*size=*/96, /*param_number=*/4),
       BufferInfo::MakeTempBuffer(1), BufferInfo::MakeTempBuffer(120)},
      11, {}));
  compile_result.program_shape =
      xla::ShapeUtil::MakeProgramShape(
          {
              xla::ShapeUtil::MakeShape(xla::F32, {1, 2}),
              xla::ShapeUtil::MakeShape(xla::S64, {3, 4}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::S32, {5}),
          },
          xla::ShapeUtil::MakeTupleShape({
              xla::ShapeUtil::MakeShape(xla::U32, {5, 6}),
              xla::ShapeUtil::MakeShape(xla::F32, {1}),
              xla::ShapeUtil::MakeShape(xla::S32, {5}),
          }))
          .ToProto();
  compile_result.entry_point = "entry_point";
  compile_result.pointer_size = 8;

  MetadataResult metadata_result;
  TF_ASSERT_OK(GenerateMetadata(opts, compile_result, &metadata_result));

  // The other fields in metadata_result are tested as part of the generated
  // header test.

  // This specific golden test checks a binary file. It can potentially run into
  // issues due to ABIs not being stable, but has not so far.
  // If we see any ABI issues, we should reconsider this specific test case.
  CompareWithGoldenFile("tensorflow/compiler/aot/codegen_test_o.golden",
                        metadata_result.object_file_data, false);

  string header;
  TF_ASSERT_OK(
      GenerateHeader(opts, config, compile_result, metadata_result, &header));

  CompareWithGoldenFile("tensorflow/compiler/aot/codegen_test_h.golden", header,
                        true);
}
}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
