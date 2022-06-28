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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc() {
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

#include "tensorflow/core/framework/op_def_util.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

OpDef FromText(const string& text) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/framework/op_def_util_test.cc", "FromText");

  OpDef op_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &op_def));
  return op_def;
}

OpDef::AttrDef ADef(const string& text) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/framework/op_def_util_test.cc", "ADef");

  OpDef::AttrDef attr_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &attr_def));
  return attr_def;
}

class ValidateOpDefTest : public ::testing::Test {
 protected:
  Status TestProto(const string& text) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/framework/op_def_util_test.cc", "TestProto");
 return ValidateOpDef(FromText(text)); }

  Status TestBuilder(const OpDefBuilder& builder) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/framework/op_def_util_test.cc", "TestBuilder");

    OpRegistrationData op_reg_data;
    Status status = builder.Finalize(&op_reg_data);
    TF_EXPECT_OK(status);
    if (!status.ok()) {
      return status;
    } else {
      return ValidateOpDef(op_reg_data.op_def);
    }
  }
};

namespace {
void ExpectFailure(const Status& status, const string& message) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectFailure");

  EXPECT_FALSE(status.ok()) << "Did not see error with: " << message;
  if (!status.ok()) {
    LOG(INFO) << "message: " << status;
    EXPECT_TRUE(absl::StrContains(status.ToString(), message))
        << "Actual: " << status << "\nExpected to contain: " << message;
  }
}
}  // namespace

TEST_F(ValidateOpDefTest, OpDefValid) {
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Input("a: int32")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Output("a: bool")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("t: type").Input("a: t")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int = 3")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5 = 3")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: numbertype")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("Uppercase")));

  TF_EXPECT_OK(TestBuilder(OpDefBuilder("Namespace>X").Attr("a: int")));
  TF_EXPECT_OK(TestBuilder(OpDefBuilder("Namespace>X>Y").Attr("a: int")));
}

TEST_F(ValidateOpDefTest, InvalidName) {
  ExpectFailure(TestBuilder(OpDefBuilder("lower").Attr("a: int")),
                "Invalid name");
  ExpectFailure(TestBuilder(OpDefBuilder("BadSuffix 7%")), "Invalid name");
  ExpectFailure(TestBuilder(OpDefBuilder(">OpName").Attr("a: int")),
                "Invalid name");
  // Can't have a dangling empty namespace
  ExpectFailure(TestBuilder(OpDefBuilder("OpName>").Attr("a: int")),
                "Invalid name");
  // Each namespace section must be Camelcased
  ExpectFailure(TestBuilder(OpDefBuilder("OpName>b").Attr("a: int")),
                "Invalid name");
  // Can't have empty namespaces
  ExpectFailure(TestBuilder(OpDefBuilder("OpName>A>>B").Attr("a: int")),
                "Invalid name");
}

TEST_F(ValidateOpDefTest, DuplicateName) {
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Input("a: int32").Input("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("DupeName").Input("a: int32").Output("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("DupeName").Output("a: int32").Output("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Input("a: int32").Attr("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Output("a: int32").Attr("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Attr("a: int").Attr("a: float")),
      "Duplicate name: a");
}

TEST_F(ValidateOpDefTest, BadAttrName) {
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude").Attr("int32: int")),
                "Attr can't have name int32 that matches a data type");
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude").Attr("float: string")),
                "Attr can't have name float that matches a data type");
}

TEST_F(ValidateOpDefTest, BadAttrType) {
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'illegal' }"),
      "Unrecognized type");
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'list(illegal)' }"),
      "Unrecognized type");
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'int extra' }"),
      "Extra ' extra' at the end");
  ExpectFailure(
      TestProto(
          "name: 'BadAttrType' attr { name: 'a' type: 'list(int extra)' }"),
      "'list(' is missing ')' in attr");
  ExpectFailure(
      TestProto(
          "name: 'BadAttrType' attr { name: 'a' type: 'list(int) extra' }"),
      "Extra ' extra' at the end");
}

TEST_F(ValidateOpDefTest, BadAttrDefault) {
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' "
                "type: 'int' default_value { s: 'x' } }"),
      "AttrValue had value with type 'string' when 'int' expected\n\t for "
      "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'int' default_value { f: 0.5 } }"),
                "AttrValue had value with type 'float' when 'int' expected\n"
                "\t for attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' type: 'int' "
                "default_value { i: 5 list { i: [2] } } }"),
      "AttrValue had value with type 'list(int)' when 'int' expected\n\t for "
      "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { f: 0.5 } }"),
      "AttrValue had value with type 'float' when 'list(int)' expected\n\t "
      "for attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' type: 'list(int)' "
                "default_value { list { i: [5] f: [0.5] } } }"),
      "AttrValue had value with type 'list(float)' when 'list(int)' "
      "expected\n\t for attr 'a'\n\t in Op 'BadAttrDef'");

  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'type' default_value { } }"),
                "AttrValue missing value with expected type 'type'\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'shape' default_value { } }"),
                "AttrValue missing value with expected type 'shape'\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'tensor' default_value { } }"),
                "AttrValue missing value with expected type 'tensor'\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");

  // default_value {} is indistinguishable from default_value{ list{} } (one
  // with an empty list) in proto3 semantics.
  TF_EXPECT_OK(
      TestProto("name: 'GoodAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { } }"));

  // Empty lists are allowed:
  TF_EXPECT_OK(
      TestProto("name: 'GoodAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { list { } } }"));
  // Builder should make the same proto:
  TF_EXPECT_OK(
      TestBuilder(OpDefBuilder("GoodAttrDef").Attr("a: list(int) = []")));

  // Unless there is a minimum length specified:
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'list(int)' has_minimum: true minimum: 2 "
                          "default_value { list { } } }"),
                "Length for attr 'a' of 0 must be at least minimum 2\n\t in Op "
                "'BadAttrDef'");
  ExpectFailure(
      TestBuilder(OpDefBuilder("GoodAttrDef").Attr("a: list(bool) >=2 = []")),
      "Length for attr 'a' of 0 must be at least minimum 2\n\t in Op "
      "'GoodAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' type: "
                          "'list(string)' has_minimum: true minimum: 2 "
                          "default_value { list { s: ['foo'] } } }"),
                "Length for attr 'a' of 1 must be at least minimum 2\n\t in Op "
                "'BadAttrDef'");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("GoodAttrDef").Attr("a: list(type) >=2 = [DT_STRING]")),
      "Length for attr 'a' of 1 must be at least minimum 2\n\t in Op "
      "'GoodAttrDef'");
}

TEST_F(ValidateOpDefTest, NoRefTypes) {
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrDef").Input("i: float_ref")),
                "Illegal use of ref type 'float_ref'. "
                "Use 'Ref(type)' instead for input 'i'");
  ExpectFailure(
      TestBuilder(OpDefBuilder("BadAttrDef").Attr("T: type = DT_INT32_REF")),
      "AttrValue must not have reference type value of int32_ref");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("BadAttrDef").Attr("T: list(type) = [DT_STRING_REF]")),
      "AttrValue must not have reference type value of string_ref");
}

TEST_F(ValidateOpDefTest, BadAttrMin) {
  ExpectFailure(TestProto("name: 'BadAttrMin' attr { name: 'a' type: 'string' "
                          "has_minimum: true minimum: 0 }"),
                "minimum for unsupported type string");
  ExpectFailure(
      TestProto("name: 'BadAttrMin' attr { name: 'a' type: 'int' default_value "
                "{ i: 2 } has_minimum: true minimum: 7 }"),
      "Value for attr 'a' of 2 must be at least minimum 7\n\t in Op "
      "'BadAttrMin'");
  ExpectFailure(
      TestProto("name: 'BadAttrMin' attr { name: 'a' "
                "type: 'list(string)' has_minimum: true minimum: -5 }"),
      "list type must have a non-negative minimum, not -5");
  TF_EXPECT_OK(
      TestProto("name: 'GoodAttrMin' attr { name: 'a' type: 'list(string)' "
                "has_minimum: true minimum: 1 }"));
  ExpectFailure(TestProto("name: 'NoHasMin' attr { name: 'a' "
                          "type: 'list(string)' minimum: 3 }"),
                "Attr 'a' with has_minimum = false but minimum 3 not equal to "
                "default of 0");
}

TEST_F(ValidateOpDefTest, BadAttrAllowed) {
  // Is in list of allowed types.
  TF_EXPECT_OK(TestBuilder(
      OpDefBuilder("GoodAttrtude").Attr("x: numbertype = DT_INT32")));
  // Not in list of allowed types.
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("BadAttrtude").Attr("x: numbertype = DT_STRING")),
      "attr 'x' of string is not in the list of allowed values");
  ExpectFailure(
      TestBuilder(OpDefBuilder("BadAttrtude")
                      .Attr("x: list(realnumbertype) = [DT_COMPLEX64]")),
      "attr 'x' of complex64 is not in the list of allowed values");
  ExpectFailure(
      TestBuilder(OpDefBuilder("BadAttrtude")
                      .Attr("x: list(realnumbertype) = [DT_COMPLEX128]")),
      "attr 'x' of complex128 is not in the list of allowed values");
  // Is in list of allowed strings.
  TF_EXPECT_OK(TestBuilder(
      OpDefBuilder("GoodAttrtude").Attr("x: {'foo', 'bar'} = 'bar'")));
  // Not in list of allowed strings.
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("BadAttrtude").Attr("x: {'foo', 'bar'} = 'baz'")),
      "attr 'x' of \"baz\" is not in the list of allowed values");
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude")
                                .Attr("x: list({'foo', 'bar'}) = ['baz']")),
                "attr 'x' of \"baz\" is not in the list of allowed values");
  ExpectFailure(TestProto("name: 'BadAttrtude' attr { name: 'a' "
                          "type: 'string' allowed_values { s: 'not list' } }"),
                "with type 'string' when 'list(string)' expected");
  ExpectFailure(
      TestProto("name: 'BadAttrtude' attr { name: 'a' "
                "type: 'string' allowed_values { list { i: [6] } } }"),
      "with type 'list(int)' when 'list(string)' expected");
}

TEST_F(ValidateOpDefTest, BadArgType) {
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type: DT_INT32 } input_arg { name: 'b' }"),
                "Missing type for input 'b'");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type: DT_INT32 } output_arg { name: 'b' }"),
                "Missing type for output 'b'");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' type: "
                "DT_INT32 type_attr: 'x' } attr { name: 'x' type: 'type' }"),
      "Exactly one of type, type_attr, type_list_attr must be set for input "
      "'a'");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type_attr: 'x' } attr { name: 'x' type: 'int' }"),
                "Attr 'x' used as type_attr for input 'a' has type int");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_attr: 'x' } attr { name: 'x' type: 'list(type)' }"),
      "Attr 'x' used as type_attr for input 'a' has type list(type)");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_list_attr: 'x' } attr { name: 'x' type: 'int' }"),
      "Attr 'x' used as type_list_attr for input 'a' has type int");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_list_attr: 'x' } attr { name: 'x' type: 'type' }"),
      "Attr 'x' used as type_list_attr for input 'a' has type type");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type_attr: 'x' }"),
                "No attr with name 'x' for input 'a'");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' number_attr: 'n' "
                "type_attr: 'x' } attr { name: 'x' type: 'list(type)' } "
                "attr { name: 'n' type: 'int' has_minimum: true minimum: 1 }"),
      "Attr 'x' used as type_attr for input 'a' has type list(type)");
  // But list(type) is fine as the type of an arg without a number_attr:
  TF_EXPECT_OK(TestProto(
      "name: 'Arg' input_arg { name: 'a' type_list_attr: 'x' } "
      "attr { name: 'x' type: 'list(type)' } attr { name: 'n' type: 'int' "
      "has_minimum: true minimum: 1 }"));

  // number_attr
  TF_EXPECT_OK(TestProto(
      "name: 'Arg' input_arg { name: 'a' type: DT_INT32 number_attr: 'n' } "
      "attr { name: 'n' type: 'int' has_minimum: true minimum: 0 }"));

  ExpectFailure(TestProto("name: 'Arg' input_arg { name: 'a' type: DT_INT32 "
                          "number_attr: 'n' }"),
                "No attr with name 'n'");
  ExpectFailure(
      TestProto(
          "name: 'Arg' input_arg { name: 'a' type: "
          "DT_INT32 number_attr: 'n' } attr { name: 'n' type: 'string' }"),
      "Attr 'n' used as length for input 'a' has type string");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' type: "
                "DT_INT32 number_attr: 'n' } attr { name: 'n' type: 'int' }"),
      "Attr 'n' used as length for input 'a' must have minimum;");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' type: DT_INT32 number_attr: "
                "'n' } attr { name: 'n' type: 'int' has_minimum: true minimum: "
                "-5 }"),
      "Attr 'n' used as length for input 'a' must have minimum >= 0;");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' number_attr: 'n' } attr { "
                "name: 'n' type: 'int' has_minimum: true minimum: 2 }"),
      "Missing type for input 'a'; in OpDef:");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' number_attr: "
                          "'n' type_list_attr: 'x' } attr { name: 'n' type: "
                          "'int' has_minimum: true minimum: 1 } attr { name: "
                          "'x' type: 'list(type)' }"),
                "Can't have both number_attr and type_list_attr for input 'a'");
}

void ExpectDifferent(const OpDef::AttrDef& a1, const OpDef::AttrDef& a2) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_5(mht_5_v, 561, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectDifferent");

  EXPECT_FALSE(AttrDefEqual(a1, a2));
  EXPECT_FALSE(AttrDefEqual(a2, a1));
  EXPECT_NE(AttrDefHash(a1), AttrDefHash(a2));
}

TEST(AttrDefUtilTest, EqualAndHash) {
  OpDef::AttrDef a = ADef(
      "name: 'foo' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }");

  EXPECT_TRUE(AttrDefEqual(a, a));
  EXPECT_EQ(AttrDefHash(a), AttrDefHash(a));

  ExpectDifferent(
      a,
      ADef("name: 'FOO' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'int32'  description: 'cool' has_minimum: true "
           "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'COOL' has_minimum: true "
           "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: false "
           "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 3 default_value { i: 2 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 2 default_value { i: 3 } allowed_values { i: 5 }"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 2 default_value { i: 2 } allowed_values { i: 6 }"));

  // Same cases but where default_value and allowed_values are not set
  a = ADef(
      "name: 'foo' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 2");
  EXPECT_TRUE(AttrDefEqual(a, a));
  EXPECT_EQ(AttrDefHash(a), AttrDefHash(a));

  ExpectDifferent(
      a,
      ADef("name: 'FOO' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 2"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'int32'  description: 'cool' has_minimum: true "
           "minimum: 2"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'COOL' has_minimum: true "
           "minimum: 2"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: false "
           "minimum: 2"));
  ExpectDifferent(
      a,
      ADef("name: 'foo' type: 'string' description: 'cool' has_minimum: true "
           "minimum: 3"));
}

protobuf::RepeatedPtrField<OpDef::AttrDef> Rep(
    const std::vector<OpDef::AttrDef>& defs) {
  protobuf::RepeatedPtrField<OpDef::AttrDef> rep;
  for (const OpDef::AttrDef& def : defs) {
    rep.Add()->MergeFrom(def);
  }
  return rep;
}

void ExpectEqual(const protobuf::RepeatedPtrField<OpDef::AttrDef>& a1,
                 const protobuf::RepeatedPtrField<OpDef::AttrDef>& a2) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_6(mht_6_v, 646, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectEqual");

  EXPECT_TRUE(RepeatedAttrDefEqual(a1, a2));
  EXPECT_TRUE(RepeatedAttrDefEqual(a2, a1));
  EXPECT_EQ(RepeatedAttrDefHash(a1), RepeatedAttrDefHash(a2));
}

void ExpectDifferent(const protobuf::RepeatedPtrField<OpDef::AttrDef>& a1,
                     const protobuf::RepeatedPtrField<OpDef::AttrDef>& a2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_7(mht_7_v, 656, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectDifferent");

  EXPECT_FALSE(RepeatedAttrDefEqual(a1, a2));
  EXPECT_FALSE(RepeatedAttrDefEqual(a2, a1));
  EXPECT_NE(RepeatedAttrDefHash(a1), RepeatedAttrDefHash(a2));
}

TEST(AttrDefUtilTest, EqualAndHash_Repeated) {
  OpDef::AttrDef a1 = ADef(
      "name: 'foo1' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }");

  // Different from a1 in name only.
  // name is special because AttrDefs are matched by name.
  OpDef::AttrDef a2 = ADef(
      "name: 'foo2' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 2 default_value { i: 2 } allowed_values { i: 5 }");

  // Different from a1 in "body" only.
  OpDef::AttrDef a3 = ADef(
      "name: 'foo1' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 3 default_value { i: 2 } allowed_values { i: 5 }");

  // Different in name and "body".
  OpDef::AttrDef a4 = ADef(
      "name: 'foo3' type: 'string' description: 'cool' has_minimum: true "
      "minimum: 3 default_value { i: 2 } allowed_values { i: 5 }");

  ExpectEqual(Rep({}), Rep({}));
  ExpectEqual(Rep({a1}), Rep({a1}));
  ExpectEqual(Rep({a1, a2}), Rep({a1, a2}));
  ExpectEqual(Rep({a1, a2}), Rep({a2, a1}));
  ExpectEqual(Rep({a1, a4}), Rep({a4, a1}));

  ExpectDifferent(Rep({a1}), Rep({}));
  ExpectDifferent(Rep({a1}), Rep({a2}));
  ExpectDifferent(Rep({a1}), Rep({a3}));
  ExpectDifferent(Rep({a1}), Rep({a4}));
  ExpectDifferent(Rep({a1}), Rep({a1, a2}));
  ExpectDifferent(Rep({a1, a2}), Rep({a1, a4}));
  ExpectDifferent(Rep({a1, a2}), Rep({a1, a2, a4}));
}

void ExpectEqual(const OpDef& o1, const OpDef& o2) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_8(mht_8_v, 701, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectEqual");

  EXPECT_TRUE(OpDefEqual(o1, o2));
  EXPECT_TRUE(OpDefEqual(o2, o1));
  EXPECT_EQ(OpDefHash(o1), OpDefHash(o2));
}

void ExpectDifferent(const OpDef& o1, const OpDef& o2) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_util_testDTcc mht_9(mht_9_v, 710, "", "./tensorflow/core/framework/op_def_util_test.cc", "ExpectDifferent");

  EXPECT_FALSE(OpDefEqual(o1, o2));
  EXPECT_FALSE(OpDefEqual(o2, o1));
  EXPECT_NE(OpDefHash(o1), OpDefHash(o2));
}

TEST(OpDefEqualityTest, EqualAndHash) {
  string a1 = "attr { name: 'a' type: 'string' } ";
  string a2 = "attr { name: 'b' type: 'string' } ";
  string a3 = "attr { name: 'c' type: 'int32' } ";
  OpDef o1 = FromText(strings::StrCat("name: 'MatMul' ", a1));
  OpDef o2 = FromText(strings::StrCat("name: 'MatMul' ", a2));
  OpDef o3 = FromText(strings::StrCat("name: 'MatMul' ", a1, a2));
  OpDef o4 = FromText(strings::StrCat("name: 'MatMul' ", a2, a1));

  ExpectEqual(o1, o1);
  ExpectEqual(o3, o4);

  ExpectDifferent(o1, o2);
  ExpectDifferent(o1, o3);
}

TEST(OpDefAttrDefaultsUnchangedTest, Foo) {
  const auto& op1 = FromText("name: 'op1' attr { name: 'n' type: 'string'}");
  const auto& op2 = FromText(
      "name: 'op2' attr { name: 'n' type: 'string' default_value: {s: 'x'}}");
  const auto& op3 = FromText(
      "name: 'op3' attr { name: 'n' type: 'string' default_value: {s: 'y'}}");

  // Adding a default value: fine.
  TF_EXPECT_OK(OpDefAttrDefaultsUnchanged(op1, op2));

  // Changing a default value: not ok.
  Status changed_attr = OpDefAttrDefaultsUnchanged(op2, op3);
  ExpectFailure(changed_attr,
                "Attr 'n' has changed it's default value; from \"x\" to \"y\"");

  // Removing a default value: not ok.
  Status removed_attr = OpDefAttrDefaultsUnchanged(op2, op1);
  ExpectFailure(removed_attr,
                "Attr 'n' has removed it's default; from \"x\" to no default");
}

}  // namespace
}  // namespace tensorflow
