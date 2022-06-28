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
class MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc() {
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

#include "tensorflow/core/framework/node_def_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"  // NOLINT
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

OpDef ToOpDef(const OpDefBuilder& builder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/framework/node_def_util_test.cc", "ToOpDef");

  OpRegistrationData op_reg_data;
  TF_EXPECT_OK(builder.Finalize(&op_reg_data));
  return op_reg_data.op_def;
}

NodeDef ToNodeDef(const string& text) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/framework/node_def_util_test.cc", "ToNodeDef");

  NodeDef node_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &node_def));
  return node_def;
}

NodeDef ToNodeDef(NodeDefBuilder&& builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/framework/node_def_util_test.cc", "ToNodeDef");

  NodeDef node_def;
  TF_EXPECT_OK(builder.Finalize(&node_def));
  return node_def;
}

void ExpectSuccess(const NodeDef& good, const OpDef& op_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/framework/node_def_util_test.cc", "ExpectSuccess");

  EXPECT_EQ(Status::OK(), ValidateNodeDef(good, op_def))
      << "NodeDef: " << SummarizeNodeDef(good)
      << "; OpDef: " << SummarizeOpDef(op_def);
}

void ExpectFailure(const NodeDef& bad, const OpDef& op_def,
                   const string& message) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/framework/node_def_util_test.cc", "ExpectFailure");

  Status status = ValidateNodeDef(bad, op_def);

  EXPECT_FALSE(status.ok()) << "NodeDef: " << SummarizeNodeDef(bad)
                            << "; OpDef: " << SummarizeOpDef(op_def);
  if (status.ok()) return;

  EXPECT_TRUE(errors::IsInvalidArgument(status))
      << status << "; NodeDef: " << SummarizeNodeDef(bad)
      << "; OpDef: " << SummarizeOpDef(op_def);

  LOG(INFO) << "Message: " << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.ToString(), message))
      << "NodeDef: " << SummarizeNodeDef(bad)
      << "; OpDef: " << SummarizeOpDef(op_def) << "\nActual error: " << status
      << "\nDoes not contain: " << message;
}

TEST(NodeDefUtilTest, In) {
  const OpDef op = ToOpDef(OpDefBuilder("In").Input("i: T").Attr("T: type"));
  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'In'
    input: 'a'
    attr {
      key: 'T'
      value { type: DT_FLOAT }
    }
  )pb");
  ExpectSuccess(node_def, op);

  EXPECT_EQ("{{node n}} = In[T=DT_FLOAT](a)", SummarizeNodeDef(node_def));

  // Mismatching Op names.
  NodeDef bad = node_def;
  bad.set_op("Wrong");
  ExpectFailure(bad, op, "NodeDef op 'Wrong' does not match Op<name=In;");

  // Missing attr
  bad = node_def;
  bad.clear_attr();
  ExpectFailure(bad, op, "NodeDef missing attr 'T' from Op<name=In;");

  // Attr has wrong type
  bad = node_def;
  bad.clear_attr();
  AddNodeAttr("T", 17, &bad);
  ExpectFailure(
      bad, op,
      "AttrValue had value with type 'int' when 'type' expected\n\t for attr "
      "'T'\n\t; NodeDef: ");

  // Wrong number of inputs
  bad = node_def;
  bad.add_input("b");
  ExpectFailure(
      bad, op,
      "NodeDef expected inputs 'float' do not match 2 inputs specified;");

  bad = node_def;
  bad.clear_input();
  ExpectFailure(
      bad, op,
      "NodeDef expected inputs 'float' do not match 0 inputs specified;");

  // Control inputs must appear after data inputs
  NodeDef good = node_def;
  good.add_input("^b");
  ExpectSuccess(node_def, op);

  bad = node_def;
  bad.clear_input();
  bad.add_input("^b");
  bad.add_input("a");
  ExpectFailure(bad, op,
                "Non-control input 'a' after control input "
                "in NodeDef:");

  bad = node_def;
  bad.add_input("^b:0");
  ExpectFailure(bad, op, "Control input '^b:0' must not have ':' in NodeDef:");
}

TEST(NodeDefUtilTest, Out) {
  const OpDef op =
      ToOpDef(OpDefBuilder("Out").Output("o: T").Attr("T: numbertype"));
  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'Out'
    attr {
      key: 'T'
      value { type: DT_INT32 }
    }
  )pb");
  ExpectSuccess(node_def, op);

  EXPECT_EQ("{{node n}} = Out[T=DT_INT32]()", SummarizeNodeDef(node_def));

  // Non-number type.
  NodeDef bad = node_def;
  bad.clear_attr();
  AddNodeAttr("T", DT_STRING, &bad);
  ExpectFailure(bad, op,
                "Value for attr 'T' of string is not in the list of allowed "
                "values: float, double, int32, uint8, int16, int8, complex64, "
                "int64, qint8, quint8, qint32, bfloat16, uint16, complex128, "
                "half, uint32, uint64");
}

TEST(NodeDefUtilTest, Enum) {
  const OpDef op = ToOpDef(OpDefBuilder("Enum").Attr("e: {'apple','orange'}"));
  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'Enum'
    attr {
      key: 'e'
      value { s: 'apple' }
    }
  )pb");
  ExpectSuccess(node_def, op);

  EXPECT_EQ("{{node n}} = Enum[e=\"apple\"]()", SummarizeNodeDef(node_def));

  NodeDef good = node_def;
  good.clear_attr();
  AddNodeAttr("e", "orange", &good);
  ExpectSuccess(good, op);

  // Non-allowed value.
  NodeDef bad = node_def;
  bad.clear_attr();
  AddNodeAttr("e", "foo", &bad);
  ExpectFailure(bad, op,
                "Value for attr 'e' of \"foo\" is not in the list of allowed "
                "values: \"apple\", \"orange\"");
}

TEST(NodeDefUtilTest, SameIn) {
  const OpDef op = ToOpDef(OpDefBuilder("SameIn")
                               .Input("i: N * T")
                               .Attr("N: int >= 2")
                               .Attr("T: {float,double}"));
  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'SameIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'N'
      value { i: 2 }
    }
    attr {
      key: 'T'
      value { type: DT_DOUBLE }
    }
  )pb");
  ExpectSuccess(node_def, op);

  EXPECT_EQ("{{node n}} = SameIn[N=2, T=DT_DOUBLE](a, b)",
            SummarizeNodeDef(node_def));

  // Illegal type
  NodeDef bad = ToNodeDef(R"pb(
    name: 'n'
    op: 'SameIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'N'
      value { i: 2 }
    }
    attr {
      key: 'T'
      value { type: DT_STRING }
    }
  )pb");
  ExpectFailure(bad, op,
                "Value for attr 'T' of string is not in the list of allowed "
                "values: float, double");

  // Too few inputs
  bad = ToNodeDef(R"pb(
    name: 'n'
    op: 'SameIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'N'
      value { i: 1 }
    }
    attr {
      key: 'T'
      value { type: DT_FLOAT }
    }
  )pb");
  ExpectFailure(bad, op, "Value for attr 'N' of 1 must be at least minimum 2");
}

TEST(NodeDefUtilTest, AnyIn) {
  const OpDef op =
      ToOpDef(OpDefBuilder("AnyIn").Input("i: T").Attr("T: list(type) >= 1"));

  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectSuccess(node_def, op);

  EXPECT_EQ("{{node n}} = AnyIn[T=[DT_INT32, DT_STRING]](a, b)",
            SummarizeNodeDef(node_def));

  const NodeDef bad = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    attr {
      key: 'T'
      value { list {} }
    }
  )pb");
  ExpectFailure(bad, op, "Length for attr 'T' of 0 must be at least minimum 1");

  // With proto3 semantics, an empty value {} is indistinguishable from a value
  // with an empty list in it. So we simply expect to get a message complaining
  // about empty list for value {}.
  const NodeDef bad2 = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    attr {
      key: 'T'
      value {}
    }
  )pb");
  ExpectFailure(bad2, op,
                "Length for attr 'T' of 0 must be at least minimum 1");
}

TEST(NodeDefUtilTest, Device) {
  const OpDef op_def1 = ToOpDef(OpDefBuilder("None"));
  const NodeDef node_def1 =
      ToNodeDef(std::move(NodeDefBuilder("d", &op_def1).Device("/cpu:17")));
  ExpectSuccess(node_def1, op_def1);
  EXPECT_EQ("{{node d}} = None[_device=\"/cpu:17\"]()",
            SummarizeNodeDef(node_def1));

  const OpDef op_def2 = ToOpDef(OpDefBuilder("WithAttr").Attr("v: int"));
  const NodeDef node_def2 = ToNodeDef(
      std::move(NodeDefBuilder("d", &op_def2).Attr("v", 7).Device("/cpu:5")));
  ExpectSuccess(node_def2, op_def2);
  EXPECT_EQ("{{node d}} = WithAttr[v=7, _device=\"/cpu:5\"]()",
            SummarizeNodeDef(node_def2));
}

void ExpectValidSyntax(const NodeDef& good) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_5(mht_5_v, 505, "", "./tensorflow/core/framework/node_def_util_test.cc", "ExpectValidSyntax");

  EXPECT_EQ(Status::OK(), ValidateExternalNodeDefSyntax(good))
      << "NodeDef: " << SummarizeNodeDef(good);
}

void ExpectInvalidSyntax(const NodeDef& bad, const string& message) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSnode_def_util_testDTcc mht_6(mht_6_v, 514, "", "./tensorflow/core/framework/node_def_util_test.cc", "ExpectInvalidSyntax");

  Status status = ValidateExternalNodeDefSyntax(bad);

  ASSERT_FALSE(status.ok()) << "NodeDef: " << SummarizeNodeDef(bad);

  EXPECT_TRUE(errors::IsInvalidArgument(status))
      << status << "; NodeDef: " << SummarizeNodeDef(bad);

  EXPECT_TRUE(absl::StrContains(StringPiece(status.ToString()), message))
      << "NodeDef: " << SummarizeNodeDef(bad) << ", " << status << ", "
      << message;
}

TEST(NodeDefUtilTest, ValidSyntax) {
  const NodeDef node_def = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectValidSyntax(node_def);

  const NodeDef node_def_namespace = ToNodeDef(R"pb(
    name: 'n'
    op: 'Project>AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectValidSyntax(node_def_namespace);

  const NodeDef node_def_explicit_inputs = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a:0'
    input: 'b:123'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectValidSyntax(node_def_explicit_inputs);

  EXPECT_EQ("{{node n}} = AnyIn[T=[DT_INT32, DT_STRING]](a:0, b:123)",
            SummarizeNodeDef(node_def_explicit_inputs));

  const NodeDef node_def_explicit_inputs_namespace = ToNodeDef(R"pb(
    name: 'Project>n'
    op: 'Project>AnyIn'
    input: 'Project>a:0'
    input: 'Project>b:123'
    input: '^Project>c'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectValidSyntax(node_def_explicit_inputs_namespace);

  EXPECT_EQ(
      "{{node Project>n}} = Project>AnyIn[T=[DT_INT32, DT_STRING]]"
      "(Project>a:0, Project>b:123, ^Project>c)",
      SummarizeNodeDef(node_def_explicit_inputs_namespace));

  const NodeDef node_def_partial_shape = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    attr {
      key: 'shp'
      value {
        shape {
          dim { size: -1 }
          dim { size: 0 }
        }
      }
    }
  )pb");
  ExpectValidSyntax(node_def_partial_shape);

  const NodeDef node_def_control_input = ToNodeDef(R"pb(
    name: 'n-'
    op: 'AnyIn'
    input: 'a'
    input: '^b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectValidSyntax(node_def_control_input);

  const NodeDef node_def_invalid_name = ToNodeDef(R"pb(
    name: 'n:0'
    op: 'AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_invalid_name, "Illegal op name 'n:0'");

  const NodeDef node_def_internal_name = ToNodeDef(R"pb(
    name: '_n'
    op: 'AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_internal_name, "Illegal op name '_n'");

  const NodeDef node_def_slash_in_name = ToNodeDef(R"pb(
    name: 'n\\'
    op: 'AnyIn'
    input: 'a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_slash_in_name, "Illegal op name 'n\\'");

  const NodeDef node_def_internal_input_name = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: '_a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_internal_input_name,
                      "Illegal op input name '_a'");

  const NodeDef node_def_input_name_slash = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a\\'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_input_name_slash, "Illegal op input name 'a\\'");

  const NodeDef node_def_invalid_control_input_name = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    input: '^b:0'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_invalid_control_input_name,
                      "Illegal op input name '^b:0'");

  const NodeDef node_def_control_input_name_slash = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a'
    input: '^b\\'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_control_input_name_slash,
                      "Illegal op input name '^b\\'");

  const NodeDef node_def_data_input_after_control = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: '^a'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_data_input_after_control,
                      "All control inputs must follow all data inputs");

  const NodeDef node_def_data_input_invalid_port = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a:b'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_data_input_invalid_port,
                      "Illegal op input name 'a:b");

  const NodeDef node_def_data_input_invalid_port2 = ToNodeDef(R"pb(
    name: 'n'
    op: 'AnyIn'
    input: 'a:00'
    input: 'b'
    attr {
      key: 'T'
      value { list { type: [ DT_INT32, DT_STRING ] } }
    }
  )pb");
  ExpectInvalidSyntax(node_def_data_input_invalid_port2,
                      "Illegal op input name 'a:00");
}

TEST(InputTypesForNode, Simple) {
  const OpDef op_def = ToOpDef(OpDefBuilder("Simple")
                                   .Input("a: float")
                                   .Input("b: int32")
                                   .Output("c: string")
                                   .Output("d: bool"));
  const NodeDef node_def = ToNodeDef(std::move(
      NodeDefBuilder("simple", &op_def).Input(FakeInput()).Input(FakeInput())));
  DataTypeVector types;
  EXPECT_TRUE(InputTypesForNode(node_def, op_def, &types).ok());
  EXPECT_EQ(types[0], DT_FLOAT);
  EXPECT_EQ(types[1], DT_INT32);

  DataType type;
  EXPECT_TRUE(InputTypeForNode(node_def, op_def, 0, &type).ok());
  EXPECT_EQ(type, DT_FLOAT);
  EXPECT_TRUE(InputTypeForNode(node_def, op_def, 1, &type).ok());
  EXPECT_EQ(type, DT_INT32);
  EXPECT_FALSE(InputTypeForNode(node_def, op_def, 2, &type).ok());
}

TEST(OutputTypesForNode, Simple) {
  const OpDef op_def = ToOpDef(OpDefBuilder("Simple")
                                   .Input("a: float")
                                   .Input("b: int32")
                                   .Output("c: string")
                                   .Output("d: bool"));
  const NodeDef node_def = ToNodeDef(std::move(
      NodeDefBuilder("simple", &op_def).Input(FakeInput()).Input(FakeInput())));
  DataTypeVector types;
  EXPECT_TRUE(OutputTypesForNode(node_def, op_def, &types).ok());
  EXPECT_EQ(types[0], DT_STRING);
  EXPECT_EQ(types[1], DT_BOOL);

  DataType type;
  EXPECT_TRUE(OutputTypeForNode(node_def, op_def, 0, &type).ok());
  EXPECT_EQ(type, DT_STRING);
  EXPECT_TRUE(OutputTypeForNode(node_def, op_def, 1, &type).ok());
  EXPECT_EQ(type, DT_BOOL);
  EXPECT_FALSE(OutputTypeForNode(node_def, op_def, 2, &type).ok());
}

TEST(OutputTypesForNode, LargeOutput) {
  const OpDef op_def = ToOpDef(OpDefBuilder("TestSplitOp")
                                   .Input("value: int64")
                                   .Output("output: num_split * int64")
                                   .Attr("num_split: int >= 1"));
  int64_t num_split = 1000000000000;
  const NodeDef node_def =
      ToNodeDef(std::move(NodeDefBuilder("test_split_op", &op_def)
                              .Input(FakeInput())
                              .Attr("num_split", num_split)));
  DataTypeVector types;
  EXPECT_FALSE(OutputTypesForNode(node_def, op_def, &types).ok());
}

TEST(OutputTypesForNode_AttrSliceOverload, Simple) {
  const OpDef op_def = ToOpDef(OpDefBuilder("Simple")
                                   .Input("a: float")
                                   .Input("b: int32")
                                   .Output("c: string")
                                   .Output("d: bool"));
  const AttrSlice attr_slice =
      AttrSlice(ToNodeDef(std::move(NodeDefBuilder("simple", &op_def)
                                        .Input(FakeInput())
                                        .Input(FakeInput()))));
  DataTypeVector types;
  EXPECT_TRUE(OutputTypesForNode(attr_slice, op_def, &types).ok());
  EXPECT_EQ(types[0], DT_STRING);
  EXPECT_EQ(types[1], DT_BOOL);
}

TEST(NameRangesForNodeTest, Simple) {
  const OpDef op_def = ToOpDef(OpDefBuilder("Simple")
                                   .Input("a: float")
                                   .Input("b: int32")
                                   .Output("c: string")
                                   .Output("d: bool"));
  NameRangeMap inputs, outputs;
  const NodeDef node_def = ToNodeDef(std::move(
      NodeDefBuilder("simple", &op_def).Input(FakeInput()).Input(FakeInput())));
  TF_EXPECT_OK(NameRangesForNode(node_def, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 1}}, {"b", {1, 2}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}, {"d", {1, 2}}}), outputs);

  EXPECT_EQ("{{node simple}} = Simple[](a, b)", SummarizeNodeDef(node_def));

  OpDef bad_op_def = op_def;
  bad_op_def.mutable_input_arg(0)->clear_type();
  EXPECT_FALSE(NameRangesForNode(node_def, bad_op_def, &inputs, &outputs).ok());
}

TEST(NameRangesForNodeTest, Polymorphic) {
  const OpDef op_def = ToOpDef(OpDefBuilder("Polymorphic")
                                   .Input("a: T")
                                   .Input("b: T")
                                   .Output("c: T")
                                   .Attr("T: type"));
  NameRangeMap inputs, outputs;
  const NodeDef node_def1 =
      ToNodeDef(std::move(NodeDefBuilder("poly", &op_def)
                              .Input(FakeInput(DT_INT32))
                              .Input(FakeInput(DT_INT32))));
  TF_EXPECT_OK(NameRangesForNode(node_def1, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 1}}, {"b", {1, 2}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}}), outputs);
  EXPECT_EQ("{{node poly}} = Polymorphic[T=DT_INT32](a, b)",
            SummarizeNodeDef(node_def1));

  const NodeDef node_def2 =
      ToNodeDef(std::move(NodeDefBuilder("poly", &op_def)
                              .Input(FakeInput(DT_BOOL))
                              .Input(FakeInput(DT_BOOL))));
  TF_EXPECT_OK(NameRangesForNode(node_def2, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 1}}, {"b", {1, 2}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}}), outputs);
  EXPECT_EQ("{{node poly}} = Polymorphic[T=DT_BOOL](a, b)",
            SummarizeNodeDef(node_def2));
}

TEST(NameRangesForNodeTest, NRepeats) {
  const OpDef op_def = ToOpDef(OpDefBuilder("NRepeats")
                                   .Input("a: N * int32")
                                   .Input("b: N * T")
                                   .Output("c: T")
                                   .Output("d: N * string")
                                   .Output("e: M * bool")
                                   .Attr("N: int")
                                   .Attr("M: int")
                                   .Attr("T: type"));
  NameRangeMap inputs, outputs;
  const NodeDef node_def1 =
      ToNodeDef(std::move(NodeDefBuilder("nr", &op_def)
                              .Input(FakeInput(4, DT_INT32))
                              .Input(FakeInput(4, DT_FLOAT))
                              .Attr("M", 3)));
  TF_EXPECT_OK(NameRangesForNode(node_def1, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 4}}, {"b", {4, 8}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}, {"d", {1, 5}}, {"e", {5, 8}}}),
            outputs);
  EXPECT_EQ(
      "{{node nr}} = NRepeats[M=3, N=4, T=DT_FLOAT](a, a:1, a:2, a:3, b, b:1, "
      "b:2, b:3)",
      SummarizeNodeDef(node_def1));

  const NodeDef node_def2 =
      ToNodeDef(std::move(NodeDefBuilder("nr", &op_def)
                              .Input(FakeInput(2, DT_INT32))
                              .Input(FakeInput(2, DT_DOUBLE))
                              .Attr("M", 7)));
  TF_EXPECT_OK(NameRangesForNode(node_def2, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 2}}, {"b", {2, 4}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}, {"d", {1, 3}}, {"e", {3, 10}}}),
            outputs);
  EXPECT_EQ("{{node nr}} = NRepeats[M=7, N=2, T=DT_DOUBLE](a, a:1, b, b:1)",
            SummarizeNodeDef(node_def2));

  NodeDef bad_node_def = node_def2;
  bad_node_def.clear_attr();
  EXPECT_FALSE(NameRangesForNode(bad_node_def, op_def, &inputs, &outputs).ok());
}

TEST(NameRangesForNodeTest, TypeList) {
  const OpDef op_def = ToOpDef(OpDefBuilder("TypeList")
                                   .Input("a: T1")
                                   .Input("b: T2")
                                   .Output("c: T2")
                                   .Output("d: T3")
                                   .Output("e: T1")
                                   .Attr("T1: list(type)")
                                   .Attr("T2: list(type)")
                                   .Attr("T3: list(type)"));
  NameRangeMap inputs, outputs;
  const NodeDef node_def1 =
      ToNodeDef(std::move(NodeDefBuilder("tl", &op_def)
                              .Input(FakeInput({DT_BOOL, DT_FLOAT}))
                              .Input(FakeInput(4, DT_FLOAT))
                              .Attr("T3", {DT_INT32, DT_DOUBLE, DT_STRING})));
  TF_EXPECT_OK(NameRangesForNode(node_def1, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 2}}, {"b", {2, 6}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 4}}, {"d", {4, 7}}, {"e", {7, 9}}}),
            outputs);
  EXPECT_EQ(
      "{{node tl}} = TypeList[T1=[DT_BOOL, DT_FLOAT],"
      " T2=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT],"
      " T3=[DT_INT32, DT_DOUBLE, DT_STRING]](a, a:1, b, b:1, b:2, b:3)",
      SummarizeNodeDef(node_def1));

  const NodeDef node_def2 =
      ToNodeDef(std::move(NodeDefBuilder("tl", &op_def)
                              .Input(FakeInput(7, DT_INT32))
                              .Input(FakeInput({DT_DOUBLE}))
                              .Attr("T3", {DT_DOUBLE, DT_STRING})));
  TF_EXPECT_OK(NameRangesForNode(node_def2, op_def, &inputs, &outputs));
  EXPECT_EQ(NameRangeMap({{"a", {0, 7}}, {"b", {7, 8}}}), inputs);
  EXPECT_EQ(NameRangeMap({{"c", {0, 1}}, {"d", {1, 3}}, {"e", {3, 10}}}),
            outputs);
  EXPECT_EQ(
      "{{node tl}} = TypeList[T1=[DT_INT32, DT_INT32, DT_INT32, DT_INT32, "
      "DT_INT32,"
      " DT_INT32, DT_INT32], T2=[DT_DOUBLE], T3=[DT_DOUBLE, DT_STRING]]"
      "(a, a:1, a:2, a:3, a:4, a:5, a:6, b)",
      SummarizeNodeDef(node_def2));

  NodeDef bad_node_def = node_def2;
  bad_node_def.clear_attr();
  EXPECT_FALSE(NameRangesForNode(bad_node_def, op_def, &inputs, &outputs).ok());
}

TEST(AddPrefixAndSuffixToNode, Enter) {
  NodeDef node_def;
  node_def.set_name("enter");
  node_def.set_op("Enter");
  AddNodeAttr("frame_name", "test_frame", &node_def);
  const string prefix = "prefix/";
  const string suffix = "/suffix";
  TF_ASSERT_OK(AddPrefixAndSuffixToNode(prefix, suffix, &node_def));
  EXPECT_EQ("prefix/enter/suffix", node_def.name());
  string frame_name;
  TF_ASSERT_OK(GetNodeAttr(node_def, "frame_name", &frame_name));
  EXPECT_EQ("prefix/test_frame/suffix", frame_name);
}

TEST(MaybeAddPrefixToColocationConstraints, Basic) {
  NodeDef node_def;
  node_def.set_name("Identity");
  node_def.set_op("Identity");
  AddNodeAttr(kColocationAttrName,
              {strings::StrCat(kColocationGroupPrefix, "Node1"),
               strings::StrCat(kColocationGroupPrefix, "Node2"),
               strings::StrCat(kColocationGroupPrefix, "Node3")},
              &node_def);

  std::unordered_set<string> match;
  match.insert("Node1");
  match.insert("Node3");
  TF_ASSERT_OK(MaybeAddPrefixToColocationConstraints(match, "fn/", &node_def));
  std::vector<string> coloc_constraints;
  TF_ASSERT_OK(GetNodeAttr(node_def, kColocationAttrName, &coloc_constraints));
  EXPECT_EQ(
      coloc_constraints,
      std::vector<string>({"loc:@fn/Node1", "loc:@Node2", "loc:@fn/Node3"}));
}

TEST(MaybeAddPrefixToColocationConstraints, NoConstraints) {
  NodeDef node_def;
  node_def.set_name("Identity");
  node_def.set_op("Identity");

  std::unordered_set<string> match;
  match.insert("Node1");
  match.insert("Node3");
  TF_ASSERT_OK(MaybeAddPrefixToColocationConstraints(match, "fn/", &node_def));
  EXPECT_FALSE(HasNodeAttr(node_def, kColocationAttrName));
}

TEST(MaybeUpdateColocationConstraintsWithMap, Basic) {
  NodeDef node_def;
  node_def.set_name("Identity");
  node_def.set_op("Identity");
  AddNodeAttr(kColocationAttrName,
              {strings::StrCat(kColocationGroupPrefix, "Node1"),
               strings::StrCat(kColocationGroupPrefix, "Node2"),
               strings::StrCat(kColocationGroupPrefix, "Node3")},
              &node_def);

  std::map<absl::string_view, absl::string_view> node_map;
  node_map["Node1"] = "Node4";
  node_map["Invalid"] = "Node5";
  TF_ASSERT_OK(MaybeUpdateColocationConstraintsWithMap(node_map, &node_def));
  std::vector<string> coloc_constraints;
  TF_ASSERT_OK(GetNodeAttr(node_def, kColocationAttrName, &coloc_constraints));
  EXPECT_EQ(coloc_constraints,
            std::vector<string>({"loc:@Node4", "loc:@Node2", "loc:@Node3"}));
}

TEST(MaybeUpdateColocationConstraintsWithMap, NoConstraints) {
  NodeDef node_def;
  node_def.set_name("Identity");
  node_def.set_op("Identity");

  std::map<absl::string_view, absl::string_view> node_map;
  node_map["Node1"] = "Node4";
  node_map["Invalid"] = "Node5";
  TF_ASSERT_OK(MaybeUpdateColocationConstraintsWithMap(node_map, &node_def));
  EXPECT_FALSE(HasNodeAttr(node_def, kColocationAttrName));
}

TEST(FormatNodeForErrorTest, Node) {
  Graph g(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("enter", "NoOp").Finalize(&g, &node));
  EXPECT_EQ("{{node enter}}", FormatNodeForError(*node));
}

TEST(FormatNodeForErrorTest, NodeDef) {
  NodeDef node_def;
  node_def.set_name("enter");
  node_def.set_op("Enter");
  AddNodeAttr("frame_name", "test_frame", &node_def);
  EXPECT_EQ("{{node enter}}", FormatNodeDefForError(node_def));
}

TEST(FormatNodeForErrorTest, NodeDefWithOriginalNames) {
  NodeDef node_def;
  node_def.set_name("enter");
  node_def.set_op("Enter");
  AddNodeAttr("frame_name", "test_frame", &node_def);
  *(node_def.mutable_experimental_debug_info()->add_original_node_names()) =
      "node_name";
  *(node_def.mutable_experimental_debug_info()->add_original_func_names()) =
      "func_name";
  EXPECT_EQ("{{function_node func_name}}{{node node_name}}",
            FormatNodeDefForError(node_def));
  *(node_def.mutable_experimental_debug_info()->add_original_node_names()) =
      "node_name2";
  *(node_def.mutable_experimental_debug_info()->add_original_func_names()) =
      "func_name2";
  EXPECT_EQ(
      "{{function_node func_name}}{{node node_name}}, "
      "{{function_node func_name2}}{{node node_name2}}",
      FormatNodeDefForError(node_def));
}

TEST(AttachDef, AllowMultipleFormattedNode) {
  NodeDef a;
  a.set_name("a");
  NodeDef b;
  b.set_name("b");
  Status s = Status(error::CANCELLED, "Error");
  Status s2 = AttachDef(s, a, true);
  EXPECT_EQ("Error\n\t [[{{node a}}]]", s2.error_message());
  Status s3 = AttachDef(s2, b, true);
  EXPECT_EQ("Error\n\t [[{{node a}}]]\n\t [[{{node b}}]]", s3.error_message());
}

TEST(AttachDef, DisallowMultipleFormattedNode) {
  NodeDef a;
  a.set_name("a");
  NodeDef b;
  b.set_name("b");
  Status s = Status(error::CANCELLED, "Error");
  Status s2 = AttachDef(s, a, false);
  EXPECT_EQ("Error\n\t [[{{node a}}]]", s2.error_message());
  Status s3 = AttachDef(s2, b, false);
  EXPECT_EQ("Error\n\t [[{{node a}}]]\n\t [[b]]", s3.error_message());
}

}  // namespace
}  // namespace tensorflow
