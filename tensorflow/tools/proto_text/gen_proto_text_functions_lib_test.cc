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
class MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/proto_text/gen_proto_text_functions_lib.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/proto_text/test.pb_text.h"
#include "tensorflow/tools/proto_text/test.pb.h"

namespace tensorflow {
namespace test {
namespace {

// Convert <input> to text depending on <short_debug>, then parse that into a
// new message using the generated parse function. Return the new message.
template <typename T>
T RoundtripParseProtoOrDie(const T& input, bool short_debug) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/tools/proto_text/gen_proto_text_functions_lib_test.cc", "RoundtripParseProtoOrDie");

  const string s = short_debug ? input.ShortDebugString() : input.DebugString();
  T t;
  EXPECT_TRUE(ProtoParseFromString(s, &t)) << "Failed to parse " << s;
  return t;
}

// Macro that takes <proto> and verifies the proto text string output
// matches DebugString calls on the proto, and verifies parsing the
// DebugString output works. It does this for regular and short
// debug strings.
#define EXPECT_TEXT_TRANSFORMS_MATCH()                               \
  EXPECT_EQ(proto.DebugString(), ProtoDebugString(proto));           \
  EXPECT_EQ(proto.ShortDebugString(), ProtoShortDebugString(proto)); \
  EXPECT_EQ(proto.DebugString(),                                     \
            RoundtripParseProtoOrDie(proto, true).DebugString());    \
  EXPECT_EQ(proto.DebugString(),                                     \
            RoundtripParseProtoOrDie(proto, false).DebugString());

// Macro for failure cases. Verifies both protobuf and proto_text to
// make sure they match.
#define EXPECT_PARSE_FAILURE(str)                  \
  EXPECT_FALSE(ProtoParseFromString(str, &proto)); \
  EXPECT_FALSE(protobuf::TextFormat::ParseFromString(str, &proto))

// Macro for success cases parsing from a string. Verifies protobuf and
// proto_text cases match.
#define EXPECT_PARSE_SUCCESS(expected, str)                          \
  do {                                                               \
    EXPECT_TRUE(ProtoParseFromString(str, &proto));                  \
    string proto_text_str = ProtoShortDebugString(proto);            \
    EXPECT_TRUE(protobuf::TextFormat::ParseFromString(str, &proto)); \
    string protobuf_str = ProtoShortDebugString(proto);              \
    EXPECT_EQ(proto_text_str, protobuf_str);                         \
    EXPECT_EQ(expected, proto_text_str);                             \
  } while (false)

// Test different cases of numeric values, including repeated values.
TEST(CreateProtoDebugStringLibTest, ValidSimpleTypes) {
  TestAllTypes proto;

  // Note that this also tests that output of fields matches tag number order,
  // since some of these fields have high tag numbers.
  proto.Clear();
  proto.set_optional_int32(-1);
  proto.set_optional_int64(-2);
  proto.set_optional_uint32(3);
  proto.set_optional_uint64(4);
  proto.set_optional_sint32(-5);
  proto.set_optional_sint64(-6);
  proto.set_optional_fixed32(-7);
  proto.set_optional_fixed64(-8);
  proto.set_optional_sfixed32(-9);
  proto.set_optional_sfixed64(-10);
  proto.set_optional_float(-12.34);
  proto.set_optional_double(-5.678);
  proto.set_optional_bool(true);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Max numeric values.
  proto.Clear();
  proto.set_optional_int32(std::numeric_limits<int32>::max());
  proto.set_optional_int64(std::numeric_limits<protobuf_int64>::max());
  proto.set_optional_uint32(std::numeric_limits<uint32>::max());
  proto.set_optional_uint64(std::numeric_limits<uint64>::max());
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::max());
  proto.set_optional_double(std::numeric_limits<double>::max());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Least positive numeric values.
  proto.Clear();
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::min());
  proto.set_optional_double(std::numeric_limits<double>::min());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Lowest numeric values.
  proto.Clear();
  proto.set_optional_int32(std::numeric_limits<int32>::lowest());
  proto.set_optional_int64(std::numeric_limits<protobuf_int64>::lowest());
  // TODO(b/67475677): Re-enable after resolving float precision issue
  // proto.set_optional_float(std::numeric_limits<float>::lowest());
  proto.set_optional_double(std::numeric_limits<double>::lowest());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // inf and -inf for float and double.
  proto.Clear();
  proto.set_optional_double(std::numeric_limits<double>::infinity());
  proto.set_optional_float(std::numeric_limits<float>::infinity());
  EXPECT_TEXT_TRANSFORMS_MATCH();
  proto.set_optional_double(-1 * std::numeric_limits<double>::infinity());
  proto.set_optional_float(-1 * std::numeric_limits<float>::infinity());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // String and bytes values.
  proto.Clear();
  for (int i = 0; i < 256; ++i) {
    proto.mutable_optional_string()->push_back(static_cast<char>(i));
    proto.mutable_optional_bytes()->push_back(static_cast<char>(i));
  }
  strings::StrAppend(proto.mutable_optional_string(), "¢€𐍈");
  proto.set_optional_cord(proto.optional_string());
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Repeated values. Include zero values to show they are retained in
  // repeateds.
  proto.Clear();
  proto.add_repeated_int32(-1);
  proto.add_repeated_int32(0);
  proto.add_repeated_int64(0);
  proto.add_repeated_int64(1);
  proto.add_repeated_uint32(-10);
  proto.add_repeated_uint32(0);
  proto.add_repeated_uint32(10);
  proto.add_repeated_uint64(-20);
  proto.add_repeated_uint64(0);
  proto.add_repeated_uint64(20);
  proto.add_repeated_sint32(-30);
  proto.add_repeated_sint32(0);
  proto.add_repeated_sint32(30);
  proto.add_repeated_sint64(-40);
  proto.add_repeated_sint64(0);
  proto.add_repeated_sint64(40);
  proto.add_repeated_fixed32(-50);
  proto.add_repeated_fixed32(0);
  proto.add_repeated_fixed32(50);
  proto.add_repeated_fixed64(-60);
  proto.add_repeated_fixed64(0);
  proto.add_repeated_fixed64(60);
  proto.add_repeated_sfixed32(-70);
  proto.add_repeated_sfixed32(0);
  proto.add_repeated_sfixed32(70);
  proto.add_repeated_sfixed64(-80);
  proto.add_repeated_sfixed64(0);
  proto.add_repeated_sfixed64(80);
  proto.add_repeated_float(-1.2345);
  proto.add_repeated_float(0);
  proto.add_repeated_float(-2.3456);
  proto.add_repeated_double(-10.2345);
  proto.add_repeated_double(0);
  proto.add_repeated_double(-20.3456);
  proto.add_repeated_bool(false);
  proto.add_repeated_bool(true);
  proto.add_repeated_bool(false);
  proto.add_repeated_string("abc");
  proto.add_repeated_string("");
  proto.add_repeated_string("def");
  proto.add_repeated_cord("abc");
  proto.add_repeated_cord("");
  proto.add_repeated_cord("def");
  proto.add_packed_repeated_int64(-1000);
  proto.add_packed_repeated_int64(0);
  proto.add_packed_repeated_int64(1000);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Proto supports [] for list values as well.
  EXPECT_PARSE_SUCCESS("repeated_int32: 1 repeated_int32: 2 repeated_int32: 3",
                       "repeated_int32: [1, 2 , 3]");

  // Test [] and also interesting bool values.
  EXPECT_PARSE_SUCCESS(("repeated_bool: false repeated_bool: false "
                        "repeated_bool: true repeated_bool: true "
                        "repeated_bool: false repeated_bool: true"),
                       "repeated_bool: [false, 0, 1, true, False, True]");

  EXPECT_PARSE_SUCCESS(("repeated_string: \"a,b\" "
                        "repeated_string: \"cdef\""),
                       "repeated_string:   [  'a,b', 'cdef'  ]  ");

  // Proto supports ' as quote character.
  EXPECT_PARSE_SUCCESS("optional_string: \"123\\\" \\'xyz\"",
                       "optional_string: '123\\\" \\'xyz'  ");

  EXPECT_PARSE_SUCCESS("optional_double: 10000", "optional_double: 1e4");

  // Error cases.
  EXPECT_PARSE_FAILURE("optional_string: '1' optional_string: '2'");
  EXPECT_PARSE_FAILURE("optional_double: 123 optional_double: 456");
  EXPECT_PARSE_FAILURE("optional_double: 0001");
  EXPECT_PARSE_FAILURE("optional_double: 000.1");
  EXPECT_PARSE_FAILURE("optional_double: a");
  EXPECT_PARSE_FAILURE("optional_double: x123");
  EXPECT_PARSE_FAILURE("optional_double: '123'");
  EXPECT_PARSE_FAILURE("optional_double: --111");
  EXPECT_PARSE_FAILURE("optional_string: 'abc\"");
  EXPECT_PARSE_FAILURE("optional_bool: truE");
  EXPECT_PARSE_FAILURE("optional_bool: FALSE");
}

TEST(CreateProtoDebugStringLibTest, NestedMessages) {
  TestAllTypes proto;

  proto.Clear();
  // Test empty message.
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_foreign_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Empty messages.
  proto.Clear();
  proto.mutable_optional_nested_message();
  proto.mutable_optional_foreign_message();
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1);
  proto.mutable_optional_foreign_message()->set_c(-1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1234);
  proto.mutable_optional_nested_message()
      ->mutable_msg();  // empty double-nested
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->set_optional_int32(1234);
  proto.mutable_optional_nested_message()->mutable_msg()->set_optional_string(
      "abc");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.mutable_optional_nested_message()->mutable_msg()->set_optional_string(
      "abc");
  proto.mutable_optional_nested_message()->set_optional_int64(1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  auto* nested = proto.add_repeated_nested_message();
  nested = proto.add_repeated_nested_message();
  nested->set_optional_int32(123);
  nested->mutable_msg();
  nested = proto.add_repeated_nested_message();
  nested->mutable_msg();
  nested->mutable_msg()->set_optional_string("abc");
  nested->set_optional_int64(1234);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // text format allows use of <> for messages.
  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message: < optional_int32: 123   >");

  // <> and {} must use same style for closing.
  EXPECT_PARSE_FAILURE("optional_nested_message: < optional_int32: 123   }");
  EXPECT_PARSE_FAILURE("optional_nested_message: { optional_int32: 123   >");

  // colon after identifier is optional for messages.
  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message < optional_int32: 123   >");

  EXPECT_PARSE_SUCCESS("optional_nested_message { optional_int32: 123 }",
                       "optional_nested_message{ optional_int32: 123   }  ");

  // Proto supports [] for list values as well.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 }"),
      "repeated_nested_message: [ { }, { optional_int32: 123  } ]");

  // Colon after repeated_nested_message is optional.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 }"),
      "repeated_nested_message [ { }, { optional_int32: 123  } ]");

  // Using the list format a:[..] twice, like a:[..] a:[..] joins the two
  // arrays.
  EXPECT_PARSE_SUCCESS(
      ("repeated_nested_message { } "
       "repeated_nested_message { optional_int32: 123 } "
       "repeated_nested_message { optional_int32: 456 }"),
      ("repeated_nested_message [ { }, { optional_int32: 123  } ]"
       "repeated_nested_message [ { optional_int32: 456  } ]"));

  // Parse errors on nested messages.
  EXPECT_PARSE_FAILURE("optional_nested_message: {optional_int32: 'abc' }");

  // Optional_nested_message appearing twice is an error.
  EXPECT_PARSE_FAILURE(
      ("optional_nested_message { optional_int32: 123 } "
       "optional_nested_message { optional_int64: 456 }"));
}

TEST(CreateProtoDebugStringLibTest, RecursiveMessage) {
  NestedTestAllTypes proto;

  NestedTestAllTypes* cur = &proto;
  for (int depth = 0; depth < 20; ++depth) {
    cur->mutable_payload()->set_optional_int32(1000 + depth);
    cur = cur->mutable_child();
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();
}

template <typename T>
T ParseProto(const string& value_text_proto) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("value_text_proto: \"" + value_text_proto + "\"");
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc mht_1(mht_1_v, 506, "", "./tensorflow/tools/proto_text/gen_proto_text_functions_lib_test.cc", "ParseProto");

  T value;
  EXPECT_TRUE(protobuf::TextFormat::ParseFromString(value_text_proto, &value))
      << value_text_proto;
  return value;
}

TestAllTypes::NestedMessage ParseNestedMessage(const string& value_text_proto) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("value_text_proto: \"" + value_text_proto + "\"");
   MHTracer_DTPStensorflowPStoolsPSproto_textPSgen_proto_text_functions_lib_testDTcc mht_2(mht_2_v, 517, "", "./tensorflow/tools/proto_text/gen_proto_text_functions_lib_test.cc", "ParseNestedMessage");

  return ParseProto<TestAllTypes::NestedMessage>(value_text_proto);
}

TEST(CreateProtoDebugStringLibTest, Map) {
  TestAllTypes proto;

  std::vector<TestAllTypes::NestedMessage> msg_values;
  msg_values.push_back(ParseNestedMessage("optional_int32: 345"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 123"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 234"));
  msg_values.push_back(ParseNestedMessage("optional_int32: 0"));

  // string->message map
  proto.Clear();
  {
    auto& map = *proto.mutable_map_string_to_message();
    map["def"] = msg_values[0];
    map["abc"] = msg_values[1];
    map["cde"] = msg_values[2];
    map[""] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int32->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int32_to_message();
    map[20] = msg_values[0];
    map[10] = msg_values[1];
    map[15] = msg_values[2];
    map[0] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int64->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_message();
    map[20] = msg_values[0];
    map[10] = msg_values[1];
    map[15] = msg_values[2];
    map[0] = msg_values[3];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // bool->message map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_message();
    map[true] = msg_values[0];
    map[false] = msg_values[1];
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // string->int64 map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_string_to_int64();
    map["def"] = 0;
    map["abc"] = std::numeric_limits<protobuf_int64>::max();
    map[""] = 20;
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // int64->string map.
  proto.Clear();
  {
    auto& map = *proto.mutable_map_int64_to_string();
    map[0] = "def";
    map[std::numeric_limits<protobuf_int64>::max()] = "";
    map[20] = "abc";
  }
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Test a map with the same key multiple times.
  EXPECT_PARSE_SUCCESS(("map_string_to_int64 { key: \"abc\" value: 5 } "
                        "map_string_to_int64 { key: \"def\" value: 2 } "
                        "map_string_to_int64 { key: \"ghi\" value: 4 }"),
                       ("map_string_to_int64: { key: 'abc' value: 1 } "
                        "map_string_to_int64: { key: 'def' value: 2 } "
                        "map_string_to_int64: { key: 'ghi' value: 3 } "
                        "map_string_to_int64: { key: 'ghi' value: 4 } "
                        "map_string_to_int64: { key: 'abc' value: 5 } "));
}

TEST(CreateProtoDebugStringLibTest, Enums) {
  TestAllTypes proto;

  proto.Clear();
  proto.set_optional_nested_enum(TestAllTypes::ZERO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.set_optional_nested_enum(TestAllTypes::FOO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.add_repeated_nested_enum(TestAllTypes::FOO);
  proto.add_repeated_nested_enum(TestAllTypes::ZERO);
  proto.add_repeated_nested_enum(TestAllTypes::BAR);
  proto.add_repeated_nested_enum(TestAllTypes::NEG);
  proto.add_repeated_nested_enum(TestAllTypes::ZERO);
  proto.set_optional_foreign_enum(ForeignEnum::FOREIGN_BAR);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Parsing from numbers works as well.
  EXPECT_PARSE_SUCCESS(
      "optional_nested_enum: BAR "   // 2
      "repeated_nested_enum: BAR "   // 2
      "repeated_nested_enum: ZERO "  // 0
      "repeated_nested_enum: FOO",   // 1
      ("repeated_nested_enum: 2 "
       "repeated_nested_enum: 0 "
       "optional_nested_enum: 2 "
       "repeated_nested_enum: 1"));

  EXPECT_PARSE_SUCCESS("", "optional_nested_enum: -0");
  // TODO(amauryfa): restore the line below when protobuf::TextFormat also
  // supports unknown enum values.
  // EXPECT_PARSE_SUCCESS("optional_nested_enum: 6", "optional_nested_enum: 6");
  EXPECT_PARSE_FAILURE("optional_nested_enum: 2147483648");  // > INT32_MAX
  EXPECT_PARSE_FAILURE("optional_nested_enum: BARNONE");
  EXPECT_PARSE_FAILURE("optional_nested_enum: 'BAR'");
  EXPECT_PARSE_FAILURE("optional_nested_enum: \"BAR\" ");

  EXPECT_EQ(string("BAR"),
            string(EnumName_TestAllTypes_NestedEnum(TestAllTypes::BAR)));
  // out of range - returns empty string (see NameOfEnum in proto library).
  EXPECT_EQ(string(""), string(EnumName_TestAllTypes_NestedEnum(
                            static_cast<TestAllTypes_NestedEnum>(123))));
}

TEST(CreateProtoDebugStringLibTest, Oneof) {
  TestAllTypes proto;

  proto.Clear();
  proto.set_oneof_string("abc");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Empty oneof_string is printed, as the setting of the value in the oneof is
  // meaningful.
  proto.Clear();
  proto.set_oneof_string("");
  EXPECT_TEXT_TRANSFORMS_MATCH();

  proto.Clear();
  proto.set_oneof_string("abc");
  proto.set_oneof_uint32(123);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Zero uint32 is printed, as the setting of the value in the oneof is
  // meaningful.
  proto.Clear();
  proto.set_oneof_uint32(0);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Zero enum value is meaningful.
  proto.Clear();
  proto.set_oneof_enum(TestAllTypes::ZERO);
  EXPECT_TEXT_TRANSFORMS_MATCH();

  // Parse a text format that lists multiple members of the oneof.
  EXPECT_PARSE_FAILURE("oneof_string: \"abc\" oneof_uint32: 13 ");
  EXPECT_PARSE_FAILURE("oneof_string: \"abc\" oneof_string: \"def\" ");
}

TEST(CreateProtoDebugStringLibTest, Comments) {
  TestAllTypes proto;

  EXPECT_PARSE_SUCCESS("optional_int64: 123 optional_string: \"#text\"",
                       ("#leading comment \n"
                        "optional_int64# comment\n"
                        ":# comment\n"
                        "123# comment\n"
                        "optional_string  # comment\n"
                        ":   # comment\n"
                        "\"#text\"#comment####\n"));

  EXPECT_PARSE_FAILURE("optional_int64:// not a valid comment\n123");
  EXPECT_PARSE_FAILURE("optional_int64:/* not a valid comment */\n123");
}

}  // namespace
}  // namespace test
}  // namespace tensorflow
