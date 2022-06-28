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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/framework/python_op_gen.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void ExpectHasSubstr(const string& s, const string& expected) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("s: \"" + s + "\"");
   mht_0_v.push_back("expected: \"" + expected + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/python/framework/python_op_gen_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'Generated ops does not contain '" << expected << "'";
}

void ExpectDoesNotHaveSubstr(const string& s, const string& expected) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + s + "\"");
   mht_1_v.push_back("expected: \"" + expected + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/python/framework/python_op_gen_test.cc", "ExpectDoesNotHaveSubstr");

  EXPECT_FALSE(absl::StrContains(s, expected))
      << "'Generated ops contains '" << expected << "'";
}

void ExpectSubstrOrder(const string& s, const string& before,
                       const string& after) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   mht_2_v.push_back("before: \"" + before + "\"");
   mht_2_v.push_back("after: \"" + after + "\"");
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_op_gen_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/python/framework/python_op_gen_test.cc", "ExpectSubstrOrder");

  int before_pos = s.find(before);
  int after_pos = s.find(after);
  ASSERT_NE(std::string::npos, before_pos);
  ASSERT_NE(std::string::npos, after_pos);
  EXPECT_LT(before_pos, after_pos) << before << "' is not before '" << after;
}

TEST(PythonOpGen, TypeAnnotateAllOps) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);

  ApiDefMap api_def_map(ops);

  std::unordered_set<string> type_annotate_ops;
  for (const auto& op : ops.op()) {
    type_annotate_ops.insert(op.name());
  }

  string code = GetPythonOps(ops, api_def_map, {}, "", type_annotate_ops);

  const string all_types =
      ", _dtypes.BFloat16, _dtypes.Bool, _dtypes.Complex128, "
      "_dtypes.Complex64, "
      "_dtypes.Float16, _dtypes.Float32, _dtypes.Float64, _dtypes.Half, "
      "_dtypes.Int16, "
      "_dtypes.Int32, _dtypes.Int64, _dtypes.Int8, _dtypes.QInt16, "
      "_dtypes.QInt32, "
      "_dtypes.QInt8, _dtypes.QUInt16, _dtypes.QUInt8, _dtypes.Resource, "
      "_dtypes.String, "
      "_dtypes.UInt16, _dtypes.UInt32, _dtypes.UInt64, _dtypes.UInt8, "
      "_dtypes.Variant)";

  const string fake_param_typevar =
      "TV_FakeParam_dtype = TypeVar(\"TV_FakeParam_dtype\"" + all_types;
  const string fake_param =
      "def fake_param_eager_fallback(dtype: TV_FakeParam_dtype, shape, name, "
      "ctx) -> _ops.Tensor[TV_FakeParam_dtype]:";
  const string fake_param_fallback =
      "def fake_param_eager_fallback(dtype: TV_FakeParam_dtype, shape, name, "
      "ctx) -> _ops.Tensor[TV_FakeParam_dtype]:";

  ExpectHasSubstr(code, fake_param_typevar);
  ExpectHasSubstr(code, fake_param);
  ExpectHasSubstr(code, fake_param_fallback);

  const string to_bool_typevar =
      "TV_ToBool_T = TypeVar(\"TV_ToBool_T\"" + all_types;
  const string to_bool_ =
      "def to_bool(input: _ops.Tensor[TV_ToBool_T], name=None) -> "
      "_ops.Tensor[_dtypes.Bool]:";
  const string to_bool_fallback =
      "def to_bool_eager_fallback(input: _ops.Tensor[TV_ToBool_T], name, ctx) "
      "-> _ops.Tensor[_dtypes.Bool]:";

  ExpectHasSubstr(code, to_bool_typevar);
  ExpectHasSubstr(code, to_bool_);
  ExpectHasSubstr(code, to_bool_fallback);
}

TEST(PythonOpGen, TypeAnnotateSingleTypeTensor) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Bar"
    input_arg {
      name: "x"
      type: DT_STRING
    }
    input_arg {
      name: "y"
      type: DT_QINT8
    }
    output_arg {
      name: "output"
      type: DT_BOOL
    }
    summary: "Summary for op Bar."
    description: "Description for op Bar."
  }
  )";

  std::unordered_set<string> type_annotate_ops{"Bar"};

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_bar =
      "def bar(x: _ops.Tensor[_dtypes.String], y: _ops.Tensor[_dtypes.QInt8], "
      "name=None) -> _ops.Tensor[_dtypes.Bool]:";
  ExpectHasSubstr(code, typed_bar);

  const string untyped_bar = "def bar(x, y, name=None):";
  ExpectDoesNotHaveSubstr(code, untyped_bar);
}

TEST(PythonOpGen, TypeAnnotateMultiTypeTensor) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Foo"
    input_arg {
      name: "x"
      type_attr: "T"
    }
    input_arg {
      name: "y"
      type_attr: "T2"
    }
    output_arg {
      name: "output"
      type_attr: "T"
    }
    attr {
      name: "T"
      type: "type"
      allowed_values {
        list {
          type: DT_UINT8
          type: DT_INT8
        }
      }
    }
    attr {
      name: "T2"
      type: "type"
      allowed_values {
        list {
          type: DT_STRING
          type: DT_FLOAT
          type: DT_DOUBLE
        }
      }
    }
    summary: "Summary for op Foo."
    description: "Description for op Foo."
  }
  )";

  std::unordered_set<string> type_annotate_ops{
      "Foo",
  };

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_foo =
      "def foo(x: _ops.Tensor[TV_Foo_T], y: _ops.Tensor[TV_Foo_T2], name=None) "
      "-> _ops.Tensor[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo);
}

TEST(PythonOpGen, GenerateCorrectTypeVars) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Foo"
    input_arg {
      name: "x"
      type_attr: "T"
    }
    input_arg {
      name: "y"
      type_attr: "T2"
    }
    output_arg {
      name: "output"
      type_attr: "T"
    }
    attr {
      name: "T"
      type: "type"
      allowed_values {
        list {
          type: DT_UINT8
          type: DT_INT8
        }
      }
    }
    attr {
      name: "T2"
      type: "type"
      allowed_values {
        list {
          type: DT_STRING
          type: DT_FLOAT
          type: DT_DOUBLE
        }
      }
    }
    summary: "Summary for op Foo."
    description: "Description for op Foo."
  }
  )";

  std::unordered_set<string> type_annotate_ops{
      "Foo",
  };

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typevars_foo = R"(
TV_Foo_T = TypeVar("TV_Foo_T", _dtypes.Int8, _dtypes.UInt8)
TV_Foo_T2 = TypeVar("TV_Foo_T2", _dtypes.Float32, _dtypes.Float64, _dtypes.String)
)";

  ExpectHasSubstr(code, typevars_foo);
}

TEST(PythonOpGen, TypeAnnotateFallback) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Foo"
    input_arg {
      name: "x"
      type_attr: "T"
    }
    input_arg {
      name: "y"
      type_attr: "T2"
    }
    output_arg {
      name: "output"
      type_attr: "T"
    }
    attr {
      name: "T"
      type: "type"
      allowed_values {
        list {
          type: DT_UINT8
          type: DT_INT8
        }
      }
    }
    attr {
      name: "T2"
      type: "type"
      allowed_values {
        list {
          type: DT_STRING
          type: DT_FLOAT
          type: DT_DOUBLE
        }
      }
    }
    summary: "Summary for op Foo."
    description: "Description for op Foo."
  }
  )";

  std::unordered_set<string> type_annotate_ops{
      "Foo",
  };

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typed_foo_fallback =
      "def foo_eager_fallback(x: _ops.Tensor[TV_Foo_T], y: "
      "_ops.Tensor[TV_Foo_T2], name, ctx) -> _ops.Tensor[TV_Foo_T]:";
  ExpectHasSubstr(code, typed_foo_fallback);
}

TEST(PythonOpGen, GenerateTypeVarAboveOp) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Foo"
    input_arg {
      name: "x"
      type_attr: "T"
    }
    input_arg {
      name: "y"
      type_attr: "T2"
    }
    output_arg {
      name: "output"
      type_attr: "T"
    }
    attr {
      name: "T"
      type: "type"
      allowed_values {
        list {
          type: DT_UINT8
          type: DT_INT8
        }
      }
    }
    attr {
      name: "T2"
      type: "type"
      allowed_values {
        list {
          type: DT_STRING
          type: DT_FLOAT
          type: DT_DOUBLE
        }
      }
    }
    summary: "Summary for op Foo."
    description: "Description for op Foo."
  }
  )";

  std::unordered_set<string> type_annotate_ops{
      "Foo",
  };

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string typevar_foo = "TV_Foo_";
  const string def_foo = "def foo";
  ExpectSubstrOrder(code, typevar_foo, def_foo);
}

TEST(PythonOpGen, TypeAnnotateDefaultParams) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "FooBar"
    input_arg {
      name: "x"
      type: DT_FLOAT
    }
    output_arg {
      name: "output"
      type: DT_BOOL
    }
    attr {
      name: "t"
      type: "type"
      allowed_values {
        list {
          type: DT_HALF
          type: DT_INT8
        }
      }
    }
    attr {
      name: "var1"
      type: "bool"
      default_value {
        b: false
      }
    }
    attr {
      name: "var2"
      type: "int"
      default_value {
        i: 0
      }
    }
    summary: "Summary for op FooBar."
    description: "Description for op FooBar."
  }
  )";

  std::unordered_set<string> type_annotate_ops{
      "FooBar",
  };

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string params =
      "def foo_bar(x: _ops.Tensor[_dtypes.Float32], t: TV_FooBar_t, "
      "var1:bool=False, var2:int=0, name=None)";
  const string params_fallback =
      "def foo_bar_eager_fallback(x: _ops.Tensor[_dtypes.Float32], t: "
      "TV_FooBar_t, var1: bool, var2: int, name, ctx)";
  ExpectHasSubstr(code, params);
  ExpectHasSubstr(code, params_fallback);
}

TEST(PythonOpGen, NoTypingSequenceTensors) {
  constexpr char kBaseOpDef[] = R"(
  op {
    name: "Baz"
    input_arg {
      name: "inputs"
      number_attr: "N"
      type_list_attr: "T"
    }
    output_arg {
      name: "output1"
      type: DT_BOOL
    }
    output_arg {
      name: "output2"
      type: DT_BOOL
    }
    attr {
      name: "T"
      type: "bool"
    }
    attr {
      name: "N"
      type: "int"
    }
    summary: "Summary for op Baz."
    description: "Description for op Baz."
  }
  )";

  std::unordered_set<string> type_annotate_ops{"Baz"};

  OpList op_defs;
  OpRegistry::Global()->Export(false, &op_defs);
  protobuf::TextFormat::ParseFromString(kBaseOpDef, &op_defs);
  ApiDefMap api_def_map(op_defs);

  string code = GetPythonOps(op_defs, api_def_map, {}, "", type_annotate_ops);

  const string baz_def_line = "def baz(inputs, name=None):";

  ExpectHasSubstr(code, baz_def_line);
}

}  // namespace
}  // namespace tensorflow
