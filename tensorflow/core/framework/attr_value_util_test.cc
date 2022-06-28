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
class MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc() {
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

#include "tensorflow/core/framework/attr_value_util.h"

#include <numeric>
#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// A few helpers to construct AttrValue protos.
template <typename T>
AttrValue V(T value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/framework/attr_value_util_test.cc", "V");

  AttrValue ret;
  SetAttrValue(value, &ret);
  return ret;
}

AttrValue P(const string& p) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("p: \"" + p + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/framework/attr_value_util_test.cc", "P");

  AttrValue ret;
  ret.set_placeholder(p);
  return ret;
}

AttrValue F(const string& name,
            std::vector<std::pair<string, AttrValue>> pairs) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/framework/attr_value_util_test.cc", "F");

  AttrValue ret;
  ret.mutable_func()->set_name(name);
  ret.mutable_func()->mutable_attr()->insert(pairs.begin(), pairs.end());
  return ret;
}

AttrValue Fs(
    std::vector<std::pair<string, std::vector<std::pair<string, AttrValue>>>>
        funcs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_3(mht_3_v, 232, "", "./tensorflow/core/framework/attr_value_util_test.cc", "Fs");

  AttrValue ret;
  for (const auto& func : funcs) {
    NameAttrList* entry = ret.mutable_list()->add_func();
    entry->set_name(func.first);
    entry->mutable_attr()->insert(func.second.begin(), func.second.end());
  }
  return ret;
}

TEST(AttrValueUtil, HasType) {
  // OK
  EXPECT_TRUE(AttrValueHasType(V(123), "int").ok());
  EXPECT_TRUE(AttrValueHasType(V(1.2), "float").ok());
  EXPECT_TRUE(AttrValueHasType(V(DT_FLOAT), "type").ok());
  EXPECT_TRUE(AttrValueHasType(F("f", {}), "func").ok());
  EXPECT_TRUE(AttrValueHasType(Fs({{"f", {}}, {"g", {}}}), "list(func)").ok());

  // not OK.
  EXPECT_FALSE(AttrValueHasType(V(123), "func").ok());
  EXPECT_FALSE(AttrValueHasType(V(1.2), "int").ok());
  EXPECT_FALSE(AttrValueHasType(V(DT_FLOAT), "shape").ok());
  EXPECT_FALSE(AttrValueHasType(F("f", {}), "string").ok());
  EXPECT_FALSE(AttrValueHasType(P("T"), "float").ok());
  EXPECT_FALSE(AttrValueHasType(V(static_cast<DataType>(1000)), "type").ok());
  std::vector<DataType> list_type({static_cast<DataType>(1000)});
  EXPECT_FALSE(AttrValueHasType(V(list_type), "list(type)").ok());
}

SubstituteFunc ReplaceTWith(const AttrValue& val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/framework/attr_value_util_test.cc", "ReplaceTWith");

  return [val](const string& placeholder, AttrValue* target) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("placeholder: \"" + placeholder + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/framework/attr_value_util_test.cc", "lambda");

    if (placeholder == "T") {
      *target = val;
      return true;
    } else {
      return false;
    }
  };
}

TEST(AttrValueUtil, Basic) {
  auto v = F("MatMul", {{"dtype", P("T")},
                        {"transpose_a", V(false)},
                        {"transpose_b", V(true)},
                        {"use_cublas", V(true)}});
  TF_EXPECT_OK(AttrValueHasType(v, "func"));
  EXPECT_TRUE(HasPlaceHolder(v));

  EXPECT_EQ(
      SummarizeAttrValue(v),
      "MatMul[dtype=$T, transpose_a=false, transpose_b=true, use_cublas=true]");

  SubstitutePlaceholders(ReplaceTWith(V(DT_FLOAT)), &v);
  EXPECT_TRUE(!HasPlaceHolder(v));
  EXPECT_EQ(SummarizeAttrValue(v),
            "MatMul[dtype=DT_FLOAT, transpose_a=false, transpose_b=true, "
            "use_cublas=true]");
}

TEST(AttrValueUtil, Shaped) {
  auto v =
      F("OpRequiresShape", {{"shape_full", V(TensorShape({1, 0}))},
                            {"shape_part", V(PartialTensorShape({-1, 1, 0}))}});
  TF_EXPECT_OK(AttrValueHasType(v, "func"));
  EXPECT_FALSE(HasPlaceHolder(v));

  EXPECT_EQ(SummarizeAttrValue(v),
            "OpRequiresShape[shape_full=[1,0], shape_part=[?,1,0]]");
}

TEST(AttrValueUtil, DeepAttr) {
  auto v = Fs({{"f", {{"T", P("T")}}}, {"g", {{"T", P("T")}}}});
  TF_EXPECT_OK(AttrValueHasType(v, "list(func)"));
  EXPECT_TRUE(HasPlaceHolder(v));

  for (int i = 0; i < 3; ++i) {
    v = F("f", {{"T", P("T")}, {"F", v}});
    EXPECT_TRUE(HasPlaceHolder(v));
  }
  EXPECT_EQ(SummarizeAttrValue(v),
            "f[F=f[F=f[F=[f[T=$T], g[T=$T]], T=$T], T=$T], T=$T]");

  SubstitutePlaceholders(ReplaceTWith(F("x", {})), &v);
  EXPECT_TRUE(!HasPlaceHolder(v));
  EXPECT_EQ(SummarizeAttrValue(v),
            "f[F=f[F=f[F=[f[T=x[]], g[T=x[]]], T=x[]], T=x[]], T=x[]]");
}

TEST(AttrValueUtil, SummarizeAttrValueDoesNotElideShortStrings) {
  AttrValue attr_value;
  SetAttrValue(string(40, '-'), &attr_value);
  EXPECT_EQ(strings::StrCat("\"", string(40, '-'), "\""),
            SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueElidesLongStrings) {
  AttrValue attr_value;
  SetAttrValue(string(80, '-'), &attr_value);
  EXPECT_EQ("\"----------...----------\"", SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueDoesNotElideShortLists) {
  std::vector<int> alist(10);
  std::iota(alist.begin(), alist.end(), 0);

  AttrValue attr_value;
  SetAttrValue(alist, &attr_value);
  EXPECT_EQ("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueElidesLongLists) {
  std::vector<int> alist(110);
  std::iota(alist.begin(), alist.end(), 0);

  AttrValue attr_value;
  SetAttrValue(alist, &attr_value);
  EXPECT_EQ("[0, 1, 2, 3, 4, 2587181569776227444, 105, 106, 107, 108, 109]",
            SummarizeAttrValue(attr_value));
}

AttrValue FromText(const string& text) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_6(mht_6_v, 363, "", "./tensorflow/core/framework/attr_value_util_test.cc", "FromText");

  AttrValue attr;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &attr));
  return attr;
}

void ExpectDifferent(const AttrValue& a1, const AttrValue& a2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_util_testDTcc mht_7(mht_7_v, 372, "", "./tensorflow/core/framework/attr_value_util_test.cc", "ExpectDifferent");

  EXPECT_FALSE(AreAttrValuesEqual(a1, a2));
  EXPECT_FALSE(AreAttrValuesEqual(a2, a1));
  EXPECT_NE(AttrValueHash(a1), AttrValueHash(a2));
}

TEST(AttrValueEquality, StringAndFuncTensors) {
  AttrValue a = FromText(R"(
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 2
          }
        }
        string_val: 'reader_dataset_ops_test/tmphtXHks/text_line.0.txt'
        string_val: 'reader_dataset_ops_test/tmphtXHks/text_line.1.txt'
      })");
  EXPECT_TRUE(AreAttrValuesEqual(a, a));
  EXPECT_EQ(AttrValueHash(a), AttrValueHash(a));

  AttrValue b = a;
  (*b.mutable_tensor()->mutable_string_val(0))[3] = '1';
  ExpectDifferent(a, b);

  AttrValue c1;
  c1.mutable_func()->set_name("func_name");
  (*c1.mutable_func()->mutable_attr())["attr1"] = a;
  (*c1.mutable_func()->mutable_attr())["attr2"] = b;
  EXPECT_TRUE(AreAttrValuesEqual(c1, c1));
  EXPECT_EQ(AttrValueHash(c1), AttrValueHash(c1));

  ExpectDifferent(c1, a);

  AttrValue c2 = c1;
  c2.mutable_func()->set_name("func_name2");
  ExpectDifferent(c1, c2);

  c2 = c1;
  (*c2.mutable_func()->mutable_attr())["attr3"] = b;
  ExpectDifferent(c1, c2);

  c2 = c1;
  (*c2.mutable_func()->mutable_attr())["attr2"] = a;
  ExpectDifferent(c1, c2);

  c2 = c1;
  c2.mutable_func()->mutable_attr()->erase("attr2");
  ExpectDifferent(c1, c2);
}

TEST(AttrValueEquality, GiantTensors) {
  AttrValue tensor = FromText(R"(
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
        }
        int_val: 0
      })");
  EXPECT_TRUE(AreAttrValuesEqual(tensor, tensor));
}

}  // namespace tensorflow
