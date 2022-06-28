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
class MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc() {
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

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb_text.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb_text.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {

// Do not construct large tensors to compute their hash or compare for equality.
constexpr int kMaxAttrValueTensorByteSize = 32 * 1024 * 1024;  // 32mb

// Limit nesting of tensors to 100 deep to prevent memory overflow.
constexpr int kMaxTensorNestDepth = 100;

// Return the size of the tensor represented by this TensorProto. If shape is
// not fully defined return -1.
int64_t TensorByteSize(const TensorProto& t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/framework/attr_value_util.cc", "TensorByteSize");

  // num_elements returns -1 if shape is not fully defined.
  int64_t num_elems = PartialTensorShape(t.tensor_shape()).num_elements();
  return num_elems < 0 ? -1 : num_elems * DataTypeSize(t.dtype());
}

// Compute TensorProto hash by creating a Tensor, serializing it as tensor
// content, and computing a hash of it's string representation. This is unsafe
// operation, because large tensors can be represented as TensorProto, but can't
// be serialized to tensor content.
uint64 TensorProtoHash(const TensorProto& tp) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/framework/attr_value_util.cc", "TensorProtoHash");

  Tensor tensor(tp.dtype());
  bool success = tensor.FromProto(tp);
  DCHECK(success);
  TensorProto p;
  tensor.AsProtoTensorContent(&p);
  return DeterministicProtoHash64(p);
}

// Do not create large tensors in memory, compute hash based on TensorProto
// string representation. Tensors with identical content potentially can have a
// different hash code if they are defined with different TensorProto
// representations.
uint64 FastTensorProtoHash(const TensorProto& tp) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/framework/attr_value_util.cc", "FastTensorProtoHash");

  if (TensorByteSize(tp) > kMaxAttrValueTensorByteSize) {
    return DeterministicProtoHash64(tp);
  } else {
    return TensorProtoHash(tp);
  }
}

bool AreTensorProtosEqual(const TensorProto& lhs, const TensorProto& rhs,
                          bool allow_false_negatives) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/framework/attr_value_util.cc", "AreTensorProtosEqual");

  // A small TensorProto can expand into a giant Tensor.  So we avoid
  // conversion to an actual Tensor if we can quickly rule out equality
  // by comparing the Tensor size since different sized Tensors are definitely
  // different.
  const int64_t lhs_tensor_bytes = TensorByteSize(lhs);
  const int64_t rhs_tensor_bytes = TensorByteSize(rhs);
  if (lhs_tensor_bytes != rhs_tensor_bytes) {
    return false;
  }

  // If the TensorProto representation expands into a much bigger Tensor,
  // we have a fast-path that first compares the protos.
  const int64_t lhs_proto_bytes = lhs.ByteSizeLong();
  const bool large_expansion =
      (lhs_proto_bytes < 512 && lhs_tensor_bytes > 4096);

  // If the tensor is very large, we'll only compare the proto representation if
  // false negatives are allowed. This may miss some equivalent tensors whose
  // actual tensor values are the same but which are described by different
  // TensorProtos. This avoids construction of large protos in memory.
  const bool only_compare_proto =
      (allow_false_negatives && lhs_tensor_bytes > kMaxAttrValueTensorByteSize);
  if (large_expansion || only_compare_proto) {
    if (AreSerializedProtosEqual(lhs, rhs))
      return true;
    else if (only_compare_proto)
      return false;
  }

  // Finally, compare them by constructing Tensors and serializing them back.
  // There are multiple equivalent representations of attr values containing
  // TensorProtos. Comparing Tensor objects is pretty tricky. This is unsafe
  // operation, because large tensors can be represented as TensorProto, but
  // can't be serialized to tensor content.
  Tensor lhs_t(lhs.dtype());
  bool success = lhs_t.FromProto(lhs);
  if (!success) {
    return false;
  }

  Tensor rhs_t(rhs.dtype());
  success = rhs_t.FromProto(rhs);
  if (!success) {
    return false;
  }

  TensorProto lhs_tp;
  lhs_t.AsProtoTensorContent(&lhs_tp);

  TensorProto rhs_tp;
  rhs_t.AsProtoTensorContent(&rhs_tp);

  return AreSerializedProtosEqual(lhs_tp, rhs_tp);
}

using TensorProtoHasher = std::function<uint64(const TensorProto&)>;

uint64 AttrValueHash(const AttrValue& a, const TensorProtoHasher& tensor_hash) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_4(mht_4_v, 318, "", "./tensorflow/core/framework/attr_value_util.cc", "AttrValueHash");

  if (a.has_tensor()) return tensor_hash(a.tensor());

  if (a.has_func()) {
    const NameAttrList& func = a.func();
    uint64 h = Hash64(func.name());
    std::map<string, AttrValue> map(func.attr().begin(), func.attr().end());
    for (const auto& pair : map) {
      h = Hash64(pair.first.data(), pair.first.size(), h);
      h = Hash64Combine(AttrValueHash(pair.second, tensor_hash), h);
    }
    return h;
  }

  // If `a` is not a tensor or func, get a hash of serialized string.
  return DeterministicProtoHash64(a);
}

string SummarizeString(const string& str) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/framework/attr_value_util.cc", "SummarizeString");

  string escaped = absl::CEscape(str);

  // If the string is long, replace the middle with ellipses.
  constexpr int kMaxStringSummarySize = 80;
  if (escaped.size() >= kMaxStringSummarySize) {
    StringPiece prefix(escaped);
    StringPiece suffix = prefix;
    prefix.remove_suffix(escaped.size() - 10);
    suffix.remove_prefix(escaped.size() - 10);
    return strings::StrCat("\"", prefix, "...", suffix, "\"");
  } else {
    return strings::StrCat("\"", escaped, "\"");
  }
}

string SummarizeTensor(const TensorProto& tensor_proto) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_6(mht_6_v, 359, "", "./tensorflow/core/framework/attr_value_util.cc", "SummarizeTensor");

  Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return strings::StrCat(
        "<Invalid TensorProto: ", tensor_proto.ShortDebugString(), ">");
  }
  return t.DebugString();
}

string SummarizeFunc(const NameAttrList& func) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_7(mht_7_v, 371, "", "./tensorflow/core/framework/attr_value_util.cc", "SummarizeFunc");

  std::vector<string> entries;
  for (const auto& p : func.attr()) {
    entries.push_back(
        strings::StrCat(p.first, "=", SummarizeAttrValue(p.second)));
  }
  std::sort(entries.begin(), entries.end());
  return strings::StrCat(func.name(), "[", absl::StrJoin(entries, ", "), "]");
}

bool ParseAttrValueHelper_TensorNestsUnderLimit(int limit, string to_parse) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("to_parse: \"" + to_parse + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_8(mht_8_v, 385, "", "./tensorflow/core/framework/attr_value_util.cc", "ParseAttrValueHelper_TensorNestsUnderLimit");

  int nests = 0;
  int maxed_out = to_parse.length();
  int open_curly = to_parse.find('{');
  int open_bracket = to_parse.find('<');
  int close_curly = to_parse.find('}');
  int close_bracket = to_parse.find('>');
  if (open_curly == -1) {
    open_curly = maxed_out;
  }
  if (open_bracket == -1) {
    open_bracket = maxed_out;
  }
  int min = std::min(open_curly, open_bracket);
  do {
    if (open_curly == maxed_out && open_bracket == maxed_out) {
      return true;
    }
    if (min == open_curly) {
      nests += 1;
      open_curly = to_parse.find('{', open_curly + 1);
      if (open_curly == -1) {
        open_curly = maxed_out;
      }
    } else if (min == open_bracket) {
      nests += 1;
      open_bracket = to_parse.find('<', open_bracket + 1);
      if (open_bracket == -1) {
        open_bracket = maxed_out;
      }
    } else if (min == close_curly) {
      nests -= 1;
      close_curly = to_parse.find('}', close_curly + 1);
      if (close_curly == -1) {
        close_curly = maxed_out;
      }
    } else if (min == close_bracket) {
      nests -= 1;
      close_bracket = to_parse.find('>', close_bracket + 1);
      if (close_bracket == -1) {
        close_bracket = maxed_out;
      }
    }
    min = std::min({open_curly, open_bracket, close_curly, close_bracket});
  } while (nests < 100);
  return false;
}

}  // namespace

string SummarizeAttrValue(const AttrValue& attr_value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_9(mht_9_v, 438, "", "./tensorflow/core/framework/attr_value_util.cc", "SummarizeAttrValue");

  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return SummarizeString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PartialTensorShape::DebugString(attr_value.shape());
    case AttrValue::kTensor:
      return SummarizeTensor(attr_value.tensor());
    case AttrValue::kList: {
      std::vector<string> pieces;
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          pieces.push_back(SummarizeString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().i(i)));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().f(i)));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          pieces.push_back(attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          pieces.push_back(EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          pieces.push_back(
              TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          pieces.push_back(SummarizeTensor(attr_value.list().tensor(i)));
        }
      } else if (attr_value.list().func_size() > 0) {
        for (int i = 0; i < attr_value.list().func_size(); ++i) {
          pieces.push_back(SummarizeFunc(attr_value.list().func(i)));
        }
      }
      constexpr int kMaxListSummarySize = 15;
      if (pieces.size() >= kMaxListSummarySize) {
        pieces[5] = strings::StrCat(Fingerprint64(
            absl::StrJoin(pieces.begin() + 5, pieces.end() - 5, ",")));
        pieces.erase(pieces.begin() + 6, pieces.end() - 5);
      }
      return strings::StrCat("[", absl::StrJoin(pieces, ", "), "]");
    }
    case AttrValue::kFunc: {
      return SummarizeFunc(attr_value.func());
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("$", attr_value.placeholder());
    case AttrValue::VALUE_NOT_SET:
      return "<Unknown AttrValue type>";
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

Status AttrValueHasType(const AttrValue& attr_value, StringPiece type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_10(mht_10_v, 512, "", "./tensorflow/core/framework/attr_value_util.cc", "AttrValueHasType");

  int num_set = 0;

#define VALIDATE_FIELD(name, type_string, oneof_case)                         \
  do {                                                                        \
    if (attr_value.has_list()) {                                              \
      if (attr_value.list().name##_size() > 0) {                              \
        if (type != "list(" type_string ")") {                                \
          return errors::InvalidArgument(                                     \
              "AttrValue had value with type 'list(" type_string ")' when '", \
              type, "' expected");                                            \
        }                                                                     \
        ++num_set;                                                            \
      }                                                                       \
    } else if (attr_value.value_case() == AttrValue::oneof_case) {            \
      if (type != type_string) {                                              \
        return errors::InvalidArgument(                                       \
            "AttrValue had value with type '" type_string "' when '", type,   \
            "' expected");                                                    \
      }                                                                       \
      ++num_set;                                                              \
    }                                                                         \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == AttrValue::kPlaceholder) {
    return errors::InvalidArgument(
        "AttrValue had value with unexpected type 'placeholder'");
  }

  // If the attr type is 'list', we expect attr_value.has_list() to be
  // true.  However, proto3's attr_value.has_list() can be false when
  // set to an empty list for GraphDef versions <= 4. So we simply
  // check if has_list is false and some other field in attr_value is
  // set to flag the error.  This test can be made more strict once
  // support for GraphDef versions <= 4 is dropped.
  if (absl::StartsWith(type, "list(") && !attr_value.has_list()) {
    if (num_set) {
      return errors::InvalidArgument(
          "AttrValue missing value with expected type '", type, "'");
    } else {
      // Indicate that we have a list, but an empty one.
      ++num_set;
    }
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  if (num_set == 0 && !absl::StartsWith(type, "list(")) {
    return errors::InvalidArgument(
        "AttrValue missing value with expected type '", type, "'");
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!DataType_IsValid(attr_value.type())) {
      return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                     attr_value.type());
    }
    if (IsRefType(attr_value.type())) {
      return errors::InvalidArgument(
          "AttrValue must not have reference type value of ",
          DataTypeString(attr_value.type()));
    }
    if (attr_value.type() == DT_INVALID) {
      return errors::InvalidArgument("AttrValue has invalid DataType");
    }
  } else if (type == "list(type)") {
    for (auto as_int : attr_value.list().type()) {
      const DataType dtype = static_cast<DataType>(as_int);
      if (!DataType_IsValid(dtype)) {
        return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                       as_int);
      }
      if (IsRefType(dtype)) {
        return errors::InvalidArgument(
            "AttrValue must not have reference type value of ",
            DataTypeString(dtype));
      }
      if (dtype == DT_INVALID) {
        return errors::InvalidArgument("AttrValue contains invalid DataType");
      }
    }
  }

  return Status::OK();
}

bool ParseAttrValue(StringPiece type, StringPiece text, AttrValue* out) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_11(mht_11_v, 613, "", "./tensorflow/core/framework/attr_value_util.cc", "ParseAttrValue");

  // Parse type.
  string field_name;
  bool is_list = absl::ConsumePrefix(&type, "list(");
  if (absl::ConsumePrefix(&type, "string")) {
    field_name = "s";
  } else if (absl::ConsumePrefix(&type, "int")) {
    field_name = "i";
  } else if (absl::ConsumePrefix(&type, "float")) {
    field_name = "f";
  } else if (absl::ConsumePrefix(&type, "bool")) {
    field_name = "b";
  } else if (absl::ConsumePrefix(&type, "type")) {
    field_name = "type";
  } else if (absl::ConsumePrefix(&type, "shape")) {
    field_name = "shape";
  } else if (absl::ConsumePrefix(&type, "tensor")) {
    field_name = "tensor";
  } else if (absl::ConsumePrefix(&type, "func")) {
    field_name = "func";
  } else if (absl::ConsumePrefix(&type, "placeholder")) {
    field_name = "placeholder";
  } else {
    return false;
  }
  if (is_list && !absl::ConsumePrefix(&type, ")")) {
    return false;
  }

  // Construct a valid text proto message to parse.
  string to_parse;
  if (is_list) {
    // TextFormat parser considers "i: 7" to be the same as "i: [7]",
    // but we only want to allow list values with [].
    StringPiece cleaned = text;
    str_util::RemoveLeadingWhitespace(&cleaned);
    str_util::RemoveTrailingWhitespace(&cleaned);
    if (cleaned.size() < 2 || cleaned[0] != '[' ||
        cleaned[cleaned.size() - 1] != ']') {
      return false;
    }
    cleaned.remove_prefix(1);
    str_util::RemoveLeadingWhitespace(&cleaned);
    if (cleaned.size() == 1) {
      // User wrote "[]", so return empty list without invoking the TextFormat
      // parse which returns an error for "i: []".
      out->Clear();
      out->mutable_list();
      return true;
    }
    to_parse = strings::StrCat("list { ", field_name, ": ", text, " }");
  } else {
    to_parse = strings::StrCat(field_name, ": ", text);
  }
  if (field_name == "tensor") {
    if (!ParseAttrValueHelper_TensorNestsUnderLimit(kMaxTensorNestDepth,
                                                    to_parse)) {
      return false;
    }
  }
  return ProtoParseFromString(to_parse, out);
}

void SetAttrValue(const AttrValue& value, AttrValue* out) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_12(mht_12_v, 679, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");
 *out = value; }

#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) { out->set_##FIELD(value); }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, FIELD)                       \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) {                     \
    out->mutable_list()->Clear(); /* create list() even if value empty */ \
    for (const auto& v : value) {                                         \
      out->mutable_list()->add_##FIELD(v);                                \
    }                                                                     \
  }

#define DEFINE_SET_ATTR_VALUE_BOTH(ARG_TYPE, FIELD) \
  DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD)        \
  DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<ARG_TYPE>, FIELD)

DEFINE_SET_ATTR_VALUE_ONE(const string&, s)
DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<string>, s)
DEFINE_SET_ATTR_VALUE_BOTH(const char*, s)
DEFINE_SET_ATTR_VALUE_BOTH(int64_t, i)
DEFINE_SET_ATTR_VALUE_BOTH(int32_t, i)
DEFINE_SET_ATTR_VALUE_BOTH(float, f)
DEFINE_SET_ATTR_VALUE_BOTH(double, f)
DEFINE_SET_ATTR_VALUE_BOTH(bool, b)
DEFINE_SET_ATTR_VALUE_LIST(const std::vector<bool>&, b)
DEFINE_SET_ATTR_VALUE_LIST(std::initializer_list<bool>, b)
DEFINE_SET_ATTR_VALUE_BOTH(DataType, type)

void SetAttrValue(const tstring& value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(gtl::ArraySlice<tstring> value, AttrValue* out) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_13(mht_13_v, 715, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void SetAttrValue(StringPiece value, AttrValue* out) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_14(mht_14_v, 725, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->set_s(value.data(), value.size());
}

void SetAttrValue(const gtl::ArraySlice<StringPiece> value, AttrValue* out) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_15(mht_15_v, 732, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void MoveAttrValue(std::vector<string>&& value, AttrValue* out) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_16(mht_16_v, 742, "", "./tensorflow/core/framework/attr_value_util.cc", "MoveAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (auto& v : value) {
    out->mutable_list()->add_s(std::move(v));
  }
}

void SetAttrValue(const TensorShape& value, AttrValue* out) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_17(mht_17_v, 752, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const TensorShapeProto& value, AttrValue* out) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_18(mht_18_v, 759, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  *out->mutable_shape() = value;
}

void SetAttrValue(const PartialTensorShape& value, AttrValue* out) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_19(mht_19_v, 766, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const gtl::ArraySlice<TensorShape> value, AttrValue* out) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_20(mht_20_v, 773, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(gtl::ArraySlice<TensorShapeProto> value, AttrValue* out) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_21(mht_21_v, 783, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_shape() = v;
  }
}

void SetAttrValue(const gtl::ArraySlice<PartialTensorShape> value,
                  AttrValue* out) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_22(mht_22_v, 794, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(const Tensor& value, AttrValue* out) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_23(mht_23_v, 804, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  if (value.NumElements() > 1) {
    value.AsProtoTensorContent(out->mutable_tensor());
  } else {
    value.AsProtoField(out->mutable_tensor());
  }
}

void SetAttrValue(const gtl::ArraySlice<Tensor> value, AttrValue* out) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_24(mht_24_v, 815, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    if (v.NumElements() > 1) {
      v.AsProtoTensorContent(out->mutable_list()->add_tensor());
    } else {
      v.AsProtoField(out->mutable_list()->add_tensor());
    }
  }
}

void SetAttrValue(const TensorProto& value, AttrValue* out) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_25(mht_25_v, 829, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  *out->mutable_tensor() = value;
}

void SetAttrValue(const gtl::ArraySlice<TensorProto> value, AttrValue* out) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_26(mht_26_v, 836, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_tensor() = v;
  }
}

void SetAttrValue(const NameAttrList& value, AttrValue* out) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_27(mht_27_v, 846, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  *out->mutable_func() = value;
}

void SetAttrValue(gtl::ArraySlice<NameAttrList> value, AttrValue* out) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_28(mht_28_v, 853, "", "./tensorflow/core/framework/attr_value_util.cc", "SetAttrValue");

  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_func() = v;
  }
}

bool AreAttrValuesEqual(const AttrValue& a, const AttrValue& b,
                        bool allow_false_negatives) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_29(mht_29_v, 864, "", "./tensorflow/core/framework/attr_value_util.cc", "AreAttrValuesEqual");

  if (a.type() != b.type()) {
    return false;
  } else if (a.type() != DT_INVALID && b.type() != DT_INVALID) {
    return a.type() == b.type();
  }

  if (a.has_tensor() != b.has_tensor()) {
    return false;
  } else if (a.has_tensor() && b.has_tensor()) {
    return AreTensorProtosEqual(a.tensor(), b.tensor(), allow_false_negatives);
  }

  // `func` field contains a nested AttrValue. Compare such AttrValues
  // recursively.
  if (a.has_func() != b.has_func()) {
    return false;
  } else if (a.has_func() && b.has_func()) {
    const NameAttrList& af = a.func();
    const NameAttrList& bf = b.func();
    if (af.name() != bf.name()) return false;
    std::unordered_map<string, AttrValue> am(af.attr().begin(),
                                             af.attr().end());
    for (const auto& bm_pair : bf.attr()) {
      const auto& iter = am.find(bm_pair.first);
      if (iter == am.end()) return false;
      if (!AreAttrValuesEqual(iter->second, bm_pair.second,
                              allow_false_negatives))
        return false;
      am.erase(iter);
    }
    if (!am.empty()) return false;
    return true;
  }

  // All other fields in AttrValue have deterministic representations.
  // It is safe to compare their serialized strings.
  return AreSerializedProtosEqual(a, b);
}

uint64 AttrValueHash(const AttrValue& a) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_30(mht_30_v, 907, "", "./tensorflow/core/framework/attr_value_util.cc", "AttrValueHash");

  return AttrValueHash(a, TensorProtoHash);
}

uint64 FastAttrValueHash(const AttrValue& a) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_31(mht_31_v, 914, "", "./tensorflow/core/framework/attr_value_util.cc", "FastAttrValueHash");

  return AttrValueHash(a, FastTensorProtoHash);
}

bool HasPlaceHolder(const AttrValue& val) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_32(mht_32_v, 921, "", "./tensorflow/core/framework/attr_value_util.cc", "HasPlaceHolder");

  switch (val.value_case()) {
    case AttrValue::kList: {
      for (const NameAttrList& func : val.list().func()) {
        for (const auto& p : func.attr()) {
          if (HasPlaceHolder(p.second)) {
            return true;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (const auto& p : val.func().attr()) {
        if (HasPlaceHolder(p.second)) {
          return true;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return true;
    default:
      break;
  }
  return false;
}

bool SubstitutePlaceholders(const SubstituteFunc& substitute,
                            AttrValue* value) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSattr_value_utilDTcc mht_33(mht_33_v, 952, "", "./tensorflow/core/framework/attr_value_util.cc", "SubstitutePlaceholders");

  switch (value->value_case()) {
    case AttrValue::kList: {
      for (NameAttrList& func : *value->mutable_list()->mutable_func()) {
        for (auto& p : *func.mutable_attr()) {
          if (!SubstitutePlaceholders(substitute, &p.second)) {
            return false;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (auto& p : *(value->mutable_func()->mutable_attr())) {
        if (!SubstitutePlaceholders(substitute, &p.second)) {
          return false;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return substitute(value->placeholder(), value);
    case AttrValue::VALUE_NOT_SET:
      return false;
    default:
      break;
  }
  return true;
}

}  // namespace tensorflow
