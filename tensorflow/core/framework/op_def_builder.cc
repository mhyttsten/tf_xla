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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc() {
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

#include "tensorflow/core/framework/op_def_builder.h"

#include <limits>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"

using ::tensorflow::strings::Scanner;

namespace tensorflow {

namespace {

string AttrError(StringPiece orig, const string& op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/framework/op_def_builder.cc", "AttrError");

  return strings::StrCat(" from Attr(\"", orig, "\") for Op ", op_name);
}

bool ConsumeAttrName(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeAttrName");

  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeListPrefix(StringPiece* sp) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeListPrefix");

  return Scanner(*sp)
      .OneLiteral("list")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeQuotedString(char quote_ch, StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("quote_ch: '" + std::string(1, quote_ch) + "'");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeQuotedString");

  const string quote_str(1, quote_ch);
  return Scanner(*sp)
      .OneLiteral(quote_str.c_str())
      .RestartCapture()
      .ScanEscapedUntil(quote_ch)
      .StopCapture()
      .OneLiteral(quote_str.c_str())
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrType(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeAttrType");

  return Scanner(*sp)
      .Many(Scanner::LOWERLETTER_DIGIT)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeAttrNumber(StringPiece* sp, int64_t* out) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeAttrNumber");

  Scanner scan(*sp);
  StringPiece match;
  StringPiece remaining;

  scan.AnySpace().RestartCapture();
  if (scan.Peek() == '-') {
    scan.OneLiteral("-");
  }
  if (!scan.Many(Scanner::DIGIT)
           .StopCapture()
           .AnySpace()
           .GetResult(&remaining, &match)) {
    return false;
  }
  int64_t value = 0;
  if (!strings::safe_strto64(match, &value)) {
    return false;
  }
  *out = value;
  *sp = remaining;
  return true;
}

#define VERIFY(expr, ...)                                                 \
  do {                                                                    \
    if (!(expr)) {                                                        \
      errors->push_back(                                                  \
          strings::StrCat(__VA_ARGS__, AttrError(orig, op_def->name()))); \
      return;                                                             \
    }                                                                     \
  } while (false)

bool ConsumeCompoundAttrType(StringPiece* sp, StringPiece* out) {
  auto capture_begin = sp->begin();
  if (absl::ConsumePrefix(sp, "numbertype") ||
      absl::ConsumePrefix(sp, "numerictype") ||
      absl::ConsumePrefix(sp, "quantizedtype") ||
      absl::ConsumePrefix(sp, "realnumbertype") ||
      absl::ConsumePrefix(sp, "realnumberictype")) {
    *out = StringPiece(capture_begin, sp->begin() - capture_begin);
    return true;
  }
  return false;
}

bool ProcessCompoundType(const StringPiece type_string, AttrValue* allowed) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_6(mht_6_v, 318, "", "./tensorflow/core/framework/op_def_builder.cc", "ProcessCompoundType");

  if (type_string == "numbertype" || type_string == "numerictype") {
    for (DataType dt : NumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "quantizedtype") {
    for (DataType dt : QuantizedTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else if (type_string == "realnumbertype" ||
             type_string == "realnumerictype") {
    for (DataType dt : RealNumberTypes()) {
      allowed->mutable_list()->add_type(dt);
    }
  } else {
    return false;
  }
  return true;
}

void FinalizeAttr(StringPiece spec, bool allow_attr_type_any, OpDef* op_def,
                  std::vector<string>* errors) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_7(mht_7_v, 342, "", "./tensorflow/core/framework/op_def_builder.cc", "FinalizeAttr");

  OpDef::AttrDef* attr = op_def->add_attr();
  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
  VERIFY(ConsumeAttrName(&spec, &tmp_name), "Trouble parsing '<name>:'");
  attr->set_name(tmp_name.data(), tmp_name.size());

  // Read "<type>" or "list(<type>)".
  bool is_list = ConsumeListPrefix(&spec);
  string type;
  StringPiece type_string;  // Used if type == "type"
  if (absl::ConsumePrefix(&spec, "string")) {
    type = "string";
  } else if (absl::ConsumePrefix(&spec, "int")) {
    type = "int";
  } else if (absl::ConsumePrefix(&spec, "float")) {
    type = "float";
  } else if (absl::ConsumePrefix(&spec, "bool")) {
    type = "bool";
  } else if (absl::ConsumePrefix(&spec, "type")) {
    type = "type";
  } else if (absl::ConsumePrefix(&spec, "shape")) {
    type = "shape";
  } else if (absl::ConsumePrefix(&spec, "tensor")) {
    type = "tensor";
  } else if (absl::ConsumePrefix(&spec, "func")) {
    type = "func";
  } else if (absl::ConsumePrefix(&spec, "any") && allow_attr_type_any) {
    type = "any";
  } else if (ConsumeCompoundAttrType(&spec, &type_string)) {
    type = "type";
    AttrValue* allowed = attr->mutable_allowed_values();
    VERIFY(ProcessCompoundType(type_string, allowed),
           "Expected to see a compound type, saw: ", type_string);
  } else if (absl::ConsumePrefix(&spec, "{")) {
    // e.g. "{ int32, float, bool }" or "{ \"foo\", \"bar\" }"
    AttrValue* allowed = attr->mutable_allowed_values();
    str_util::RemoveLeadingWhitespace(&spec);
    if (absl::StartsWith(spec, "\"") || absl::StartsWith(spec, "'")) {
      type = "string";  // "{ \"foo\", \"bar\" }" or "{ 'foo', 'bar' }"
      while (true) {
        StringPiece escaped_string;
        VERIFY(ConsumeQuotedString('"', &spec, &escaped_string) ||
                   ConsumeQuotedString('\'', &spec, &escaped_string),
               "Trouble parsing allowed string at '", spec, "'");
        string unescaped;
        string error;
        VERIFY(absl::CUnescape(escaped_string, &unescaped, &error),
               "Trouble unescaping \"", escaped_string,
               "\", got error: ", error);
        allowed->mutable_list()->add_s(unescaped);
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
                 "Expected , or } after strings in list, not: '", spec, "'");
          break;
        }
      }
    } else {  // "{ bool, numbertype, string }"
      type = "type";
      while (true) {
        VERIFY(ConsumeAttrType(&spec, &type_string),
               "Trouble parsing type string at '", spec, "'");
        if (ProcessCompoundType(type_string, allowed)) {
          // Processed a compound type.
        } else {
          DataType dt;
          VERIFY(DataTypeFromString(type_string, &dt),
                 "Unrecognized type string '", type_string, "'");
          allowed->mutable_list()->add_type(dt);
        }
        if (absl::ConsumePrefix(&spec, ",")) {
          str_util::RemoveLeadingWhitespace(&spec);
          if (absl::ConsumePrefix(&spec, "}"))
            break;  // Allow ending with ", }".
        } else {
          VERIFY(absl::ConsumePrefix(&spec, "}"),
                 "Expected , or } after types in list, not: '", spec, "'");
          break;
        }
      }
    }
  } else {  // if spec.Consume("{")
    VERIFY(false, "Trouble parsing type string at '", spec, "'");
  }
  str_util::RemoveLeadingWhitespace(&spec);

  // Write the type into *attr.
  if (is_list) {
    VERIFY(absl::ConsumePrefix(&spec, ")"),
           "Expected ) to close 'list(', not: '", spec, "'");
    str_util::RemoveLeadingWhitespace(&spec);
    attr->set_type(strings::StrCat("list(", type, ")"));
  } else {
    attr->set_type(type);
  }

  // Read optional minimum constraint at the end.
  if ((is_list || type == "int") && absl::ConsumePrefix(&spec, ">=")) {
    int64_t min_limit = -999;
    VERIFY(ConsumeAttrNumber(&spec, &min_limit),
           "Could not parse integer lower limit after '>=', found '", spec,
           "' instead");
    attr->set_has_minimum(true);
    attr->set_minimum(min_limit);
  }

  // Parse default value, if present.
  if (absl::ConsumePrefix(&spec, "=")) {
    str_util::RemoveLeadingWhitespace(&spec);
    VERIFY(ParseAttrValue(attr->type(), spec, attr->mutable_default_value()),
           "Could not parse default value '", spec, "'");
  } else {
    VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");
  }
}

#undef VERIFY

string InOutError(bool is_output, StringPiece orig, const string& op_name) {
  return strings::StrCat(" from ", is_output ? "Output" : "Input", "(\"", orig,
                         "\") for Op ", op_name);
}

bool ConsumeInOutName(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_8(mht_8_v, 474, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeInOutName");

  return Scanner(*sp)
      .One(Scanner::LOWERLETTER)
      .Any(Scanner::LOWERLETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutRefOpen(StringPiece* sp) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_9(mht_9_v, 488, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeInOutRefOpen");

  return Scanner(*sp)
      .OneLiteral("Ref")
      .AnySpace()
      .OneLiteral("(")
      .AnySpace()
      .GetResult(sp);
}

bool ConsumeInOutRefClose(StringPiece* sp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_10(mht_10_v, 500, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeInOutRefClose");

  return Scanner(*sp).OneLiteral(")").AnySpace().GetResult(sp);
}

bool ConsumeInOutNameOrType(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_11(mht_11_v, 507, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeInOutNameOrType");

  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeInOutTimesType(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_12(mht_12_v, 519, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeInOutTimesType");

  return Scanner(*sp)
      .OneLiteral("*")
      .AnySpace()
      .RestartCapture()
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .GetResult(sp, out);
}

bool ConsumeControlOutName(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_13(mht_13_v, 534, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeControlOutName");

  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .GetResult(sp, out);
}

#define VERIFY(expr, ...)                                             \
  do {                                                                \
    if (!(expr)) {                                                    \
      errors->push_back(strings::StrCat(                              \
          __VA_ARGS__, InOutError(is_output, orig, op_def->name()))); \
      return;                                                         \
    }                                                                 \
  } while (false)

void FinalizeInputOrOutput(StringPiece spec, bool is_output, OpDef* op_def,
                           std::vector<string>* errors) {
  OpDef::ArgDef* arg =
      is_output ? op_def->add_output_arg() : op_def->add_input_arg();

  StringPiece orig(spec);

  // Parse "<name>:" at the beginning.
  StringPiece tmp_name;
  VERIFY(ConsumeInOutName(&spec, &tmp_name), "Trouble parsing 'name:'");
  arg->set_name(tmp_name.data(), tmp_name.size());

  // Detect "Ref(...)".
  if (ConsumeInOutRefOpen(&spec)) {
    arg->set_is_ref(true);
  }

  {  // Parse "<name|type>" or "<name>*<name|type>".
    StringPiece first, second, type_or_attr;
    VERIFY(ConsumeInOutNameOrType(&spec, &first),
           "Trouble parsing either a type or an attr name at '", spec, "'");
    if (ConsumeInOutTimesType(&spec, &second)) {
      arg->set_number_attr(first.data(), first.size());
      type_or_attr = second;
    } else {
      type_or_attr = first;
    }
    DataType dt;
    if (DataTypeFromString(type_or_attr, &dt)) {
      arg->set_type(dt);
    } else {
      const OpDef::AttrDef* attr = FindAttr(type_or_attr, *op_def);
      VERIFY(attr != nullptr, "Reference to unknown attr '", type_or_attr, "'");
      if (attr->type() == "type") {
        arg->set_type_attr(type_or_attr.data(), type_or_attr.size());
      } else {
        VERIFY(attr->type() == "list(type)", "Reference to attr '",
               type_or_attr, "' with type ", attr->type(),
               " that isn't type or list(type)");
        arg->set_type_list_attr(type_or_attr.data(), type_or_attr.size());
      }
    }
  }

  // Closing ) for Ref(.
  if (arg->is_ref()) {
    VERIFY(ConsumeInOutRefClose(&spec),
           "Did not find closing ')' for 'Ref(', instead found: '", spec, "'");
  }

  // Should not have anything else.
  VERIFY(spec.empty(), "Extra '", spec, "' unparsed at the end");

  // Int attrs that are the length of an input or output get a default
  // minimum of 1.
  if (!arg->number_attr().empty()) {
    OpDef::AttrDef* attr = FindAttrMutable(arg->number_attr(), op_def);
    if (attr != nullptr && !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  } else if (!arg->type_list_attr().empty()) {
    // If an input or output has type specified by a list(type) attr,
    // it gets a default minimum of 1 as well.
    OpDef::AttrDef* attr = FindAttrMutable(arg->type_list_attr(), op_def);
    if (attr != nullptr && attr->type() == "list(type)" &&
        !attr->has_minimum()) {
      attr->set_has_minimum(true);
      attr->set_minimum(1);
    }
  }

  // If the arg's dtype is resource we should mark the op as stateful as it
  // likely touches a resource manager. This deliberately doesn't cover inputs /
  // outputs which resolve to resource via Attrs as those mostly operate on
  // resource handles as an opaque type (as opposed to ops which explicitly take
  // / produce resources).
  if (arg->type() == DT_RESOURCE) {
    op_def->set_is_stateful(true);
  }
}

#undef VERIFY

string ControlOutError(StringPiece orig, const string& op_name) {
  return strings::StrCat(" from ControlOutput(\"", orig, "\") for Op ",
                         op_name);
}

void FinalizeControlOutput(StringPiece name, OpDef* op_def,
                           std::vector<string>* errors) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_14(mht_14_v, 644, "", "./tensorflow/core/framework/op_def_builder.cc", "FinalizeControlOutput");

  StringPiece orig(name);

  // Parse control output name.
  StringPiece tmp_name;
  if (!ConsumeControlOutName(&orig, &tmp_name)) {
    errors->push_back(strings::StrCat("Trouble parsing 'name:'",
                                      ControlOutError(orig, op_def->name())));
  }

  *op_def->add_control_output() = string(tmp_name.data(), tmp_name.size());
}

int num_leading_spaces(StringPiece s) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_15(mht_15_v, 660, "", "./tensorflow/core/framework/op_def_builder.cc", "num_leading_spaces");

  size_t i = 0;
  while (i < s.size() && s[i] == ' ') {
    ++i;
  }
  return i;
}

bool ConsumeDocNameColon(StringPiece* sp, StringPiece* out) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_16(mht_16_v, 671, "", "./tensorflow/core/framework/op_def_builder.cc", "ConsumeDocNameColon");

  return Scanner(*sp)
      .One(Scanner::LETTER)
      .Any(Scanner::LETTER_DIGIT_UNDERSCORE)
      .StopCapture()
      .AnySpace()
      .OneLiteral(":")
      .AnySpace()
      .GetResult(sp, out);
}

bool IsDocNameColon(StringPiece s) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_17(mht_17_v, 685, "", "./tensorflow/core/framework/op_def_builder.cc", "IsDocNameColon");

  return ConsumeDocNameColon(&s, nullptr /* out */);
}

void FinalizeDoc(const string& text, OpDef* op_def,
                 std::vector<string>* errors) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_18(mht_18_v, 694, "", "./tensorflow/core/framework/op_def_builder.cc", "FinalizeDoc");

  std::vector<string> lines = str_util::Split(text, '\n');

  // Remove trailing spaces.
  for (string& line : lines) {
    absl::StripTrailingAsciiWhitespace(&line);
  }

  // First non-blank line -> summary.
  int l = 0;
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;
  if (static_cast<size_t>(l) < lines.size()) {
    op_def->set_summary(lines[l]);
    ++l;
  }
  while (static_cast<size_t>(l) < lines.size() && lines[l].empty()) ++l;

  // Lines until we see name: -> description.
  int start_l = l;
  while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
    ++l;
  }
  int end_l = l;
  // Trim trailing blank lines from the description.
  while (start_l < end_l && lines[end_l - 1].empty()) --end_l;
  string desc = absl::StrJoin(
      gtl::ArraySlice<string>(lines.data() + start_l, end_l - start_l), "\n");
  if (!desc.empty()) op_def->set_description(desc);

  // name: description
  //   possibly continued on the next line
  //   if so, we remove the minimum indent
  StringPiece name;
  std::vector<StringPiece> description;
  while (static_cast<size_t>(l) < lines.size()) {
    description.clear();
    description.push_back(lines[l]);
    ConsumeDocNameColon(&description.back(), &name);
    ++l;
    while (static_cast<size_t>(l) < lines.size() && !IsDocNameColon(lines[l])) {
      description.push_back(lines[l]);
      ++l;
    }
    // Remove any trailing blank lines.
    while (!description.empty() && description.back().empty()) {
      description.pop_back();
    }
    // Compute the minimum indent of all lines after the first.
    int min_indent = -1;
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) {
        int indent = num_leading_spaces(description[i]);
        if (min_indent < 0 || indent < min_indent) min_indent = indent;
      }
    }
    // Remove min_indent spaces from all lines after the first.
    for (size_t i = 1; i < description.size(); ++i) {
      if (!description[i].empty()) description[i].remove_prefix(min_indent);
    }
    // Concatenate lines into a single string.
    const string complete(absl::StrJoin(description, "\n"));

    // Find name.
    bool found = false;
    for (int i = 0; !found && i < op_def->input_arg_size(); ++i) {
      if (op_def->input_arg(i).name() == name) {
        op_def->mutable_input_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->output_arg_size(); ++i) {
      if (op_def->output_arg(i).name() == name) {
        op_def->mutable_output_arg(i)->set_description(complete);
        found = true;
      }
    }
    for (int i = 0; !found && i < op_def->attr_size(); ++i) {
      if (op_def->attr(i).name() == name) {
        op_def->mutable_attr(i)->set_description(complete);
        found = true;
      }
    }
    if (!found) {
      errors->push_back(
          strings::StrCat("No matching input/output/attr for name '", name,
                          "' from Doc() for Op ", op_def->name()));
      return;
    }
  }
}

}  // namespace

OpDefBuilder::OpDefBuilder(string op_name) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_19(mht_19_v, 791, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::OpDefBuilder");

  op_def()->set_name(std::move(op_name));
}

OpDefBuilder& OpDefBuilder::Attr(string spec) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_20(mht_20_v, 799, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Attr");

  attrs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Input(string spec) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_21(mht_21_v, 808, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Input");

  inputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::Output(string spec) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("spec: \"" + spec + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_22(mht_22_v, 817, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Output");

  outputs_.push_back(std::move(spec));
  return *this;
}

OpDefBuilder& OpDefBuilder::ControlOutput(string name) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_23(mht_23_v, 826, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::ControlOutput");

  control_outputs_.push_back(std::move(name));
  return *this;
}

OpDefBuilder& OpDefBuilder::Doc(string text) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("text: \"" + text + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_24(mht_24_v, 835, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Doc");

#ifndef TF_LEAN_BINARY
  if (!doc_.empty()) {
    errors_.push_back(
        strings::StrCat("Extra call to Doc() for Op ", op_def()->name()));
  } else {
    doc_ = std::move(text);
  }
#endif
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsCommutative() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_25(mht_25_v, 850, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetIsCommutative");

  op_def()->set_is_commutative(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsAggregate() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_26(mht_26_v, 858, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetIsAggregate");

  op_def()->set_is_aggregate(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsStateful() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_27(mht_27_v, 866, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetIsStateful");

  op_def()->set_is_stateful(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetAllowsUninitializedInput() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_28(mht_28_v, 874, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetAllowsUninitializedInput");

  op_def()->set_allows_uninitialized_input(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::SetIsDistributedCommunication() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_29(mht_29_v, 882, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetIsDistributedCommunication");

  op_def()->set_is_distributed_communication(true);
  return *this;
}

OpDefBuilder& OpDefBuilder::Deprecated(int version, string explanation) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("explanation: \"" + explanation + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_30(mht_30_v, 891, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Deprecated");

  if (op_def()->has_deprecation()) {
    errors_.push_back(
        strings::StrCat("Deprecated called twice for Op ", op_def()->name()));
  } else {
    OpDeprecation* deprecation = op_def()->mutable_deprecation();
    deprecation->set_version(version);
    deprecation->set_explanation(std::move(explanation));
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::SetTypeConstructor(OpTypeConstructor c) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_31(mht_31_v, 906, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetTypeConstructor");

  op_reg_data_.type_ctor = c;
  return *this;
}

OpDefBuilder& OpDefBuilder::SetForwardTypeFn(ForwardTypeInferenceFn f) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_32(mht_32_v, 914, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetForwardTypeFn");

  op_reg_data_.fwd_type_fn = f;
  return *this;
}

OpDefBuilder& OpDefBuilder::SetShapeFn(OpShapeInferenceFn fn) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_33(mht_33_v, 922, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::SetShapeFn");

  if (op_reg_data_.shape_inference_fn != nullptr) {
    errors_.push_back(
        strings::StrCat("SetShapeFn called twice for Op ", op_def()->name()));
  } else {
    op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
  }
  return *this;
}

OpDefBuilder& OpDefBuilder::AllowAttrTypeAny() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_34(mht_34_v, 935, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::AllowAttrTypeAny");

  allow_attr_type_any_ = true;
  return *this;
}

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_def_builderDTcc mht_35(mht_35_v, 943, "", "./tensorflow/core/framework/op_def_builder.cc", "OpDefBuilder::Finalize");

  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
  for (StringPiece attr : attrs_) {
    FinalizeAttr(attr, allow_attr_type_any_, op_def, &errors);
  }
  for (StringPiece input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (StringPiece output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  for (StringPiece control_output : control_outputs_) {
    FinalizeControlOutput(control_output, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);

  if (op_reg_data->type_ctor != nullptr) {
    TF_RETURN_IF_ERROR(op_reg_data->type_ctor(op_def));
  }

  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(absl::StrJoin(errors, "\n"));
}

}  // namespace tensorflow
