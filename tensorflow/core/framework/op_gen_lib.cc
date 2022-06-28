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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc() {
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

#include "tensorflow/core/framework/op_gen_lib.h"

#include <algorithm>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {

string WordWrap(StringPiece prefix, StringPiece str, int width) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/framework/op_gen_lib.cc", "WordWrap");

  const string indent_next_line = "\n" + Spaces(prefix.size());
  width -= prefix.size();
  string result;
  strings::StrAppend(&result, prefix);

  while (!str.empty()) {
    if (static_cast<int>(str.size()) <= width) {
      // Remaining text fits on one line.
      strings::StrAppend(&result, str);
      break;
    }
    auto space = str.rfind(' ', width);
    if (space == StringPiece::npos) {
      // Rather make a too-long line and break at a space.
      space = str.find(' ');
      if (space == StringPiece::npos) {
        strings::StrAppend(&result, str);
        break;
      }
    }
    // Breaking at character at position <space>.
    StringPiece to_append = str.substr(0, space);
    str.remove_prefix(space + 1);
    // Remove spaces at break.
    while (str_util::EndsWith(to_append, " ")) {
      to_append.remove_suffix(1);
    }
    while (absl::ConsumePrefix(&str, " ")) {
    }

    // Go on to the next line.
    strings::StrAppend(&result, to_append);
    if (!str.empty()) strings::StrAppend(&result, indent_next_line);
  }

  return result;
}

bool ConsumeEquals(StringPiece* description) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/framework/op_gen_lib.cc", "ConsumeEquals");

  if (absl::ConsumePrefix(description, "=")) {
    while (absl::ConsumePrefix(description,
                               " ")) {  // Also remove spaces after "=".
    }
    return true;
  }
  return false;
}

// Split `*orig` into two pieces at the first occurrence of `split_ch`.
// Returns whether `split_ch` was found. Afterwards, `*before_split`
// contains the maximum prefix of the input `*orig` that doesn't
// contain `split_ch`, and `*orig` contains everything after the
// first `split_ch`.
static bool SplitAt(char split_ch, StringPiece* orig,
                    StringPiece* before_split) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("split_ch: '" + std::string(1, split_ch) + "'");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/framework/op_gen_lib.cc", "SplitAt");

  auto pos = orig->find(split_ch);
  if (pos == StringPiece::npos) {
    *before_split = *orig;
    *orig = StringPiece();
    return false;
  } else {
    *before_split = orig->substr(0, pos);
    orig->remove_prefix(pos + 1);
    return true;
  }
}

// Does this line start with "<spaces><field>:" where "<field>" is
// in multi_line_fields? Sets *colon_pos to the position of the colon.
static bool StartsWithFieldName(StringPiece line,
                                const std::vector<string>& multi_line_fields) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_3(mht_3_v, 283, "", "./tensorflow/core/framework/op_gen_lib.cc", "StartsWithFieldName");

  StringPiece up_to_colon;
  if (!SplitAt(':', &line, &up_to_colon)) return false;
  while (absl::ConsumePrefix(&up_to_colon, " "))
    ;  // Remove leading spaces.
  for (const auto& field : multi_line_fields) {
    if (up_to_colon == field) {
      return true;
    }
  }
  return false;
}

static bool ConvertLine(StringPiece line,
                        const std::vector<string>& multi_line_fields,
                        string* ml) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_4(mht_4_v, 301, "", "./tensorflow/core/framework/op_gen_lib.cc", "ConvertLine");

  // Is this a field we should convert?
  if (!StartsWithFieldName(line, multi_line_fields)) {
    return false;
  }
  // Has a matching field name, so look for "..." after the colon.
  StringPiece up_to_colon;
  StringPiece after_colon = line;
  SplitAt(':', &after_colon, &up_to_colon);
  while (absl::ConsumePrefix(&after_colon, " "))
    ;  // Remove leading spaces.
  if (!absl::ConsumePrefix(&after_colon, "\"")) {
    // We only convert string fields, so don't convert this line.
    return false;
  }
  auto last_quote = after_colon.rfind('\"');
  if (last_quote == StringPiece::npos) {
    // Error: we don't see the expected matching quote, abort the conversion.
    return false;
  }
  StringPiece escaped = after_colon.substr(0, last_quote);
  StringPiece suffix = after_colon.substr(last_quote + 1);
  // We've now parsed line into '<up_to_colon>: "<escaped>"<suffix>'

  string unescaped;
  if (!absl::CUnescape(escaped, &unescaped, nullptr)) {
    // Error unescaping, abort the conversion.
    return false;
  }
  // No more errors possible at this point.

  // Find a string to mark the end that isn't in unescaped.
  string end = "END";
  for (int s = 0; unescaped.find(end) != string::npos; ++s) {
    end = strings::StrCat("END", s);
  }

  // Actually start writing the converted output.
  strings::StrAppend(ml, up_to_colon, ": <<", end, "\n", unescaped, "\n", end);
  if (!suffix.empty()) {
    // Output suffix, in case there was a trailing comment in the source.
    strings::StrAppend(ml, suffix);
  }
  strings::StrAppend(ml, "\n");
  return true;
}

string PBTxtToMultiline(StringPiece pbtxt,
                        const std::vector<string>& multi_line_fields) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_5(mht_5_v, 352, "", "./tensorflow/core/framework/op_gen_lib.cc", "PBTxtToMultiline");

  string ml;
  // Probably big enough, since the input and output are about the
  // same size, but just a guess.
  ml.reserve(pbtxt.size() * (17. / 16));
  StringPiece line;
  while (!pbtxt.empty()) {
    // Split pbtxt into its first line and everything after.
    SplitAt('\n', &pbtxt, &line);
    // Convert line or output it unchanged
    if (!ConvertLine(line, multi_line_fields, &ml)) {
      strings::StrAppend(&ml, line, "\n");
    }
  }
  return ml;
}

// Given a single line of text `line` with first : at `colon`, determine if
// there is an "<<END" expression after the colon and if so return true and set
// `*end` to everything after the "<<".
static bool FindMultiline(StringPiece line, size_t colon, string* end) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_6(mht_6_v, 375, "", "./tensorflow/core/framework/op_gen_lib.cc", "FindMultiline");

  if (colon == StringPiece::npos) return false;
  line.remove_prefix(colon + 1);
  while (absl::ConsumePrefix(&line, " ")) {
  }
  if (absl::ConsumePrefix(&line, "<<")) {
    *end = string(line);
    return true;
  }
  return false;
}

string PBTxtFromMultiline(StringPiece multiline_pbtxt) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_7(mht_7_v, 390, "", "./tensorflow/core/framework/op_gen_lib.cc", "PBTxtFromMultiline");

  string pbtxt;
  // Probably big enough, since the input and output are about the
  // same size, but just a guess.
  pbtxt.reserve(multiline_pbtxt.size() * (33. / 32));
  StringPiece line;
  while (!multiline_pbtxt.empty()) {
    // Split multiline_pbtxt into its first line and everything after.
    if (!SplitAt('\n', &multiline_pbtxt, &line)) {
      strings::StrAppend(&pbtxt, line);
      break;
    }

    string end;
    auto colon = line.find(':');
    if (!FindMultiline(line, colon, &end)) {
      // Normal case: not a multi-line string, just output the line as-is.
      strings::StrAppend(&pbtxt, line, "\n");
      continue;
    }

    // Multi-line case:
    //     something: <<END
    // xx
    // yy
    // END
    // Should be converted to:
    //     something: "xx\nyy"

    // Output everything up to the colon ("    something:").
    strings::StrAppend(&pbtxt, line.substr(0, colon + 1));

    // Add every line to unescaped until we see the "END" string.
    string unescaped;
    bool first = true;
    while (!multiline_pbtxt.empty()) {
      SplitAt('\n', &multiline_pbtxt, &line);
      if (absl::ConsumePrefix(&line, end)) break;
      if (first) {
        first = false;
      } else {
        unescaped.push_back('\n');
      }
      strings::StrAppend(&unescaped, line);
      line = StringPiece();
    }

    // Escape what we extracted and then output it in quotes.
    strings::StrAppend(&pbtxt, " \"", absl::CEscape(unescaped), "\"", line,
                       "\n");
  }
  return pbtxt;
}

static void StringReplace(const string& from, const string& to, string* s) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("from: \"" + from + "\"");
   mht_8_v.push_back("to: \"" + to + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_8(mht_8_v, 449, "", "./tensorflow/core/framework/op_gen_lib.cc", "StringReplace");

  // Split *s into pieces delimited by `from`.
  std::vector<string> split;
  string::size_type pos = 0;
  while (pos < s->size()) {
    auto found = s->find(from, pos);
    if (found == string::npos) {
      split.push_back(s->substr(pos));
      break;
    } else {
      split.push_back(s->substr(pos, found - pos));
      pos = found + from.size();
      if (pos == s->size()) {  // handle case where `from` is at the very end.
        split.push_back("");
      }
    }
  }
  // Join the pieces back together with a new delimiter.
  *s = absl::StrJoin(split, to);
}

static void RenameInDocs(const string& from, const string& to,
                         ApiDef* api_def) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("from: \"" + from + "\"");
   mht_9_v.push_back("to: \"" + to + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_9(mht_9_v, 476, "", "./tensorflow/core/framework/op_gen_lib.cc", "RenameInDocs");

  const string from_quoted = strings::StrCat("`", from, "`");
  const string to_quoted = strings::StrCat("`", to, "`");
  for (int i = 0; i < api_def->in_arg_size(); ++i) {
    if (!api_def->in_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_in_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < api_def->out_arg_size(); ++i) {
    if (!api_def->out_arg(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_out_arg(i)->mutable_description());
    }
  }
  for (int i = 0; i < api_def->attr_size(); ++i) {
    if (!api_def->attr(i).description().empty()) {
      StringReplace(from_quoted, to_quoted,
                    api_def->mutable_attr(i)->mutable_description());
    }
  }
  if (!api_def->summary().empty()) {
    StringReplace(from_quoted, to_quoted, api_def->mutable_summary());
  }
  if (!api_def->description().empty()) {
    StringReplace(from_quoted, to_quoted, api_def->mutable_description());
  }
}

namespace {

// Initializes given ApiDef with data in OpDef.
void InitApiDefFromOpDef(const OpDef& op_def, ApiDef* api_def) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_10(mht_10_v, 511, "", "./tensorflow/core/framework/op_gen_lib.cc", "InitApiDefFromOpDef");

  api_def->set_graph_op_name(op_def.name());
  api_def->set_visibility(ApiDef::VISIBLE);

  auto* endpoint = api_def->add_endpoint();
  endpoint->set_name(op_def.name());

  for (const auto& op_in_arg : op_def.input_arg()) {
    auto* api_in_arg = api_def->add_in_arg();
    api_in_arg->set_name(op_in_arg.name());
    api_in_arg->set_rename_to(op_in_arg.name());
    api_in_arg->set_description(op_in_arg.description());

    *api_def->add_arg_order() = op_in_arg.name();
  }
  for (const auto& op_out_arg : op_def.output_arg()) {
    auto* api_out_arg = api_def->add_out_arg();
    api_out_arg->set_name(op_out_arg.name());
    api_out_arg->set_rename_to(op_out_arg.name());
    api_out_arg->set_description(op_out_arg.description());
  }
  for (const auto& op_attr : op_def.attr()) {
    auto* api_attr = api_def->add_attr();
    api_attr->set_name(op_attr.name());
    api_attr->set_rename_to(op_attr.name());
    if (op_attr.has_default_value()) {
      *api_attr->mutable_default_value() = op_attr.default_value();
    }
    api_attr->set_description(op_attr.description());
  }
  api_def->set_summary(op_def.summary());
  api_def->set_description(op_def.description());
}

// Updates base_arg based on overrides in new_arg.
void MergeArg(ApiDef::Arg* base_arg, const ApiDef::Arg& new_arg) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_11(mht_11_v, 549, "", "./tensorflow/core/framework/op_gen_lib.cc", "MergeArg");

  if (!new_arg.rename_to().empty()) {
    base_arg->set_rename_to(new_arg.rename_to());
  }
  if (!new_arg.description().empty()) {
    base_arg->set_description(new_arg.description());
  }
}

// Updates base_attr based on overrides in new_attr.
void MergeAttr(ApiDef::Attr* base_attr, const ApiDef::Attr& new_attr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_12(mht_12_v, 562, "", "./tensorflow/core/framework/op_gen_lib.cc", "MergeAttr");

  if (!new_attr.rename_to().empty()) {
    base_attr->set_rename_to(new_attr.rename_to());
  }
  if (new_attr.has_default_value()) {
    *base_attr->mutable_default_value() = new_attr.default_value();
  }
  if (!new_attr.description().empty()) {
    base_attr->set_description(new_attr.description());
  }
}

// Updates base_api_def based on overrides in new_api_def.
Status MergeApiDefs(ApiDef* base_api_def, const ApiDef& new_api_def) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_13(mht_13_v, 578, "", "./tensorflow/core/framework/op_gen_lib.cc", "MergeApiDefs");

  // Merge visibility
  if (new_api_def.visibility() != ApiDef::DEFAULT_VISIBILITY) {
    base_api_def->set_visibility(new_api_def.visibility());
  }
  // Merge endpoints
  if (new_api_def.endpoint_size() > 0) {
    base_api_def->clear_endpoint();
    std::copy(
        new_api_def.endpoint().begin(), new_api_def.endpoint().end(),
        protobuf::RepeatedFieldBackInserter(base_api_def->mutable_endpoint()));
  }
  // Merge args
  for (const auto& new_arg : new_api_def.in_arg()) {
    bool found_base_arg = false;
    for (int i = 0; i < base_api_def->in_arg_size(); ++i) {
      auto* base_arg = base_api_def->mutable_in_arg(i);
      if (base_arg->name() == new_arg.name()) {
        MergeArg(base_arg, new_arg);
        found_base_arg = true;
        break;
      }
    }
    if (!found_base_arg) {
      return errors::FailedPrecondition("Argument ", new_arg.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  for (const auto& new_arg : new_api_def.out_arg()) {
    bool found_base_arg = false;
    for (int i = 0; i < base_api_def->out_arg_size(); ++i) {
      auto* base_arg = base_api_def->mutable_out_arg(i);
      if (base_arg->name() == new_arg.name()) {
        MergeArg(base_arg, new_arg);
        found_base_arg = true;
        break;
      }
    }
    if (!found_base_arg) {
      return errors::FailedPrecondition("Argument ", new_arg.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  // Merge arg order
  if (new_api_def.arg_order_size() > 0) {
    // Validate that new arg_order is correct.
    if (new_api_def.arg_order_size() != base_api_def->arg_order_size()) {
      return errors::FailedPrecondition(
          "Invalid number of arguments ", new_api_def.arg_order_size(), " for ",
          base_api_def->graph_op_name(),
          ". Expected: ", base_api_def->arg_order_size());
    }
    if (!std::is_permutation(new_api_def.arg_order().begin(),
                             new_api_def.arg_order().end(),
                             base_api_def->arg_order().begin())) {
      return errors::FailedPrecondition(
          "Invalid arg_order: ", absl::StrJoin(new_api_def.arg_order(), ", "),
          " for ", base_api_def->graph_op_name(),
          ". All elements in arg_order override must match base arg_order: ",
          absl::StrJoin(base_api_def->arg_order(), ", "));
    }

    base_api_def->clear_arg_order();
    std::copy(
        new_api_def.arg_order().begin(), new_api_def.arg_order().end(),
        protobuf::RepeatedFieldBackInserter(base_api_def->mutable_arg_order()));
  }
  // Merge attributes
  for (const auto& new_attr : new_api_def.attr()) {
    bool found_base_attr = false;
    for (int i = 0; i < base_api_def->attr_size(); ++i) {
      auto* base_attr = base_api_def->mutable_attr(i);
      if (base_attr->name() == new_attr.name()) {
        MergeAttr(base_attr, new_attr);
        found_base_attr = true;
        break;
      }
    }
    if (!found_base_attr) {
      return errors::FailedPrecondition("Attribute ", new_attr.name(),
                                        " not defined in base api for ",
                                        base_api_def->graph_op_name());
    }
  }
  // Merge summary
  if (!new_api_def.summary().empty()) {
    base_api_def->set_summary(new_api_def.summary());
  }
  // Merge description
  auto description = new_api_def.description().empty()
                         ? base_api_def->description()
                         : new_api_def.description();

  if (!new_api_def.description_prefix().empty()) {
    description =
        strings::StrCat(new_api_def.description_prefix(), "\n", description);
  }
  if (!new_api_def.description_suffix().empty()) {
    description =
        strings::StrCat(description, "\n", new_api_def.description_suffix());
  }
  base_api_def->set_description(description);
  return Status::OK();
}
}  // namespace

ApiDefMap::ApiDefMap(const OpList& op_list) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_14(mht_14_v, 689, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::ApiDefMap");

  for (const auto& op : op_list.op()) {
    ApiDef api_def;
    InitApiDefFromOpDef(op, &api_def);
    map_[op.name()] = api_def;
  }
}

ApiDefMap::~ApiDefMap() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_15(mht_15_v, 700, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::~ApiDefMap");
}

Status ApiDefMap::LoadFileList(Env* env, const std::vector<string>& filenames) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_16(mht_16_v, 705, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::LoadFileList");

  for (const auto& filename : filenames) {
    TF_RETURN_IF_ERROR(LoadFile(env, filename));
  }
  return Status::OK();
}

Status ApiDefMap::LoadFile(Env* env, const string& filename) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_17(mht_17_v, 716, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::LoadFile");

  if (filename.empty()) return Status::OK();
  string contents;
  TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &contents));
  Status status = LoadApiDef(contents);
  if (!status.ok()) {
    // Return failed status annotated with filename to aid in debugging.
    return errors::CreateWithUpdatedMessage(
        status, strings::StrCat("Error parsing ApiDef file ", filename, ": ",
                                status.error_message()));
  }
  return Status::OK();
}

Status ApiDefMap::LoadApiDef(const string& api_def_file_contents) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("api_def_file_contents: \"" + api_def_file_contents + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_18(mht_18_v, 734, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::LoadApiDef");

  const string contents = PBTxtFromMultiline(api_def_file_contents);
  ApiDefs api_defs;
  TF_RETURN_IF_ERROR(
      proto_utils::ParseTextFormatFromString(contents, &api_defs));
  for (const auto& api_def : api_defs.op()) {
    // Check if the op definition is loaded. If op definition is not
    // loaded, then we just skip this ApiDef.
    if (map_.find(api_def.graph_op_name()) != map_.end()) {
      // Overwrite current api def with data in api_def.
      TF_RETURN_IF_ERROR(MergeApiDefs(&map_[api_def.graph_op_name()], api_def));
    }
  }
  return Status::OK();
}

void ApiDefMap::UpdateDocs() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_19(mht_19_v, 753, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::UpdateDocs");

  for (auto& name_and_api_def : map_) {
    auto& api_def = name_and_api_def.second;
    CHECK_GT(api_def.endpoint_size(), 0);
    const string canonical_name = api_def.endpoint(0).name();
    if (api_def.graph_op_name() != canonical_name) {
      RenameInDocs(api_def.graph_op_name(), canonical_name, &api_def);
    }
    for (const auto& in_arg : api_def.in_arg()) {
      if (in_arg.name() != in_arg.rename_to()) {
        RenameInDocs(in_arg.name(), in_arg.rename_to(), &api_def);
      }
    }
    for (const auto& out_arg : api_def.out_arg()) {
      if (out_arg.name() != out_arg.rename_to()) {
        RenameInDocs(out_arg.name(), out_arg.rename_to(), &api_def);
      }
    }
    for (const auto& attr : api_def.attr()) {
      if (attr.name() != attr.rename_to()) {
        RenameInDocs(attr.name(), attr.rename_to(), &api_def);
      }
    }
  }
}

const tensorflow::ApiDef* ApiDefMap::GetApiDef(const string& name) const {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_gen_libDTcc mht_20(mht_20_v, 783, "", "./tensorflow/core/framework/op_gen_lib.cc", "ApiDefMap::GetApiDef");

  return gtl::FindOrNull(map_, name);
}
}  // namespace tensorflow
