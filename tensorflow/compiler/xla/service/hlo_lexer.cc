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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_lexer.h"

#include <limits>
#include <string>

#include "absl/base/casts.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {
namespace {

using absl::string_view;

constexpr int kEOF = -1;
constexpr int kError = -2;

// [a-zA-Z0-9_.-]
bool IsIdentifierChar(char c) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "IsIdentifierChar");

  return absl::ascii_isalnum(static_cast<unsigned char>(c)) || c == '-' ||
         c == '.' || c == '_';
}

}  // namespace

int HloLexer::GetNextChar() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::GetNextChar");

  int current_char = PeekCurrentChar();
  if (current_char != kEOF && current_char != kError) {
    current_ptr_++;
  }
  return current_char;
}

int HloLexer::PeekCurrentChar() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_2(mht_2_v, 233, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::PeekCurrentChar");

  if (current_ptr_ == buf_.end()) {
    return kEOF;
  }
  char current_char = *current_ptr_;
  if (current_char == 0) {
    // '\0' should not appear in the middle of the string.
    return kError;
  }
  return static_cast<unsigned char>(current_char);
}

bool HloLexer::CanDereference(const char* ptr) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("ptr: \"" + (ptr == nullptr ? std::string("nullptr") : std::string((char*)ptr)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::CanDereference");

  return ptr < buf_.end() && ptr >= buf_.begin();
}

absl::string_view HloLexer::StringViewFromPointers(const char* begin,
                                                   const char* end) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("begin: \"" + (begin == nullptr ? std::string("nullptr") : std::string((char*)begin)) + "\"");
   mht_4_v.push_back("end: \"" + (end == nullptr ? std::string("nullptr") : std::string((char*)end)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_4(mht_4_v, 259, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::StringViewFromPointers");

  CHECK(begin <= end);
  CHECK(begin == buf_.end() || CanDereference(begin));
  CHECK(end == buf_.end() || CanDereference(end));
  return absl::string_view(begin, end - begin);
}

TokKind HloLexer::LookAhead() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_5(mht_5_v, 269, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LookAhead");

  if (GetKind() == TokKind::kEof || GetKind() == TokKind::kError) {
    return GetKind();
  }

  const char* old_current_ptr = current_ptr_;
  TokenState old_token_state = token_state_;
  Lex();
  TokKind kind = GetKind();
  token_state_ = old_token_state;
  current_ptr_ = old_current_ptr;
  return kind;
}

TokKind HloLexer::LexToken() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_6(mht_6_v, 286, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexToken");

  while (true) {
    token_state_.token_start = current_ptr_;

    int current_char = GetNextChar();
    switch (current_char) {
      default:
        // [a-zA-Z_]
        if (absl::ascii_isalpha(static_cast<unsigned char>(current_char)) ||
            current_char == '_') {
          return LexIdentifier();
        }
        return TokKind::kError;
      case kEOF:
        // Hit the end of the input buffer.
        return TokKind::kEof;
      case kError:
        // Hit an invalid character in the input buffer.
        return TokKind::kError;
      case ' ':
      case '\t':
      case '\n':
      case '\r':
        // Ignore whitespace.
        continue;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case '-':
      case '?':
        if (current_char == '-' && PeekCurrentChar() == '>') {
          current_ptr_++;
          return TokKind::kArrow;
        }
        return LexNumberOrPattern();
      case '=':
        return TokKind::kEqual;
      case '<':
        if (current_char == '<' && PeekCurrentChar() == '=') {
          current_ptr_++;
          return TokKind::kLeq;
        }
        return TokKind::kError;
      case ',':
        return TokKind::kComma;
      case '%':
        return LexPercent();
      case ':':
        return TokKind::kColon;
      case '*':
        return TokKind::kAsterisk;
      case '[':
        return TokKind::kLsquare;
      case ']':
        return TokKind::kRsquare;
      case '{':
        return TokKind::kLbrace;
      case '}':
        return TokKind::kRbrace;
      case '(':
        return TokKind::kLparen;
      case ')':
        return TokKind::kRparen;
      case '/': {
        if (PeekCurrentChar() == '*') {
          // This is the start of a /*...*/ delimited comment. Save the current
          // location in case the comment is unterminated so the error message
          // will point to the beginning of the comment.
          const char* comment_start = current_ptr_;
          current_ptr_++;
          // Advance until '*/' is found.
          while (true) {
            int current = GetNextChar();
            if (current == '*' && PeekCurrentChar() == '/') {
              // End of comment.
              current_ptr_++;
              break;
            }
            if (current == kEOF) {
              // Unterminated comment.
              current_ptr_ = comment_start;
              return TokKind::kError;
            }
            if (current == kError) {
              return TokKind::kError;
            }
          }
          // Return no token for the comment. Keep lexing.
          continue;
        } else if (PeekCurrentChar() == '/') {
          // This is the start of a '//' delimited comment. Throw away
          // everything until end of line or file. The end-of-line character(s)
          // are left unlexed in the buffer which is harmless because these are
          // skipped later by the lexer. This approach enables support for
          // different end-of-line encodings.
          while (true) {
            int current = PeekCurrentChar();
            if (current == kEOF || current == '\n' || current == '\r') {
              break;
            }
            if (current == kError) {
              return TokKind::kError;
            }
            current_ptr_++;
          }
          continue;
        }
        // A lone '/' is an error.
        return TokKind::kError;
      }
      case '.':
        if (PeekCurrentChar() == '.') {
          current_ptr_++;
          if (PeekCurrentChar() == '.') {
            current_ptr_++;
            return TokKind::kDots;
          }
        }
        return TokKind::kError;
      case '"':
        return LexString();
    }
  }
}

absl::optional<int64_t> HloLexer::LexNanPayload(absl::string_view& consumable) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_7(mht_7_v, 421, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexNanPayload");

  static LazyRE2 payload_pattern = {R"(\(0x[0-9a-fA-F]+\))"};
  if (!RE2::Consume(&consumable, *payload_pattern)) {
    return absl::nullopt;
  }
  auto slice = StringViewFromPointers(current_ptr_, consumable.begin());
  current_ptr_ = consumable.begin();
  CHECK(absl::StartsWith(slice, "(0x"));
  slice.remove_prefix(std::strlen("(0x"));
  CHECK(absl::EndsWith(slice, ")"));
  slice.remove_suffix(std::strlen(")"));
  uint64_t payload_value;
  if (tensorflow::strings::HexStringToUint64(slice, &payload_value)) {
    if (payload_value <= 0 || payload_value > NanPayloadBitMask<double>()) {
      LOG(ERROR) << "NaN payload out of range: " << payload_value;
      return absl::nullopt;
    }
    return payload_value;
  }
  return absl::nullopt;
}

// Lex a shape, name, keyword, attribute name, the dim labels pattern, and
// other identifiers.
//
// shape    ::= ([a-zA-Z0-9_]*[0-9]*)\[([0-9,]*)\](?:\s*{([0-9,]*)})?
// name     ::= [a-zA-Z_][a-zA-Z0-9_.-]*:
// keyword  ::= HloModule, ENTRY, ...
// attribute_name ::= condition, body, dimensions, ...
// dim_labels_pattern ::= [0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,}
// identifiers ::= other cases that match [a-zA-Z_][a-zA-Z0-9_.-]*
TokKind HloLexer::LexIdentifier() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_8(mht_8_v, 455, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexIdentifier");

  while (IsIdentifierChar(PeekCurrentChar())) {
    current_ptr_++;
  }

  // If followed by ':', it's a name.
  if (PeekCurrentChar() == ':') {
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    current_ptr_++;  // skip ':'
    return TokKind::kName;
  }

  // If followed by '=', it's a attribute name.
  if (PeekCurrentChar() == '=') {
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    current_ptr_++;  // skip '='
    return TokKind::kAttributeName;
  }

  absl::string_view identifier =
      StringViewFromPointers(token_state_.token_start, current_ptr_);

  // Primitive type strings are reserved words. The exception is 'tuple' whose
  // type is represented using nested parentheses without the string 'tuple'.
  if (primitive_util::IsPrimitiveTypeName(identifier)) {
    PrimitiveType primitive_type =
        primitive_util::StringToPrimitiveType(identifier).ValueOrDie();
    if (primitive_type != TUPLE) {
      token_state_.primitive_type_val = primitive_type;
      return TokKind::kPrimitiveType;
    }
  }

  if (identifier == "nan") {
    absl::optional<int64_t> payload;
    if (PeekCurrentChar() == '(') {
      absl::string_view consumable =
          StringViewFromPointers(current_ptr_, buf_.end());
      payload = LexNanPayload(consumable);
      if (!payload.has_value()) {
        return TokKind::kError;
      }
    }
    token_state_.decimal_val = NanWithSignAndPayload<double>(
        /*sign=*/false, payload.value_or(QuietNanWithoutPayload<double>()));
    return TokKind::kDecimal;
  }

  // See if this is a keyword.
#define KEYWORD(STR)            \
  do {                          \
    if (identifier == #STR) {   \
      return TokKind::kw_##STR; \
    }                           \
  } while (false)

  KEYWORD(true);
  KEYWORD(false);
  KEYWORD(inf);
  KEYWORD(HloModule);
  KEYWORD(ENTRY);
  KEYWORD(ROOT);
  KEYWORD(maximal);
  KEYWORD(replicated);
  KEYWORD(manual);
  KEYWORD(last_tile_dim_replicate);

#undef KEYWORD

  {
    absl::string_view consumable =
        StringViewFromPointers(token_state_.token_start, buf_.end());
    static LazyRE2 dim_labels_pattern = {
        R"([0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,})"};
    if (RE2::Consume(&consumable, *dim_labels_pattern)) {
      current_ptr_ = consumable.begin();
      token_state_.str_val.assign(token_state_.token_start, current_ptr_);
      return TokKind::kDimLabels;
    }
  }

  token_state_.str_val = std::string(identifier);
  return TokKind::kIdent;
}

// Lex names after a % character.
// name ::= [a-zA-Z_][a-zA-Z0-9_.-]*
TokKind HloLexer::LexPercent() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_9(mht_9_v, 545, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexPercent");

  const char* name_start = current_ptr_;
  if (absl::ascii_isalpha(static_cast<unsigned char>(PeekCurrentChar())) ||
      PeekCurrentChar() == '_') {
    current_ptr_++;
    while (IsIdentifierChar(PeekCurrentChar())) {
      current_ptr_++;
    }
    token_state_.str_val.assign(name_start, current_ptr_);
    return TokKind::kName;
  }
  return TokKind::kError;
}

// Lex integer and floating-point values, -inf, and patterns for dim labels,
// dxd (e.g. 1x2x3), and pad.
//
// fp with exp ::= [-]?([0-9]+|[0-9]+[.][0-9]*|[0-9]*[.][0-9]+)([eE][+-]?[0-9]+)
// fp without exp ::= [-]?([0-9]+[.][0-9]*|[0-9]*[.][0-9]+)
// dim_labels_pattern ::= [0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,}
// dxd_pattern ::= [0-9]+(x[0-9]+)+
// pad_pattern ::=
//   [-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?(x[-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?)*
// int ::=  [-]?[0-9]+
// negative inf ::= '-inf'
TokKind HloLexer::LexNumberOrPattern() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_10(mht_10_v, 573, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexNumberOrPattern");

  absl::string_view consumable =
      StringViewFromPointers(token_state_.token_start, buf_.end());
  static LazyRE2 float_pattern = {
      R"([-]?((\d+|\d+[.]\d*|\d*[.]\d+)([eE][+-]?\d+))|[-]?(\d+[.]\d*|\d*[.]\d+))"};
  if (RE2::Consume(&consumable, *float_pattern)) {
    current_ptr_ = consumable.begin();
    CHECK(absl::SimpleAtod(std::string(token_state_.token_start, current_ptr_),
                           &token_state_.decimal_val));
    return TokKind::kDecimal;
  }

  static LazyRE2 dim_labels_pattern = {
      R"([0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,})"};
  static LazyRE2 dxd_pattern = {R"([0-9]+(x[0-9]+)+)"};
  static LazyRE2 pad_pattern = {
      R"([-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?(x[-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?)*)"};

  if (RE2::Consume(&consumable, *dim_labels_pattern)) {
    current_ptr_ = consumable.begin();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kDimLabels;
  }

  if (RE2::Consume(&consumable, *dxd_pattern)) {
    current_ptr_ = consumable.begin();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kDxD;
  }

  if (RE2::Consume(&consumable, *pad_pattern)) {
    current_ptr_ = consumable.begin();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kPad;
  }

  static LazyRE2 int_pattern = {R"([-]?\d+)"};
  if (RE2::Consume(&consumable, *int_pattern)) {
    current_ptr_ = consumable.begin();
    auto slice = StringViewFromPointers(token_state_.token_start, current_ptr_);
    if (absl::SimpleAtoi(slice, &token_state_.int64_val)) {
      return TokKind::kInt;
    }
    uint64_t uint64_val;
    if (absl::SimpleAtoi(slice, &uint64_val)) {
      token_state_.int64_val = absl::bit_cast<int64_t>(uint64_val);
      return TokKind::kInt;
    }
    LOG(ERROR) << "Failed to parse int literal: " << slice;
    return TokKind::kError;
  }

  static LazyRE2 neg_inf = {"-inf"};
  if (RE2::Consume(&consumable, *neg_inf)) {
    current_ptr_ = consumable.begin();
    return TokKind::kNegInf;
  }

  static LazyRE2 neg_nan = {"-nan"};
  if (RE2::Consume(&consumable, *neg_nan)) {
    current_ptr_ = consumable.begin();

    absl::optional<int64_t> payload;
    if (PeekCurrentChar() == '(') {
      payload = LexNanPayload(consumable);
      if (!payload.has_value()) {
        return TokKind::kError;
      }
    }
    token_state_.decimal_val = NanWithSignAndPayload<double>(
        /*sign=*/true, payload.value_or(QuietNanWithoutPayload<double>()));
    return TokKind::kDecimal;
  }

  return TokKind::kError;
}

std::pair<unsigned, unsigned> HloLexer::GetLineAndColumn(LocTy location) const {
  unsigned line_no = 1;
  const char* start = buf_.begin();
  const char* ptr = start;
  if (line_no_cache_.last_query && CanDereference(line_no_cache_.last_query) &&
      line_no_cache_.last_query <= location) {
    ptr = line_no_cache_.last_query;
    line_no = line_no_cache_.line_no_of_query;
  }
  for (; ptr != location; ptr++) {
    CHECK_LT(ptr, buf_.end());
    if (*ptr == '\n') {
      line_no++;
    }
  }

  // Update the line number cache.
  line_no_cache_.last_query = ptr;
  line_no_cache_.line_no_of_query = line_no;
  size_t line_offset = StringViewFromPointers(start, ptr).rfind('\n');
  if (line_offset == absl::string_view::npos) {
    line_offset = 0;
  }
  return {line_no, ptr - start - line_offset};
}

absl::string_view HloLexer::GetLine(LocTy loc) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_11(mht_11_v, 679, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::GetLine");

  if (!CanDereference(loc)) {
    return "LINE OUT OF RANGE";
  }
  size_t line_start = StringViewFromPointers(buf_.begin(), loc + 1).rfind('\n');
  const char* start = line_start == absl::string_view::npos
                          ? buf_.begin()
                          : buf_.begin() + line_start + 1;
  size_t line_end = StringViewFromPointers(loc, buf_.end()).find('\n');
  const char* end =
      line_end == absl::string_view::npos ? buf_.end() : loc + line_end;

  return StringViewFromPointers(start, end);
}

// Lexes quoted string with escaping characters. If matched, the quoted string
// will be unescaped and stored to token_state_.str_val.
TokKind HloLexer::LexString() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_12(mht_12_v, 699, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "HloLexer::LexString");

  absl::string_view consumable =
      StringViewFromPointers(token_state_.token_start, buf_.end());
  static LazyRE2 escaping_pattern = {R"("([^"\\]|\\.)*")"};
  if (RE2::Consume(&consumable, *escaping_pattern)) {
    current_ptr_ = consumable.begin();
    absl::string_view raw =
        StringViewFromPointers(token_state_.token_start + 1, current_ptr_ - 1);
    std::string error;
    if (!absl::CUnescape(raw, &token_state_.str_val, &error)) {
      LOG(ERROR) << "Failed unescaping string: " << raw << ". error: " << error;
      return TokKind::kError;
    }
    return TokKind::kString;
  }
  return TokKind::kError;
}

std::string TokKindToString(TokKind kind) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTcc mht_13(mht_13_v, 720, "", "./tensorflow/compiler/xla/service/hlo_lexer.cc", "TokKindToString");

  switch (kind) {
    case TokKind::kEof:
      return "kEof";
    case TokKind::kError:
      return "kError";
    case TokKind::kEqual:
      return "kEqaul";
    case TokKind::kComma:
      return "kComma";
    case TokKind::kColon:
      return "kColon";
    case TokKind::kAsterisk:
      return "kAsterisk";
    case TokKind::kLsquare:
      return "kLsquare";
    case TokKind::kRsquare:
      return "kRsquare";
    case TokKind::kLbrace:
      return "kLbrace";
    case TokKind::kRbrace:
      return "kRbrace";
    case TokKind::kLparen:
      return "kLparen";
    case TokKind::kRparen:
      return "kRparen";
    case TokKind::kArrow:
      return "kArrow";
    case TokKind::kLeq:
      return "kLeq";
    case TokKind::kw_HloModule:
      return "kw_HloModule";
    case TokKind::kw_ENTRY:
      return "kw_ENTRY";
    case TokKind::kw_ROOT:
      return "kw_ROOT";
    case TokKind::kw_true:
      return "kw_true";
    case TokKind::kw_false:
      return "kw_false";
    case TokKind::kw_maximal:
      return "kw_maximal";
    case TokKind::kw_replicated:
      return "kw_replicated";
    case TokKind::kw_manual:
      return "kw_manual";
    case TokKind::kw_last_tile_dim_replicate:
      return "kw_last_tile_dim_replicate";
    case TokKind::kw_inf:
      return "kw_inf";
    case TokKind::kNegInf:
      return "kNegInf";
    case TokKind::kPrimitiveType:
      return "kPrimitiveType";
    case TokKind::kName:
      return "kName";
    case TokKind::kAttributeName:
      return "kAttributeName";
    case TokKind::kDimLabels:
      return "kDimLabels";
    case TokKind::kDxD:
      return "kDxD";
    case TokKind::kPad:
      return "kPad";
    case TokKind::kIdent:
      return "kIdent";
    case TokKind::kString:
      return "kString";
    case TokKind::kInt:
      return "kInt";
    case TokKind::kDecimal:
      return "kDecimal";
    case TokKind::kDots:
      return "kDots";
  }
}

}  // namespace xla
