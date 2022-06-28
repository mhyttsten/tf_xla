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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh() {
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


#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

// Defines different kinds of tokens used by the HLO lexer.
//
// You shouldn't need to use this directly unless you're using HloLexer
// directly, and you probably don't need to do that.  Use hlo_parser instead.
enum class TokKind {
  // Markers
  kEof,
  kError,

  // Tokens with no info.
  kEqual,     // =
  kComma,     // ,
  kColon,     // :
  kAsterisk,  // *
  kLsquare,
  kRsquare,  // [  ]
  kLbrace,
  kRbrace,  // {  }
  kLparen,
  kRparen,  // (  )
  kDots,    // ...

  kArrow,  // ->
  kLeq,    // <=

  // Keywords
  kw_HloModule,
  kw_ENTRY,
  kw_ROOT,
  kw_true,
  kw_false,
  kw_maximal,
  kw_replicated,
  kw_manual,
  kw_last_tile_dim_replicate,
  kw_inf,

  kNegInf,  // -inf

  // Typed tokens.
  kPrimitiveType,  // F32, PRED, etc.
  kName,           // %foo
  kAttributeName,  // dimensions=
  kDimLabels,      // [0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,}
  kDxD,            // [0-9]+(x[0-9]+)+
  kPad,            // [0-9]+_[0-9]+(_[0-9]+)?(x[0-9]+_[0-9]+(_[0-9]+)?)*
  kIdent,          // other identifiers
  kString,         // "abcd\"\n"
  kInt,            // 42
  kDecimal,        // 4.2
};

std::string TokKindToString(TokKind kind);

// Lexer for the HloModule::ToString() format text.
//
// This class is meant to be used by hlo_parser.cc.  You shouldn't need to use
// it directly.
class HloLexer {
 public:
  explicit HloLexer(absl::string_view buf) : buf_(buf) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buf: \"" + std::string(buf.data(), buf.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_0(mht_0_v, 260, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "HloLexer");

    current_ptr_ = buf_.begin();
  }

  TokKind Lex() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_1(mht_1_v, 267, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "Lex");
 return token_state_.current_kind = LexToken(); }

  TokKind GetKind() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_2(mht_2_v, 272, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetKind");
 return token_state_.current_kind; }
  std::string GetStrVal() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_3(mht_3_v, 276, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetStrVal");

    switch (GetKind()) {
      case TokKind::kName:
      case TokKind::kAttributeName:
      case TokKind::kDimLabels:
      case TokKind::kDxD:
      case TokKind::kPad:
      case TokKind::kString:
      case TokKind::kIdent:
        return token_state_.str_val;
      default:
        LOG(FATAL) << "This token does not have string value";
    }
  }
  int64_t GetInt64Val() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_4(mht_4_v, 293, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetInt64Val");

    CHECK(GetKind() == TokKind::kInt) << TokKindToString(GetKind());
    return token_state_.int64_val;
  }
  double GetDecimalVal() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_5(mht_5_v, 300, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetDecimalVal");

    CHECK(GetKind() == TokKind::kDecimal);
    return token_state_.decimal_val;
  }
  PrimitiveType GetPrimitiveTypeVal() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_6(mht_6_v, 307, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetPrimitiveTypeVal");

    CHECK(GetKind() == TokKind::kPrimitiveType);
    return token_state_.primitive_type_val;
  }

  typedef const char* LocTy;

  // Returns the location of the current token.
  LocTy GetLoc() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_lexerDTh mht_7(mht_7_v, 318, "", "./tensorflow/compiler/xla/service/hlo_lexer.h", "GetLoc");
 return token_state_.token_start; }

  // Returns the line and column of a location in the buffer.
  std::pair<unsigned, unsigned> GetLineAndColumn(LocTy location) const;

  // Returns the whole line given the location.
  absl::string_view GetLine(LocTy loc) const;

  // Looks ahead one token and returns it. Lexer state is unchanged.
  TokKind LookAhead();

 private:
  // Returns the current character. If it's neither the end of input buffer nor
  // an invalid character, moves the pointer forward.
  int GetNextChar();

  // Returns the current character.
  int PeekCurrentChar() const;

  // Creates string_view with the given begin and end. Exits if the begin > end,
  // or it's out of the range of the current buffer.
  absl::string_view StringViewFromPointers(const char* begin,
                                           const char* end) const;

  // Returns true if the given ptr is dereferenceable within the range of the
  // current buffer.
  bool CanDereference(const char* ptr) const;

  TokKind LexToken();

  TokKind LexIdentifier();
  TokKind LexPercent();
  TokKind LexShape();
  TokKind LexConstant();
  TokKind LexNumberOrPattern();
  TokKind LexString();

  absl::optional<int64_t> LexNanPayload(absl::string_view& consumable);

  absl::string_view buf_;
  const char* current_ptr_;

  // Information about the current token.
  struct TokenState {
    const char* token_start = nullptr;
    TokKind current_kind;
    std::string str_val;
    int64_t int64_val;
    double decimal_val;
    PrimitiveType primitive_type_val;
  };
  TokenState token_state_;

  struct LineNoCacheTy {
    const char* last_query;
    unsigned line_no_of_query;
  };
  // This caches the line number of the previous query.
  mutable LineNoCacheTy line_no_cache_{nullptr, 0};
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_LEXER_H_
