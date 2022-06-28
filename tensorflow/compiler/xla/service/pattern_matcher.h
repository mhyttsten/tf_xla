/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh() {
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


#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/utility/utility.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

// A pattern matcher for HloInstructions, Shapes, and Layouts.
//
// The Match function's first argument must be HloInstruction*, Shape*, or
// Layout*. The second argument is a pattern that will be matched against the
// first argument, as described below.
//
// Patterns are constructed using the match::Op, match::Shape, or match::Layout
// functions. By default, the returned patterns will match any HloInstruction,
// Shape, or Layout, respectively. However the match can be made more specific
// by using the pattern's modifier methods, for example:
//
//   match::Op().WithOpcode(HloOpcode::kAdd).WithOperand(
//     0, match::Op().WithOpcode(HloOpcode::kConstant))
//
// This pattern will match Add instructions whose first operand is a constant.
//
// Each pattern type has the following modifiers, which are described where
// nontrivial.
//
//   Op():
//     - Is: is the given HloInstruction* (i.e. pointer equality)
//     - WithName
//     - WithOpcode
//     - WithoutOpcode: anything other than the given opcode
//     - WithShape: instr's shape matches the given pattern
//     - WithShapeEqualTo: instr's shape is equal to the given Shape
//     - WithShapeCompatibleTo: instr's shape is compatible with the given Shape
//     - WithElementType: instr.shape().element_type() matches the given type
//     - WithNumOperands
//     - WithOperand: operand at the given index matches the given pattern
//     - WithOperandIfPresent: instr has > i operands, and the i'th one matches
//       the given pattern
//     - IsConstant
//     - IsNonConstant
//     - IsConstantScalar/IsEffectiveConstantScalar: Optionally accepts a value,
//       e.g. IsConstantScalar() or IsConstantScalar(42).
//     - WithFusionKind
//     - WithTupleIndex: get-tuple-element operations with the given tuple index
//     - WithOneUse: Instruction is used as an operand exactly once.
//     - WithOneUser: Instruction is used by exactly one other instruction, but
//       is possibly used more than once as an operand (e.g. multiply(x,x)).
//     - WithComparisonDirection: instr has the given direction
//     - WithPredicate: Instruction matches an arbitrary function you pass.
//       Function must have signature `bool(const HloInstruction*)`.
//
//   Shape():
//     - EqualTo
//     - CompatibleTo
//     - IsScalar/IsEffectiveScalar/IsArray/IsTuple
//     - IsDenseArray
//     - WithLayout: layout shape's layout matches the given pattern (e.g.
//       Layout().WithDenseFormat())
//     - WithLayoutEqualTo: shape's layout equals the argument (i.e. another
//       Layout, but not the result of Layout().foo())
//     - WithSubshape: shape is a tuple whose subshape matches the given pattern
//       (e.g. Shape().IsScalar()).
//     - WithSubshapeEqualTo: shape is a tuple with a subshape equal to the arg
//       (i.e. another Shape, but not the result of Shape().foo())
//     - WithElementType: shape is an array/scalar with the given elem type
//     - WithRank: shape is an array/scalar with the given rank
//
//  Layout():
//     - EqualTo
//     - WithDenseFormat
//
// Op(), Shape(), and Layout() may be passed an argument of type
// HloInstruction**, Shape**, or Layout**, respectively, or const versions of
// these pointers. If the pattern is matched, the address of the matched value
// will be "captured" and stored at this location.
//
// For example:
//   HloInstruction* foo = ...;
//   HloInstruction* matched_operand;
//   CHECK(Match(foo,
//               match::Op().WithOperand(0, match::Op(&matched_operand))));
//
// Helpers are provided for most HLO instructions. These helpers can be called
// with no arguments, in which case they will match any instruction matching the
// opcode. They may also be called with matches for the operands and with an
// optional capture. (The capture must be the first argument.) Some examples of
// these helpers and their equivalents are provided below.

// Example nullary instruction:
//   Parameter()                    == Op().WithOpcode(HloOpcode::kParameter)
//   Parameter(&a)                  == Op(&a).WithOpcode(HloOpcode::kParameter)
//
// Example unary instruction:
//   Abs()                          == Op().WithOpcode(HloOpcode::kAbs)
//   Abs(Op(&a))                    == Op().WithOpcode(HloOpcode::kAbs)
//                                         .WithOperand(0, Op(&a)))
//   Abs(&a, Op(&b))                == Op(&a).WithOpcode(HloOpcode::kAbs)
//                                           .WithOperand(0, Op(&b))
//
// Commutative binary instructions have a special form that accepts either order
// of args, e.g.:
//
//   AddAnyOrder(Parameter(1), Abs()) ==
//     Op().WithOpcode(HloOpcode::kAdd)
//         .WithBinaryOperandsAnyOrder(Op().WithParameterNum(1), Abs());
//
//   MultiplyAnyOrder(&a, Parameter(), Abs())  // Captures the mul in `a`.
//
// The following additional helpers are provided.  In all cases, `&a` is
// optional.
//
//   ConstantScalar(&a)               == Op(&a).IsConstantScalar();
//   ConstantScalar(&a, v)            == Op(&a).IsConstantScalar(v);
//   ConstantEffectiveScalar(&a)      == Op(&a).IsConstantEffectiveScalar();
//   ConstantEffectiveScalar(&a, v)   == Op(&a).IsConstantEffectiveScalar(&a, v)
//   NonConstant(&a)                  == Op(&a).IsNonConstant()
//   GetTupleElement(&a, b, index)    == Op(&a).WithTupleIndex(index)
//                                             .WithOperand(0, b);
//   Parameter(&a, n)                 == Op(&a).WithParameterNum(n);

struct MatchOption {
  // If true, actually capture matched item into the user pointer.
  bool capture;

  // An explanation for why we failed to match is streamed here, if not-null.
  std::ostream* explain_os;
};

template <typename Value, typename Pattern>
bool Match(Value* value, const Pattern& pattern,
           MatchOption option = {/*.capture=*/true, /*.explain_os=*/nullptr}) {
  if (option.capture) {
    auto new_option = option;
    new_option.capture = false;
    if (!pattern.Match(value, new_option)) {
      return false;
    }
  }
  return pattern.Match(value, option);
}

namespace match {

namespace detail {

// Macro for streaming to option.explain_os if it's not null.
//
//   EXPLAIN << "value of foo(): " << foo()
//
#pragma push_macro("EXPLAIN")
#define EXPLAIN \
  if (option.explain_os) *option.explain_os

// kIndentInc is the additional number of spaces that we indent by when we
// increase the indent "by one".
enum {
  kIndentInc = 2,
};

// Writes a newline and then `indent` spaces.
//
// We follow an unintuitive convention in this file's pretty-printers: Indents
// are performed by the caller, not the callee.  For example, if you want to
// print
//
//   foo:
//    - bar
//
// you'd do:
//
//  Foo::DescribeTo(std::ostream* os, int64_t indent) {
//    *os << "foo:";
//    Indent(os, indent)  // Create a newline at the *current* indent level.
//    *os << " - ";
//    bar.DescribeTo(os, indent + 3);  // + 3 because strlen(" * ") == 3.
//  }
//
//  Bar::DescribeTo(std::ostream* os, int64_t indent) { *os << "bar"; }
//
// Notice that Bar::DescribeTo() does not call Indent; the indenting is
// performed by Foo.  This convention allows the caller to decide whether a
// matcher is preceded by a newline, which is important e.g. for the AllOf
// matcher.
//
// (Incidentally, indenting in Match's explanations is handled differently.
// Indents are a common case in DescribeTo [we're printing a whole tree], but
// they're a special case in Match [we're printing only a path through the tree
// that encounters a failing node]. Indents in Match only appear when we
// encounter a failing disjunction, so we just handle them as a special case
// there.)
inline void Indent(std::ostream* os, int64_t indent) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_0(mht_0_v, 390, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Indent");

  *os << "\n";
  for (int64_t i = 0; i < indent; ++i) {
    *os << " ";
  }
}

// SFINAE template that determines whether T declares a static member
// kIsTrivialMatcher.
//
// Trivial matchers get special treatment.  For example, when printing
// a conjunction of matchers, we don't print "and" after a trivial matcher. This
// yields e.g.
//    "a shape compatible with f32[1,2]"
// rather than
//    "a shape AND compatible with f32[1,2]"
template <typename T, typename Dummy = void>
struct IsTrivialMatcher {
  static constexpr bool value = false;
};
template <typename T>
struct IsTrivialMatcher<T,
                        typename std::enable_if<T::kIsTrivialMatcher>::type> {
  static constexpr bool value = true;
};

template <typename Item, typename... Patterns>
class AllOfPattern {
 public:
  explicit AllOfPattern(const Patterns&... patterns) : patterns_(patterns...) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_1(mht_1_v, 422, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "AllOfPattern");
}

  bool Match(const Item* item, MatchOption option) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_2(mht_2_v, 427, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    // This invariant is guaranteed by the top-level Match and AnyOf.
    DCHECK(matched || !option.capture);
    return matched;
  }

  bool Match(Item* item, MatchOption option) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_3(mht_3_v, 437, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    bool matched = MatchImpl(item, option, std::integral_constant<size_t, 0>());
    // This invariant is guaranteed by the top-level Match and AnyOf.
    DCHECK(matched || !option.capture);
    return matched;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_4(mht_4_v, 447, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }

  // Accessor for patterns_.  Please don't use this outside of this file.
  const std::tuple<Patterns...>& patterns() const { return patterns_; }

 private:
  template <typename ItemType, size_t index>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, index>) const {
    // We don't need to do any EXPLAINing here; it's all correctly handled by
    // our sub-matchers (if any fail).
    return std::get<index>(patterns_).Match(item, option) &&
           MatchImpl(item, option, std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option,
                 std::integral_constant<size_t, sizeof...(Patterns)>) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_5(mht_5_v, 469, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    return true;
  }

  // Pretty-printing a conjunction has some special cases to make it easy to
  // read in the simple (common) case.
  //
  // If sizeof...(Patterns) == 1, prints as e.g.
  //
  //   a shape
  //
  // If sizeof...(Patterns) == 2 and patterns_[0] is a trivial matcher (e.g. "a
  // shape") prints as
  //
  //   a shape compatible with f32[1,2]
  //
  // If sizeof...(Patterns) > 2 and patterns_[0] is a trivial matcher, prints as
  //
  //   a shape:
  //    * compatible with f32[1,2] AND
  //    * that represents a scalar
  //
  // Otherwise prints as:
  //
  //   all of:
  //    * foo AND
  //    * bar
  //
  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64_t indent) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_6(mht_6_v, 502, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeToImpl");

    constexpr bool first_is_trivial =
        IsTrivialMatcher<typename std::remove_reference<decltype(std::get<0>(
            patterns_))>::type>::value;
    constexpr bool is_last = index == sizeof...(Patterns) - 1;
    const auto& submatcher = std::get<index>(patterns_);

    auto print_bulleted_item = [&] {
      *os << " * ";
      submatcher.DescribeTo(os, indent + 3);
      if (!is_last) {
        *os << " AND";
        Indent(os, indent);
      }
    };

    if (index == 0) {
      if (first_is_trivial || is_last) {
        submatcher.DescribeTo(os, indent + kIndentInc);
        if (sizeof...(Patterns) > 2) {
          *os << ":";
          Indent(os, indent);
        }
      } else {
        *os << "all of:";
        Indent(os, indent);
        print_bulleted_item();
      }
    } else if (first_is_trivial && index == 1 && sizeof...(Patterns) == 2) {
      *os << " ";
      submatcher.DescribeTo(os, indent);
    } else {
      print_bulleted_item();
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }

  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64_t indent) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_7(mht_7_v, 544, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeToImpl");
}

  std::tuple<Patterns...> patterns_;
};

}  // namespace detail

// Returns a pattern that represents the conjunction of all input patterns. All
// patterns need to match in order to have the AllOf pattern match.
template <typename Item, typename... Patterns>
auto AllOf(const Patterns&... patterns) {
  return detail::AllOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}

// AllOf<AllOf<A, B...>, X, Y, ...> => AllOf<A, B, ..., X, Y, ...>.
//
// This transformation is necessary for good pretty-printing.
template <typename Item, typename... InnerPs, typename... OuterPs>
auto AllOf(const detail::AllOfPattern<Item, InnerPs...>& inner_p,
           const OuterPs&... outer_ps) {
  // Invoke constructor of AllOfPattern<Item, InnerPs..., OuterPs...>.
  auto make_all_of = [](const InnerPs&... inner_ps,
                        const OuterPs&... outer_ps) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_8(mht_8_v, 570, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "lambda");

    return detail::AllOfPattern<typename std::remove_const<Item>::type,
                                InnerPs..., OuterPs...>(inner_ps...,
                                                        outer_ps...);
  };
  return absl::apply(make_all_of, std::tuple_cat(inner_p.patterns(),
                                                 std::make_tuple(outer_ps...)));
}

namespace detail {

template <typename LayoutType, typename Impl>
class LayoutPattern;

// The base LayoutPattern implementation. Matches only if the layout is not
// nullptr.
class LayoutPatternBaseImpl {
 public:
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_9(mht_9_v, 591, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (layout == nullptr) {
      EXPLAIN << "Layout is null";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_10(mht_10_v, 602, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "a layout";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// A LayoutPattern implementation that matches only if the layout equals a
// Layout proto.
class LayoutPatternEqualImpl {
 public:
  explicit constexpr LayoutPatternEqualImpl(const ::xla::Layout* layout)
      : layout_(layout) {}

  bool Match(const ::xla::Layout* layout, MatchOption option) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_11(mht_11_v, 619, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!LayoutUtil::Equal(*layout_, *layout)) {
      EXPLAIN << "Layout " << LayoutUtil::HumanString(*layout)
              << " is not equal to expected "
              << LayoutUtil::HumanString(*layout_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_12(mht_12_v, 632, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "equal to " << LayoutUtil::HumanString(*layout_);
  }

 private:
  const ::xla::Layout* layout_;
};

// A LayoutPattern implementation that matches only if the layout has a given
// format.
class LayoutPatternFormatImpl {
 public:
  explicit constexpr LayoutPatternFormatImpl(Format format) : format_(format) {}

  bool Match(const ::xla::Layout* layout, MatchOption option) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_13(mht_13_v, 649, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (layout->format() != format_) {
      EXPLAIN << "Layout has format " << Format_Name(layout->format())
              << " but expected " << Format_Name(format_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_14(mht_14_v, 661, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with format " << Format_Name(format_);
  }

 private:
  Format format_;
};

// A pattern that matches Layouts.
template <typename LayoutType, typename Impl>
class LayoutPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_15(mht_15_v, 677, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "AppendImpl");

    auto new_allof = AllOf<::xla::Layout>(impl_, std::move(new_impl));
    return LayoutPattern<LayoutType, decltype(new_allof)>(std::move(new_allof),
                                                          matched_layout_);
  }

 public:
  explicit constexpr LayoutPattern(const Impl& impl,
                                   LayoutType** matched_layout)
      : impl_(impl), matched_layout_(matched_layout) {}

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(const ::xla::Layout* layout, MatchOption option) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_16(mht_16_v, 692, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  // Returns true and captures the layout iff it matches the pattern.
  bool Match(::xla::Layout* layout, MatchOption option) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_17(mht_17_v, 706, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(layout, option)) {
      if (option.capture && matched_layout_) {
        *matched_layout_ = layout;
      }
      return true;
    }
    return false;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_18(mht_18_v, 719, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    impl_.DescribeTo(os, indent);
  }

  // Modifies the pattern to match only if the layout equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr auto EqualTo(const ::xla::Layout* layout) const {
    return AppendImpl(LayoutPatternEqualImpl(layout));
  }

  // Modifies the pattern to match only if the layout has a dense format.
  constexpr auto WithDenseFormat() const {
    return AppendImpl(LayoutPatternFormatImpl(DENSE));
  }

 private:
  Impl impl_;
  LayoutType** matched_layout_;
};

template <typename Item, typename... Patterns>
class AnyOfPattern {
 public:
  explicit AnyOfPattern(const Patterns&... patterns) : patterns_(patterns...) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_19(mht_19_v, 745, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "AnyOfPattern");
}

  bool Match(const Item* item, MatchOption option) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_20(mht_20_v, 750, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(item, option);
  }

  bool Match(Item* item, MatchOption option) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_21(mht_21_v, 757, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(item, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_22(mht_22_v, 764, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "any of:";
    Indent(os, indent);
    DescribeToImpl(os, std::integral_constant<size_t, 0>(), indent);
  }

 private:
  template <typename ItemType>
  bool MatchImpl(ItemType* item, MatchOption option) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_23(mht_23_v, 775, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    // If we're generating an explanation, buffer it until we know we failed.
    absl::optional<std::stringstream> explanation;
    MatchOption new_option = option;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }
    bool rv = MatchRecursiveImpl(item, new_option,
                                 std::integral_constant<size_t, 0>());
    if (!rv && option.explain_os) {
      EXPLAIN << "None of the following matchers succeeded:";
      EXPLAIN << explanation->str();
    }
    return rv;
  }

  template <typename ItemType, size_t index>
  bool MatchRecursiveImpl(ItemType* item, MatchOption option,
                          std::integral_constant<size_t, index>) const {
    auto new_option = option;
    new_option.capture = false;

    absl::optional<std::stringstream> explanation;
    if (option.explain_os) {
      new_option.explain_os = &explanation.emplace();
    }

    // Try to match the sub-pattern without capturing behavior.
    if (std::get<index>(patterns_).Match(item, new_option)) {
      // Capture the branch.
      if (option.capture) {
        // TODO(timshen): Currently the behavior can be exponential. Optimize it
        // with memoization or recording the matched sub-pattern index, if it
        // takes too long to run.
        //
        // Specifically, the "memoization" approach is to create an empty
        // container with the key (pattern, instruction), and value as whether
        // matched or not.
        //
        // Alternatively, we may run the pattern matching with captures off, but
        // instead record a "trace" somewhere, indicating how exactly the
        // pattern matches the input. For example, the trace information for
        // AnyOf will be a runtime number indicate which sub-pattern is matched.
        // Then we run another pass to do captures only with the help of the
        // trace.
        bool matched = std::get<index>(patterns_).Match(item, option);
        DCHECK(matched);
      }
      return true;
    }
    if (option.explain_os) {
      EXPLAIN << "\nMatcher #" << index + 1;
      EXPLAIN << "\n - ";
      std::get<index>(patterns_).DescribeTo(option.explain_os, /*indent=*/3);
      EXPLAIN << "\nfailed with";
      EXPLAIN << "\n - ";
      EXPLAIN << absl::StrReplaceAll(explanation->str(), {{"\n", "\n   "}});
    }
    return MatchRecursiveImpl(item, option,
                              std::integral_constant<size_t, index + 1>());
  }

  template <typename ItemType>
  bool MatchRecursiveImpl(
      ItemType* item, MatchOption option,
      std::integral_constant<size_t, sizeof...(Patterns)>) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_24(mht_24_v, 843, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchRecursiveImpl");

    return false;
  }

  template <size_t index>
  void DescribeToImpl(std::ostream* os, std::integral_constant<size_t, index>,
                      int64_t indent) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_25(mht_25_v, 852, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeToImpl");

    *os << " - ";
    std::get<index>(patterns_).DescribeTo(os, indent + 3);
    if (index != sizeof...(Patterns) - 1) {
      *os << " OR";
      Indent(os, indent);
    }
    DescribeToImpl(os, std::integral_constant<size_t, index + 1>(), indent);
  }

  void DescribeToImpl(std::ostream* os,
                      std::integral_constant<size_t, sizeof...(Patterns)>,
                      int64_t indent) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_26(mht_26_v, 867, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeToImpl");
}

  std::tuple<Patterns...> patterns_;
};

}  // namespace detail

// Returns a pattern that represents the logical disjunction of the input
// patterns. The returned pattern matches from left to right, and stops on the
// first match.
template <typename Item, typename... Patterns>
auto AnyOf(const Patterns&... patterns) {
  return detail::AnyOfPattern<typename std::remove_const<Item>::type,
                              Patterns...>(patterns...);
}

// Creates a layout pattern that will capture the matched layout in the
// argument.
inline constexpr auto Layout(const ::xla::Layout** matched_layout = nullptr) {
  return detail::LayoutPattern<const ::xla::Layout,
                               detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}

// Creates a layout pattern that will capture the matched layout in the
// argument.
inline constexpr auto Layout(::xla::Layout** matched_layout) {
  return detail::LayoutPattern<::xla::Layout, detail::LayoutPatternBaseImpl>(
      detail::LayoutPatternBaseImpl(), matched_layout);
}

namespace detail {

template <typename ShapeType, typename Impl>
class ShapePattern;

// The base ShapePattern implementation. Matches only if the shape is not
// nullptr.
class ShapePatternBaseImpl {
 public:
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_27(mht_27_v, 910, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (shape == nullptr) {
      EXPLAIN << "Shape is null";
    }
    return shape != nullptr;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_28(mht_28_v, 920, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "a shape";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// A ShapePattern implementation that matches only if the shape equals a Shape
// proto.
class ShapePatternEqualImpl {
 public:
  explicit constexpr ShapePatternEqualImpl(const ::xla::Shape* shape)
      : shape_(shape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_29(mht_29_v, 937, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!ShapeUtil::Equal(*shape_, *shape)) {
      EXPLAIN << "Shape not equal to "
              << ShapeUtil::HumanStringWithLayout(*shape_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_30(mht_30_v, 949, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "equal to " << ShapeUtil::HumanStringWithLayout(*shape_);
  }

 private:
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape is compatible to
// a Shape proto.
class ShapePatternCompatibleImpl {
 public:
  explicit constexpr ShapePatternCompatibleImpl(const ::xla::Shape* shape)
      : shape_(shape) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_31(mht_31_v, 967, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!ShapeUtil::Compatible(*shape_, *shape)) {
      EXPLAIN << "Shape not compatible with "
              << ShapeUtil::HumanString(*shape_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_32(mht_32_v, 979, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "compatible with " << ShapeUtil::HumanString(*shape_);
  }

 private:
  const ::xla::Shape* shape_;
};

// A ShapePattern implementation that matches only if the shape has a given
// element type.
class ShapePatternElementTypeImpl {
 public:
  explicit constexpr ShapePatternElementTypeImpl(PrimitiveType element_type)
      : element_type_(element_type) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_33(mht_33_v, 997, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (shape->element_type() != element_type_) {
      EXPLAIN << "Shape does not have element type "
              << PrimitiveType_Name(element_type_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_34(mht_34_v, 1009, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with element type " << PrimitiveType_Name(element_type_);
  }

 private:
  PrimitiveType element_type_;
};

// A ShapePattern implementation that matches only if the shape has a given
// list of dimensions.
class ShapePatternDimsImpl {
 public:
  explicit ShapePatternDimsImpl(absl::Span<const int64_t> dims)
      : dims_(dims.begin(), dims.end()) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_35(mht_35_v, 1025, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ShapePatternDimsImpl");
}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_36(mht_36_v, 1030, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (shape->dimensions() != dims_) {
      EXPLAIN << "Shape does not have dimensions [" << absl::StrJoin(dims_, ",")
              << "]";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_37(mht_37_v, 1042, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with dimensions [" << absl::StrJoin(dims_, ",") << "]";
  }

 private:
  absl::InlinedVector<int64_t, 8> dims_;
};

// A ShapePattern implementation that matches only if the shape is scalar.
class ShapePatternIsScalarImpl {
 public:
  explicit constexpr ShapePatternIsScalarImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_38(mht_38_v, 1058, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!ShapeUtil::IsScalar(*shape)) {
      EXPLAIN << "Shape is not a scalar";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_39(mht_39_v, 1069, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "that represents a scalar";
  }
};

// A ShapePattern implementation that matches only if the shape is an array
class ShapePatternIsArrayImpl {
 public:
  explicit constexpr ShapePatternIsArrayImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_40(mht_40_v, 1082, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!shape->IsArray()) {
      EXPLAIN << "Shape is not an array";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_41(mht_41_v, 1093, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "that represents an array";
  }
};

// A ShapePattern implementation that matches only if the shape is a tuple.
class ShapePatternIsTupleImpl {
 public:
  explicit constexpr ShapePatternIsTupleImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_42(mht_42_v, 1106, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!shape->IsTuple()) {
      EXPLAIN << "Shape is not a tuple";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_43(mht_43_v, 1117, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "that represents a tuple";
  }
};

// A ShapePattern implementation that matches only if the shape is an effective
// scalar.
class ShapePatternEffectiveScalarImpl {
 public:
  explicit constexpr ShapePatternEffectiveScalarImpl() {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_44(mht_44_v, 1131, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!ShapeUtil::IsEffectiveScalar(*shape)) {
      EXPLAIN << "Shape is not an effective scalar";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_45(mht_45_v, 1142, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "that is an effective scalar";
  }
};

// A ShapePattern implementation that matches only if the shape has a given
// rank.
class ShapePatternRankImpl {
 public:
  explicit constexpr ShapePatternRankImpl(int64_t rank) : rank_(rank) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_46(mht_46_v, 1156, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (shape->rank() != rank_) {
      if (rank_ == 0) {
        EXPLAIN << "Shape is not a scalar";
      } else {
        EXPLAIN << "Shape does not have rank " << rank_;
      }
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_47(mht_47_v, 1171, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    if (rank_ == 0) {
      *os << "that is a scalar";
    } else {
      *os << "that has " << rank_ << " dimension" << (rank_ != 1 ? "s" : "");
    }
  }

 private:
  int64_t rank_;
};

// A ShapePattern implementation that matches only if the shape has a layout
// that matches a given pattern.
template <typename LayoutType, typename LayoutImpl>
class ShapePatternLayoutImpl {
 public:
  explicit constexpr ShapePatternLayoutImpl(
      const LayoutPattern<LayoutType, LayoutImpl>& layout)
      : layout_(layout) {}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_48(mht_48_v, 1195, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return LayoutUtil::HasLayout(*shape) &&
           layout_.Match(&shape->layout(), option);
  }

  bool Match(::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_49(mht_49_v, 1203, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!LayoutUtil::HasLayout(*shape)) {
      EXPLAIN << "Shape does not have a layout";
      return false;
    }
    if (!layout_.Match(shape->mutable_layout(), option)) {
      EXPLAIN << "\nin layout";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_50(mht_50_v, 1218, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with";
    Indent(os, indent + kIndentInc);
    layout_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  LayoutPattern<LayoutType, LayoutImpl> layout_;
};

// A ShapePattern implementation that matches only if the shape has a subshape
// that matches a given pattern.
template <typename SubshapeType, typename SubshapeImpl>
class ShapePatternSubshapeImpl {
 public:
  explicit ShapePatternSubshapeImpl(
      ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape)
      : index_(index), subshape_(subshape) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_51(mht_51_v, 1239, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ShapePatternSubshapeImpl");
}

  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_52(mht_52_v, 1244, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(shape, option);
  }

  bool Match(::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_53(mht_53_v, 1251, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(shape, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_54(mht_54_v, 1258, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with subshape at index " << ShapeIndex(index_) << " which is";
    Indent(os, indent + kIndentInc);
    subshape_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  ::xla::Shape* GetSubshape(::xla::Shape* shape) const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_55(mht_55_v, 1268, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "GetSubshape");

    return ShapeUtil::GetMutableSubshape(shape, index_);
  }
  const ::xla::Shape* GetSubshape(const ::xla::Shape* shape) const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_56(mht_56_v, 1274, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "GetSubshape");

    return &ShapeUtil::GetSubshape(*shape, index_);
  }

  template <typename ShapeType>
  bool MatchImpl(ShapeType* shape, MatchOption option) const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_57(mht_57_v, 1282, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (!ShapeUtil::IndexIsValid(*shape, index_)) {
      EXPLAIN << "No subshape at " << ShapeIndex(index_);
      return false;
    }
    if (!subshape_.Match(GetSubshape(shape), option)) {
      EXPLAIN << "\nin subshape at " << ShapeIndex(index_);
      return false;
    }
    return true;
  }

  ShapeIndexView index_;
  ShapePattern<SubshapeType, SubshapeImpl> subshape_;
};

// A pattern that matches Shapes.
template <typename ShapeType, typename Impl>
class ShapePattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_58(mht_58_v, 1306, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "AppendImpl");

    auto new_all_of = AllOf<::xla::Shape>(impl_, std::move(new_impl));
    return ShapePattern<ShapeType, decltype(new_all_of)>(std::move(new_all_of),
                                                         matched_shape_);
  }

 public:
  explicit constexpr ShapePattern(const Impl& impl, ShapeType** matched_shape)
      : impl_(impl), matched_shape_(matched_shape) {}

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(const ::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_59(mht_59_v, 1320, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    if (shape) {
      EXPLAIN << "\nin "
              << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                      : ShapeUtil::HumanString(*shape));
    }
    return false;
  }

  // Returns true and captures the shape iff it matches the pattern.
  bool Match(::xla::Shape* shape, MatchOption option) const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_60(mht_60_v, 1339, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(shape, option)) {
      if (option.capture && matched_shape_) {
        *matched_shape_ = shape;
      }
      return true;
    }
    EXPLAIN << "\nin "
            << (shape->has_layout() ? ShapeUtil::HumanStringWithLayout(*shape)
                                    : ShapeUtil::HumanString(*shape));
    return false;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_61(mht_61_v, 1355, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    return impl_.DescribeTo(os, indent);
  }

  // Modifies the pattern to match only if the shape equals the given proto.
  // The layout must outlive the returned pattern.
  constexpr auto EqualTo(const ::xla::Shape* shape) const {
    return AppendImpl(ShapePatternEqualImpl(shape));
  }

  // Modifies the pattern to match only if the shape is compatible to the given
  // proto. The layout must outlive the returned pattern.
  constexpr auto CompatibleTo(const ::xla::Shape* shape) const {
    return AppendImpl(ShapePatternCompatibleImpl(shape));
  }

  // Modifies the pattern to match only if the shape has the given element type.
  constexpr auto WithElementType(PrimitiveType element_type) const {
    return AppendImpl(ShapePatternElementTypeImpl(element_type));
  }

  constexpr auto WithDims(absl::Span<const int64_t> dims) const {
    return AppendImpl(ShapePatternDimsImpl(dims));
  }

  // Modifies the pattern to match only if the shape is scalar.
  constexpr auto IsScalar() const {
    return AppendImpl(ShapePatternIsScalarImpl());
  }

  // Modifies the pattern to match only if the shape is an array.
  constexpr auto IsArray() const {
    return AppendImpl(ShapePatternIsArrayImpl());
  }

  // Modifies the pattern to match only if the shape is a tuple.
  constexpr auto IsTuple() const {
    return AppendImpl(ShapePatternIsTupleImpl());
  }

  constexpr auto IsEffectiveScalar() const {
    return AppendImpl(ShapePatternEffectiveScalarImpl());
  }

  // Modifies the pattern to match only if the shape has the given rank.
  constexpr auto WithRank(int64_t rank) const {
    return AppendImpl(ShapePatternRankImpl(rank));
  }

  // Modifies the pattern to match only if the shape has a layout that matches
  // the given pattern.
  template <typename LayoutType, typename LayoutImpl>
  auto WithLayout(const LayoutPattern<LayoutType, LayoutImpl>& layout) const {
    return AppendImpl(ShapePatternLayoutImpl<LayoutType, LayoutImpl>(layout));
  }

  constexpr auto WithLayoutEqualTo(const ::xla::Layout* layout) const {
    return WithLayout(Layout().EqualTo(layout));
  }

  constexpr auto IsDenseArray() const {
    return WithLayout(Layout().WithDenseFormat());
  }

  // Modifies the pattern to match only if the shape has a subshape that matches
  // the given pattern.
  template <typename SubshapeType, typename SubshapeImpl>
  auto WithSubshape(
      ShapeIndexView index,
      const ShapePattern<SubshapeType, SubshapeImpl>& subshape) const {
    return AppendImpl(
        ShapePatternSubshapeImpl<SubshapeType, SubshapeImpl>(index, subshape));
  }

  ShapePattern<ShapeType,
               AllOfPattern<::xla::Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternEqualImpl>>>>
  WithSubshapeEqualTo(ShapeIndexView index, const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .EqualTo(shape));
  }

  ShapePattern<ShapeType,
               AllOfPattern<::xla::Shape, Impl,
                            ShapePatternSubshapeImpl<
                                const ::xla::Shape,
                                AllOfPattern<::xla::Shape, ShapePatternBaseImpl,
                                             ShapePatternCompatibleImpl>>>>
  WithSubshapeCompatibleTo(ShapeIndexView index,
                           const ::xla::Shape* shape) const {
    return WithSubshape(index,
                        ShapePattern<const ::xla::Shape, ShapePatternBaseImpl>(
                            ShapePatternBaseImpl(), nullptr)
                            .CompatibleTo(shape));
  }

 private:
  Impl impl_;
  ShapeType** matched_shape_;
};

}  // namespace detail

// Creates a shape pattern that will capture the matched layout in the argument.
inline constexpr auto Shape(const ::xla::Shape** matched_shape = nullptr) {
  return detail::ShapePattern<const ::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}

// Creates a shape pattern that will capture the matched layout in the argument.
inline constexpr auto Shape(::xla::Shape** matched_shape) {
  return detail::ShapePattern<::xla::Shape, detail::ShapePatternBaseImpl>(
      detail::ShapePatternBaseImpl(), matched_shape);
}

namespace detail {

// Overloads to get a const or non-const operand out of an instruction.
inline HloInstruction* HloOperand(HloInstruction* instr, int64_t idx) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_62(mht_62_v, 1481, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloOperand");

  return instr->mutable_operand(idx);
}
inline const HloInstruction* HloOperand(const HloInstruction* instr,
                                        int64_t idx) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_63(mht_63_v, 1488, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloOperand");

  return instr->operand(idx);
}

// Pretty-printer for HloInstruction.  Sort of like ToShortString, but with
// fewer %s and more shapes.
inline std::string InstToString(const HloInstruction* inst) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_64(mht_64_v, 1497, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "InstToString");

  return inst->ToString(
      HloPrintOptions().set_print_metadata(false).set_print_percent(false));
}

template <typename HloInstructionType, typename Impl>
class HloInstructionPattern;

// The base HloInstructionPattern implementation. Matches only if the
// instruction is not nullptr.
class HloInstructionPatternBaseImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_65(mht_65_v, 1512, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (inst == nullptr) {
      EXPLAIN << "HloInstruction* is null";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_66(mht_66_v, 1523, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "an HloInstruction";
  }

  static constexpr bool kIsTrivialMatcher = true;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given name.
class HloInstructionPatternNameImpl {
 public:
  explicit HloInstructionPatternNameImpl(absl::string_view name)
      : name_(name) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_67(mht_67_v, 1539, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloInstructionPatternNameImpl");
}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_68(mht_68_v, 1544, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (inst->name() != name_) {
      EXPLAIN << "HloInstruction not named \"" << name_ << "\"";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_69(mht_69_v, 1555, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "named \"" << name_ << "\"";
  }

 private:
  absl::string_view name_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// equals a particular pointer.
class HloInstructionIsImpl {
 public:
  explicit HloInstructionIsImpl(const HloInstruction* inst) : inst_(inst) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_70(mht_70_v, 1570, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloInstructionIsImpl");
}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_71(mht_71_v, 1575, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (inst != inst_) {
      EXPLAIN << "HloInstruction " << std::hex << std::nouppercase
              << std::showbase << reinterpret_cast<uint64_t>(inst) << " is not "
              << reinterpret_cast<uint64_t>(inst_) << " ("
              << InstToString(inst_) << ")";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_72(mht_72_v, 1589, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which is " << std::hex << std::nouppercase << std::showbase
        << reinterpret_cast<uint64_t>(inst_) << " (" << InstToString(inst_)
        << ")";
  }

 private:
  const HloInstruction* inst_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a given opcode.
class HloInstructionPatternOpcodeImpl {
 public:
  explicit constexpr HloInstructionPatternOpcodeImpl(HloOpcode opcode,
                                                     bool invert)
      : opcode_(opcode), invert_(invert) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_73(mht_73_v, 1610, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (invert_ && inst->opcode() == opcode_) {
      EXPLAIN << "HloInstruction has opcode " << HloOpcodeString(opcode_)
              << ", expected anything else";
      return false;
    }
    if (!invert_ && inst->opcode() != opcode_) {
      EXPLAIN << "HloInstruction doesn't have opcode "
              << HloOpcodeString(opcode_);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_74(mht_74_v, 1627, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    if (!invert_) {
      *os << "with opcode " << HloOpcodeString(opcode_);
    } else {
      *os << "with any opcode other than " << HloOpcodeString(opcode_);
    }
  }

 private:
  HloOpcode opcode_;
  bool invert_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has one of a given list of custom call targets.
class HloInstructionCustomCallTargetImpl {
 public:
  explicit HloInstructionCustomCallTargetImpl(
      absl::Span<const absl::string_view> custom_call_targets)
      : custom_call_targets_(custom_call_targets.begin(),
                             custom_call_targets.end()) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_75(mht_75_v, 1650, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloInstructionCustomCallTargetImpl");
}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_76(mht_76_v, 1655, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (inst->opcode() != HloOpcode::kCustomCall ||
        !absl::c_linear_search(custom_call_targets_,
                               inst->custom_call_target())) {
      if (custom_call_targets_.size() == 1) {
        EXPLAIN << "HloInstruction is not a custom call with a target '"
                << custom_call_targets_.front() << "'";
      } else {
        EXPLAIN << "HloInstruction is not a custom call with a target in {"
                << absl::StrJoin(custom_call_targets_, ", ") << "}";
      }
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_77(mht_77_v, 1674, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    if (custom_call_targets_.size() == 1) {
      *os << "custom call with target '" << custom_call_targets_.front() << "'";
    } else {
      *os << "custom call with target in {"
          << absl::StrJoin(custom_call_targets_, ", ") << "}";
    }
  }

 private:
  absl::InlinedVector<std::string, 1> custom_call_targets_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has the given number of operands.
class HloInstructionPatternNumOperandsImpl {
 public:
  explicit constexpr HloInstructionPatternNumOperandsImpl(int64_t num_operands)
      : num_operands_(num_operands) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_78(mht_78_v, 1697, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (inst->operand_count() != num_operands_) {
      EXPLAIN << "HloInstruction doesn't have " << num_operands_ << " operands";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_79(mht_79_v, 1708, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with " << num_operands_ << " operand"
        << (num_operands_ != 1 ? "s" : "");
  }

 private:
  int64_t num_operands_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has a shape that matches a given pattern.
template <typename ShapeType, typename ShapeImpl>
class HloInstructionPatternShapeImpl {
 public:
  explicit constexpr HloInstructionPatternShapeImpl(
      const ShapePattern<ShapeType, ShapeImpl>& shape)
      : shape_(shape) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_80(mht_80_v, 1729, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!shape_.Match(&inst->shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_81(mht_81_v, 1740, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!shape_.Match(inst->mutable_shape(), option)) {
      EXPLAIN << "\nin output shape";
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_82(mht_82_v, 1751, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "outputting";
    Indent(os, indent + kIndentInc);
    shape_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  ShapePattern<ShapeType, ShapeImpl> shape_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// has an operand that matches a given pattern.
template <typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandImpl {
 public:
  explicit constexpr HloInstructionPatternOperandImpl(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : operand_index_(operand_index), operand_(operand) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_83(mht_83_v, 1774, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_84(mht_84_v, 1781, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_85(mht_85_v, 1788, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with operand " << operand_index_ << " which is:";
    Indent(os, indent + kIndentInc);
    operand_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_86(mht_86_v, 1799, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (operand_index_ >= inst->operand_count()) {
      EXPLAIN << "desired operand index " << operand_index_
              << " is out of bounds";
      return false;
    }
    if (!operand_.Match(HloOperand(inst, operand_index_), option)) {
      EXPLAIN << "\nin operand " << operand_index_;
      return false;
    }
    return true;
  }

  int64_t operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};

// An HloInstructionPattern implementation that matches if the instruction has
// fewer than i+1 operands, or if the i'th operand matches a given pattern.
template <typename OperandType, typename OperandImpl>
class HloInstructionPatternOperandIfPresentImpl {
 public:
  explicit constexpr HloInstructionPatternOperandIfPresentImpl(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand)
      : operand_index_(operand_index), operand_(operand) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_87(mht_87_v, 1829, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_88(mht_88_v, 1836, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_89(mht_89_v, 1843, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "either with fewer than " << operand_index_ + 1 << " operand"
        << (operand_index_ + 1 != 1 ? "s" : "") << ", or with an operand "
        << operand_index_ << " which is:";
    Indent(os, indent + kIndentInc);
    operand_.DescribeTo(os, indent + kIndentInc);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_90(mht_90_v, 1856, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (operand_index_ >= inst->operand_count()) {
      return true;
    }
    if (!operand_.Match(HloOperand(inst, operand_index_), option)) {
      EXPLAIN << "\nin operand " << operand_index_;
      return false;
    }
    return true;
  }

  int64_t operand_index_;
  HloInstructionPattern<OperandType, OperandImpl> operand_;
};

// Matches a binary instruction whose operands come in any order.
template <typename OperandType1, typename OperandImpl1, typename OperandType2,
          typename OperandImpl2>
class HloInstructionPatternBinaryOperandsAnyOrderImpl {
 public:
  explicit constexpr HloInstructionPatternBinaryOperandsAnyOrderImpl(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2)
      : op1_(op1), op2_(op2) {}

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_91(mht_91_v, 1884, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_92(mht_92_v, 1891, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_93(mht_93_v, 1898, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with two operands in either order:";
    Indent(os, indent);
    *os << " - ";
    op1_.DescribeTo(os, indent + 3);
    Indent(os, indent);
    *os << " - ";
    op2_.DescribeTo(os, indent + 3);
  }

 private:
  HloInstruction* operand(HloInstruction* inst, int64_t idx) const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_94(mht_94_v, 1912, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "operand");

    return inst->mutable_operand(idx);
  }
  const HloInstruction* operand(const HloInstruction* inst, int64_t idx) const {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_95(mht_95_v, 1918, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "operand");

    return inst->operand(idx);
  }

  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_96(mht_96_v, 1926, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    // We could implement this using AnyOf and AllOf matchers, but the templates
    // get pretty difficult to debug, since any compile error herein becomes
    // not-an-error via SFINAE.  Also this way lets us give better messages on
    // failure.
    if (inst->operand_count() != 2) {
      EXPLAIN << "HloInstruction did not have two operands";
      return false;
    }

    // If we're not generating explanations, this is pretty simple.
    if (!option.explain_os) {
      auto try_match = [&](int64_t idx1, int64_t idx2) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_97(mht_97_v, 1941, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "lambda");

        MatchOption new_option = option;
        new_option.capture = false;
        if (op1_.Match(operand(inst, idx1), new_option) &&
            op2_.Match(operand(inst, idx2), new_option)) {
          if (option.capture) {
            bool matched = op1_.Match(operand(inst, idx1), option) &&
                           op2_.Match(operand(inst, idx2), option);
            DCHECK(matched);
          }
          return true;
        }
        return false;
      };
      return try_match(0, 1) || try_match(1, 0);
    }

    // If we are generating explanations, we have some work to do in order to
    // generate a helpful error.
    //
    // First, try all four operand/matcher combinations, recording the
    // failure explanations separately from option.explain_os. matches[i][j]
    // tells us if matcher_i matches operand j.
    bool matches[/*matcher*/ 2][/*operand*/ 2];
    std::stringstream explanations[/*matcher*/ 2][/*operand*/ 2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        MatchOption new_option = option;
        new_option.capture = false;
        new_option.explain_os = &explanations[i][j];
        matches[i][j] = i == 0 ? op1_.Match(operand(inst, j), new_option)
                               : op2_.Match(operand(inst, j), new_option);
      }
    }

    // Check if the match succeeded.
    for (int i = 0; i < 2; ++i) {
      if (matches[0][i] && matches[1][(i + 1) % 2]) {
        // Rerun the matches with capture enabled if necessary.
        if (option.capture) {
          auto* operand1 = operand(inst, i);
          auto* operand2 = operand(inst, (i + 1) % 2);
          bool matched =
              op1_.Match(operand1, option) && op2_.Match(operand2, option);
          DCHECK(matched);
        }
        return true;
      }
    }

    auto describe_matcher = [&](int matcher_idx) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_98(mht_98_v, 1994, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "lambda");

      EXPLAIN << "\n - ";
      if (matcher_idx == 0) {
        op1_.DescribeTo(option.explain_os, /*indent=*/3);
      } else {
        CHECK_EQ(matcher_idx, 1);
        op2_.DescribeTo(option.explain_os, /*indent=*/3);
      }
      for (int i = 0; i < 2; ++i) {
        if (matches[matcher_idx][/*operand*/ i]) {
          continue;
        }
        EXPLAIN << "\ndoes not match " << (i == 0 ? "LHS" : "RHS") << ":\n";
        EXPLAIN << " - ";
        EXPLAIN << absl::StrReplaceAll(
            explanations[matcher_idx][/*operand*/ i].str(), {{"\n", "\n   "}});
      }
    };

    // If we failed to match, one of the following is true:
    //  1. op1 (op2) matches neither LHS nor RHS, or
    //  2. op1 and op2 both match LHS (RHS), but neither matches RHS (LHS).
    // We print different explanations depending on which case we're in.

    // Case 1.
    bool wrote_explanation = false;
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (!matches[i][0] && !matches[i][1]) {
        EXPLAIN << "HloInstruction's operands (ignoring order) did not match "
                << (i == 0 ? "first" : "second") << " matcher.  Specifically,";
        describe_matcher(i);
        wrote_explanation = true;
      }
    }

    // Case 2.
    for (int i = 0; !wrote_explanation && i < 2; ++i) {
      if (matches[/*matcher*/ 0][/*operand*/ i] &&
          matches[/*matcher*/ 1][/*operand*/ i]) {
        CHECK(!matches[0][(i + 1) % 2]);
        CHECK(!matches[1][(i + 1) % 2]);
        CHECK(!wrote_explanation);
        EXPLAIN << "HloInstruction's " << (i == 1 ? "LHS" : "RHS")
                << " operand did not match either of the two matchers.  "
                   "Specifically,";
        describe_matcher(0);
        EXPLAIN << "\nand";
        describe_matcher(1);
        wrote_explanation = true;
      }
    }

    CHECK(wrote_explanation);
    return false;
  }

  HloInstructionPattern<OperandType1, OperandImpl1> op1_;
  HloInstructionPattern<OperandType2, OperandImpl2> op2_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a fusion node with a particular kind.
class HloInstructionPatternFusionKindImpl {
 public:
  explicit constexpr HloInstructionPatternFusionKindImpl(
      ::xla::HloInstruction::FusionKind kind)
      : kind_(kind) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_99(mht_99_v, 2065, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_100(mht_100_v, 2072, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_101(mht_101_v, 2079, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "with fusion kind " << ToString(kind_);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_102(mht_102_v, 2088, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (inst->opcode() != HloOpcode::kFusion) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_)
              << "; it's not a fusion";
      return false;
    }
    if (inst->fusion_kind() != kind_) {
      EXPLAIN << "HloInstruction does not have fusion kind " << ToString(kind_);
      return false;
    }
    return true;
  }

  ::xla::HloInstruction::FusionKind kind_;
};

// An HloInstructionPattern implementation that matches only if the instruction
// is a kGetTupleElement with a particular tuple index.
class HloInstructionPatternTupleIndexImpl {
 public:
  explicit constexpr HloInstructionPatternTupleIndexImpl(int64_t tuple_index)
      : tuple_index_(tuple_index) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_103(mht_103_v, 2114, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_104(mht_104_v, 2121, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_105(mht_105_v, 2128, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which is a GTE with index " << tuple_index_;
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_106(mht_106_v, 2137, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (inst->opcode() != HloOpcode::kGetTupleElement) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_
              << "; it's not a GTE at all";
      return false;
    }
    if (inst->tuple_index() != tuple_index_) {
      EXPLAIN << "HloInstruction is not a GTE with index " << tuple_index_;
      return false;
    }
    return true;
  }

  int64_t tuple_index_;
};

class HloInstructionPatternParameterNumImpl {
 public:
  explicit constexpr HloInstructionPatternParameterNumImpl(
      int64_t parameter_num)
      : parameter_num_(parameter_num) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_107(mht_107_v, 2162, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_108(mht_108_v, 2169, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_109(mht_109_v, 2176, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which is parameter " << parameter_num_;
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_110(mht_110_v, 2185, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (inst->opcode() != HloOpcode::kParameter ||
        inst->parameter_number() != parameter_num_) {
      EXPLAIN << "HloInstruction is not parameter " << parameter_num_;
      return false;
    }
    return true;
  }

  int64_t parameter_num_;
};

// Superclass that contains common code used by Op::WithOneUse() and
// Op::WithOneUser().
class HloInstructionPatternOneUseOrUserImpl {
 protected:
  bool MatchOneUser(const HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_111(mht_111_v, 2204, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchOneUser");

    if (inst->user_count() != 1) {
      EXPLAIN << "HloInstruction has " << inst->user_count()
              << " users, but expected exactly one.";
      if (inst->user_count() > 1) {
        EXPLAIN << "\nAll users:";
        for (const HloInstruction* user : inst->users()) {
          EXPLAIN << "\n - " << InstToString(user);
        }
      }
      return false;
    }
    return true;
  }
};

class HloInstructionPatternOneUseImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_112(mht_112_v, 2226, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (!MatchOneUser(inst, option)) {
      return false;
    }

    int64_t use_count = absl::c_count_if(
        inst->users()[0]->operands(),
        [&](const HloInstruction* operand) { return operand == inst; });
    if (use_count != 1) {
      EXPLAIN << "HloInstruction is used " << use_count
              << " times by its user, but is expected to be used just once: "
              << InstToString(inst->users()[0]);
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_113(mht_113_v, 2246, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which has exactly one use";
  }
};

class HloInstructionPatternOneUserImpl
    : public HloInstructionPatternOneUseOrUserImpl {
 public:
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_114(mht_114_v, 2257, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchOneUser(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_115(mht_115_v, 2264, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which has exactly one user (but possibly is used multiple times by "
           "that instruction)";
  }
};

class HloInstructionPatternComparisonDirectionImpl {
 public:
  explicit constexpr HloInstructionPatternComparisonDirectionImpl(
      ComparisonDirection direction)
      : direction_(direction) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_116(mht_116_v, 2279, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_117(mht_117_v, 2286, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_118(mht_118_v, 2293, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which has comparison direction "
        << ComparisonDirectionToString(direction_);
  }

 private:
  template <typename HloInstructionType>
  bool MatchImpl(HloInstructionType* inst, MatchOption option) const {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_119(mht_119_v, 2303, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    if (inst->opcode() != HloOpcode::kCompare ||
        inst->comparison_direction() != direction_) {
      EXPLAIN << "HloInstruction is not comparison "
              << ComparisonDirectionToString(direction_);
      return false;
    }
    return true;
  }

  ComparisonDirection direction_;
};

class HloInstructionPredicateImpl {
 public:
  explicit HloInstructionPredicateImpl(
      std::function<bool(const HloInstruction*)> fn)
      : fn_(std::move(fn)) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_120(mht_120_v, 2323, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "HloInstructionPredicateImpl");
}

  bool Match(const HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_121(mht_121_v, 2328, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    bool match = fn_(inst);
    if (!match) {
      EXPLAIN << "HloInstruction does not match user-specified predicate";
    }
    return match;
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_122(mht_122_v, 2339, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which matches a user-specified predicate";
  }

 private:
  std::function<bool(const HloInstruction*)> fn_;
};

// Matches a constant scalar or effective scalar, optionally with a given value.
template <typename ScalarTy>
class HloConstantScalarImpl {
 public:
  explicit constexpr HloConstantScalarImpl(bool match_effective_scalar)
      : val_(absl::nullopt), match_effective_scalar_(match_effective_scalar) {}

  constexpr HloConstantScalarImpl(ScalarTy val, bool match_effective_scalar)
      : val_(val), match_effective_scalar_(match_effective_scalar) {}

  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_123(mht_123_v, 2360, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_124(mht_124_v, 2367, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    return MatchImpl(inst, option);
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_125(mht_125_v, 2374, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    *os << "which is a constant "
        << (match_effective_scalar_ ? "effective " : "") << "scalar";
    if (val_.has_value()) {
      *os << " with value " << *val_;
    }
  }

 private:
  template <typename InstTy>
  bool MatchImpl(InstTy* inst, MatchOption option) const {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_126(mht_126_v, 2387, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "MatchImpl");

    const auto* const_inst = DynCast<HloConstantInstruction>(inst);
    if (!const_inst) {
      EXPLAIN << "HloInstruction is not a constant";
      return false;
    }
    if (match_effective_scalar_ &&
        !ShapeUtil::IsEffectiveScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not an effective scalar";
      return false;
    }
    if (!match_effective_scalar_ && !ShapeUtil::IsScalar(inst->shape())) {
      EXPLAIN << "HloInstruction is not a scalar";
      return false;
    }
    if (!val_.has_value()) {
      return true;
    }

    auto const_inst_scalar_or = const_inst->literal().Reshape({});
    if (!const_inst_scalar_or.ok()) {
      EXPLAIN << "could not convert matched literal to effective scalar";
      return false;
    }
    Literal const_inst_scalar = std::move(const_inst_scalar_or).ValueOrDie();
    if (!const_inst_scalar.IsEqualAt({}, *val_)) {
      EXPLAIN << "HloInstruction's constant value "
              << const_inst_scalar.ToStringWithoutShape()
              << " did not match expected value " << *val_;
      return false;
    }
    return true;
  }

  absl::optional<ScalarTy> val_;
  bool match_effective_scalar_;
};

// A pattern that matches HloInstructions.
template <typename HloInstructionType, typename Impl>
class HloInstructionPattern {
 private:
  template <typename NewImpl>
  auto AppendImpl(NewImpl new_impl) const {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_127(mht_127_v, 2433, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "AppendImpl");

    auto new_allof = AllOf<::xla::HloInstruction>(impl_, std::move(new_impl));
    return HloInstructionPattern<HloInstructionType, decltype(new_allof)>(
        std::move(new_allof), matched_inst_);
  }

 public:
  explicit constexpr HloInstructionPattern(const Impl& impl,
                                           HloInstructionType** matched_inst)
      : impl_(impl), matched_inst_(matched_inst) {}

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(const ::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_128(mht_128_v, 2448, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    if (inst != nullptr) {
      EXPLAIN << "\nin " << InstToString(inst);
    }
    return false;
  }

  // Returns true and captures the instruction iff it matches the pattern.
  bool Match(::xla::HloInstruction* inst, MatchOption option) const {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_129(mht_129_v, 2465, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Match");

    if (impl_.Match(inst, option)) {
      if (option.capture && matched_inst_) {
        *matched_inst_ = inst;
      }
      return true;
    }
    EXPLAIN << "\nin " << InstToString(inst);
    return false;
  }

  // Modifies the pattern to match only if the instruction has the given name.
  auto WithName(absl::string_view name) const {
   std::vector<std::string> mht_130_v;
   mht_130_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_130(mht_130_v, 2481, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithName");

    return AppendImpl(HloInstructionPatternNameImpl(name));
  }

  // Modifies the pattern to match only if the instruction has the given opcode.
  auto WithOpcode(HloOpcode opcode) const {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_131(mht_131_v, 2489, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithOpcode");

    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, false));
  }

  // Modifies the pattern to match only the custom call with a given target.
  auto WithCustomCallTarget(absl::string_view custom_call_target) const {
   std::vector<std::string> mht_132_v;
   mht_132_v.push_back("custom_call_target: \"" + std::string(custom_call_target.data(), custom_call_target.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_132(mht_132_v, 2498, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithCustomCallTarget");

    return AppendImpl(HloInstructionCustomCallTargetImpl({custom_call_target}));
  }

  // Modifies the pattern to match a custom call with one of the given targets.
  auto WithCustomCallTarget(
      absl::Span<const absl::string_view> custom_call_targets) const {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_133(mht_133_v, 2507, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithCustomCallTarget");

    return AppendImpl(HloInstructionCustomCallTargetImpl(custom_call_targets));
  }

  auto WithNumOperands(int64_t num_operands) const {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_134(mht_134_v, 2514, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithNumOperands");

    return AppendImpl(HloInstructionPatternNumOperandsImpl(num_operands));
  }

  // Modifies the pattern to match only if the instruction does not have the
  // given opcode.
  auto WithoutOpcode(HloOpcode opcode) const {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_135(mht_135_v, 2523, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithoutOpcode");

    return AppendImpl(HloInstructionPatternOpcodeImpl(opcode, true));
  }

  constexpr auto Is(const HloInstruction* instr) const {
    return AppendImpl(HloInstructionIsImpl(instr));
  }

  // Modifies the pattern to match only if the instruction is a constant.
  constexpr auto IsConstant() const { return WithOpcode(HloOpcode::kConstant); }

  constexpr auto IsConstantScalar() const {
    return AppendImpl(
        HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/false));
  }

  // This does not check that T has the same type as the instruction, so e.g.
  // IsConstantScalar(1.0) may match a constant of shape int32_t[].
  template <typename ScalarTy>
  constexpr auto IsConstantScalar(const ScalarTy& val) const {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, /*match_effective_scalar=*/false));
  }

  constexpr auto IsConstantEffectiveScalar() const {
    return AppendImpl(
        HloConstantScalarImpl</*Dummy*/ int>(/*match_effective_scalar=*/true));
  }

  template <typename ScalarTy>
  constexpr auto IsConstantEffectiveScalar(const ScalarTy& val) const {
    return AppendImpl(
        HloConstantScalarImpl<ScalarTy>(val, /*match_effective_scalar=*/true));
  }

  // Modifies the pattern to match only if the instruction is not a constant.
  constexpr auto IsNonConstant() const {
    return WithoutOpcode(HloOpcode::kConstant);
  }

  // Modifies the pattern to match only if the instruction has a shape that
  // matches the given pattern.
  template <typename ShapeType, typename ShapeImpl>
  constexpr auto WithShape(
      const ShapePattern<ShapeType, ShapeImpl>& shape) const {
    return AppendImpl(
        HloInstructionPatternShapeImpl<ShapeType, ShapeImpl>(shape));
  }

  // Because we only specify the shape's element type and dims, this is
  // effectivley checking shape-compatible-to, not shape-equal-to.  Perhaps this
  // function should be called WithShapeCompatibleTo, but the short name is
  // nice, and there's no ambiguity because there's no layout in the args!
  constexpr auto WithShape(PrimitiveType ty, absl::Span<const int64_t> dims) {
    return WithShape(Shape().WithElementType(ty).WithDims(dims));
  }

  // Make this a templated function to work around gcc 4.9.4 template infinite
  // recursion bug.
  template <typename Dummy = void>
  constexpr auto WithShapeEqualTo(const ::xla::Shape* shape) const {
    return WithShape(Shape().EqualTo(shape));
  }

  // Make this a templated function to work around gcc 4.9.4 template infinite
  // recursion bug.
  template <typename Dummy = void>
  constexpr auto WithShapeCompatibleTo(const ::xla::Shape* shape) const {
    return WithShape(Shape().CompatibleTo(shape));
  }

  // Modifies the pattern to match only if the instruction's shape's element
  // type matches the given pattern.
  constexpr auto WithElementType(PrimitiveType ty) {
    return WithShape(Shape().WithElementType(ty));
  }

  // Modifies the pattern to match only if the instruction has an operand that
  // matches the given pattern.
  template <typename OperandType, typename OperandImpl>
  constexpr auto WithOperand(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const {
    return AppendImpl(
        HloInstructionPatternOperandImpl<OperandType, OperandImpl>(
            operand_index, operand));
  }

  // Modifies the pattern to match only if
  //  - the instruction has fewer than i+1 operands, or
  //  - the i'th operand matches the given pattern.
  template <typename OperandType, typename OperandImpl>
  constexpr auto WithOperandIfPresent(
      int64_t operand_index,
      const HloInstructionPattern<OperandType, OperandImpl>& operand) const {
    return AppendImpl(
        HloInstructionPatternOperandIfPresentImpl<OperandType, OperandImpl>(
            operand_index, operand));
  }

  template <typename OperandType1, typename OperandImpl1, typename OperandType2,
            typename OperandImpl2>
  constexpr auto WithBinaryOperandsAnyOrder(
      const HloInstructionPattern<OperandType1, OperandImpl1>& op1,
      const HloInstructionPattern<OperandType2, OperandImpl2>& op2) const {
    return AppendImpl(
        HloInstructionPatternBinaryOperandsAnyOrderImpl<
            OperandType1, OperandImpl1, OperandType2, OperandImpl2>(op1, op2));
  }

  // Modifies the pattern to match only if the instruction is a fusion node with
  // the given kind.
  constexpr auto WithFusionKind(HloInstruction::FusionKind kind) const {
    return AppendImpl(HloInstructionPatternFusionKindImpl(kind));
  }

  // Modifies the pattern to match only if the instruction is a
  // get-tuple-element with the given tuple index.
  constexpr auto WithTupleIndex(int64_t tuple_index) const {
    return AppendImpl(HloInstructionPatternTupleIndexImpl(tuple_index));
  }

  // Modifies the pattern to match only if the instruction is a parameter
  // with the given parameter number.
  constexpr auto WithParameterNum(int64_t parameter_num) const {
    return AppendImpl(HloInstructionPatternParameterNumImpl(parameter_num));
  }

  // Modifies the pattern to match if the instruction is used exactly once.
  // Does not match if the instruction is used twice by the same user (e.g.
  // multiply(x,x)).
  constexpr auto WithOneUse() const {
    return AppendImpl(HloInstructionPatternOneUseImpl());
  }

  // Modifies the pattern to match if the instruction is used by exactly one
  // other instruction.  Will match if the instruction is used twice, so long as
  // it's by the same user (e.g.  multiply(x,x)).
  constexpr auto WithOneUser() const {
    return AppendImpl(HloInstructionPatternOneUserImpl());
  }

  // Modifies the pattern to match only if the instruction has the given
  // comparison direction.
  auto WithComparisonDirection(ComparisonDirection direction) const {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_136(mht_136_v, 2670, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithComparisonDirection");

    return AppendImpl(HloInstructionPatternComparisonDirectionImpl(direction));
  }

  auto WithPredicate(std::function<bool(const HloInstruction*)> fn) const {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_137(mht_137_v, 2677, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "WithPredicate");

    return AppendImpl(HloInstructionPredicateImpl(std::move(fn)));
  }

  void DescribeTo(std::ostream* os, int64_t indent = 0) const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_138(mht_138_v, 2684, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "DescribeTo");

    impl_.DescribeTo(os, indent);
  }

 private:
  Impl impl_;
  HloInstructionType** matched_inst_;
};

}  // namespace detail

// Creates an instruction pattern that will capture the matched instruction in
// the argument.
inline constexpr auto Op(const ::xla::HloInstruction** matched_inst = nullptr) {
  return detail::HloInstructionPattern<const ::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}

// Creates an instruction pattern that will capture the matched instruction in
// the argument.
inline constexpr auto Op(::xla::HloInstruction** matched_inst) {
  return detail::HloInstructionPattern<::xla::HloInstruction,
                                       detail::HloInstructionPatternBaseImpl>(
      detail::HloInstructionPatternBaseImpl(), matched_inst);
}

// Helpers for nullary instructions.
#define XLA_NULLOP_PATTERN(NAME)                                     \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); } \
                                                                     \
  template <typename HloInstructionType>                             \
  inline auto NAME(HloInstructionType** matched_inst) {              \
    return Op(matched_inst).WithOpcode(HloOpcode::k##NAME);          \
  }
XLA_NULLOP_PATTERN(Constant)
XLA_NULLOP_PATTERN(Parameter)
XLA_NULLOP_PATTERN(Iota)
XLA_NULLOP_PATTERN(Rng)
XLA_NULLOP_PATTERN(PartitionId)
XLA_NULLOP_PATTERN(ReplicaId)
#undef XLA_NULLOP_PATTERN

// Helpers for unary instructions.
#define XLA_UNOP_PATTERN(NAME)                                       \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); } \
                                                                     \
  template <typename Arg>                                            \
  inline auto NAME(Arg&& arg) {                                      \
    return Op()                                                      \
        .WithOpcode(HloOpcode::k##NAME)                              \
        .WithOperand(0, std::forward<Arg>(arg));                     \
  }                                                                  \
                                                                     \
  template <typename HloInstructionType, typename Arg>               \
  inline auto NAME(HloInstructionType** matched_inst, Arg&& arg) {   \
    return Op(matched_inst)                                          \
        .WithOpcode(HloOpcode::k##NAME)                              \
        .WithOperand(0, std::forward<Arg>(arg));                     \
  }
XLA_UNOP_PATTERN(Abs)
XLA_UNOP_PATTERN(RoundNearestAfz)
XLA_UNOP_PATTERN(Bitcast)
XLA_UNOP_PATTERN(BitcastConvert)
XLA_UNOP_PATTERN(Broadcast)
XLA_UNOP_PATTERN(Ceil)
XLA_UNOP_PATTERN(Convert)
XLA_UNOP_PATTERN(Copy)
XLA_UNOP_PATTERN(Cos)
XLA_UNOP_PATTERN(AllReduce)
XLA_UNOP_PATTERN(Exp)
XLA_UNOP_PATTERN(Fft)
XLA_UNOP_PATTERN(Floor)
XLA_UNOP_PATTERN(GetTupleElement)
XLA_UNOP_PATTERN(Imag)
XLA_UNOP_PATTERN(Infeed)
XLA_UNOP_PATTERN(IsFinite)
XLA_UNOP_PATTERN(Log)
XLA_UNOP_PATTERN(Not)
XLA_UNOP_PATTERN(Negate)
XLA_UNOP_PATTERN(Real)
XLA_UNOP_PATTERN(Recv)
XLA_UNOP_PATTERN(RecvDone)
XLA_UNOP_PATTERN(ReducePrecision)
XLA_UNOP_PATTERN(Reshape)
XLA_UNOP_PATTERN(Reverse)
XLA_UNOP_PATTERN(Rsqrt)
XLA_UNOP_PATTERN(SendDone)
XLA_UNOP_PATTERN(Sign)
XLA_UNOP_PATTERN(Sin)
XLA_UNOP_PATTERN(Slice)
XLA_UNOP_PATTERN(Sqrt)
XLA_UNOP_PATTERN(Tanh)
XLA_UNOP_PATTERN(Transpose)
#undef XLA_UNOP_PATTERN

// Helpers for binary instructions.
#define XLA_BINOP_PATTERN(NAME)                                               \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }          \
                                                                              \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return Op()                                                               \
        .WithOpcode(HloOpcode::k##NAME)                                       \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs));                              \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return Op(matched_inst)                                                   \
        .WithOpcode(HloOpcode::k##NAME)                                       \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs));                              \
  }

#define XLA_COMMUTATIVE_BINOP_PATTERN(NAME)                                \
  XLA_BINOP_PATTERN(NAME)                                                  \
                                                                           \
  template <typename HloInstructionType, typename Lhs, typename Rhs>       \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs, \
                             Rhs&& rhs) {                                  \
    return Op(matched_inst)                                                \
        .WithOpcode(HloOpcode::k##NAME)                                    \
        .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),                \
                                    std::forward<Rhs>(rhs));               \
  }                                                                        \
  template <typename Lhs, typename Rhs>                                    \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs) {                       \
    return NAME##AnyOrder<const HloInstruction>(                           \
        nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));          \
  }
XLA_COMMUTATIVE_BINOP_PATTERN(Add)
XLA_BINOP_PATTERN(Atan2)
XLA_BINOP_PATTERN(Divide)
XLA_BINOP_PATTERN(Complex)
XLA_BINOP_PATTERN(Compare)
XLA_BINOP_PATTERN(Convolution)
XLA_BINOP_PATTERN(Dot)
XLA_BINOP_PATTERN(Gather)
XLA_COMMUTATIVE_BINOP_PATTERN(Maximum)
XLA_COMMUTATIVE_BINOP_PATTERN(Minimum)
XLA_COMMUTATIVE_BINOP_PATTERN(Multiply)
XLA_BINOP_PATTERN(Outfeed)
XLA_BINOP_PATTERN(Pad)
XLA_BINOP_PATTERN(Power)
XLA_BINOP_PATTERN(Remainder)
XLA_BINOP_PATTERN(Send)
XLA_BINOP_PATTERN(Subtract)
XLA_COMMUTATIVE_BINOP_PATTERN(And)
XLA_COMMUTATIVE_BINOP_PATTERN(Or)
XLA_BINOP_PATTERN(ShiftLeft)
XLA_BINOP_PATTERN(ShiftRightArithmetic)
XLA_BINOP_PATTERN(ShiftRightLogical)
#undef XLA_COMMUTATIVE_BINOP_PATTERN
#undef XLA_BINOP_PATTERN

// Helpers for ternary instructions.
#define XLA_TERNOP_PATTERN(NAME)                                       \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }   \
                                                                       \
  template <typename Arg0, typename Arg1, typename Arg2>               \
  inline auto NAME(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {            \
    return Op()                                                        \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }                                                                    \
                                                                       \
  template <typename HloInstructionType, typename Arg0, typename Arg1, \
            typename Arg2>                                             \
  inline auto NAME(HloInstructionType** matched_inst, Arg0&& arg0,     \
                   Arg1&& arg1, Arg2&& arg2) {                         \
    return Op(matched_inst)                                            \
        .WithOpcode(HloOpcode::k##NAME)                                \
        .WithOperand(0, std::forward<Arg0>(arg0))                      \
        .WithOperand(1, std::forward<Arg1>(arg1))                      \
        .WithOperand(2, std::forward<Arg2>(arg2));                     \
  }
XLA_TERNOP_PATTERN(Clamp);
XLA_TERNOP_PATTERN(Scatter);
XLA_TERNOP_PATTERN(Select);
XLA_TERNOP_PATTERN(SelectAndScatter);
#undef XLA_TERNOP_PATTERN

namespace detail {
template <typename Matcher, typename FirstArg>
inline auto WithOperands(Matcher&& m, int64_t operand_num,
                         FirstArg&& first_arg) {
  return m.WithOperand(operand_num, std::forward<FirstArg>(first_arg));
}

template <typename Matcher, typename FirstArg, typename... Args>
inline auto WithOperands(Matcher&& m, int64_t operand_num, FirstArg&& first_arg,
                         Args&&... args) {
  return WithOperands(
      m.WithOperand(operand_num, std::forward<FirstArg>(first_arg)),
      operand_num + 1, std::forward<Args>(args)...);
}
}  // namespace detail

#define XLA_VARIADIC_OP_PATTERN(NAME)                                         \
  inline auto NAME() { return Op().WithOpcode(HloOpcode::k##NAME); }          \
                                                                              \
  template <typename... Args>                                                 \
  inline auto NAME(Args&&... args) {                                          \
    return detail::WithOperands(                                              \
        Op().WithOpcode(HloOpcode::k##NAME).WithNumOperands(sizeof...(Args)), \
        /*operand_num=*/0, std::forward<Args>(args)...);                      \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename... Args>                    \
  inline auto NAME(HloInstructionType** matched_inst, Args&&... args) {       \
    return detail::WithOperands(Op(matched_inst)                              \
                                    .WithOpcode(HloOpcode::k##NAME)           \
                                    .WithNumOperands(sizeof...(Args)),        \
                                /*operand_num=*/0,                            \
                                std::forward<Args>(args)...);                 \
  }

// We could implement all ops as "variadic" ops, but it would make the
// already-bad compile errors even worse.
XLA_VARIADIC_OP_PATTERN(AfterAll);
XLA_VARIADIC_OP_PATTERN(Concatenate);
XLA_VARIADIC_OP_PATTERN(Conditional);
XLA_VARIADIC_OP_PATTERN(DynamicSlice)
XLA_VARIADIC_OP_PATTERN(DynamicUpdateSlice)
XLA_VARIADIC_OP_PATTERN(Fusion);
XLA_VARIADIC_OP_PATTERN(Map)
XLA_VARIADIC_OP_PATTERN(Reduce);
XLA_VARIADIC_OP_PATTERN(ReduceWindow)
XLA_VARIADIC_OP_PATTERN(Sort);
XLA_VARIADIC_OP_PATTERN(Tuple);
XLA_VARIADIC_OP_PATTERN(Call);

// CustomCall doesn't use the XLA_VARIADIC_OP_PATTERN macro so that you can
// optionally pass a string_view for the custom_call_target before the other
// operands.
inline auto CustomCall() {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_139(mht_139_v, 2926, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "CustomCall");
 return Op().WithOpcode(HloOpcode::kCustomCall); }

template <typename HloInstructionType>
auto CustomCall(HloInstructionType** matched_inst) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_140(mht_140_v, 2932, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "CustomCall");

  return Op(matched_inst).WithOpcode(HloOpcode::kCustomCall);
}

template <
    typename Arg0, typename... Args,
    typename std::enable_if<
        !std::is_convertible<Arg0, absl::string_view>::value &&
        !std::is_convertible<Arg0, HloInstruction**>::value &&
        !std::is_convertible<Arg0, const HloInstruction**>::value>::type* =
        nullptr>
auto CustomCall(Arg0&& arg0, Args&&... args) {
  return detail::WithOperands(CustomCall().WithNumOperands(sizeof...(Args) + 1),
                              /*operand_num=*/0, std::forward<Arg0>(arg0),
                              std::forward<Args>(args)...);
}
template <typename... Args>
auto CustomCall(absl::string_view custom_call_target, Args&&... args) {
  return CustomCall(std::forward<Args>(args)...)
      .WithCustomCallTarget(custom_call_target);
}

template <typename HloInstructionType, typename Arg0, typename... Args,
          typename std::enable_if<!std::is_convertible<
              Arg0, absl::string_view>::value>::type* = nullptr>
auto CustomCall(HloInstructionType** matched_inst, Arg0&& arg0,
                Args&&... args) {
  return detail::WithOperands(
      CustomCall(matched_inst).WithNumOperands(sizeof...(Args) + 1),
      /*operand_num=*/0, std::forward<Arg0>(arg0), std::forward<Args>(args)...);
}
template <typename HloInstructionType, typename... Args>
auto CustomCall(HloInstructionType** matched_inst,
                absl::string_view custom_call_target, Args&&... args) {
  return CustomCall(matched_inst, std::forward<Args>(args)...)
      .WithCustomCallTarget(custom_call_target);
}

// Helpers for comparison instructions.
#define XLA_COMPARE_PATTERN(NAME)                                             \
  inline auto NAME() {                                                        \
    return Op()                                                               \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }                                                                           \
                                                                              \
  template <typename Lhs, typename Rhs>                                       \
  inline auto NAME(Lhs&& lhs, Rhs&& rhs) {                                    \
    return Op()                                                               \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs))                               \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }                                                                           \
                                                                              \
  template <typename HloInstructionType, typename Lhs, typename Rhs>          \
  inline auto NAME(HloInstructionType** matched_inst, Lhs&& lhs, Rhs&& rhs) { \
    return Op(matched_inst)                                                   \
        .WithOpcode(HloOpcode::kCompare)                                      \
        .WithOperand(0, std::forward<Lhs>(lhs))                               \
        .WithOperand(1, std::forward<Rhs>(rhs))                               \
        .WithComparisonDirection(ComparisonDirection::k##NAME);               \
  }

#define XLA_COMMUTATIVE_COMPARE_PATTERN(NAME)                              \
  XLA_COMPARE_PATTERN(NAME)                                                \
                                                                           \
  template <typename HloInstructionType, typename Lhs, typename Rhs>       \
  inline auto NAME##AnyOrder(HloInstructionType** matched_inst, Lhs&& lhs, \
                             Rhs&& rhs) {                                  \
    return Op(matched_inst)                                                \
        .WithOpcode(HloOpcode::kCompare)                                   \
        .WithBinaryOperandsAnyOrder(std::forward<Lhs>(lhs),                \
                                    std::forward<Rhs>(rhs));               \
  }                                                                        \
  template <typename Lhs, typename Rhs>                                    \
  inline auto NAME##AnyOrder(Lhs&& lhs, Rhs&& rhs) {                       \
    return NAME##AnyOrder<const HloInstruction>(                           \
        nullptr, std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));          \
  }

XLA_COMMUTATIVE_COMPARE_PATTERN(Eq);
XLA_COMMUTATIVE_COMPARE_PATTERN(Ne);
XLA_COMPARE_PATTERN(Ge);
XLA_COMPARE_PATTERN(Gt);
XLA_COMPARE_PATTERN(Le);
XLA_COMPARE_PATTERN(Lt);

// Helpers for matching non-constant instructions.
inline auto NonConstant() {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_141(mht_141_v, 3024, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "NonConstant");
 return Op().IsNonConstant(); }

template <typename HloInstructionType>
inline auto NonConstant(HloInstructionType** matched_inst) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_142(mht_142_v, 3030, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "NonConstant");

  return Op(matched_inst).IsNonConstant();
}

// Add overloads for GetTupleElement which take a int64_t specifying which tuple
// element is selected.
template <typename Arg>
inline auto GetTupleElement(Arg&& arg, int64_t tuple_index) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_143(mht_143_v, 3040, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "GetTupleElement");

  return Op()
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}

template <typename HloInstructionType, typename Arg>
inline auto GetTupleElement(HloInstructionType** matched_inst, Arg&& arg,
                            int64_t tuple_index) {
  return Op(matched_inst)
      .WithOpcode(HloOpcode::kGetTupleElement)
      .WithOperand(0, std::forward<Arg>(arg))
      .WithTupleIndex(tuple_index);
}

// Add overloads for Parameter which take an int64_t specifying the parameter
// number.
inline auto Parameter(int64_t parameter_num) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_144(mht_144_v, 3061, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Parameter");

  return Op().WithOpcode(HloOpcode::kParameter).WithParameterNum(parameter_num);
}
template <typename HloInstructionType>
inline auto Parameter(HloInstructionType** matched_inst,
                      int64_t parameter_num) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_145(mht_145_v, 3069, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "Parameter");

  return Op(matched_inst)
      .WithOpcode(HloOpcode::kParameter)
      .WithParameterNum(parameter_num);
}

inline auto ConstantScalar() {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_146(mht_146_v, 3078, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantScalar");
 return Op().IsConstantScalar(); }

template <typename HloInstructionType>
inline auto ConstantScalar(HloInstructionType** matched_inst) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_147(mht_147_v, 3084, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantScalar");

  return Op(matched_inst).IsConstantScalar();
}

template <typename ScalarTy>
inline auto ConstantScalar(ScalarTy val) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_148(mht_148_v, 3092, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantScalar");

  return Op().IsConstantScalar(val);
}

template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantScalar(HloInstructionType** matched_inst, ScalarTy val) {
  return Op(matched_inst).IsConstantScalar(val);
}

inline auto ConstantEffectiveScalar() {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_149(mht_149_v, 3104, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantEffectiveScalar");

  return Op().IsConstantEffectiveScalar();
}

template <typename HloInstructionType>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_150(mht_150_v, 3112, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantEffectiveScalar");

  return Op(matched_inst).IsConstantEffectiveScalar();
}

template <typename ScalarTy>
inline auto ConstantEffectiveScalar(ScalarTy val) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSpattern_matcherDTh mht_151(mht_151_v, 3120, "", "./tensorflow/compiler/xla/service/pattern_matcher.h", "ConstantEffectiveScalar");

  return Op().IsConstantEffectiveScalar(val);
}

template <typename HloInstructionType, typename ScalarTy>
inline auto ConstantEffectiveScalar(HloInstructionType** matched_inst,
                                    ScalarTy val) {
  return Op(matched_inst).IsConstantEffectiveScalar(val);
}

}  // namespace match

}  // namespace xla

#undef EXPLAIN
#pragma pop_macro("EXPLAIN")
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PATTERN_MATCHER_H_
