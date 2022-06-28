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

#ifndef TENSORFLOW_CORE_PLATFORM_MACROS_H_
#define TENSORFLOW_CORE_PLATFORM_MACROS_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSmacrosDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSmacrosDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSmacrosDTh() {
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


// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define TF_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define TF_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define TF_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define TF_ATTRIBUTE_UNUSED __attribute__((unused))
#define TF_ATTRIBUTE_COLD __attribute__((cold))
#define TF_ATTRIBUTE_WEAK __attribute__((weak))
#define TF_PACKED __attribute__((packed))
#define TF_MUST_USE_RESULT __attribute__((warn_unused_result))
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(_MSC_VER)
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN __declspec(noreturn)
#define TF_ATTRIBUTE_ALWAYS_INLINE __forceinline
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_ATTRIBUTE_WEAK
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#else
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN
#define TF_ATTRIBUTE_ALWAYS_INLINE
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_ATTRIBUTE_WEAK
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// Control visibility outside .so
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_EXPORT __declspec(dllexport)
#else
#define TF_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __has_builtin
#define TF_HAS_BUILTIN(x) __has_builtin(x)
#else
#define TF_HAS_BUILTIN(x) 0
#endif

// C++11-style attributes (N2761)
#if defined(__has_cpp_attribute)
// Safely checks if an attribute is supported. Equivalent to
// ABSL_HAVE_CPP_ATTRIBUTE.
#define TF_HAS_CPP_ATTRIBUTE(n) __has_cpp_attribute(n)
#else
#define TF_HAS_CPP_ATTRIBUTE(n) 0
#endif

// [[clang::annotate("x")]] allows attaching custom strings (e.g. "x") to
// declarations (variables, functions, fields, etc.) for use by tools. They are
// represented in the Clang AST (as AnnotateAttr nodes) and in LLVM IR, but not
// in final output.
#if TF_HAS_CPP_ATTRIBUTE(clang::annotate)
#define TF_ATTRIBUTE_ANNOTATE(str) [[clang::annotate(str)]]
#else
#define TF_ATTRIBUTE_ANNOTATE(str)
#endif

// A variable declaration annotated with the `TF_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer.
#if TF_HAS_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define TF_CONST_INIT [[clang::require_constant_initialization]]
#else
#define TF_CONST_INIT
#endif

// Compilers can be told that a certain branch is not likely to be taken
// (for instance, a CHECK failure), and use that information in static
// analysis. Giving it this information can help it optimize for the
// common case in the absence of better information (ie.
// -fprofile-arcs).
#if TF_HAS_BUILTIN(__builtin_expect) || (defined(__GNUC__) && __GNUC__ >= 3)
#define TF_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define TF_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define TF_PREDICT_FALSE(x) (x)
#define TF_PREDICT_TRUE(x) (x)
#endif

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

// The TF_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression TF_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
#define TF_ARRAYSIZE(a)         \
  ((sizeof(a) / sizeof(*(a))) / \
   static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define TF_FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
#endif
#endif

#ifndef TF_FALLTHROUGH_INTENDED
#define TF_FALLTHROUGH_INTENDED \
  do {                          \
  } while (0)
#endif

namespace tensorflow {
namespace internal {
template <typename T>
void remove_unused_variable_compiler_warning(const T&){
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSmacrosDTh mht_0(mht_0_v, 324, "", "./tensorflow/core/platform/macros.h", "remove_unused_variable_compiler_warning");
};
}
}  // namespace tensorflow
#define TF_UNUSED_VARIABLE(x) \
  tensorflow::internal::remove_unused_variable_compiler_warning(x)

#endif  // TENSORFLOW_CORE_PLATFORM_MACROS_H_
