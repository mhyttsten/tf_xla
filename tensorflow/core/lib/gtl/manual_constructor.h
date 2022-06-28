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

// ManualConstructor statically-allocates space in which to store some
// object, but does not initialize it.  You can then call the constructor
// and destructor for the object yourself as you see fit.  This is useful
// for memory management optimizations, where you want to initialize and
// destroy an object multiple times but only allocate it once.
//
// (When I say ManualConstructor statically allocates space, I mean that
// the ManualConstructor object itself is forced to be the right size.)

#ifndef TENSORFLOW_LIB_GTL_MANUAL_CONSTRUCTOR_H_
#define TENSORFLOW_LIB_GTL_MANUAL_CONSTRUCTOR_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh() {
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


#include <stddef.h>
#include <new>
#include <utility>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace gtl {
namespace internal {

//
// Provides a char array with the exact same alignment as another type. The
// first parameter must be a complete type, the second parameter is how many
// of that type to provide space for.
//
//   TF_LIB_GTL_ALIGNED_CHAR_ARRAY(struct stat, 16) storage_;
//
// Because MSVC and older GCCs require that the argument to their alignment
// construct to be a literal constant integer, we use a template instantiated
// at all the possible powers of two.
#ifndef SWIG
template <int alignment, int size>
struct AlignType {};
template <int size>
struct AlignType<0, size> {
  typedef char result[size];
};
#if defined(_MSC_VER)
#define TF_LIB_GTL_ALIGN_ATTRIBUTE(X) __declspec(align(X))
#define TF_LIB_GTL_ALIGN_OF(T) __alignof(T)
#elif defined(COMPILER_GCC3) || __GNUC__ >= 3 || defined(__APPLE__) || \
    defined(COMPILER_ICC) || defined(OS_NACL) || defined(__clang__)
#define TF_LIB_GTL_ALIGN_ATTRIBUTE(X) __attribute__((aligned(X)))
#define TF_LIB_GTL_ALIGN_OF(T) __alignof__(T)
#endif

#if defined(TF_LIB_GTL_ALIGN_ATTRIBUTE)

#define TF_LIB_GTL_ALIGNTYPE_TEMPLATE(X)                     \
  template <int size>                                        \
  struct AlignType<X, size> {                                \
    typedef TF_LIB_GTL_ALIGN_ATTRIBUTE(X) char result[size]; \
  }

TF_LIB_GTL_ALIGNTYPE_TEMPLATE(1);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(2);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(4);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(8);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(16);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(32);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(64);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(128);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(256);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(512);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(1024);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(2048);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(4096);
TF_LIB_GTL_ALIGNTYPE_TEMPLATE(8192);
// Any larger and MSVC++ will complain.

#define TF_LIB_GTL_ALIGNED_CHAR_ARRAY(T, Size)                          \
  typename tensorflow::gtl::internal::AlignType<TF_LIB_GTL_ALIGN_OF(T), \
                                                sizeof(T) * Size>::result

#undef TF_LIB_GTL_ALIGNTYPE_TEMPLATE
#undef TF_LIB_GTL_ALIGN_ATTRIBUTE

#else  // defined(TF_LIB_GTL_ALIGN_ATTRIBUTE)
#error "You must define TF_LIB_GTL_ALIGNED_CHAR_ARRAY for your compiler."
#endif  // defined(TF_LIB_GTL_ALIGN_ATTRIBUTE)

#else  // !SWIG

// SWIG can't represent alignment and doesn't care about alignment on data
// members (it works fine without it).
template <typename Size>
struct AlignType {
  typedef char result[Size];
};
#define TF_LIB_GTL_ALIGNED_CHAR_ARRAY(T, Size) \
  tensorflow::gtl::internal::AlignType<Size * sizeof(T)>::result

// Enough to parse with SWIG, will never be used by running code.
#define TF_LIB_GTL_ALIGN_OF(Type) 16

#endif  // !SWIG

}  // namespace internal
}  // namespace gtl

template <typename Type>
class ManualConstructor {
 public:
  // No constructor or destructor because one of the most useful uses of
  // this class is as part of a union, and members of a union cannot have
  // constructors or destructors.  And, anyway, the whole point of this
  // class is to bypass these.

  // Support users creating arrays of ManualConstructor<>s.  This ensures that
  // the array itself has the correct alignment.
  static void* operator new[](size_t size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_0(mht_0_v, 298, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "lambda");

    return port::AlignedMalloc(size, TF_LIB_GTL_ALIGN_OF(Type));
  }
  static void operator delete[](void* mem) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_1(mht_1_v, 304, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "lambda");
 port::AlignedFree(mem); }

  inline Type* get() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_2(mht_2_v, 309, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "get");
 return reinterpret_cast<Type*>(space_); }
  inline const Type* get() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_3(mht_3_v, 313, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "get");

    return reinterpret_cast<const Type*>(space_);
  }

  inline Type* operator->() { return get(); }
  inline const Type* operator->() const { return get(); }

  inline Type& operator*() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_4(mht_4_v, 323, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "*");
 return *get(); }
  inline const Type& operator*() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_5(mht_5_v, 327, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "*");
 return *get(); }

  inline void Init() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSgtlPSmanual_constructorDTh mht_6(mht_6_v, 332, "", "./tensorflow/core/lib/gtl/manual_constructor.h", "Init");
 new (space_) Type; }

// Init() constructs the Type instance using the given arguments
// (which are forwarded to Type's constructor). In C++11, Init() can
// take any number of arguments of any type, and forwards them perfectly.
// On pre-C++11 platforms, it can take up to 11 arguments, and may not be
// able to forward certain kinds of arguments.
//
// Note that Init() with no arguments performs default-initialization,
// not zero-initialization (i.e it behaves the same as "new Type;", not
// "new Type();"), so it will leave non-class types uninitialized.
#ifdef LANG_CXX11
  template <typename... Ts>
  inline void Init(Ts&&... args) {                 // NOLINT
    new (space_) Type(std::forward<Ts>(args)...);  // NOLINT
  }
#else   // !defined(LANG_CXX11)
  template <typename T1>
  inline void Init(const T1& p1) {
    new (space_) Type(p1);
  }

  template <typename T1, typename T2>
  inline void Init(const T1& p1, const T2& p2) {
    new (space_) Type(p1, p2);
  }

  template <typename T1, typename T2, typename T3>
  inline void Init(const T1& p1, const T2& p2, const T3& p3) {
    new (space_) Type(p1, p2, p3);
  }

  template <typename T1, typename T2, typename T3, typename T4>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4) {
    new (space_) Type(p1, p2, p3, p4);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5) {
    new (space_) Type(p1, p2, p3, p4, p5);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6) {
    new (space_) Type(p1, p2, p3, p4, p5, p6);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9, typename T10>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9, const T10& p10) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
  }

  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8, typename T9, typename T10,
            typename T11>
  inline void Init(const T1& p1, const T2& p2, const T3& p3, const T4& p4,
                   const T5& p5, const T6& p6, const T7& p7, const T8& p8,
                   const T9& p9, const T10& p10, const T11& p11) {
    new (space_) Type(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
  }
#endif  // LANG_CXX11

  inline void Destroy() { get()->~Type(); }

 private:
  TF_LIB_GTL_ALIGNED_CHAR_ARRAY(Type, 1) space_;
};

#undef TF_LIB_GTL_ALIGNED_CHAR_ARRAY
#undef TF_LIB_GTL_ALIGN_OF

}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_GTL_MANUAL_CONSTRUCTOR_H_
