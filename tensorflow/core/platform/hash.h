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

// Simple hash functions used for internal data structures

#ifndef TENSORFLOW_CORE_PLATFORM_HASH_H_
#define TENSORFLOW_CORE_PLATFORM_HASH_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPShashDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPShashDTh() {
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
#include <stdint.h>

#include <functional>
#include <string>

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_0(mht_0_v, 205, "", "./tensorflow/core/platform/hash.h", "Hash64");

  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const char* data) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_1(mht_1_v, 213, "", "./tensorflow/core/platform/hash.h", "Hash64");
 return Hash64(data, ::strlen(data)); }

inline uint64 Hash64(const std::string& str) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_2(mht_2_v, 219, "", "./tensorflow/core/platform/hash.h", "Hash64");

  return Hash64(str.data(), str.size());
}

inline uint64 Hash64(const tstring& str) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("str: \"" + (std::string)str + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_3(mht_3_v, 227, "", "./tensorflow/core/platform/hash.h", "Hash64");

  return Hash64(str.data(), str.size());
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_4(mht_4_v, 234, "", "./tensorflow/core/platform/hash.h", "Hash64Combine");

  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

// Combine two hashes in an order-independent way. This operation should be
// associative and compute the same hash for a collection of elements
// independent of traversal order. Note that it is better to combine hashes
// symmetrically with addition rather than XOR, since (x^x) == 0 but (x+x) != 0.
inline uint64 Hash64CombineUnordered(uint64 a, uint64 b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPShashDTh mht_5(mht_5_v, 245, "", "./tensorflow/core/platform/hash.h", "Hash64CombineUnordered");
 return a + b; }

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
//
// The second type argument is only used for SFNIAE below.
template <typename T, typename = void>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  size_t operator()(T value) const {
    // This works around a defect in the std::hash C++ spec that isn't fixed in
    // (at least) gcc 4.8.4:
    // http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2148
    //
    // We should be able to remove this and use the default
    // tensorflow::hash<EnumTy>() once we stop building with GCC versions old
    // enough to not have this defect fixed.
    return std::hash<uint64>()(static_cast<uint64>(value));
  }
};

template <typename T>
struct hash<T*> {
  size_t operator()(const T* t) const {
    // Hash pointers as integers, but bring more entropy to the lower bits.
    size_t k = static_cast<size_t>(reinterpret_cast<uintptr_t>(t));
    return k + (k >> 6);
  }
};

template <>
struct hash<string> {
  size_t operator()(const string& s) const {
    return static_cast<size_t>(Hash64(s));
  }
};

template <>
struct hash<tstring> {
  size_t operator()(const tstring& s) const {
    return static_cast<size_t>(Hash64(s.data(), s.size()));
  }
};

template <>
struct hash<StringPiece> {
  size_t operator()(StringPiece sp) const {
    return static_cast<size_t>(Hash64(sp.data(), sp.size()));
  }
};
using StringPieceHasher = ::tensorflow::hash<StringPiece>;

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& p) const {
    return Hash64Combine(hash<T>()(p.first), hash<U>()(p.second));
  }
};

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::tstring> {
  size_t operator()(const tensorflow::tstring& s) const {
    return static_cast<size_t>(tensorflow::Hash64(s.data(), s.size()));
  }
};
}  // namespace std

#endif  // TENSORFLOW_CORE_PLATFORM_HASH_H_
