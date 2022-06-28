/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_
#define TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh() {
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


#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) ||                  \
    defined(_WIN32)
#define TF_TSTRING_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define TF_TSTRING_LITTLE_ENDIAN 0
#else
#error "Unable to detect endianness."
#endif

#if defined(__clang__) || \
    (defined(__GNUC__) && \
     ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
static inline uint32_t TF_swap32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}

#elif defined(_MSC_VER)
static inline uint32_t TF_swap32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}

#elif defined(__APPLE__)
static inline uint32_t TF_swap32(uint32_t host_int) {
  return OSSwapInt32(host_int);
}

#else
static inline uint32_t TF_swap32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else   // defined(__GLIBC__)
  return (((host_int & uint32_t{0xFF}) << 24) |
          ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) |
          ((host_int & uint32_t{0xFF000000}) >> 24));
#endif  // defined(__GLIBC__)
}
#endif

#if TF_TSTRING_LITTLE_ENDIAN
#define TF_le32toh(x) x
#else  // TF_TSTRING_LITTLE_ENDIAN
#define TF_le32toh(x) TF_swap32(x)
#endif  // TF_TSTRING_LITTLE_ENDIAN

static inline size_t TF_align16(size_t i) { return (i + 0xF) & ~0xF; }

static inline size_t TF_max(size_t a, size_t b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_0(mht_0_v, 242, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_max");
 return a > b ? a : b; }
static inline size_t TF_min(size_t a, size_t b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_1(mht_1_v, 246, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_min");
 return a < b ? a : b; }

typedef enum TF_TString_Type {  // NOLINT
  TF_TSTR_SMALL = 0x00,
  TF_TSTR_LARGE = 0x01,
  TF_TSTR_OFFSET = 0x02,
  TF_TSTR_VIEW = 0x03,
  TF_TSTR_TYPE_MASK = 0x03
} TF_TString_Type;

typedef struct TF_TString_Large {  // NOLINT
  size_t size;
  size_t cap;
  char *ptr;
} TF_TString_Large;

typedef struct TF_TString_Offset {  // NOLINT
  uint32_t size;
  uint32_t offset;
  uint32_t count;
} TF_TString_Offset;

typedef struct TF_TString_View {  // NOLINT
  size_t size;
  const char *ptr;
} TF_TString_View;

typedef struct TF_TString_Raw {  // NOLINT
  uint8_t raw[24];
} TF_TString_Raw;

typedef union TF_TString_Union {  // NOLINT
  TF_TString_Large large;
  TF_TString_Offset offset;
  TF_TString_View view;
  TF_TString_Raw raw;
} TF_TString_Union;

enum {
  TF_TString_SmallCapacity =
      (sizeof(TF_TString_Union) - sizeof(/* null delim */ char) -
       sizeof(/* uint8_t size */ uint8_t)),
};

typedef struct TF_TString_Small {  // NOLINT
  uint8_t size;
  char str[TF_TString_SmallCapacity + sizeof(/* null delim */ char)];
} TF_TString_Small;

typedef struct TF_TString {  // NOLINT
  union {
    // small conflicts with '#define small char' in RpcNdr.h for MSVC, so we use
    // smll instead.
    TF_TString_Small smll;
    TF_TString_Large large;
    TF_TString_Offset offset;
    TF_TString_View view;
    TF_TString_Raw raw;
  } u;
} TF_TString;

// TODO(dero): Fix for OSS, and add C only build test.
// _Static_assert(CHAR_BIT == 8);
// _Static_assert(sizeof(TF_TString) == 24);

static inline TF_TString_Type TF_TString_GetType(const TF_TString *str) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_2(mht_2_v, 314, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_GetType");

  return (TF_TString_Type)(str->u.raw.raw[0] & TF_TSTR_TYPE_MASK);  // NOLINT
}

// XXX(dero): For the big-endian case, this function could potentially be more
// performant and readable by always storing the string size as little-endian
// and always byte-swapping on big endian, resulting in a simple 'bswap'+'shr'
// (for architectures that have a bswap op).
static inline size_t TF_TString_ToActualSizeT(size_t size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_3(mht_3_v, 325, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_ToActualSizeT");

#if TF_TSTRING_LITTLE_ENDIAN
  return size >> 2;
#else   // TF_TSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);

  return (((mask << 2) & size) >> 2) | (~mask & size);
#endif  // TF_TSTRING_LITTLE_ENDIAN
}

static inline size_t TF_TString_ToInternalSizeT(size_t size,
                                                TF_TString_Type type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_4(mht_4_v, 340, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_ToInternalSizeT");

#if TF_TSTRING_LITTLE_ENDIAN
  return (size << 2) | type;
#else   // TF_TSTRING_LITTLE_ENDIAN
  // 0xFF000000 or 0xFF00000000000000 depending on platform
  static const size_t mask = ~((~(size_t)0) >> 8);

  return (mask & (size << 2)) | (~mask & size) |
         ((size_t)type << ((sizeof(size_t) - 1) * 8));  // NOLINT
#endif  // TF_TSTRING_LITTLE_ENDIAN
}

static inline void TF_TString_Init(TF_TString *str) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_5(mht_5_v, 355, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Init");

  memset(str->u.raw.raw, 0, sizeof(TF_TString_Raw));
}

static inline void TF_TString_Dealloc(TF_TString *str) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_6(mht_6_v, 362, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Dealloc");

  if (TF_TString_GetType(str) == TF_TSTR_LARGE &&
      str->u.large.ptr != NULL) {  // NOLINT
    free(str->u.large.ptr);
    TF_TString_Init(str);
  }
}

static inline size_t TF_TString_GetSize(const TF_TString *str) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_7(mht_7_v, 373, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_GetSize");

  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.size >> 2;
    case TF_TSTR_LARGE:
      return TF_TString_ToActualSizeT(str->u.large.size);
    case TF_TSTR_OFFSET:
      return TF_le32toh(str->u.offset.size) >> 2;
    case TF_TSTR_VIEW:
      return TF_TString_ToActualSizeT(str->u.view.size);
    default:
      return 0;  // Unreachable.
  }
}

static inline size_t TF_TString_GetCapacity(const TF_TString *str) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_8(mht_8_v, 391, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_GetCapacity");

  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return TF_TString_SmallCapacity;
    case TF_TSTR_LARGE:
      return str->u.large.cap;
    case TF_TSTR_OFFSET:
    case TF_TSTR_VIEW:
    default:
      return 0;
  }
}

static inline const char *TF_TString_GetDataPointer(const TF_TString *str) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_9(mht_9_v, 407, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_GetDataPointer");

  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.str;
    case TF_TSTR_LARGE:
      return str->u.large.ptr;
    case TF_TSTR_OFFSET:
      return (const char *)str + str->u.offset.offset;  // NOLINT
    case TF_TSTR_VIEW:
      return str->u.view.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

static inline char *TF_TString_ResizeUninitialized(TF_TString *str,
                                                   size_t new_size) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_10(mht_10_v, 427, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_ResizeUninitialized");

  size_t curr_size = TF_TString_GetSize(str);
  size_t copy_size = TF_min(new_size, curr_size);

  TF_TString_Type curr_type = TF_TString_GetType(str);
  const char *curr_ptr = TF_TString_GetDataPointer(str);

  // Case: SMALL/LARGE/VIEW/OFFSET -> SMALL
  if (new_size <= TF_TString_SmallCapacity) {
    str->u.smll.size = (uint8_t)((new_size << 2) | TF_TSTR_SMALL);  // NOLINT
    str->u.smll.str[new_size] = '\0';

    if (curr_type != TF_TSTR_SMALL && copy_size) {
      memcpy(str->u.smll.str, curr_ptr, copy_size);
    }

    if (curr_type == TF_TSTR_LARGE) {
      free((void *)curr_ptr);  // NOLINT
    }

    // We do not clear out the newly excluded region.

    return str->u.smll.str;
  }

  // Case: SMALL/LARGE/VIEW/OFFSET -> LARGE
  size_t new_cap;
  size_t curr_cap = TF_TString_GetCapacity(str);

  if (new_size < curr_size && new_size < curr_cap / 2) {
    // TODO(dero): Replace with shrink_to_fit flag.
    new_cap = TF_align16(curr_cap / 2 + 1) - 1;
  } else if (new_size > curr_cap) {
    new_cap = TF_align16(new_size + 1) - 1;
  } else {
    new_cap = curr_cap;
  }

  char *new_ptr;
  if (new_cap == curr_cap) {
    new_ptr = str->u.large.ptr;
  } else if (curr_type == TF_TSTR_LARGE) {
    new_ptr = (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    if (copy_size) {
      memcpy(new_ptr, curr_ptr, copy_size);
    }
  }

  str->u.large.size = TF_TString_ToInternalSizeT(new_size, TF_TSTR_LARGE);
  str->u.large.ptr = new_ptr;
  str->u.large.ptr[new_size] = '\0';
  str->u.large.cap = new_cap;

  return str->u.large.ptr;
}

static inline char *TF_TString_GetMutableDataPointer(TF_TString *str) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_11(mht_11_v, 488, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_GetMutableDataPointer");

  switch (TF_TString_GetType(str)) {
    case TF_TSTR_SMALL:
      return str->u.smll.str;
    case TF_TSTR_OFFSET:
    case TF_TSTR_VIEW:
      // Convert OFFSET/VIEW to SMALL/LARGE
      TF_TString_ResizeUninitialized(str, TF_TString_GetSize(str));
      return (TF_TString_GetType(str) == TF_TSTR_SMALL) ? str->u.smll.str
                                                        : str->u.large.ptr;
    case TF_TSTR_LARGE:
      return str->u.large.ptr;
    default:
      // Unreachable.
      return NULL;  // NOLINT
  }
}

static inline void TF_TString_Reserve(TF_TString *str, size_t new_cap) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_12(mht_12_v, 509, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Reserve");

  TF_TString_Type curr_type = TF_TString_GetType(str);

  if (new_cap <= TF_TString_SmallCapacity) {
    // We do nothing, we let Resize/GetMutableDataPointer handle the
    // conversion to SMALL from VIEW/OFFSET when the need arises.
    // In the degenerate case, where new_cap <= TF_TString_SmallCapacity,
    // curr_size > TF_TString_SmallCapacity, and the type is VIEW/OFFSET, we
    // defer the malloc to Resize/GetMutableDataPointer.
    return;
  }

  if (curr_type == TF_TSTR_LARGE && new_cap <= str->u.large.cap) {
    // We handle reduced cap in resize.
    return;
  }

  // Case: VIEW/OFFSET -> LARGE or grow an existing LARGE type
  size_t curr_size = TF_TString_GetSize(str);
  const char *curr_ptr = TF_TString_GetDataPointer(str);

  // Since VIEW and OFFSET types are read-only, their capacity is effectively 0.
  // So we make sure we have enough room in the VIEW and OFFSET cases.
  new_cap = TF_align16(TF_max(new_cap, curr_size) + 1) - 1;

  if (curr_type == TF_TSTR_LARGE) {
    str->u.large.ptr =
        (char *)realloc(str->u.large.ptr, new_cap + 1);  // NOLINT
  } else {
    // Convert to Large
    char *new_ptr = (char *)malloc(new_cap + 1);  // NOLINT
    memcpy(new_ptr, curr_ptr, curr_size);

    str->u.large.size = TF_TString_ToInternalSizeT(curr_size, TF_TSTR_LARGE);
    str->u.large.ptr = new_ptr;
    str->u.large.ptr[curr_size] = '\0';
  }

  str->u.large.cap = new_cap;
}

static inline void TF_TString_ReserveAmortized(TF_TString *str,
                                               size_t new_cap) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_13(mht_13_v, 554, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_ReserveAmortized");

  const size_t curr_cap = TF_TString_GetCapacity(str);
  if (new_cap > curr_cap) {
    TF_TString_Reserve(str, new_cap > 2 * curr_cap ? new_cap : 2 * curr_cap);
  }
}

static inline char *TF_TString_Resize(TF_TString *str, size_t new_size,
                                      char c) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_14(mht_14_v, 566, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Resize");

  size_t curr_size = TF_TString_GetSize(str);
  char *cstr = TF_TString_ResizeUninitialized(str, new_size);

  if (new_size > curr_size) {
    memset(cstr + curr_size, c, new_size - curr_size);
  }

  return cstr;
}

static inline void TF_TString_AssignView(TF_TString *dst, const char *src,
                                         size_t size) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_15(mht_15_v, 582, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_AssignView");

  TF_TString_Dealloc(dst);

  dst->u.view.size = TF_TString_ToInternalSizeT(size, TF_TSTR_VIEW);
  dst->u.view.ptr = src;
}

static inline void TF_TString_AppendN(TF_TString *dst, const char *src,
                                      size_t src_size) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_16(mht_16_v, 594, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_AppendN");

  if (!src_size) return;

  size_t dst_size = TF_TString_GetSize(dst);

  // For append use cases, we want to ensure amortized growth.
  TF_TString_ReserveAmortized(dst, dst_size + src_size);
  char *dst_c = TF_TString_ResizeUninitialized(dst, dst_size + src_size);

  memcpy(dst_c + dst_size, src, src_size);
}

static inline void TF_TString_Append(TF_TString *dst, const TF_TString *src) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_17(mht_17_v, 609, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Append");

  const char *src_c = TF_TString_GetDataPointer(src);
  size_t size = TF_TString_GetSize(src);

  TF_TString_AppendN(dst, src_c, size);
}

static inline void TF_TString_Copy(TF_TString *dst, const char *src,
                                   size_t size) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_18(mht_18_v, 621, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Copy");

  char *dst_c = TF_TString_ResizeUninitialized(dst, size);

  if (size) memcpy(dst_c, src, size);
}

static inline void TF_TString_Assign(TF_TString *dst, const TF_TString *src) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_19(mht_19_v, 630, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Assign");

  if (dst == src) return;

  TF_TString_Dealloc(dst);

  switch (TF_TString_GetType(src)) {
    case TF_TSTR_SMALL:
    case TF_TSTR_VIEW:
      *dst = *src;
      return;
    case TF_TSTR_LARGE: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_Copy(dst, src_c, size);
    }
      return;
    case TF_TSTR_OFFSET: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

static inline void TF_TString_Move(TF_TString *dst, TF_TString *src) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPSctstring_internalDTh mht_20(mht_20_v, 662, "", "./tensorflow/core/platform/ctstring_internal.h", "TF_TString_Move");

  if (dst == src) return;

  TF_TString_Dealloc(dst);

  switch (TF_TString_GetType(src)) {
    case TF_TSTR_SMALL:
    case TF_TSTR_VIEW:
      *dst = *src;
      return;
    case TF_TSTR_LARGE:
      *dst = *src;
      TF_TString_Init(src);
      return;
    case TF_TSTR_OFFSET: {
      const char *src_c = TF_TString_GetDataPointer(src);
      size_t size = TF_TString_GetSize(src);

      TF_TString_AssignView(dst, src_c, size);
    }
      return;
    default:
      return;  // Unreachable.
  }
}

#endif  // TENSORFLOW_CORE_PLATFORM_CTSTRING_INTERNAL_H_
