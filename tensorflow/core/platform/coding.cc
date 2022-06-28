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
class MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc() {
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

#include "tensorflow/core/platform/coding.h"

#include "tensorflow/core/platform/byte_order.h"

namespace tensorflow {
namespace core {

void EncodeFixed16(char* buf, uint16 value) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/platform/coding.cc", "EncodeFixed16");

  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
  }
}

void EncodeFixed32(char* buf, uint32 value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/platform/coding.cc", "EncodeFixed32");

  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
  }
}

void EncodeFixed64(char* buf, uint64 value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/platform/coding.cc", "EncodeFixed64");

  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
    buf[4] = (value >> 32) & 0xff;
    buf[5] = (value >> 40) & 0xff;
    buf[6] = (value >> 48) & 0xff;
    buf[7] = (value >> 56) & 0xff;
  }
}

void PutFixed16(string* dst, uint16 value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/platform/coding.cc", "PutFixed16");

  char buf[sizeof(value)];
  EncodeFixed16(buf, value);
  dst->append(buf, sizeof(buf));
}

void PutFixed32(string* dst, uint32 value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_4(mht_4_v, 248, "", "./tensorflow/core/platform/coding.cc", "PutFixed32");

  char buf[sizeof(value)];
  EncodeFixed32(buf, value);
  dst->append(buf, sizeof(buf));
}

void PutFixed64(string* dst, uint64 value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_5(mht_5_v, 257, "", "./tensorflow/core/platform/coding.cc", "PutFixed64");

  char buf[sizeof(value)];
  EncodeFixed64(buf, value);
  dst->append(buf, sizeof(buf));
}

char* EncodeVarint32(char* dst, uint32 v) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dst: \"" + (dst == nullptr ? std::string("nullptr") : std::string((char*)dst)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_6(mht_6_v, 267, "", "./tensorflow/core/platform/coding.cc", "EncodeVarint32");

  // Operate on characters as unsigneds
  unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
  static const int B = 128;
  if (v < (1 << 7)) {
    *(ptr++) = v;
  } else if (v < (1 << 14)) {
    *(ptr++) = v | B;
    *(ptr++) = v >> 7;
  } else if (v < (1 << 21)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = v >> 14;
  } else if (v < (1 << 28)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = v >> 21;
  } else {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = (v >> 21) | B;
    *(ptr++) = v >> 28;
  }
  return reinterpret_cast<char*>(ptr);
}

void PutVarint32(string* dst, uint32 v) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/platform/coding.cc", "PutVarint32");

  char buf[5];
  char* ptr = EncodeVarint32(buf, v);
  dst->append(buf, ptr - buf);
}

void PutVarint32(tstring* dst, uint32 v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_8(mht_8_v, 307, "", "./tensorflow/core/platform/coding.cc", "PutVarint32");

  char buf[5];
  char* ptr = EncodeVarint32(buf, v);
  dst->append(buf, ptr - buf);
}

char* EncodeVarint64(char* dst, uint64 v) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("dst: \"" + (dst == nullptr ? std::string("nullptr") : std::string((char*)dst)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_9(mht_9_v, 317, "", "./tensorflow/core/platform/coding.cc", "EncodeVarint64");

  static const int B = 128;
  unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
  while (v >= B) {
    *(ptr++) = (v & (B - 1)) | B;
    v >>= 7;
  }
  *(ptr++) = static_cast<unsigned char>(v);
  return reinterpret_cast<char*>(ptr);
}

void PutVarint64(string* dst, uint64 v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_10(mht_10_v, 331, "", "./tensorflow/core/platform/coding.cc", "PutVarint64");

  char buf[10];
  char* ptr = EncodeVarint64(buf, v);
  dst->append(buf, ptr - buf);
}

void PutVarint64(tstring* dst, uint64 v) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_11(mht_11_v, 340, "", "./tensorflow/core/platform/coding.cc", "PutVarint64");

  char buf[10];
  char* ptr = EncodeVarint64(buf, v);
  dst->append(buf, ptr - buf);
}

int VarintLength(uint64_t v) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_12(mht_12_v, 349, "", "./tensorflow/core/platform/coding.cc", "VarintLength");

  int len = 1;
  while (v >= 128) {
    v >>= 7;
    len++;
  }
  return len;
}

const char* GetVarint32Ptr(const char* p, const char* limit, uint32* value) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("p: \"" + (p == nullptr ? std::string("nullptr") : std::string((char*)p)) + "\"");
   mht_13_v.push_back("limit: \"" + (limit == nullptr ? std::string("nullptr") : std::string((char*)limit)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_13(mht_13_v, 363, "", "./tensorflow/core/platform/coding.cc", "GetVarint32Ptr");

  if (p < limit) {
    uint32 result = *(reinterpret_cast<const unsigned char*>(p));
    if ((result & 128) == 0) {
      *value = result;
      return p + 1;
    }
  }
  return GetVarint32PtrFallback(p, limit, value);
}

const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                   uint32* value) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("p: \"" + (p == nullptr ? std::string("nullptr") : std::string((char*)p)) + "\"");
   mht_14_v.push_back("limit: \"" + (limit == nullptr ? std::string("nullptr") : std::string((char*)limit)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_14(mht_14_v, 380, "", "./tensorflow/core/platform/coding.cc", "GetVarint32PtrFallback");

  uint32 result = 0;
  for (uint32 shift = 0; shift <= 28 && p < limit; shift += 7) {
    uint32 byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

bool GetVarint32(StringPiece* input, uint32* value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_15(mht_15_v, 400, "", "./tensorflow/core/platform/coding.cc", "GetVarint32");

  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GetVarint32Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = StringPiece(q, limit - q);
    return true;
  }
}

const char* GetVarint64Ptr(const char* p, const char* limit, uint64* value) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("p: \"" + (p == nullptr ? std::string("nullptr") : std::string((char*)p)) + "\"");
   mht_16_v.push_back("limit: \"" + (limit == nullptr ? std::string("nullptr") : std::string((char*)limit)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_16(mht_16_v, 417, "", "./tensorflow/core/platform/coding.cc", "GetVarint64Ptr");

  uint64 result = 0;
  for (uint32 shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64 byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

bool GetVarint64(StringPiece* input, uint64* value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSplatformPScodingDTcc mht_17(mht_17_v, 437, "", "./tensorflow/core/platform/coding.cc", "GetVarint64");

  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GetVarint64Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = StringPiece(q, limit - q);
    return true;
  }
}

}  // namespace core
}  // namespace tensorflow
