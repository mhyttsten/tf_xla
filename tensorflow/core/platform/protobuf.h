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

#ifndef TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
#define TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh() {
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


#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

// Import whatever namespace protobuf comes from into the
// ::tensorflow::protobuf namespace.
//
// TensorFlow code should use the ::tensorflow::protobuf namespace to
// refer to all protobuf APIs.

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/map.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/field_comparator.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "google/protobuf/util/type_resolver_util.h"

namespace tensorflow {

namespace protobuf = ::google::protobuf;
using protobuf_int64 = ::google::protobuf::int64;
using protobuf_uint64 = ::google::protobuf::uint64;
extern const char* kProtobufInt64Typename;
extern const char* kProtobufUint64Typename;

// Parses a protocol buffer contained in a string in the binary wire format.
// Returns true on success. Note: Unlike protobuf's builtin ParseFromString,
// this function has no size restrictions on the total size of the encoded
// protocol buffer.
bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                         const std::string& serialized);
bool ParseProtoUnlimited(protobuf::MessageLite* proto, const void* serialized,
                         size_t size);
inline bool ParseProtoUnlimited(protobuf::MessageLite* proto,
                                const tstring& serialized) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("serialized: \"" + (std::string)serialized + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_0(mht_0_v, 232, "", "./tensorflow/core/platform/protobuf.h", "ParseProtoUnlimited");

  return ParseProtoUnlimited(proto, serialized.data(), serialized.size());
}

// Returns the string value for the value of a string or bytes protobuf field.
inline const std::string& ProtobufStringToString(const std::string& s) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/platform/protobuf.h", "ProtobufStringToString");

  return s;
}

// Set <dest> to <src>. Swapping is allowed, as <src> does not need to be
// preserved.
inline void SetProtobufStringSwapAllowed(std::string* src, std::string* dest) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_2(mht_2_v, 250, "", "./tensorflow/core/platform/protobuf.h", "SetProtobufStringSwapAllowed");

  *dest = std::move(*src);
}

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
// These versions of ProtobufStringToString and SetProtobufString get used by
// tools/proto_text's generated code.  They have the same name as the versions
// in core/platform/protobuf.h, so the generation code doesn't need to determine
// if the type is Cord or string at generation time.
inline std::string ProtobufStringToString(const absl::Cord& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_3(mht_3_v, 262, "", "./tensorflow/core/platform/protobuf.h", "ProtobufStringToString");

  return std::string(s);
}
inline void SetProtobufStringSwapAllowed(std::string* src, absl::Cord* dest) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_4(mht_4_v, 268, "", "./tensorflow/core/platform/protobuf.h", "SetProtobufStringSwapAllowed");

  dest->CopyFrom(*src);
}
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

inline bool SerializeToTString(const protobuf::MessageLite& proto,
                               tstring* output) {
  size_t size = proto.ByteSizeLong();
  output->resize_uninitialized(size);
  return proto.SerializeWithCachedSizesToArray(
      reinterpret_cast<uint8*>(output->data()));
}

inline bool ParseFromTString(const tstring& input,
                             protobuf::MessageLite* proto) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("input: \"" + (std::string)input + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSprotobufDTh mht_5(mht_5_v, 286, "", "./tensorflow/core/platform/protobuf.h", "ParseFromTString");

  return proto->ParseFromArray(input.data(), static_cast<int>(input.size()));
}

// Analogue to StringOutputStream for tstring.
class TStringOutputStream : public protobuf::io::ZeroCopyOutputStream {
 public:
  explicit TStringOutputStream(tstring* target);
  ~TStringOutputStream() override = default;

  TStringOutputStream(const TStringOutputStream&) = delete;
  void operator=(const TStringOutputStream&) = delete;

  bool Next(void** data, int* size) override;
  void BackUp(int count) override;
  int64_t ByteCount() const override;

 private:
  static constexpr int kMinimumSize = 16;

  tstring* target_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROTOBUF_H_
