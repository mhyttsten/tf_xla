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
class MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc() {
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
#include "tensorflow/core/lib/strings/proto_serialization.h"

#include <cstring>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// Helper for deterministic serialization.
class DeterministicSerializer {
 public:
  explicit DeterministicSerializer(const protobuf::MessageLite& msg)
      : DeterministicSerializer(msg, msg.ByteSizeLong()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "DeterministicSerializer");
}

  DeterministicSerializer(const protobuf::MessageLite& msg, size_t size)
      : size_(size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "DeterministicSerializer");

    char* ptr = space_;
    if (size_ > sizeof(space_)) {
      ptr = new char[size_];
      alloc_.reset(ptr);
    }
    bool ok = SerializeToBufferDeterministic(msg, ptr, size_);
    DCHECK(ok);
  }

  size_t size() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "size");
 return size_; }
  const char* data() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "data");
 return alloc_ == nullptr ? space_ : alloc_.get(); }

 private:
  // Avoid InlinedVector since it causes 2x slowdown in the compilation
  // of graphs containing large tensors in debug mode.
  static constexpr int kInlinedBufferSize = 256;
  const size_t size_;
  std::unique_ptr<char[]> alloc_;
  char space_[kInlinedBufferSize];
};
}  // namespace

bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
                                    string* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_4(mht_4_v, 241, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "SerializeToStringDeterministic");

  const size_t size = msg.ByteSizeLong();
  DCHECK_LE(size, static_cast<size_t>(INT_MAX));
  *result = string(size, '\0');
  return SerializeToBufferDeterministic(msg, const_cast<char*>(result->data()),
                                        result->size());
}

bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
                                    char* buffer, size_t size) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_5(mht_5_v, 254, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "SerializeToBufferDeterministic");

  DCHECK(msg.ByteSizeLong() == size && size <= static_cast<size_t>(INT_MAX));
  protobuf::io::ArrayOutputStream array_stream(buffer, size);
  protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  msg.SerializeWithCachedSizes(&output_stream);
  return !output_stream.HadError() &&
         size == static_cast<size_t>(output_stream.ByteCount());
}

bool AreSerializedProtosEqual(const protobuf::MessageLite& x,
                              const protobuf::MessageLite& y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_6(mht_6_v, 268, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "AreSerializedProtosEqual");

  const size_t size = x.ByteSizeLong();
  if (size != y.ByteSizeLong()) return false;
  if (size == 0) return true;
  DeterministicSerializer x_serialized(x, size);
  DeterministicSerializer y_serialized(y, size);
  return memcmp(x_serialized.data(), y_serialized.data(), size) == 0;
}

uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto,
                                uint64 seed) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_7(mht_7_v, 281, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "DeterministicProtoHash64");

  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size(), seed);
}

uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSstringsPSproto_serializationDTcc mht_8(mht_8_v, 289, "", "./tensorflow/core/lib/strings/proto_serialization.cc", "DeterministicProtoHash64");

  DeterministicSerializer serialized(proto);
  return Hash64(serialized.data(), serialized.size());
}

}  // namespace tensorflow
