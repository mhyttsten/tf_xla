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
class MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc() {
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

#include "tensorflow/core/platform/tensor_coding.h"

#include <vector>

#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
#include "strings/cord_varint.h"
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

namespace tensorflow {
namespace port {

void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/platform/tensor_coding.cc", "AssignRefCounted");

  out->assign(src.data(), src.size());
}

void EncodeStringList(const tstring* strings, int64_t n, string* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/platform/tensor_coding.cc", "EncodeStringList");

  out->clear();
  for (int i = 0; i < n; ++i) {
    core::PutVarint32(out, strings[i].size());
  }
  for (int i = 0; i < n; ++i) {
    out->append(strings[i]);
  }
}

bool DecodeStringList(const string& src, tstring* strings, int64_t n) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("src: \"" + src + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/platform/tensor_coding.cc", "DecodeStringList");

  std::vector<uint32> sizes(n);
  StringPiece reader(src);
  int64_t tot = 0;
  for (auto& v : sizes) {
    if (!core::GetVarint32(&reader, &v)) return false;
    tot += v;
  }
  if (tot != static_cast<int64_t>(reader.size())) {
    return false;
  }

  tstring* data = strings;
  for (int64_t i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.size()) {
      return false;
    }
    data->assign(reader.data(), size);
    reader.remove_prefix(size);
  }

  return true;
}

void CopyFromArray(string* s, const char* base, size_t bytes) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("base: \"" + (base == nullptr ? std::string("nullptr") : std::string((char*)base)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/platform/tensor_coding.cc", "CopyFromArray");

  s->assign(base, bytes);
}

class StringListEncoderImpl : public StringListEncoder {
 public:
  explicit StringListEncoderImpl(string* out) : out_(out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/platform/tensor_coding.cc", "StringListEncoderImpl");
}
  ~StringListEncoderImpl() override = default;

  void Append(const protobuf::MessageLite& m) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_5(mht_5_v, 266, "", "./tensorflow/core/platform/tensor_coding.cc", "Append");

    core::PutVarint32(out_, m.ByteSizeLong());
    tensorflow::string serialized_message;
    m.AppendToString(&serialized_message);
    strings::StrAppend(&rest_, serialized_message);
  }

  void Append(const string& s) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/platform/tensor_coding.cc", "Append");

    core::PutVarint32(out_, s.length());
    strings::StrAppend(&rest_, s);
  }

  void Finalize() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_7(mht_7_v, 285, "", "./tensorflow/core/platform/tensor_coding.cc", "Finalize");
 strings::StrAppend(out_, rest_); }

 private:
  string* out_;
  string rest_;
};

class StringListDecoderImpl : public StringListDecoder {
 public:
  explicit StringListDecoderImpl(const string& in) : reader_(in) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("in: \"" + in + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_8(mht_8_v, 298, "", "./tensorflow/core/platform/tensor_coding.cc", "StringListDecoderImpl");
}
  ~StringListDecoderImpl() override = default;

  bool ReadSizes(std::vector<uint32>* sizes) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_9(mht_9_v, 304, "", "./tensorflow/core/platform/tensor_coding.cc", "ReadSizes");

    int64_t total = 0;
    for (auto& size : *sizes) {
      if (!core::GetVarint32(&reader_, &size)) return false;
      total += size;
    }
    if (total != static_cast<int64_t>(reader_.size())) {
      return false;
    }
    return true;
  }

  const char* Data(uint32 size) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_10(mht_10_v, 319, "", "./tensorflow/core/platform/tensor_coding.cc", "Data");

    const char* data = reader_.data();
    reader_.remove_prefix(size);
    return data;
  }

 private:
  StringPiece reader_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out) {
  return std::unique_ptr<StringListEncoder>(new StringListEncoderImpl(out));
}

std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in) {
  return std::unique_ptr<StringListDecoder>(new StringListDecoderImpl(in));
}

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
void AssignRefCounted(StringPiece src, core::RefCounted* obj, absl::Cord* out) {
  obj->Ref();
  *out = absl::MakeCordFromExternal(src, [obj] { obj->Unref(); });
}

void EncodeStringList(const tstring* strings, int64_t n, absl::Cord* out) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_11(mht_11_v, 346, "", "./tensorflow/core/platform/tensor_coding.cc", "EncodeStringList");

  out->Clear();
  for (int i = 0; i < n; ++i) {
    ::strings::CordAppendVarint(strings[i].size(), out);
  }
  for (int i = 0; i < n; ++i) {
    out->Append(strings[i]);
  }
}

bool DecodeStringList(const absl::Cord& src, string* strings, int64_t n) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_12(mht_12_v, 359, "", "./tensorflow/core/platform/tensor_coding.cc", "DecodeStringList");

  std::vector<uint32> sizes(n);
  CordReader reader(src);
  int64_t tot = 0;
  for (auto& v : sizes) {
    if (!::strings::CordReaderReadVarint(&reader, &v)) return false;
    tot += v;
  }
  if (tot != reader.Available()) {
    return false;
  }
  string* data = strings;
  for (int i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.Available()) {
      return false;
    }
    gtl::STLStringResizeUninitialized(data, size);
    reader.ReadN(size, gtl::string_as_array(data));
  }
  return true;
}

bool DecodeStringList(const absl::Cord& src, tstring* strings, int64_t n) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_13(mht_13_v, 385, "", "./tensorflow/core/platform/tensor_coding.cc", "DecodeStringList");

  std::vector<uint32> sizes(n);
  CordReader reader(src);
  int64_t tot = 0;
  for (auto& v : sizes) {
    if (!::strings::CordReaderReadVarint(&reader, &v)) return false;
    tot += v;
  }
  if (tot != reader.Available()) {
    return false;
  }
  tstring* data = strings;
  for (int i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.Available()) {
      return false;
    }
    data->resize_uninitialized(size);
    reader.ReadN(size, data->data());
  }
  return true;
}

void CopyFromArray(absl::Cord* c, const char* base, size_t bytes) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("base: \"" + (base == nullptr ? std::string("nullptr") : std::string((char*)base)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_14(mht_14_v, 412, "", "./tensorflow/core/platform/tensor_coding.cc", "CopyFromArray");

  c->CopyFrom(base, bytes);
}

class CordStringListEncoderImpl : public StringListEncoder {
 public:
  explicit CordStringListEncoderImpl(absl::Cord* out) : out_(out) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_15(mht_15_v, 421, "", "./tensorflow/core/platform/tensor_coding.cc", "CordStringListEncoderImpl");
}
  ~CordStringListEncoderImpl() override = default;

  void Append(const protobuf::MessageLite& m) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_16(mht_16_v, 427, "", "./tensorflow/core/platform/tensor_coding.cc", "Append");

    ::strings::CordAppendVarint(m.ByteSizeLong(), out_);
    m.AppendToString(&rest_);
  }

  void Append(const string& s) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_17(mht_17_v, 436, "", "./tensorflow/core/platform/tensor_coding.cc", "Append");

    ::strings::CordAppendVarint(s.length(), out_);
    rest_.append(s.data(), s.size());
  }

  void Finalize() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_18(mht_18_v, 444, "", "./tensorflow/core/platform/tensor_coding.cc", "Finalize");
 out_->Append(rest_); }

 private:
  absl::Cord* out_;
  string rest_;
};

class CordStringListDecoderImpl : public StringListDecoder {
 public:
  explicit CordStringListDecoderImpl(const absl::Cord& in) : reader_(in) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_19(mht_19_v, 456, "", "./tensorflow/core/platform/tensor_coding.cc", "CordStringListDecoderImpl");
}
  ~CordStringListDecoderImpl() override = default;

  bool ReadSizes(std::vector<uint32>* sizes) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_20(mht_20_v, 462, "", "./tensorflow/core/platform/tensor_coding.cc", "ReadSizes");

    int64_t total = 0;
    for (auto& size : *sizes) {
      if (!::strings::CordReaderReadVarint(&reader_, &size)) return false;
      total += size;
    }
    if (total != static_cast<int64_t>(reader_.Available())) {
      return false;
    }
    return true;
  }

  const char* Data(uint32 size) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSplatformPStensor_codingDTcc mht_21(mht_21_v, 477, "", "./tensorflow/core/platform/tensor_coding.cc", "Data");

    tmp_.resize(size);
    reader_.ReadN(size, tmp_.data());
    return tmp_.data();
  }

 private:
  CordReader reader_;
  std::vector<char> tmp_;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(absl::Cord* out) {
  return std::unique_ptr<StringListEncoder>(new CordStringListEncoderImpl(out));
}

std::unique_ptr<StringListDecoder> NewStringListDecoder(const absl::Cord& in) {
  return std::unique_ptr<StringListDecoder>(new CordStringListDecoderImpl(in));
}

#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

}  // namespace port
}  // namespace tensorflow
