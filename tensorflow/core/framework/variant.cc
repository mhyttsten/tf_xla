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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc() {
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

#include "tensorflow/core/framework/variant.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {

Variant::~Variant() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/framework/variant.cc", "Variant::~Variant");
 ResetMemory(); }

bool Variant::Decode(VariantTensorData data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_1(mht_1_v, 198, "", "./tensorflow/core/framework/variant.cc", "Variant::Decode");

  if (!is_empty()) {
    return GetValue()->Decode(std::move(data));
  }
  return true;
}

template <>
void* Variant::get() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/framework/variant.cc", "Variant::get");

  if (is_empty()) {
    return nullptr;
  }
  return GetValue()->RawPtr();
}

template <>
const void* Variant::get() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/framework/variant.cc", "Variant::get");

  if (is_empty()) {
    return nullptr;
  }
  return GetValue()->RawPtr();
}

template <>
string TypeNameVariant(const VariantTensorDataProto& value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_4(mht_4_v, 231, "", "./tensorflow/core/framework/variant.cc", "TypeNameVariant");

  return value.type_name();
}

template <>
void EncodeVariant(const VariantTensorDataProto& value,
                   VariantTensorData* data) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_5(mht_5_v, 240, "", "./tensorflow/core/framework/variant.cc", "EncodeVariant");

  data->FromConstProto(value);
}

template <>
bool DecodeVariant(VariantTensorData* data, VariantTensorDataProto* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_6(mht_6_v, 248, "", "./tensorflow/core/framework/variant.cc", "DecodeVariant");

  data->ToProto(value);
  return true;
}

template <>
void EncodeVariant(const VariantTensorDataProto& value, string* buf) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_7(mht_7_v, 257, "", "./tensorflow/core/framework/variant.cc", "EncodeVariant");

  value.SerializeToString(buf);
}

template <>
bool DecodeVariant(string* buf, VariantTensorDataProto* value) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_8(mht_8_v, 265, "", "./tensorflow/core/framework/variant.cc", "DecodeVariant");

  return value->ParseFromString(*buf);
}

void EncodeVariantList(const Variant* variant_array, int64_t n,
                       std::unique_ptr<port::StringListEncoder> e) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_9(mht_9_v, 273, "", "./tensorflow/core/framework/variant.cc", "EncodeVariantList");

  for (int i = 0; i < n; ++i) {
    string s;
    variant_array[i].Encode(&s);
    e->Append(s);
  }
  e->Finalize();
}

bool DecodeVariantList(std::unique_ptr<port::StringListDecoder> d,
                       Variant* variant_array, int64_t n) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariantDTcc mht_10(mht_10_v, 286, "", "./tensorflow/core/framework/variant.cc", "DecodeVariantList");

  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  for (int i = 0; i < n; ++i) {
    if (variant_array[i].is_empty()) {
      variant_array[i] = VariantTensorDataProto();
    }
    // TODO(ebrevdo): Replace with StringPiece?  Any way to make this a
    // zero-copy operation that keeps a reference to the data in d?
    string str(d->Data(sizes[i]), sizes[i]);
    if (!variant_array[i].Decode(std::move(str))) return false;
    if (!DecodeUnaryVariant(&variant_array[i])) {
      LOG(ERROR) << "Could not decode variant with type_name: \""
                 << variant_array[i].TypeName()
                 << "\".  Perhaps you forgot to register a "
                    "decoder via REGISTER_UNARY_VARIANT_DECODE_FUNCTION?";
      return false;
    }
  }
  return true;
}

}  // end namespace tensorflow
