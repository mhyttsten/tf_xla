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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc() {
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

#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

VariantTensorData::VariantTensorData(VariantTensorDataProto proto) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::VariantTensorData");

  FromProto(std::move(proto));
}

int VariantTensorData::tensors_size() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::tensors_size");
 return tensors_.size(); }

const Tensor& VariantTensorData::tensors(int index) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_2(mht_2_v, 204, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::tensors");

  return tensors_[index];
}

const std::vector<Tensor>& VariantTensorData::tensors() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_3(mht_3_v, 211, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::tensors");

  return tensors_;
}

Tensor* VariantTensorData::add_tensors() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_4(mht_4_v, 218, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::add_tensors");

  tensors_.emplace_back();
  return &(tensors_[tensors_.size() - 1]);
}

void VariantTensorData::ToProto(VariantTensorDataProto* proto) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_5(mht_5_v, 226, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::ToProto");

  proto->set_type_name(type_name());
  proto->set_metadata(metadata_);
  proto->clear_tensors();
  for (const auto& tensor : tensors_) {
    tensor.AsProtoField(proto->mutable_tensors()->Add());
  }
}

bool VariantTensorData::FromProto(VariantTensorDataProto proto) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_6(mht_6_v, 238, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::FromProto");

  // TODO(ebrevdo): Do this lazily.
  set_type_name(proto.type_name());
  set_metadata(proto.metadata());
  for (const auto& tensor : proto.tensors()) {
    Tensor tmp;
    if (!tmp.FromProto(tensor)) return false;
    tensors_.push_back(tmp);
  }
  return true;
}

bool VariantTensorData::FromConstProto(const VariantTensorDataProto& proto) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_7(mht_7_v, 253, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::FromConstProto");

  set_type_name(proto.type_name());
  set_metadata(proto.metadata());
  for (const auto& tensor : proto.tensors()) {
    Tensor tmp;
    if (!tmp.FromProto(tensor)) return false;
    tensors_.push_back(tmp);
  }
  return true;
}

string VariantTensorData::SerializeAsString() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_8(mht_8_v, 267, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::SerializeAsString");

  VariantTensorDataProto proto;
  ToProto(&proto);
  return proto.SerializeAsString();
}

bool VariantTensorData::SerializeToString(string* buf) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_9(mht_9_v, 276, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::SerializeToString");

  VariantTensorDataProto proto;
  ToProto(&proto);
  return proto.SerializeToString(buf);
}

bool VariantTensorData::ParseFromString(string s) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_10(mht_10_v, 286, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::ParseFromString");

  VariantTensorDataProto proto;
  const bool status = proto.ParseFromString(s);
  if (status) FromProto(std::move(proto));
  return status;
}

string VariantTensorData::DebugString() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_11(mht_11_v, 296, "", "./tensorflow/core/framework/variant_tensor_data.cc", "VariantTensorData::DebugString");

  string repeated_field = "";
  for (const auto& t : tensors_) {
    repeated_field =
        strings::StrCat(repeated_field, " tensors: ", t.DebugString());
  }
  return strings::StrCat("type_name: ", type_name(), " metadata: ", metadata_,
                         repeated_field);
}

string ProtoDebugString(const VariantTensorData& object) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTcc mht_12(mht_12_v, 309, "", "./tensorflow/core/framework/variant_tensor_data.cc", "ProtoDebugString");

  return object.DebugString();
}

}  // namespace tensorflow
