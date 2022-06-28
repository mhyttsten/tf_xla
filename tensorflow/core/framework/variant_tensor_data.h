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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh() {
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


#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class VariantTensorDataProto;

// The serialization format for Variant objects. Objects with references to
// other Tensors can simply store those tensors in the `tensors` field, and
// serialize other metadata content in to the `metadata` field. Objects can
// optionally set the `type_name` for type-checking before deserializing an
// object.
//
// This is the native C++ class equivalent of VariantTensorDataProto. They are
// separate so that kernels do not need to depend on protos.
class VariantTensorData {
 public:
  VariantTensorData() = default;

  // TODO(b/118823936): This silently returns if the proto is invalid.
  // Consider calling FromProto explicitly instead.
  VariantTensorData(VariantTensorDataProto proto);

  // Name of the type of objects being serialized.
  const std::string& type_name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/framework/variant_tensor_data.h", "type_name");
 return type_name_; }
  void set_type_name(const std::string& type_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_1(mht_1_v, 221, "", "./tensorflow/core/framework/variant_tensor_data.h", "set_type_name");
 type_name_ = type_name; }

  template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value>
  struct PODResolver {};

  // Portions of the object that are not Tensors.
  // Directly supported types include string POD types.
  template <typename T>
  void set_metadata(const T& value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/framework/variant_tensor_data.h", "set_metadata");

    SetMetadata<T>(value, PODResolver<T>());
  }

  template <typename T>
  bool get_metadata(T* value) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_3(mht_3_v, 240, "", "./tensorflow/core/framework/variant_tensor_data.h", "get_metadata");

    return GetMetadata<T>(value, PODResolver<T>());
  }

  std::string& metadata_string() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_4(mht_4_v, 247, "", "./tensorflow/core/framework/variant_tensor_data.h", "metadata_string");
 return metadata_; }

  const std::string& metadata_string() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_5(mht_5_v, 252, "", "./tensorflow/core/framework/variant_tensor_data.h", "metadata_string");
 return metadata_; }

  // Tensors contained within objects being serialized.
  int tensors_size() const;
  const Tensor& tensors(int index) const;
  const std::vector<Tensor>& tensors() const;
  Tensor* add_tensors();

  // A more general version of add_tensors. Parameters are perfectly forwarded
  // to the constructor of the tensor added here.
  template <typename... TensorConstructorArgs>
  Tensor* add_tensor(TensorConstructorArgs&&... args);

  // Conversion to and from VariantTensorDataProto
  void ToProto(VariantTensorDataProto* proto) const;
  // This allows optimizations via std::move.
  bool FromProto(VariantTensorDataProto proto);
  bool FromConstProto(const VariantTensorDataProto& proto);

  // Serialization via VariantTensorDataProto
  std::string SerializeAsString() const;
  bool SerializeToString(std::string* buf);
  bool ParseFromString(std::string s);

  std::string DebugString() const;

 public:
  std::string type_name_;
  std::string metadata_;
  std::vector<Tensor> tensors_;

 private:
  template <typename T>
  void SetMetadata(const std::string& value,
                   PODResolver<T, false /* is_pod */>) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_6(mht_6_v, 290, "", "./tensorflow/core/framework/variant_tensor_data.h", "SetMetadata");

    metadata_ = value;
  }

  template <typename T>
  bool GetMetadata(std::string* value,
                   PODResolver<T, false /* is_pod */>) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_7(mht_7_v, 299, "", "./tensorflow/core/framework/variant_tensor_data.h", "GetMetadata");

    *value = metadata_;
    return true;
  }

  template <typename T>
  void SetMetadata(const T& value, PODResolver<T, true /* is_pod */>) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_8(mht_8_v, 308, "", "./tensorflow/core/framework/variant_tensor_data.h", "SetMetadata");

    metadata_.assign(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  template <typename T>
  bool GetMetadata(T* value, PODResolver<T, true /* is_pod */>) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_tensor_dataDTh mht_9(mht_9_v, 316, "", "./tensorflow/core/framework/variant_tensor_data.h", "GetMetadata");

    if (metadata_.size() != sizeof(T)) return false;
    std::copy_n(metadata_.data(), sizeof(T), reinterpret_cast<char*>(value));
    return true;
  }
};

// For backwards compatibility for when this was a proto
std::string ProtoDebugString(const VariantTensorData& object);

template <typename... TensorConstructorArgs>
Tensor* VariantTensorData::add_tensor(TensorConstructorArgs&&... args) {
  tensors_.emplace_back(std::forward<TensorConstructorArgs>(args)...);
  return &tensors_.back();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
