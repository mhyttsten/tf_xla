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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc() {
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
#include "tensorflow/core/kernels/tensor_list.h"

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/coding.h"

namespace tensorflow {

TensorList::~TensorList() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/kernels/tensor_list.cc", "TensorList::~TensorList");

  if (tensors_) tensors_->Unref();
}

void TensorList::Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/kernels/tensor_list.cc", "TensorList::Encode");

  data->set_type_name(TypeName());
  std::vector<size_t> invalid_indices;
  for (size_t i = 0; i < tensors().size(); i++) {
    if (tensors().at(i).dtype() != DT_INVALID) {
      *data->add_tensors() = tensors().at(i);
    } else {
      invalid_indices.push_back(i);
    }
  }
  string metadata;
  // TODO(b/118838800): Add a proto for storing the metadata.
  // Metadata format:
  // <num_invalid_tensors><invalid_indices><element_dtype><element_shape_proto>
  core::PutVarint64(&metadata, static_cast<uint64>(invalid_indices.size()));
  for (size_t i : invalid_indices) {
    core::PutVarint64(&metadata, static_cast<uint64>(i));
  }
  core::PutVarint64(&metadata, static_cast<uint64>(element_dtype));
  core::PutVarint64(&metadata, static_cast<uint64>(max_num_elements));
  TensorShapeProto element_shape_proto;
  element_shape.AsProto(&element_shape_proto);
  element_shape_proto.AppendToString(&metadata);
  data->set_metadata(metadata);
}

static Status TensorListDeviceCopy(
    const TensorList& from, TensorList* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/tensor_list.cc", "TensorListDeviceCopy");

  to->element_shape = from.element_shape;
  to->element_dtype = from.element_dtype;
  to->max_num_elements = from.max_num_elements;
  to->tensors().reserve(from.tensors().size());
  for (const Tensor& t : from.tensors()) {
    to->tensors().emplace_back(t.dtype());
    if (t.dtype() != DT_INVALID) {
      TF_RETURN_IF_ERROR(copy(t, &to->tensors().back()));
    }
  }
  return Status::OK();
}

#define REGISTER_LIST_COPY(DIRECTION)                                         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION(TensorList, DIRECTION, \
                                                       TensorListDeviceCopy)

REGISTER_LIST_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_LIST_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(TensorList, TensorList::kTypeName);

bool TensorList::Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_listDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/kernels/tensor_list.cc", "TensorList::Decode");

  // TODO(srbs): Change the signature to Decode(VariantTensorData data) so
  // that we do not have to copy each tensor individually below. This would
  // require changing VariantTensorData::tensors() as well.
  string metadata;
  data.get_metadata(&metadata);
  uint64 scratch;
  StringPiece iter(metadata);
  std::vector<size_t> invalid_indices;
  core::GetVarint64(&iter, &scratch);
  size_t num_invalid_tensors = static_cast<size_t>(scratch);
  invalid_indices.resize(num_invalid_tensors);
  for (size_t i = 0; i < num_invalid_tensors; i++) {
    core::GetVarint64(&iter, &scratch);
    invalid_indices[i] = static_cast<size_t>(scratch);
  }

  size_t total_num_tensors = data.tensors().size() + num_invalid_tensors;
  tensors().reserve(total_num_tensors);
  std::vector<size_t>::iterator invalid_indices_it = invalid_indices.begin();
  std::vector<Tensor>::const_iterator tensors_it = data.tensors().begin();
  for (size_t i = 0; i < total_num_tensors; i++) {
    if (invalid_indices_it != invalid_indices.end() &&
        *invalid_indices_it == i) {
      tensors().emplace_back(Tensor(DT_INVALID));
      invalid_indices_it++;
    } else if (tensors_it != data.tensors().end()) {
      tensors().emplace_back(*tensors_it);
      tensors_it++;
    } else {
      // VariantTensorData is corrupted.
      return false;
    }
  }

  core::GetVarint64(&iter, &scratch);
  element_dtype = static_cast<DataType>(scratch);
  core::GetVarint64(&iter, &scratch);
  max_num_elements = static_cast<int>(scratch);
  TensorShapeProto element_shape_proto;
  element_shape_proto.ParseFromString(string(iter.data(), iter.size()));
  element_shape = PartialTensorShape(element_shape_proto);
  return true;
}

const char TensorList::kTypeName[] = "tensorflow::TensorList";

}  // namespace tensorflow
