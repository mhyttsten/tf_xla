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
class MHTracer_DTPStensorflowPScorePSdataPScompression_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPScompression_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPScompression_utilsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/compression_utils.h"

#include <limits>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

Status CompressElement(const std::vector<Tensor>& element,
                       CompressedElement* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPScompression_utilsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/data/compression_utils.cc", "CompressElement");

  // Step 1: Determine the total uncompressed size. This requires serializing
  // non-memcopyable tensors, which we save to use again later.
  std::vector<TensorProto> non_memcpy_components;
  size_t total_size = 0;
  for (auto& component : element) {
    if (DataTypeCanUseMemcpy(component.dtype())) {
      const TensorBuffer* buffer = DMAHelper::buffer(&component);
      if (buffer) {
        total_size += buffer->size();
      }
    } else {
      non_memcpy_components.emplace_back();
      component.AsProtoTensorContent(&non_memcpy_components.back());
      total_size += non_memcpy_components.back().ByteSizeLong();
    }
  }

  // Step 2: Write the tensor data to a buffer, and compress that buffer.
  // We use tstring for access to resize_uninitialized.
  tstring uncompressed;
  uncompressed.resize_uninitialized(total_size);
  // Position in `uncompressed` to write the next component.
  char* position = uncompressed.mdata();
  int non_memcpy_component_index = 0;
  for (auto& component : element) {
    CompressedComponentMetadata* metadata =
        out->mutable_component_metadata()->Add();
    metadata->set_dtype(component.dtype());
    component.shape().AsProto(metadata->mutable_tensor_shape());
    if (DataTypeCanUseMemcpy(component.dtype())) {
      const TensorBuffer* buffer = DMAHelper::buffer(&component);
      if (buffer) {
        memcpy(position, buffer->data(), buffer->size());
        metadata->set_tensor_size_bytes(buffer->size());
      }
    } else {
      TensorProto& proto = non_memcpy_components[non_memcpy_component_index++];
      proto.SerializeToArray(position, proto.ByteSizeLong());
      metadata->set_tensor_size_bytes(proto.ByteSizeLong());
    }
    position += metadata->tensor_size_bytes();
  }
  if (total_size > kuint32max) {
    return errors::OutOfRange("Encountered dataset element of size ",
                              total_size, ", exceeding the 4GB Snappy limit.");
  }
  DCHECK_EQ(position, uncompressed.mdata() + total_size);

  if (!port::Snappy_Compress(uncompressed.mdata(), total_size,
                             out->mutable_data())) {
    return errors::Internal("Failed to compress using snappy.");
  }
  VLOG(3) << "Compressed element from " << total_size << " bytes to "
          << out->data().size() << " bytes";
  return Status::OK();
}

Status UncompressElement(const CompressedElement& compressed,
                         std::vector<Tensor>* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPScompression_utilsDTcc mht_1(mht_1_v, 259, "", "./tensorflow/core/data/compression_utils.cc", "UncompressElement");

  int num_components = compressed.component_metadata_size();
  out->clear();
  out->reserve(num_components);

  // Step 1: Prepare the memory that we will uncompress into.
  std::vector<struct iovec> iov(num_components);
  // We use tstring for access to resize_uninitialized.
  std::vector<tstring> tensor_proto_strs;
  // num_components is a conservative estimate. It is important to reserve
  // vector space so that the vector doesn't resize itself, which could
  // invalidate pointers to its strings' data.
  tensor_proto_strs.reserve(num_components);
  int64_t total_size = 0;
  for (int i = 0; i < num_components; ++i) {
    const CompressedComponentMetadata& metadata =
        compressed.component_metadata(i);
    if (DataTypeCanUseMemcpy(metadata.dtype())) {
      out->emplace_back(metadata.dtype(), metadata.tensor_shape());
      TensorBuffer* buffer = DMAHelper::buffer(&out->back());
      if (buffer) {
        iov[i].iov_base = buffer->data();
        iov[i].iov_len = buffer->size();
      } else {
        iov[i].iov_base = nullptr;
        iov[i].iov_len = 0;
      }
    } else {
      // Allocate an empty Tensor. We will fill it out later after
      // uncompressing into the tensor_proto_str.
      out->emplace_back();
      tensor_proto_strs.emplace_back();
      tstring& tensor_proto_str = tensor_proto_strs.back();
      tensor_proto_str.resize_uninitialized(metadata.tensor_size_bytes());
      iov[i].iov_base = tensor_proto_str.mdata();
      iov[i].iov_len = tensor_proto_str.size();
    }
    total_size += iov[i].iov_len;
  }

  // Step 2: Uncompress into the iovec.
  const std::string& compressed_data = compressed.data();
  size_t uncompressed_size;
  if (!port::Snappy_GetUncompressedLength(
          compressed_data.data(), compressed_data.size(), &uncompressed_size)) {
    return errors::Internal(
        "Could not get snappy uncompressed length. Compressed data size: ",
        compressed_data.size());
  }
  if (uncompressed_size != static_cast<size_t>(total_size)) {
    return errors::Internal(
        "Uncompressed size mismatch. Snappy expects ", uncompressed_size,
        " whereas the tensor metadata suggests ", total_size);
  }
  if (!port::Snappy_UncompressToIOVec(compressed_data.data(),
                                      compressed_data.size(), iov.data(),
                                      num_components)) {
    return errors::Internal("Failed to perform snappy decompression.");
  }

  // Step 3: Deserialize tensor proto strings to tensors.
  int tensor_proto_strs_index = 0;
  for (int i = 0; i < num_components; ++i) {
    if (DataTypeCanUseMemcpy(compressed.component_metadata(i).dtype())) {
      continue;
    }
    TensorProto tp;
    if (!tp.ParseFromString(tensor_proto_strs[tensor_proto_strs_index++])) {
      return errors::Internal("Could not parse TensorProto");
    }
    if (!out->at(i).FromProto(tp)) {
      return errors::Internal("Could not parse Tensor");
    }
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
