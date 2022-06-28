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
class MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc() {
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

#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

namespace {

// Byte-swap a buffer in place.
//
// Args:
//  buff: pointer to the buffer to be modified IN PLACE.
//  size: size of bytes in this buffer.
//  dtype: type of data in this buffer.
//  num_of_elem: number of data in this buffer, set to -1 if it
//               could not be obtained directly from tensor data.
//               If num_of_elem is -1, this function will calculate
//               the number of data based on size and dtype.
// Returns: Status::OK() on success, -1 otherwise
Status ByteSwapBuffer(char* buff, size_t size, DataType dtype,
                      int num_of_elem) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buff: \"" + (buff == nullptr ? std::string("nullptr") : std::string((char*)buff)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/util/tensor_bundle/byte_swap.cc", "ByteSwapBuffer");

  int array_len = num_of_elem;
  size_t bytes_per_elem = 0;

  switch (dtype) {
    // Types that don't need byte-swapping
    case DT_STRING:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_BOOL:
    case DT_UINT8:
    case DT_INT8:
      return Status::OK();

    // 16-bit types
    case DT_BFLOAT16:
    case DT_HALF:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:
      bytes_per_elem = 2;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // 32-bit types
    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
    case DT_UINT32:
      bytes_per_elem = 4;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // 64-bit types
    case DT_INT64:
    case DT_DOUBLE:
    case DT_UINT64:
      bytes_per_elem = 8;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      break;

    // Complex types need special handling
    case DT_COMPLEX64:
      bytes_per_elem = 4;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      array_len *= 2;
      break;

    case DT_COMPLEX128:
      bytes_per_elem = 8;
      array_len = (array_len == -1) ? size / bytes_per_elem : array_len;
      array_len *= 2;
      break;

    // Types that ought to be supported in the future
    case DT_RESOURCE:
    case DT_VARIANT:
      return errors::Unimplemented(
          "Byte-swapping not yet implemented for tensors with dtype ", dtype);

    // Byte-swapping shouldn't make sense for other dtypes.
    default:
      return errors::Unimplemented(
          "Byte-swapping not supported for tensors with dtype ", dtype);
  }

  TF_RETURN_IF_ERROR(ByteSwapArray(buff, bytes_per_elem, array_len));
  return Status::OK();
}

}  // namespace

Status ByteSwapArray(char* array, size_t bytes_per_elem, int array_len) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("array: \"" + (array == nullptr ? std::string("nullptr") : std::string((char*)array)) + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc mht_1(mht_1_v, 288, "", "./tensorflow/core/util/tensor_bundle/byte_swap.cc", "ByteSwapArray");

  if (bytes_per_elem == 1) {
    // No-op
    return Status::OK();
  } else if (bytes_per_elem == 2) {
    auto array_16 = reinterpret_cast<uint16_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_16[i] = BYTE_SWAP_16(array_16[i]);
    }
    return Status::OK();
  } else if (bytes_per_elem == 4) {
    auto array_32 = reinterpret_cast<uint32_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_32[i] = BYTE_SWAP_32(array_32[i]);
    }
    return Status::OK();
  } else if (bytes_per_elem == 8) {
    auto array_64 = reinterpret_cast<uint64_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_64[i] = BYTE_SWAP_64(array_64[i]);
    }
    return Status::OK();
  } else {
    return errors::Unimplemented("Byte-swapping of ", bytes_per_elem,
                                 "-byte values not supported.");
  }
}

Status ByteSwapTensor(Tensor* t) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc mht_2(mht_2_v, 319, "", "./tensorflow/core/util/tensor_bundle/byte_swap.cc", "ByteSwapTensor");

  char* buff = const_cast<char*>((t->tensor_data().data()));
  return ByteSwapBuffer(buff, t->tensor_data().size(), t->dtype(),
                        t->NumElements());
}

Status ByteSwapTensorContent(MetaGraphDef* meta_graph_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePSbyte_swapDTcc mht_3(mht_3_v, 328, "", "./tensorflow/core/util/tensor_bundle/byte_swap.cc", "ByteSwapTensorContent");

  for (auto& function : *meta_graph_def->mutable_graph_def()
                             ->mutable_library()
                             ->mutable_function()) {
    for (auto& node : (*function.mutable_node_def())) {
      if (node.op() == "Const") {
        auto node_iterator = node.mutable_attr()->find("value");
        if (node_iterator != node.mutable_attr()->end()) {
          AttrValue node_value = node_iterator->second;
          if (node_value.has_tensor()) {
            auto tsize = node_value.mutable_tensor()->tensor_content().size();
            auto p_type = node_value.mutable_tensor()->dtype();
            // Swap only when there is something in tensor_content field
            if (tsize != 0 && DataTypeCanUseMemcpy(p_type)) {
              Tensor parsed(p_type);
              DCHECK(parsed.FromProto(*node_value.mutable_tensor()));
              if (!parsed.tensor_data().empty()) {
                TF_RETURN_IF_ERROR(ByteSwapTensor(&parsed));
                (*node.mutable_attr())["value"]
                    .mutable_tensor()
                    ->set_tensor_content(
                        string(reinterpret_cast<const char*>(
                                   parsed.tensor_data().data()),
                               parsed.tensor_data().size()));
              } else {
                void* copy = tensorflow::port::Malloc(tsize);
                memcpy(copy,
                       string(node_value.mutable_tensor()->tensor_content())
                           .data(),
                       tsize);
                TF_RETURN_IF_ERROR(
                    ByteSwapBuffer((char*)copy, tsize, p_type, -1));
                (*node.mutable_attr())["value"]
                    .mutable_tensor()
                    ->set_tensor_content(
                        string(reinterpret_cast<const char*>(copy), tsize));
                tensorflow::port::Free(copy);
              }
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
