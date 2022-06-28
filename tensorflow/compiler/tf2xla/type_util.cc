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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPStype_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStype_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPStype_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/type_util.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status DataTypeToPrimitiveType(DataType data_type, xla::PrimitiveType* type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStype_utilDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/tf2xla/type_util.cc", "DataTypeToPrimitiveType");

  switch (data_type) {
    case tensorflow::DT_BOOL:
      *type = xla::PRED;
      return Status::OK();
    case tensorflow::DT_INT8:
    case tensorflow::DT_QINT8:
      *type = xla::S8;
      return Status::OK();
    case tensorflow::DT_INT16:
    case tensorflow::DT_QINT16:
      *type = xla::S16;
      return Status::OK();
    case tensorflow::DT_INT32:
    case tensorflow::DT_QINT32:
      *type = xla::S32;
      return Status::OK();
    case tensorflow::DT_INT64:
      *type = xla::S64;
      return Status::OK();
    case tensorflow::DT_UINT8:
    case tensorflow::DT_QUINT8:
      *type = xla::U8;
      return Status::OK();
    case tensorflow::DT_UINT16:
    case tensorflow::DT_QUINT16:
      *type = xla::U16;
      return Status::OK();
    case tensorflow::DT_UINT32:
      *type = xla::U32;
      return Status::OK();
    case tensorflow::DT_UINT64:
      *type = xla::U64;
      return Status::OK();
    case tensorflow::DT_BFLOAT16:
      *type = xla::BF16;
      return Status::OK();
    case tensorflow::DT_HALF:
      *type = xla::F16;
      return Status::OK();
    case tensorflow::DT_FLOAT:
      *type = xla::F32;
      return Status::OK();
    case tensorflow::DT_DOUBLE:
      *type = xla::F64;
      return Status::OK();
    case tensorflow::DT_COMPLEX64:
      *type = xla::C64;
      return Status::OK();
    case tensorflow::DT_COMPLEX128:
      *type = xla::C128;
      return Status::OK();
    default:
      return errors::InvalidArgument(
          "Unsupported type in DataTypeToPrimitiveType: '",
          DataTypeString(data_type), "'");
  }
}

StatusOr<DataType> EncodePrimitiveTypeAsDataType(xla::PrimitiveType type) {
  static const absl::flat_hash_map<xla::PrimitiveType, DataType>&
      data_type_map = *new absl::flat_hash_map<xla::PrimitiveType, DataType>({
          {xla::PRED, DT_BOOL},
          {xla::BF16, DT_BFLOAT16},
          {xla::F16, DT_HALF},
          {xla::F32, DT_FLOAT},
          {xla::F64, DT_DOUBLE},
          {xla::C64, DT_COMPLEX64},
          {xla::S8, DT_INT8},
          {xla::S16, DT_INT16},
          {xla::S32, DT_INT32},
          {xla::S64, DT_INT64},
          {xla::U8, DT_UINT8},
          {xla::U16, DT_UINT16},
          {xla::U32, DT_UINT32},
          {xla::U64, DT_UINT64},
          {xla::C128, DT_COMPLEX128},
      });

  auto it = data_type_map.find(type);
  if (it == data_type_map.end()) {
    return errors::InvalidArgument(
        "Unsupported type in PrimitiveTypeToDataType ", type);
  }
  return it->second;
}

}  // namespace tensorflow
