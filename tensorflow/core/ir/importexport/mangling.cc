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
class MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/mangling.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/importexport/parse_text_proto.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::DataType;
using tensorflow::Status;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::errors::FailedPrecondition;

namespace mlir {
namespace tfg {
namespace mangling_util {
namespace {

const char kAttributePrefix[] = "tf.";
const char kDataTypePrefix[] = "tfdtype$";
const char kTensorShapePrefix[] = "tfshape$";
const char kTensorPrefix[] = "tftensor$";

}  // namespace

std::string MangleAttributeName(absl::string_view str) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/ir/importexport/mangling.cc", "MangleAttributeName");

  return absl::StrCat(kAttributePrefix, str);
}

bool IsMangledAttributeName(absl::string_view str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/ir/importexport/mangling.cc", "IsMangledAttributeName");

  return absl::StartsWith(str, kAttributePrefix);
}

absl::string_view DemangleAttributeName(absl::string_view str) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/ir/importexport/mangling.cc", "DemangleAttributeName");

  DCHECK(IsMangledAttributeName(str));
  return str.substr(std::strlen(kAttributePrefix));
}

MangledKind GetMangledKind(absl::string_view str) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/ir/importexport/mangling.cc", "GetMangledKind");

  if (absl::StartsWith(str, kDataTypePrefix)) {
    return MangledKind::kDataType;
  } else if (absl::StartsWith(str, kTensorShapePrefix)) {
    return MangledKind::kTensorShape;
  } else if (absl::StartsWith(str, kTensorPrefix)) {
    return MangledKind::kTensor;
  } else {
    return MangledKind::kUnknown;
  }
}

std::string MangleShape(const TensorShapeProto& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/ir/importexport/mangling.cc", "MangleShape");

  return absl::StrCat(kTensorShapePrefix, shape.ShortDebugString());
}

Status DemangleShape(absl::string_view str, TensorShapeProto* proto) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_5(mht_5_v, 263, "", "./tensorflow/core/ir/importexport/mangling.cc", "DemangleShape");

  return ParseTextProto(str, kTensorShapePrefix, proto);
}

std::string MangleTensor(const TensorProto& tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_6(mht_6_v, 270, "", "./tensorflow/core/ir/importexport/mangling.cc", "MangleTensor");

  return absl::StrCat(kTensorPrefix, tensor.ShortDebugString());
}

Status DemangleTensor(absl::string_view str, TensorProto* proto) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/ir/importexport/mangling.cc", "DemangleTensor");

  return ParseTextProto(str, kTensorPrefix, proto);
}

std::string MangleDataType(const DataType& dtype) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_8(mht_8_v, 285, "", "./tensorflow/core/ir/importexport/mangling.cc", "MangleDataType");

  return absl::StrCat(kDataTypePrefix, DataType_Name(dtype));
}

Status DemangleDataType(absl::string_view str, DataType* proto) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("str: \"" + std::string(str.data(), str.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSirPSimportexportPSmanglingDTcc mht_9(mht_9_v, 293, "", "./tensorflow/core/ir/importexport/mangling.cc", "DemangleDataType");

  absl::string_view pbtxt;
  TF_RETURN_IF_ERROR(ConsumePrefix(str, kDataTypePrefix, &pbtxt));
  // NOLINTNEXTLINE: redundant string conversion for divergence in OSS API.
  if (!DataType_Parse(std::string(pbtxt), proto)) {
    return FailedPrecondition("Could not parse TFDataType mangled proto");
  }
  return Status::OK();
}

}  // namespace mangling_util
}  // namespace tfg
}  // namespace mlir
