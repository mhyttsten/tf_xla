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
class MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc() {
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

#include "tensorflow/core/util/padding.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status GetPaddingFromString(StringPiece str_value, Padding* value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/util/padding.cc", "GetPaddingFromString");

  if (str_value == "SAME") {
    *value = SAME;
  } else if (str_value == "VALID") {
    *value = VALID;
  } else if (str_value == "EXPLICIT") {
    *value = EXPLICIT;
  } else {
    return errors::NotFound(str_value, " is not an allowed padding type");
  }
  return Status::OK();
}

Status CheckValidPadding(Padding padding_type,
                         const std::vector<int64_t>& explicit_paddings,
                         int num_dims, TensorFormat data_format) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/util/padding.cc", "CheckValidPadding");

  if (padding_type == Padding::EXPLICIT) {
    const int num_paddings = explicit_paddings.size();
    if (num_paddings != 2 * num_dims) {
      return errors::InvalidArgument(
          "explicit_paddings attribute must contain ", 2 * num_dims,
          " values, but got: ", explicit_paddings.size());
    }
    for (int64_t padding_value : explicit_paddings) {
      if (padding_value < 0) {
        return errors::InvalidArgument(
            "All elements of explicit_paddings must be nonnegative");
      }
    }
    const int32_t batch_index = GetTensorBatchDimIndex(num_dims, data_format);
    const int32_t depth_index = GetTensorFeatureDimIndex(num_dims, data_format);
    if (explicit_paddings[2 * batch_index] != 0 ||
        explicit_paddings[2 * batch_index + 1] != 0 ||
        explicit_paddings[2 * depth_index] != 0 ||
        explicit_paddings[2 * depth_index + 1] != 0) {
      return errors::InvalidArgument(
          "Nonzero explicit padding in the batch or depth dimensions is not "
          "supported");
    }
  } else if (!explicit_paddings.empty()) {
    return errors::InvalidArgument(
        "explicit_paddings attribute must be empty if the padding attribute is "
        "not EXPLICIT");
  }
  return Status::OK();
}

string GetPaddingAttrString() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/util/padding.cc", "GetPaddingAttrString");
 return "padding: {'SAME', 'VALID'}"; }

string GetPaddingAttrStringWithExplicit() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/util/padding.cc", "GetPaddingAttrStringWithExplicit");

  return "padding: {'SAME', 'VALID', 'EXPLICIT'}";
}

string GetExplicitPaddingsAttrString() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSpaddingDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/util/padding.cc", "GetExplicitPaddingsAttrString");

  return "explicit_paddings: list(int) = []";
}

}  // end namespace tensorflow
