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
class MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/env_var.h"

#include <stdlib.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

Status ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val,
                          bool* value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/util/env_var.cc", "ReadBoolFromEnvVar");

  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  string str_value = absl::AsciiStrToLower(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return Status::OK();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadInt64FromEnvVar(StringPiece env_var_name, int64_t default_val,
                           int64_t* value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/util/env_var.cc", "ReadInt64FromEnvVar");

  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  if (strings::safe_strto64(tf_env_var_val, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into int64: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadFloatFromEnvVar(StringPiece env_var_name, float default_val,
                           float* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/util/env_var.cc", "ReadFloatFromEnvVar");

  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return Status::OK();
  }
  if (strings::safe_strtof(tf_env_var_val, value)) {
    return Status::OK();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into float: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

Status ReadStringFromEnvVar(StringPiece env_var_name, StringPiece default_val,
                            string* value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc mht_3(mht_3_v, 257, "", "./tensorflow/core/util/env_var.cc", "ReadStringFromEnvVar");

  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val != nullptr) {
    *value = tf_env_var_val;
  } else {
    *value = string(default_val);
  }
  return Status::OK();
}

Status ReadStringsFromEnvVar(StringPiece env_var_name, StringPiece default_val,
                             std::vector<string>* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSenv_varDTcc mht_4(mht_4_v, 271, "", "./tensorflow/core/util/env_var.cc", "ReadStringsFromEnvVar");

  string str_val;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar(env_var_name, default_val, &str_val));
  *value = str_util::Split(str_val, ',');
  return Status::OK();
}

}  // namespace tensorflow
