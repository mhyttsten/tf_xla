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
class MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc() {
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
#include "tensorflow/core/data/name_utils.h"

#include "absl/strings/str_join.h"

namespace tensorflow {
namespace data {
namespace name_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kDefaultDatasetDebugStringPrefix[] = "";

constexpr char kDataset[] = "Dataset";
constexpr char kOp[] = "Op";
constexpr char kVersion[] = "V";

string OpName(const string& dataset_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("dataset_type: \"" + dataset_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/data/name_utils.cc", "OpName");

  return OpName(dataset_type, OpNameParams());
}

string OpName(const string& dataset_type, const OpNameParams& params) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dataset_type: \"" + dataset_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/data/name_utils.cc", "OpName");

  if (params.op_version == 1) {
    return strings::StrCat(dataset_type, kDataset);
  }
  return strings::StrCat(dataset_type, kDataset, kVersion, params.op_version);
}

string ArgsToString(const std::vector<string>& args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/data/name_utils.cc", "ArgsToString");

  if (args.empty()) {
    return "";
  }
  return strings::StrCat("(", absl::StrJoin(args, ", "), ")");
}

string DatasetDebugString(const string& dataset_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("dataset_type: \"" + dataset_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/data/name_utils.cc", "DatasetDebugString");

  return DatasetDebugString(dataset_type, DatasetDebugStringParams());
}

string DatasetDebugString(const string& dataset_type,
                          const DatasetDebugStringParams& params) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("dataset_type: \"" + dataset_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_4(mht_4_v, 238, "", "./tensorflow/core/data/name_utils.cc", "DatasetDebugString");

  OpNameParams op_name_params;
  op_name_params.op_version = params.op_version;
  string op_name = OpName(dataset_type, op_name_params);
  return strings::StrCat(op_name, kOp, ArgsToString(params.args), kDelimiter,
                         params.dataset_prefix, kDataset);
}

string IteratorPrefix(const string& dataset_type, const string& prefix) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("dataset_type: \"" + dataset_type + "\"");
   mht_5_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/data/name_utils.cc", "IteratorPrefix");

  return IteratorPrefix(dataset_type, prefix, IteratorPrefixParams());
}

string IteratorPrefix(const string& dataset_type, const string& prefix,
                      const IteratorPrefixParams& params) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dataset_type: \"" + dataset_type + "\"");
   mht_6_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSname_utilsDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/data/name_utils.cc", "IteratorPrefix");

  if (params.op_version == 1) {
    return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                           dataset_type);
  }
  return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                         dataset_type, kVersion, params.op_version);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
