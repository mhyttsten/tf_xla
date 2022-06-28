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

#ifndef TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
#define TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_util_commonDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_util_commonDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_util_commonDTh() {
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


#include <string>
#include <unordered_map>
#include <vector>

namespace tensorflow {
enum class RowPartitionType {
  FIRST_DIM_SIZE,
  VALUE_ROWIDS,
  ROW_LENGTHS,
  ROW_SPLITS,
  ROW_LIMITS,
  ROW_STARTS
};

inline std::string RowPartitionTypeToString(
    RowPartitionType row_partition_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_util_commonDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/util/ragged_to_dense_util_common.h", "RowPartitionTypeToString");

  switch (row_partition_type) {
    case RowPartitionType::FIRST_DIM_SIZE:
      return "FIRST_DIM_SIZE";
    case RowPartitionType::VALUE_ROWIDS:
      return "VALUE_ROWIDS";
    case RowPartitionType::ROW_LENGTHS:
      return "ROW_LENGTHS";
    case RowPartitionType::ROW_SPLITS:
      return "ROW_SPLITS";
    case RowPartitionType::ROW_LIMITS:
      return "ROW_LIMITS";
    case RowPartitionType::ROW_STARTS:
      return "ROW_STARTS";
    default:
      return "UNKNOWN ROW PARTITION TYPE";
  }
}

inline std::vector<RowPartitionType> GetRowPartitionTypesHelper(
    const std::vector<std::string>& row_partition_type_strings) {
  static const auto kStringToType =
      new std::unordered_map<std::string, RowPartitionType>(
          {{"FIRST_DIM_SIZE", RowPartitionType::FIRST_DIM_SIZE},
           {"VALUE_ROWIDS", RowPartitionType::VALUE_ROWIDS},
           {"ROW_LENGTHS", RowPartitionType::ROW_LENGTHS},
           {"ROW_SPLITS", RowPartitionType::ROW_SPLITS},
           {"ROW_LIMITS", RowPartitionType::ROW_LIMITS},
           {"ROW_STARTS", RowPartitionType::ROW_STARTS}});
  std::vector<RowPartitionType> result;
  for (const auto& type_str : row_partition_type_strings) {
    const auto iter = kStringToType->find(type_str);
    if (iter == kStringToType->end()) {
      break;
    }
    result.push_back(iter->second);
  }
  return result;
}

inline int GetRaggedRank(
    const std::vector<RowPartitionType>& row_partition_types) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_util_commonDTh mht_1(mht_1_v, 247, "", "./tensorflow/core/util/ragged_to_dense_util_common.h", "GetRaggedRank");

  if (row_partition_types.empty()) {
    return 0;
  }
  if (row_partition_types[0] == RowPartitionType::FIRST_DIM_SIZE) {
    return row_partition_types.size() - 1;
  }
  return row_partition_types.size();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
