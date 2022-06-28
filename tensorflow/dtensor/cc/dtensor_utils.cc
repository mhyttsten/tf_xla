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
class MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/dtensor_utils.h"

#include <cstdlib>

#include "absl/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace dtensor {

// LINT.IfChange
int ClientId() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "ClientId");

  char* client_id_str = std::getenv("DTENSOR_CLIENT_ID");
  if (client_id_str == nullptr) return 0;
  int client_id;
  if (absl::SimpleAtoi(client_id_str, &client_id)) return client_id;
  LOG(WARNING) << "Invalid DTENSOR_CLIENT_ID, using the default value 0.";
  return 0;
}
// LINT.ThenChange(//tensorflow/dtensor/python/dtensor_device.py)

// LINT.IfChange
int NumClients() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "NumClients");

  char* num_clients_str = std::getenv("DTENSOR_NUM_CLIENTS");
  if (num_clients_str == nullptr) return 1;
  int num_clients;
  if (absl::SimpleAtoi(num_clients_str, &num_clients)) return num_clients;
  LOG(WARNING) << "Invalid DTENSOR_NUM_CLIENTS, using the default value 1.";
  return 1;
}
// LINT.ThenChange(//tensorflow/dtensor/python/dtensor_device.py)

bool LogOnAllTasks() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_2(mht_2_v, 223, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "LogOnAllTasks");

  char* dtensor_log_on_all_tasks_str = std::getenv("DTENSOR_LOG_ON_ALL_TASKS");
  if (dtensor_log_on_all_tasks_str == nullptr) return false;
  return true;
}

bool LogOpByOp() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_3(mht_3_v, 232, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "LogOpByOp");

  char* dtensor_log_op_by_op_str = std::getenv("DTENSOR_LOG_OP_BY_OP");
  if (dtensor_log_op_by_op_str == nullptr) return false;
  return true;
}

int LayoutPropagationMaxSteps() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_4(mht_4_v, 241, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "LayoutPropagationMaxSteps");

  char* dtensor_layout_propagation_max_steps_str =
      std::getenv("DTENSOR_LAYOUT_PROPAGATION_MAX_STEPS");
  if (dtensor_layout_propagation_max_steps_str == nullptr) return 500;
  int dtensor_layout_propagation_max_steps;
  if (absl::SimpleAtoi(dtensor_layout_propagation_max_steps_str,
                       &dtensor_layout_propagation_max_steps))
    return dtensor_layout_propagation_max_steps;
  LOG(WARNING) << "Invalid DTENSOR_LAYOUT_PROPAGATION_MAX_STEPS, using "
                  "the default value 500.";
  return 500;
}

bool EnableMixedPrecisionReduce() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_5(mht_5_v, 257, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "EnableMixedPrecisionReduce");

  char* dtensor_enable_mixed_precision_reduce_str =
      std::getenv("DTENSOR_ENABLE_MIXED_PRECISION_REDUCE");
  if (dtensor_enable_mixed_precision_reduce_str == nullptr) return false;
  return true;
}

bool DoNotFuseReduceScatter() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_6(mht_6_v, 267, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "DoNotFuseReduceScatter");

  char* dtensor_do_not_fuse_reduce_scatter_str =
      std::getenv("DTENSOR_DO_NOT_FUSE_REDUCE_SCATTER");
  if (dtensor_do_not_fuse_reduce_scatter_str == nullptr) return false;
  return true;
}

int ReduceInBfloat16MaxGroupSize() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSccPSdtensor_utilsDTcc mht_7(mht_7_v, 277, "", "./tensorflow/dtensor/cc/dtensor_utils.cc", "ReduceInBfloat16MaxGroupSize");

  char* dtensor_reduce_in_bfloat16_max_group_size_str =
      std::getenv("DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE");
  if (dtensor_reduce_in_bfloat16_max_group_size_str == nullptr) return 8;
  int dtensor_reduce_in_bfloat16_max_group_size;
  if (absl::SimpleAtoi(dtensor_reduce_in_bfloat16_max_group_size_str,
                       &dtensor_reduce_in_bfloat16_max_group_size))
    return dtensor_reduce_in_bfloat16_max_group_size;
  LOG(WARNING) << "Invalid DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE, using "
                  "the default value 8.";
  return 8;
}

}  // namespace dtensor
}  // namespace tensorflow
