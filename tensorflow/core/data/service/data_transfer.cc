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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc() {
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

#include "tensorflow/core/data/service/data_transfer.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {

namespace {
mutex* get_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/data/service/data_transfer.cc", "get_lock");

  static mutex lock(LINKER_INITIALIZED);
  return &lock;
}

using DataTransferServerFactories =
    std::unordered_map<std::string,
                       std::function<std::shared_ptr<DataTransferServer>(
                           DataTransferServer::GetElementT)>>;
DataTransferServerFactories& transfer_server_factories() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/data/service/data_transfer.cc", "transfer_server_factories");

  static auto& factories = *new DataTransferServerFactories();
  return factories;
}

using DataTransferClientFactories =
    std::unordered_map<std::string, DataTransferClient::FactoryT>;
DataTransferClientFactories& transfer_client_factories() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/data/service/data_transfer.cc", "transfer_client_factories");

  static auto& factories = *new DataTransferClientFactories();
  return factories;
}
}  // namespace

GetElementResult GetElementResult::Copy() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/data/service/data_transfer.cc", "GetElementResult::Copy");

  GetElementResult copy;
  copy.components = components;
  copy.element_index = element_index;
  copy.end_of_sequence = end_of_sequence;
  copy.skip = skip;
  return copy;
}

size_t GetElementResult::EstimatedMemoryUsageBytes() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_4(mht_4_v, 245, "", "./tensorflow/core/data/service/data_transfer.cc", "GetElementResult::EstimatedMemoryUsageBytes");

  size_t size_bytes = components.size() * sizeof(Tensor) + sizeof(int64) +
                      sizeof(bool) + sizeof(bool);
  for (const Tensor& tensor : components) {
    size_bytes += tensor.AllocatedBytes();
  }
  return size_bytes;
}

void DataTransferServer::Register(
    std::string name,
    std::function<std::shared_ptr<DataTransferServer>(GetElementT)> factory) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_5(mht_5_v, 260, "", "./tensorflow/core/data/service/data_transfer.cc", "DataTransferServer::Register");

  mutex_lock l(*get_lock());
  if (!transfer_server_factories().insert({name, factory}).second) {
    LOG(ERROR)
        << "Two data transfer server factories are being registered with name "
        << name << ". Which one gets used is undefined.";
  }
}

Status DataTransferServer::Build(std::string name, GetElementT get_element,
                                 std::shared_ptr<DataTransferServer>* out) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_6(mht_6_v, 274, "", "./tensorflow/core/data/service/data_transfer.cc", "DataTransferServer::Build");

  mutex_lock l(*get_lock());
  auto it = transfer_server_factories().find(name);
  if (it != transfer_server_factories().end()) {
    *out = it->second(get_element);
    return Status::OK();
  }

  std::vector<string> available_names;
  for (const auto& factory : transfer_server_factories()) {
    available_names.push_back(factory.first);
  }

  return errors::NotFound(
      "No data transfer server factory has been registered for name ", name,
      ". The available names are: [ ", absl::StrJoin(available_names, ", "),
      " ]");
}

void DataTransferClient::Register(std::string name, FactoryT factory) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_7(mht_7_v, 297, "", "./tensorflow/core/data/service/data_transfer.cc", "DataTransferClient::Register");

  mutex_lock l(*get_lock());
  if (!transfer_client_factories().insert({name, factory}).second) {
    LOG(ERROR)
        << "Two data transfer client factories are being registered with name "
        << name << ". Which one gets used is undefined.";
  }
}

Status DataTransferClient::Build(std::string name, Config config,
                                 std::unique_ptr<DataTransferClient>* out) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdata_transferDTcc mht_8(mht_8_v, 311, "", "./tensorflow/core/data/service/data_transfer.cc", "DataTransferClient::Build");

  mutex_lock l(*get_lock());
  auto it = transfer_client_factories().find(name);
  if (it != transfer_client_factories().end()) {
    return it->second(config, out);
  }

  std::vector<string> available_names;
  for (const auto& factory : transfer_client_factories()) {
    available_names.push_back(factory.first);
  }

  return errors::NotFound(
      "No data transfer client factory has been registered for name ", name,
      ". The available names are: [ ", absl::StrJoin(available_names, ", "),
      " ]");
}

}  // namespace data
}  // namespace tensorflow
