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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc() {
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

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
mutex* get_server_factory_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "get_server_factory_lock");

  static mutex server_factory_lock(LINKER_INITIALIZED);
  return &server_factory_lock;
}

typedef std::unordered_map<string, ServerFactory*> ServerFactories;
ServerFactories* server_factories() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "server_factories");

  static ServerFactories* factories = new ServerFactories;
  return factories;
}
}  // namespace

/* static */
void ServerFactory::Register(const string& server_type,
                             ServerFactory* factory) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("server_type: \"" + server_type + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "ServerFactory::Register");

  mutex_lock l(*get_server_factory_lock());
  if (!server_factories()->insert({server_type, factory}).second) {
    LOG(ERROR) << "Two server factories are being registered under "
               << server_type;
  }
}

/* static */
Status ServerFactory::GetFactory(const ServerDef& server_def,
                                 ServerFactory** out_factory) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "ServerFactory::GetFactory");

  mutex_lock l(*get_server_factory_lock());
  for (const auto& server_factory : *server_factories()) {
    if (server_factory.second->AcceptsOptions(server_def)) {
      *out_factory = server_factory.second;
      return Status::OK();
    }
  }

  std::vector<string> server_names;
  for (const auto& server_factory : *server_factories()) {
    server_names.push_back(server_factory.first);
  }

  return errors::NotFound(
      "No server factory registered for the given ServerDef: ",
      server_def.DebugString(), "\nThe available server factories are: [ ",
      absl::StrJoin(server_names, ", "), " ]");
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "NewServer");

  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, ServerFactory::Options(), out_server);
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServerWithOptions(const ServerDef& server_def,
                            const ServerFactory::Options& options,
                            std::unique_ptr<ServerInterface>* out_server) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSserver_libDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/distributed_runtime/server_lib.cc", "NewServerWithOptions");

  ServerFactory* factory;
  TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, options, out_server);
}

}  // namespace tensorflow
