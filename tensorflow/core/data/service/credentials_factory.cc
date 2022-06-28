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
class MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc() {
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

#include "tensorflow/core/data/service/credentials_factory.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {

namespace {
mutex* get_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/data/service/credentials_factory.cc", "get_lock");

  static mutex lock(LINKER_INITIALIZED);
  return &lock;
}

using CredentialsFactories =
    std::unordered_map<std::string, CredentialsFactory*>;
CredentialsFactories& credentials_factories() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/data/service/credentials_factory.cc", "credentials_factories");

  static auto& factories = *new CredentialsFactories();
  return factories;
}
}  // namespace

void CredentialsFactory::Register(CredentialsFactory* factory) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/data/service/credentials_factory.cc", "CredentialsFactory::Register");

  mutex_lock l(*get_lock());
  if (!credentials_factories().insert({factory->Protocol(), factory}).second) {
    LOG(ERROR)
        << "Two credentials factories are being registered with protocol "
        << factory->Protocol() << ". Which one gets used is undefined.";
  }
}

Status CredentialsFactory::Get(absl::string_view protocol,
                               CredentialsFactory** out) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("protocol: \"" + std::string(protocol.data(), protocol.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/data/service/credentials_factory.cc", "CredentialsFactory::Get");

  mutex_lock l(*get_lock());
  auto it = credentials_factories().find(std::string(protocol));
  if (it != credentials_factories().end()) {
    *out = it->second;
    return Status::OK();
  }

  std::vector<string> available_types;
  for (const auto& factory : credentials_factories()) {
    available_types.push_back(factory.first);
  }

  return errors::NotFound("No credentials factory has been registered for ",
                          "protocol ", protocol,
                          ". The available types are: [ ",
                          absl::StrJoin(available_types, ", "), " ]");
}

Status CredentialsFactory::CreateServerCredentials(
    absl::string_view protocol,
    std::shared_ptr<::grpc::ServerCredentials>* out) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("protocol: \"" + std::string(protocol.data(), protocol.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/data/service/credentials_factory.cc", "CredentialsFactory::CreateServerCredentials");

  CredentialsFactory* factory;
  TF_RETURN_IF_ERROR(CredentialsFactory::Get(protocol, &factory));
  TF_RETURN_IF_ERROR(factory->CreateServerCredentials(out));
  return Status::OK();
}

Status CredentialsFactory::CreateClientCredentials(
    absl::string_view protocol,
    std::shared_ptr<::grpc::ChannelCredentials>* out) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("protocol: \"" + std::string(protocol.data(), protocol.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_5(mht_5_v, 265, "", "./tensorflow/core/data/service/credentials_factory.cc", "CredentialsFactory::CreateClientCredentials");

  CredentialsFactory* factory;
  TF_RETURN_IF_ERROR(CredentialsFactory::Get(protocol, &factory));
  TF_RETURN_IF_ERROR(factory->CreateClientCredentials(out));
  return Status::OK();
}

bool CredentialsFactory::Exists(absl::string_view protocol) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("protocol: \"" + std::string(protocol.data(), protocol.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/data/service/credentials_factory.cc", "CredentialsFactory::Exists");

  mutex_lock l(*get_lock());
  return credentials_factories().find(std::string(protocol)) !=
         credentials_factories().end();
}

class InsecureCredentialsFactory : public CredentialsFactory {
 public:
  std::string Protocol() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_7(mht_7_v, 287, "", "./tensorflow/core/data/service/credentials_factory.cc", "Protocol");
 return "grpc"; }

  Status CreateServerCredentials(
      std::shared_ptr<::grpc::ServerCredentials>* out) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_8(mht_8_v, 293, "", "./tensorflow/core/data/service/credentials_factory.cc", "CreateServerCredentials");

    *out = ::grpc::InsecureServerCredentials();
    return Status::OK();
  }

  Status CreateClientCredentials(
      std::shared_ptr<::grpc::ChannelCredentials>* out) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_9(mht_9_v, 302, "", "./tensorflow/core/data/service/credentials_factory.cc", "CreateClientCredentials");

    *out = ::grpc::InsecureChannelCredentials();
    return Status::OK();
  }
};

class InsecureCredentialsRegistrar {
 public:
  InsecureCredentialsRegistrar() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePScredentials_factoryDTcc mht_10(mht_10_v, 313, "", "./tensorflow/core/data/service/credentials_factory.cc", "InsecureCredentialsRegistrar");

    auto factory = new InsecureCredentialsFactory();
    CredentialsFactory::Register(factory);
  }
};
static InsecureCredentialsRegistrar registrar;

}  // namespace data
}  // namespace tensorflow
