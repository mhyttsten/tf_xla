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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc() {
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

#include "tensorflow/core/common_runtime/session_factory.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

static mutex* get_session_factory_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/common_runtime/session_factory.cc", "get_session_factory_lock");

  static mutex session_factory_lock(LINKER_INITIALIZED);
  return &session_factory_lock;
}

typedef std::unordered_map<string, SessionFactory*> SessionFactories;
SessionFactories* session_factories() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/common_runtime/session_factory.cc", "session_factories");

  static SessionFactories* factories = new SessionFactories;
  return factories;
}

}  // namespace

void SessionFactory::Register(const string& runtime_type,
                              SessionFactory* factory) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("runtime_type: \"" + runtime_type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/common_runtime/session_factory.cc", "SessionFactory::Register");

  mutex_lock l(*get_session_factory_lock());
  if (!session_factories()->insert({runtime_type, factory}).second) {
    LOG(ERROR) << "Two session factories are being registered "
               << "under" << runtime_type;
  }
}

namespace {
const string RegisteredFactoriesErrorMessageLocked() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/common_runtime/session_factory.cc", "RegisteredFactoriesErrorMessageLocked");

  std::vector<string> factory_types;
  for (const auto& session_factory : *session_factories()) {
    factory_types.push_back(session_factory.first);
  }
  return strings::StrCat("Registered factories are {",
                         absl::StrJoin(factory_types, ", "), "}.");
}
string SessionOptionsToString(const SessionOptions& options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/common_runtime/session_factory.cc", "SessionOptionsToString");

  return strings::StrCat("target: \"", options.target,
                         "\" config: ", options.config.ShortDebugString());
}
}  // namespace

Status SessionFactory::GetFactory(const SessionOptions& options,
                                  SessionFactory** out_factory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsession_factoryDTcc mht_5(mht_5_v, 254, "", "./tensorflow/core/common_runtime/session_factory.cc", "SessionFactory::GetFactory");

  mutex_lock l(*get_session_factory_lock());  // could use reader lock

  std::vector<std::pair<string, SessionFactory*>> candidate_factories;
  for (const auto& session_factory : *session_factories()) {
    if (session_factory.second->AcceptsOptions(options)) {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " accepts target: " << options.target;
      candidate_factories.push_back(session_factory);
    } else {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " does not accept target: " << options.target;
    }
  }

  if (candidate_factories.size() == 1) {
    *out_factory = candidate_factories[0].second;
    return Status::OK();
  } else if (candidate_factories.size() > 1) {
    // NOTE(mrry): This implementation assumes that the domains (in
    // terms of acceptable SessionOptions) of the registered
    // SessionFactory implementations do not overlap. This is fine for
    // now, but we may need an additional way of distinguishing
    // different runtimes (such as an additional session option) if
    // the number of sessions grows.
    // TODO(mrry): Consider providing a system-default fallback option
    // in this case.
    std::vector<string> factory_types;
    factory_types.reserve(candidate_factories.size());
    for (const auto& candidate_factory : candidate_factories) {
      factory_types.push_back(candidate_factory.first);
    }
    return errors::Internal(
        "Multiple session factories registered for the given session "
        "options: {",
        SessionOptionsToString(options), "} Candidate factories are {",
        absl::StrJoin(factory_types, ", "), "}. ",
        RegisteredFactoriesErrorMessageLocked());
  } else {
    return errors::NotFound(
        "No session factory registered for the given session options: {",
        SessionOptionsToString(options), "} ",
        RegisteredFactoriesErrorMessageLocked());
  }
}

}  // namespace tensorflow
