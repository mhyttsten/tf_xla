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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc() {
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

#include "tensorflow/core/public/session.h"

#include <string>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

auto* session_created = monitoring::Gauge<bool, 0>::New(
    "/tensorflow/core/session_created", "True if a session was created.");

}  // namespace

Session::Session() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/common_runtime/session.cc", "Session::Session");
}

Session::~Session() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/common_runtime/session.cc", "Session::~Session");
}

Status Session::Run(const RunOptions& run_options,
                    const std::vector<std::pair<string, Tensor> >& inputs,
                    const std::vector<string>& output_tensor_names,
                    const std::vector<string>& target_tensor_names,
                    std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/common_runtime/session.cc", "Session::Run");

  return errors::Unimplemented(
      "Run with options is not supported for this session.");
}

Status Session::PRunSetup(const std::vector<string>& input_names,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          string* handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/common_runtime/session.cc", "Session::PRunSetup");

  return errors::Unimplemented(
      "Partial run is not supported for this session.");
}

Status Session::PRun(const string& handle,
                     const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_names,
                     std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/common_runtime/session.cc", "Session::PRun");

  return errors::Unimplemented(
      "Partial run is not supported for this session.");
}

Session* NewSession(const SessionOptions& options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_5(mht_5_v, 247, "", "./tensorflow/core/common_runtime/session.cc", "NewSession");

  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  session_created->GetCell()->Set(true);
  Session* out_session;
  Status s = NewSession(options, &out_session);
  if (!s.ok()) {
    LOG(ERROR) << "Failed to create session: " << s;
    return nullptr;
  }
  return out_session;
}

Status NewSession(const SessionOptions& options, Session** out_session) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_6(mht_6_v, 264, "", "./tensorflow/core/common_runtime/session.cc", "NewSession");

  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << "Failed to get session factory: " << s;
    return s;
  }
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  session_created->GetCell()->Set(true);
  s = factory->NewSession(options, out_session);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << "Failed to create session: " << s;
  }
  return s;
}

Status Reset(const SessionOptions& options,
             const std::vector<string>& containers) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsessionDTcc mht_7(mht_7_v, 288, "", "./tensorflow/core/common_runtime/session.cc", "Reset");

  SessionFactory* factory;
  TF_RETURN_IF_ERROR(SessionFactory::GetFactory(options, &factory));
  return factory->Reset(options, containers);
}

}  // namespace tensorflow
