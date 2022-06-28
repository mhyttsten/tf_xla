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
class MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc() {
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
#include "tensorflow/c/experimental/ops/gen/common/controller.h"

#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace generator {

Controller::Controller(PathConfig path_config, Env* env)
    : env_(env), path_config_(path_config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_0(mht_0_v, 197, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::Controller");

  // Load the Op and API definitions
  InitializeOpApi();

  // Convert the Op and API definitions to the internal data model
  BuildModel();
}
Controller::~Controller() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_1(mht_1_v, 207, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::~Controller");
 delete api_def_map_; }

const void Controller::WriteFile(const string& file_path,
                                 const SourceCode& code) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_2(mht_2_v, 214, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::WriteFile");

  TF_CHECK_OK(WriteStringToFile(env_, file_path, code.Render())) << file_path;
}

const std::vector<OpSpec>& Controller::GetModelOps() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_3(mht_3_v, 221, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::GetModelOps");

  return operators_;
}

void Controller::InitializeOpApi() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_4(mht_4_v, 228, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::InitializeOpApi");

  OpRegistry::Global()->Export(false, &op_list_);

  // Load matching API defs for each Op. Paths are visited in order, allowing
  // python/api_def_Xyz.pbtxt to override base/api_def_Xyz.pbtxt, for example.
  api_def_map_ = new ApiDefMap(op_list_);
  for (const auto& op : op_list_.op()) {
    for (const auto& dir : path_config_.api_dirs) {
      const string file_name = absl::Substitute("api_def_$0.pbtxt", op.name());
      const string file_path = io::JoinPath(dir, file_name);
      if (env_->FileExists(file_path).ok()) {
        TF_CHECK_OK(api_def_map_->LoadFile(env_, file_path)) << file_path;
      } else {
        // API defs are currently used for only optional pieces.
      }
    }
  }

  // Doc strings (summary, description) typically come from the API def.
  api_def_map_->UpdateDocs();
}

void Controller::BuildModel() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSopsPSgenPScommonPScontrollerDTcc mht_5(mht_5_v, 253, "", "./tensorflow/c/experimental/ops/gen/common/controller.cc", "Controller::BuildModel");

  // Build the internal data model for the requested ops
  for (const auto& op_name : path_config_.op_names) {
    const OpDef* op_def = nullptr;
    TF_CHECK_OK(OpRegistry::Global()->LookUpOpDef(op_name, &op_def));
    CHECK(op_def != nullptr);  // Crash OK

    const ApiDef* api_def = api_def_map_->GetApiDef(op_name);
    CHECK(api_def != nullptr);  // Crash OK

    operators_.push_back(OpSpec::Create(*op_def, *api_def));
  }
}

}  // namespace generator
}  // namespace tensorflow
