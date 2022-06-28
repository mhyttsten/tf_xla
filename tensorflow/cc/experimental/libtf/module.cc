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
class MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc {
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
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc() {
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
#include "tensorflow/cc/experimental/libtf/module.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
namespace tf {
namespace libtf {
namespace impl {

using tensorflow::libexport::TFPackage;
using tf::libtf::runtime::Runtime;

// TODO(danielellis): Fill in with implementations.

// Builds a vector of runtime representations of `SavedObject`s from a
// SavedModel. These are returned as a flat list.  The full hierarchy building
// and initialization should be done in a later pass.
tensorflow::StatusOr<std::vector<Handle>> BuildObjects(TFPackage& tf_package) {
  std::vector<Handle> objects;
  const tensorflow::SavedObjectGraph object_graph = tf_package.GetObjectGraph();
  for (auto& node : object_graph.nodes()) {
    if (node.kind_case() == tensorflow::SavedObject::kUserObject) {
      tensorflow::StatusOr<Handle> result = BuildSavedUserObject(node);
      if (result.ok()) {
        objects.push_back(*result);
      } else {
        return result.status();
      }
    }
  }
  return objects;
}

tensorflow::StatusOr<Handle> BuildSavedUserObject(
    tensorflow::SavedObject saved_object_proto) {
  if (saved_object_proto.kind_case() != tensorflow::SavedObject::kUserObject) {
    return tensorflow::errors::InvalidArgument("Not a UserObject.");
  }

  std::string identifier = saved_object_proto.user_object().identifier();
  if (identifier == "trackable_list_wrapper") {
    tf::libtf::List user_list;
    // TODO(b/191267013): Populate with values.
    return user_list;
  }
  if (identifier == "trackable_dict_wrapper") {
    tf::libtf::Dictionary user_dict;
    // TODO(b/191267013): Populate with values.
    return user_dict;
  }
  if (identifier == "signature_map") {
    tf::libtf::Dictionary signature_map;
    // TODO(b/191267013): Populate with values.
    return signature_map;
  }
  if (identifier == "_generic_user_object") {
    tf::libtf::Dictionary user_object;
    // TODO(b/191267013): Populate with values.
    return user_object;
  }
  return tensorflow::errors::Unimplemented(absl::StrCat(
      "UserObject with identifier '", identifier, "' not implemented."));
}

// Register all available concrete functions from a SavedModel into a runtime.
tensorflow::Status RegisterConcreteFunctions(Runtime runtime,
                                             TFPackage tf_package) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc mht_0(mht_0_v, 251, "", "./tensorflow/cc/experimental/libtf/module.cc", "RegisterConcreteFunctions");

  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Initialize any variables found in the SavedModel and attach them to the
// appropriate object representation in the runtime.
tensorflow::Status InitializeVariables(Runtime runtime, TFPackage tf_package,
                                       std::vector<Handle> objects) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc mht_1(mht_1_v, 261, "", "./tensorflow/cc/experimental/libtf/module.cc", "InitializeVariables");

  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Register concrete functions with their associated polymorphic functions.
tensorflow::Status SetupPolymorphicFunctions(Runtime runtime,
                                             TFPackage tf_package,
                                             std::vector<Handle> objects) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc mht_2(mht_2_v, 271, "", "./tensorflow/cc/experimental/libtf/module.cc", "SetupPolymorphicFunctions");

  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Register any captures with their associated higher-level functions.
tensorflow::Status SetupFunctionCaptures(Runtime runtime, TFPackage tf_package,
                                         std::vector<Handle> objects) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSexperimentalPSlibtfPSmoduleDTcc mht_3(mht_3_v, 280, "", "./tensorflow/cc/experimental/libtf/module.cc", "SetupFunctionCaptures");

  return tensorflow::errors::Unimplemented("Not implemented.");
}

// Takes a flat list of Handles and builds them into the hierarchical
// representation defined by the SavedModel.
tensorflow::StatusOr<Handle> BuildObjectHierarchy(TFPackage tf_package,
                                                  std::vector<Handle> objects) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

tensorflow::StatusOr<Handle> BuildProgram(Runtime runtime,
                                          TFPackage& tf_package) {
  return tensorflow::errors::Unimplemented("Not implemented.");
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
