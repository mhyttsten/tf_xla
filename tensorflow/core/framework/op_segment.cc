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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc() {
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

#include "tensorflow/core/framework/op_segment.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

OpSegment::Item::~Item() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::Item::~Item");

  for (const auto& kv : name_kernel) delete kv.second;
}

OpSegment::OpSegment() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::OpSegment");
}

OpSegment::~OpSegment() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::~OpSegment");

  for (const auto& kv : sessions_) delete kv.second;
}

Status OpSegment::FindOrCreate(const string& session_handle,
                               const string& node_name, OpKernel** kernel,
                               CreateKernelFn create_fn) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("session_handle: \"" + session_handle + "\"");
   mht_3_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::FindOrCreate");

  {
    mutex_lock l(mu_);
    auto item = gtl::FindPtrOrNull(sessions_, session_handle);
    if (item == nullptr) {
      return errors::NotFound("Session ", session_handle, " is not found.");
    }
    *kernel = gtl::FindPtrOrNull(item->name_kernel, node_name);
    if (*kernel != nullptr) {
      return Status::OK();
    }
  }
  Status s = create_fn(kernel);
  if (!s.ok()) {
    LOG(ERROR) << "Create kernel failed: " << s;
    return s;
  }
  {
    mutex_lock l(mu_);
    auto item = gtl::FindPtrOrNull(sessions_, session_handle);
    if (item == nullptr) {
      return errors::NotFound("Session ", session_handle, " is not found.");
    }
    OpKernel** p_kernel = &(item->name_kernel[node_name]);
    if (*p_kernel == nullptr) {
      *p_kernel = *kernel;  // Inserts 'kernel' in the map.
    } else {
      delete *kernel;
      *kernel = *p_kernel;
    }
  }
  return Status::OK();
}

void OpSegment::AddHold(const string& session_handle) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("session_handle: \"" + session_handle + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::AddHold");

  mutex_lock l(mu_);
  Item** item = &sessions_[session_handle];
  if (*item == nullptr) {
    *item = new Item;  // num_holds == 1
  } else {
    ++((*item)->num_holds);
  }
}

void OpSegment::RemoveHold(const string& session_handle) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("session_handle: \"" + session_handle + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_5(mht_5_v, 272, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::RemoveHold");

  Item* item = nullptr;
  {
    mutex_lock l(mu_);
    auto siter = sessions_.find(session_handle);
    if (siter == sessions_.end()) {
      VLOG(1) << "Session " << session_handle << " is not found.";
      return;
    }
    item = siter->second;
    if (--(item->num_holds) > 0) {
      return;
    } else {
      sessions_.erase(siter);
    }
  }
  delete item;
}

bool OpSegment::ShouldOwnKernel(FunctionLibraryRuntime* lib,
                                const string& node_op) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("node_op: \"" + node_op + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segmentDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/framework/op_segment.cc", "OpSegment::ShouldOwnKernel");

  // OpSegment should not own kernel if the node is stateless, or a function.
  return lib->IsStateful(node_op) &&
         lib->GetFunctionLibraryDefinition()->Find(node_op) == nullptr &&
         node_op != "PartitionedCall" && node_op != "StatefulPartitionedCall";
}

}  // end namespace tensorflow
