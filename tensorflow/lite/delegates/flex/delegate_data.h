/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh() {
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


#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/flex/buffer_map.h"
#include "tensorflow/lite/delegates/flex/subgraph_resource.h"

namespace tflite {
namespace flex {

// Data kept by the Flex delegate for the lifetime of an Interpreter.
//
// Note: This class is *not* thread-safe; any dependent delegates should not be
// used concurrently.
class DelegateData {
 public:
  DelegateData();
  ~DelegateData();

  // Prepare the necessary EagerContext and data for execution.
  // This must be called at least once before execution. After preparation
  // succeeds, redundant calls will be ignored (even if the session_options
  // differ).
  // When `main_subgraph` parameter is provided, this function will register
  // FunctionDefs associated with each of the subgraphs attached to the
  // `main_subgraph` which is delegated by 'flex_delegate'.
  // 'flex_delegate' should always be non-null when 'main_subgraph' is
  // non-null.
  tensorflow::Status Prepare(const tensorflow::SessionOptions& session_options,
                             Subgraph* main_subgraph = nullptr,
                             TfLiteDelegate* flex_delegate = nullptr);

  // The EagerContext that is required for execution of Flex Ops.
  // Note: The context is lazily created after the first call to |Prepare()|.
  tensorflow::EagerContext* GetEagerContext() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh mht_0(mht_0_v, 221, "", "./tensorflow/lite/delegates/flex/delegate_data.h", "GetEagerContext");
 return eager_context_; }

  tensorflow::CancellationManager* GetCancellationManager() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh mht_1(mht_1_v, 226, "", "./tensorflow/lite/delegates/flex/delegate_data.h", "GetCancellationManager");

    return cancellation_manager_;
  }

  void SetCancellationManager(
      tensorflow::CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh mht_2(mht_2_v, 234, "", "./tensorflow/lite/delegates/flex/delegate_data.h", "SetCancellationManager");

    cancellation_manager_ = cancellation_manager;
  }

  // Map from TF Lite tensor index to TensorFlow tensor for a given context.
  BufferMap* GetBufferMap(const TfLiteContext* context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegate_dataDTh mht_3(mht_3_v, 242, "", "./tensorflow/lite/delegates/flex/delegate_data.h", "GetBufferMap");

    return &buffer_map_[context];
  }

  // Returns the mapping between tensor index and last node index for a given
  // context.
  std::map<int, int>* GetTensorReleaseMap(const TfLiteContext* context) {
    return &tensor_release_map_[context];
  }

 private:
  // Will be null until Prepare() is called and completes successfully.
  tensorflow::EagerContext* eager_context_ = nullptr;
  // Not owned by DelegateData.
  tensorflow::CancellationManager* cancellation_manager_ = nullptr;
  // TODO(b/112439500): Clean up stale BufferMap instances after adding the
  // necessary cleanup hook from a TfLiteContext to a TfLiteDelegate.
  std::unordered_map<const TfLiteContext*, BufferMap> buffer_map_;
  // Maps between context and the tensor release map. The map will be filled
  // during delegate initialization, and queried during eval to look up tensor
  // lifetime information.
  std::unordered_map<const TfLiteContext*, std::map<int, int>>
      tensor_release_map_;
};

// Creates a `TFLiteSubgraphResource` for each subgraph (execpt
// for main subgraph) in the model and adds it in the eager context's resource
// manager. It also registers FunctionDefs in the function library runtime for
// subgraphs which are used by a list of flex ops.
tensorflow::Status RegisterFunctionDefForSubgraphs(
    Subgraph& main_subgraph,
    const std::function<tensorflow::Status(
        const std::vector<std::unique_ptr<Subgraph>>&,
        std::set<std::string>* result)>& select_subgraphs_to_register,
    tensorflow::ResourceMgr* resource_mgr,
    tensorflow::EagerContext* eager_context, TfLiteDelegate* flex_delegate);

}  // namespace flex
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_DATA_H_
