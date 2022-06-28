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
class MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc() {
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

#include <stddef.h>

#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace call_once_kernel {

// CallOnce operator is a control flow op to invoke other subgraph in the graph
// in order to conduct the given graph's initialization tasks, for example, hash
// table initialization and variable initialization.
//
// This operator will invoke the subgraph for initialization in the first run
// and become no-op after the first run in an interpreter's life cycle.

struct OpData {
  // Subgraph index to be invoked once in a life cycle by this CallOnce op.
  int init_subgraph_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/kernels/call_once.cc", "Init");

  auto* op_data = new OpData;
  const auto* params = reinterpret_cast<const TfLiteCallOnceParams*>(buffer);
  op_data->init_subgraph_index = params->init_subgraph_index;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc mht_1(mht_1_v, 225, "", "./tensorflow/lite/kernels/call_once.cc", "Free");

  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/kernels/call_once.cc", "Prepare");

  const OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  // Return early if the initialization graph is already invoked.
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;

  auto* subgraphs = this_subgraph->GetSubgraphs();

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 0);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 0);

  TF_LITE_ENSURE(context, op_data->init_subgraph_index < subgraphs->size());

  // Ensures that there are no input and output tensors in the subgraph.
  Subgraph* init_subgraph = (*subgraphs)[op_data->init_subgraph_index].get();
  TF_LITE_ENSURE_EQ(context, init_subgraph->inputs().size(), 0);
  TF_LITE_ENSURE_EQ(context, init_subgraph->outputs().size(), 0);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc mht_3(mht_3_v, 261, "", "./tensorflow/lite/kernels/call_once.cc", "Eval");

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);

  // The initialization graph should be invoked once in a life cycle.
  resource::InitializationStatusMap* map =
      &this_subgraph->initialization_status_map();
  resource::InitializationStatus* status =
      resource::GetInitializationStatus(map, op_data->init_subgraph_index);
  if (status->IsInitialized()) return kTfLiteOk;

  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph& init_subgraph = *(*subgraphs)[op_data->init_subgraph_index];

  TF_LITE_ENSURE_OK(context, init_subgraph.AllocateTensors());
  TF_LITE_ENSURE_OK(context, init_subgraph.Invoke());
  TF_LITE_ENSURE_OK(context, init_subgraph.ReleaseNonPersistentMemory());

  // Mark the invocation completed.
  status->MarkInitializationIsDone();
  return kTfLiteOk;
}

}  // namespace call_once_kernel

TfLiteRegistration* Register_CALL_ONCE() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPScall_onceDTcc mht_4(mht_4_v, 290, "", "./tensorflow/lite/kernels/call_once.cc", "Register_CALL_ONCE");

  static TfLiteRegistration r = {call_once_kernel::Init, call_once_kernel::Free,
                                 call_once_kernel::Prepare,
                                 call_once_kernel::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
