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
class MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// The builtin inputs provide a mechanism to generate simple TensorFlow graphs
// and feed them as inputs to Grappler. This can be used for quick experiments
// or to derive small regression tests.

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"

namespace tensorflow {
namespace grappler {

// Make a program with specified number of stages and "width" ops per stage.
namespace {
GraphDef CreateGraphDef(int num_stages, int width, int tensor_size,
                        bool use_multiple_devices, bool insert_queue,
                        const std::vector<string>& device_names) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.cc", "CreateGraphDef");

  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // x is from the feed.
  const int batch_size = tensor_size < 0 ? 1 : tensor_size;
  Output x = RandomNormal(s.WithOpName("x").WithDevice("/CPU:0"),
                          {batch_size, 1}, DataType::DT_FLOAT);

  // Create stages.
  std::vector<Output> last_stage;
  last_stage.push_back(x);
  for (int i = 0; i < num_stages; i++) {
    std::vector<Output> this_stage;
    for (int j = 0; j < width; j++) {
      if (last_stage.size() == 1) {
        Output unary_op =
            Sign(s.WithDevice(
                     device_names[use_multiple_devices ? j % device_names.size()
                                                       : 0]),
                 last_stage[0]);
        this_stage.push_back(unary_op);
      } else {
        Output combine =
            AddN(s.WithDevice(
                     device_names[use_multiple_devices ? j % device_names.size()
                                                       : 0]),
                 last_stage);
        this_stage.push_back(combine);
      }
    }
    last_stage = this_stage;
  }

  if (insert_queue) {
    FIFOQueue queue(s.WithOpName("queue").WithDevice("/CPU:0"),
                    {DataType::DT_FLOAT});
    QueueEnqueue enqueue(s.WithOpName("enqueue").WithDevice("/CPU:0"), queue,
                         last_stage);
    QueueDequeue dequeue(s.WithOpName("dequeue").WithDevice("/CPU:0"), queue,
                         {DataType::DT_FLOAT});
    QueueClose cancel(s.WithOpName("cancel").WithDevice("/CPU:0"), queue,
                      QueueClose::CancelPendingEnqueues(true));
    last_stage = {dequeue[0]};
  }

  // Create output.
  AddN output(s.WithOpName("y").WithDevice("/CPU:0"), last_stage);

  GraphDef def;
  TF_CHECK_OK(s.ToGraphDef(&def));
  return def;
}
}  // namespace

TrivialTestGraphInputYielder::TrivialTestGraphInputYielder(
    int num_stages, int width, int tensor_size, bool insert_queue,
    const std::vector<string>& device_names)
    : num_stages_(num_stages),
      width_(width),
      tensor_size_(tensor_size),
      insert_queue_(insert_queue),
      device_names_(device_names) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc mht_1(mht_1_v, 268, "", "./tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.cc", "TrivialTestGraphInputYielder::TrivialTestGraphInputYielder");
}

bool TrivialTestGraphInputYielder::NextItem(GrapplerItem* item) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPStrivial_test_graph_input_yielderDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.cc", "TrivialTestGraphInputYielder::NextItem");

  GrapplerItem r;
  r.id = strings::StrCat("ns:", num_stages_, "/",  // wrap
                         "w:", width_, "/",        // wrap
                         "ts:", tensor_size_);
  r.graph = CreateGraphDef(num_stages_, width_, tensor_size_,
                           true /*use_multiple_devices*/, insert_queue_,
                           device_names_);
  // If the batch size is variable, we need to choose a value to create a feed
  const int batch_size = tensor_size_ < 0 ? 1 : tensor_size_;
  Tensor x(DT_FLOAT, TensorShape({batch_size, 1}));
  r.feed.push_back(std::make_pair("x", x));
  r.fetch.push_back("y");

  if (insert_queue_) {
    QueueRunnerDef queue_runner;
    queue_runner.set_queue_name("queue");
    queue_runner.set_cancel_op_name("cancel");
    *queue_runner.add_enqueue_op_name() = "enqueue";
    r.queue_runners.push_back(queue_runner);
  }

  *item = std::move(r);
  return true;
}

}  // end namespace grappler
}  // end namespace tensorflow
