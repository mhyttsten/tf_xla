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
class MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPSfile_input_yielderDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPSfile_input_yielderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPSfile_input_yielderDTcc() {
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

#include "tensorflow/core/grappler/inputs/file_input_yielder.h"

#include <memory>
#include <unordered_set>
#include <utility>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {

FileInputYielder::FileInputYielder(const std::vector<string>& filenames,
                                   size_t max_iterations)
    : filenames_(filenames),
      current_file_(0),
      current_iteration_(0),
      max_iterations_(max_iterations),
      bad_inputs_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPSfile_input_yielderDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/grappler/inputs/file_input_yielder.cc", "FileInputYielder::FileInputYielder");

  CHECK_GT(filenames.size(), 0) << "List of filenames is empty.";
}

bool FileInputYielder::NextItem(GrapplerItem* item) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSinputsPSfile_input_yielderDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/grappler/inputs/file_input_yielder.cc", "FileInputYielder::NextItem");

  if (filenames_.size() == bad_inputs_) {
    // All the input files are bad, give up.
    return false;
  }

  if (current_file_ >= filenames_.size()) {
    if (current_iteration_ >= max_iterations_) {
      return false;
    } else {
      ++current_iteration_;
      current_file_ = 0;
      bad_inputs_ = 0;
    }
  }

  const string& filename = filenames_[current_file_];
  ++current_file_;

  if (!Env::Default()->FileExists(filename).ok()) {
    LOG(WARNING) << "Skipping non existent file " << filename;
    // Attempt to process the next item on the list
    bad_inputs_ += 1;
    return NextItem(item);
  }

  LOG(INFO) << "Loading model from " << filename;

  MetaGraphDef metagraph;
  Status s = ReadBinaryProto(Env::Default(), filename, &metagraph);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), filename, &metagraph);
  }
  if (!s.ok()) {
    LOG(WARNING) << "Failed to read MetaGraphDef from " << filename << ": "
                 << s.ToString();
    // Attempt to process the next item on the list
    bad_inputs_ += 1;
    return NextItem(item);
  }

  if (metagraph.collection_def().count("train_op") == 0 ||
      !metagraph.collection_def().at("train_op").has_node_list() ||
      metagraph.collection_def().at("train_op").node_list().value_size() == 0) {
    LOG(ERROR) << "No train op specified";
    bad_inputs_ += 1;
    metagraph = MetaGraphDef();
    return NextItem(item);
  } else {
    std::unordered_set<string> train_ops;
    for (const string& val :
         metagraph.collection_def().at("train_op").node_list().value()) {
      train_ops.insert(NodeName(val));
    }
    std::unordered_set<string> train_ops_found;
    for (auto& node : metagraph.graph_def().node()) {
      if (train_ops.find(node.name()) != train_ops.end()) {
        train_ops_found.insert(node.name());
      }
    }
    if (train_ops_found.size() != train_ops.size()) {
      for (const auto& train_op : train_ops) {
        if (train_ops_found.find(train_op) != train_ops_found.end()) {
          LOG(ERROR) << "Non existent train op specified: " << train_op;
        }
      }
      bad_inputs_ += 1;
      metagraph = MetaGraphDef();
      return NextItem(item);
    }
  }

  const string id =
      strings::StrCat(Fingerprint64(metagraph.SerializeAsString()));

  ItemConfig cfg;
  std::unique_ptr<GrapplerItem> new_item =
      GrapplerItemFromMetaGraphDef(id, metagraph, cfg);
  if (new_item == nullptr) {
    bad_inputs_ += 1;
    metagraph = MetaGraphDef();
    return NextItem(item);
  }

  *item = std::move(*new_item);
  return true;
}

}  // end namespace grappler
}  // end namespace tensorflow
