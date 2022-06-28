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
class MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc {
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
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

Operation::Operation(Node* n) : inputs_(GetInputs(n)), node_(n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_0(mht_0_v, 190, "", "./tensorflow/cc/framework/ops.cc", "Operation::Operation");
}

Output Operation::input(int32_t i) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_1(mht_1_v, 195, "", "./tensorflow/cc/framework/ops.cc", "Operation::input");

  CHECK_NOTNULL(node_);
  CHECK_GE(i, 0);
  CHECK_LT(i, node_->num_inputs());
  // Handle the case where the input was unknown at the time this
  // Operation was constructed.
  if (inputs_[i].first == nullptr && inputs_[i].second == -1) {
    for (const Edge* e : node_->in_edges()) {
      if (e->IsControlEdge()) continue;
      if (e->dst_input() == i) {
        return Output(e->src(), e->src_output());
      }
    }
  }
  return Output(inputs_[i].first, inputs_[i].second);
}

Output Operation::output(int32_t i) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_2(mht_2_v, 215, "", "./tensorflow/cc/framework/ops.cc", "Operation::output");

  CHECK_NOTNULL(node_);
  CHECK_GE(i, 0);
  CHECK_LT(i, node_->num_outputs());
  return Output(node_, i);
}

uint64 Operation::hash(int32_t index) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_3(mht_3_v, 225, "", "./tensorflow/cc/framework/ops.cc", "Operation::hash");

  return ::tensorflow::Hash64(reinterpret_cast<const char*>(&node_),
                              sizeof(Node*), index);
}

Operation::Inputs Operation::GetInputs(Node* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_4(mht_4_v, 233, "", "./tensorflow/cc/framework/ops.cc", "Operation::GetInputs");

  Operation::Inputs inputs;
  if (node != nullptr) {
    inputs.resize(node->num_inputs(), {nullptr, -1});
    for (const Edge* e : node->in_edges()) {
      if (e->IsControlEdge()) continue;
      inputs[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  return inputs;
}

Input::Initializer::Initializer(
    const std::initializer_list<Input::Initializer>& v) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSframeworkPSopsDTcc mht_5(mht_5_v, 249, "", "./tensorflow/cc/framework/ops.cc", "Input::Initializer::Initializer");

  if (v.size() < 1) {
    // Empty initializer list defaults to float tensor with shape (0,)
    tensor = Tensor(DT_FLOAT, TensorShape{0});
    return;
  }
  auto const& first = *v.begin();
  // Check to make sure that the constituent Initializers are all the same
  // type and same shape.
  for (auto const& e : v) {
    if (e.tensor.dtype() != first.tensor.dtype()) {
      status = errors::InvalidArgument(
          "Initializer list components should all have the same type");
      return;
    }
    if (!TensorShape{e.tensor.shape()}.IsSameSize(
            TensorShape{first.tensor.shape()})) {
      status = errors::InvalidArgument(
          "Initializer list components should all have the same shape");
      return;
    }
  }

  // Form the new shape.
  TensorShape shape{static_cast<int64_t>(v.size())};
  shape.AppendShape(TensorShape{first.tensor.shape()});

  Tensor t(first.tensor.dtype(), shape);

  // Collate the constituent Tensors.
  size_t offset = 0;
  for (auto const& e : v) {
    Tensor elem = e.tensor;
    if (first.tensor.dtype() == DT_STRING) {
      for (int i = 0; i < elem.NumElements(); ++i) {
        t.flat<tstring>()(offset + i) = elem.flat<tstring>()(i);
      }
      offset += elem.NumElements();
    } else {
      std::copy_n(elem.tensor_data().data(), elem.TotalBytes(),
                  const_cast<char*>(t.tensor_data().data()) + offset);
      offset += elem.TotalBytes();
    }
  }
  tensor = t;
}

}  // namespace tensorflow
