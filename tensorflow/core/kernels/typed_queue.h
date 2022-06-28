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

#ifndef TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
#define TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh() {
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


#include <deque>
#include <queue>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// TypedQueue builds on QueueBase, with backing class (SubQueue)
// known and stored within.  Shared methods that need to have access
// to the backed data sit in this class.
template <typename SubQueue>
class TypedQueue : public QueueBase {
 public:
  TypedQueue(const int32_t capacity, const DataTypeVector& component_dtypes,
             const std::vector<TensorShape>& component_shapes,
             const string& name);

  virtual Status Initialize();  // Must be called before any other method.

  int64_t MemoryUsed() const override;

 protected:
  std::vector<SubQueue> queues_ TF_GUARDED_BY(mu_);
};  // class TypedQueue

template <typename SubQueue>
TypedQueue<SubQueue>::TypedQueue(
    int32_t capacity, const DataTypeVector& component_dtypes,
    const std::vector<TensorShape>& component_shapes, const string& name)
    : QueueBase(capacity, component_dtypes, component_shapes, name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/typed_queue.h", "TypedQueue<SubQueue>::TypedQueue");
}

template <typename SubQueue>
Status TypedQueue<SubQueue>::Initialize() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/typed_queue.h", "TypedQueue<SubQueue>::Initialize");

  if (component_dtypes_.empty()) {
    return errors::InvalidArgument("Empty component types for queue ", name_);
  }
  if (!component_shapes_.empty() &&
      component_dtypes_.size() != component_shapes_.size()) {
    return errors::InvalidArgument(
        "Different number of component types.  ",
        "Types: ", DataTypeSliceString(component_dtypes_),
        ", Shapes: ", ShapeListString(component_shapes_));
  }

  mutex_lock lock(mu_);
  queues_.reserve(num_components());
  for (int i = 0; i < num_components(); ++i) {
    queues_.push_back(SubQueue());
  }
  return Status::OK();
}

template <typename SubQueue>
inline int64_t SizeOf(const SubQueue& sq) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_2(mht_2_v, 251, "", "./tensorflow/core/kernels/typed_queue.h", "SizeOf");

  static_assert(sizeof(SubQueue) != sizeof(SubQueue), "SubQueue size unknown.");
  return 0;
}

template <>
inline int64_t SizeOf(const std::deque<Tensor>& sq) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/typed_queue.h", "SizeOf");

  if (sq.empty()) {
    return 0;
  }
  return sq.size() * sq.front().AllocatedBytes();
}

template <>
inline int64_t SizeOf(const std::vector<Tensor>& sq) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_4(mht_4_v, 271, "", "./tensorflow/core/kernels/typed_queue.h", "SizeOf");

  if (sq.empty()) {
    return 0;
  }
  return sq.size() * sq.front().AllocatedBytes();
}

using TensorPair = std::pair<int64_t, Tensor>;

template <typename U, typename V>
int64_t SizeOf(const std::priority_queue<TensorPair, U, V>& sq) {
  if (sq.empty()) {
    return 0;
  }
  return sq.size() * (sizeof(TensorPair) + sq.top().second.AllocatedBytes());
}

template <typename SubQueue>
inline int64_t TypedQueue<SubQueue>::MemoryUsed() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_queueDTh mht_5(mht_5_v, 292, "", "./tensorflow/core/kernels/typed_queue.h", "TypedQueue<SubQueue>::MemoryUsed");

  int memory_size = 0;
  mutex_lock l(mu_);
  for (const auto& sq : queues_) {
    memory_size += SizeOf(sq);
  }
  return memory_size;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TYPED_QUEUE_H_
