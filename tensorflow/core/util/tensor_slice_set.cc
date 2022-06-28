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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc() {
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

#include "tensorflow/core/util/tensor_slice_set.h"

#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

TensorSliceSet::TensorSliceSet(const TensorShape& shape, DataType type)
    : shape_(shape), type_(type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/util/tensor_slice_set.cc", "TensorSliceSet::TensorSliceSet");
}

TensorSliceSet::~TensorSliceSet() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/util/tensor_slice_set.cc", "TensorSliceSet::~TensorSliceSet");
}

Status TensorSliceSet::Register(const TensorSlice& slice, const string& tag) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/util/tensor_slice_set.cc", "TensorSliceSet::Register");

  TensorShape result_shape;
  TF_RETURN_IF_ERROR(slice.SliceTensorShape(shape_, &result_shape));
  string str = slice.DebugString();

  if (slices_.empty()) {
    slices_hull_ = slice;
  } else {
    // We check if there is any intersection between this slice and any of the
    // registered slices.
    if (slices_hull_.Overlaps(slice)) {
      for (const auto& x : slices_) {
        if (slice.Overlaps(x.second.slice)) {
          return errors::Internal("Overlapping slices: existing slice = ",
                                  x.first, ", new slice = ", str);
        }
      }
    }
    // No overlap: we can now insert the slice
    slices_hull_.UpdateToCover(slice);
  }

  TensorSliceSet::SliceInfo info = {slice, tag, result_shape.num_elements()};
  slices_.insert(std::make_pair(str, info));
  return Status::OK();
}

bool TensorSliceSet::QueryMeta(
    const TensorSlice& slice,
    std::vector<std::pair<TensorSlice, string>>* results) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/util/tensor_slice_set.cc", "TensorSliceSet::QueryMeta");

  results->clear();
  Status s;
  string str = slice.DebugString();
  // First we check if there is an exactly match (this is the dominant case).
  const TensorSliceSet::SliceInfo* info = gtl::FindOrNull(slices_, str);
  if (info) {
    results->emplace_back(std::make_pair(info->slice, info->tag));
    return true;
  } else {
    // We didn't find any exact match but there is still a possibility that
    // multiple existing slices can be patched together to output the slice.
    // We figure this out by computing the intersection of each of the existing
    // slices with the query slice, and check if the union of all these
    // intersections cover the entire slice. We rely on the fact that the
    // existing slices don't have any intersection among themselves.
    TensorShape target_shape;
    Status s;
    s = slice.SliceTensorShape(shape_, &target_shape);
    if (!s.ok()) {
      LOG(WARNING) << s;
      return false;
    }
    int64_t total_size = target_shape.num_elements();

    int64_t overlap_size = 0;
    TensorSlice intersection;
    TensorShape inter_shape;
    for (const auto& x : slices_) {
      if (slice.Intersect(x.second.slice, &intersection)) {
        s = intersection.SliceTensorShape(shape_, &inter_shape);
        if (!s.ok()) {
          LOG(WARNING) << s;
          return false;
        }
        overlap_size += inter_shape.num_elements();
        results->emplace_back(std::make_pair(x.second.slice, x.second.tag));
      }
    }
    if (total_size == overlap_size) {
      // We have it!
      return true;
    } else {
      // We don't have all the data for the asked tensor slice
      results->clear();
      return false;
    }
  }
}

Status RegisterTensorSlice(
    const string& name, const TensorShape& shape, DataType type,
    const string& tag, const TensorSlice& slice,
    std::unordered_map<string, TensorSliceSet*>* tensor_slices) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   mht_4_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_setDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/util/tensor_slice_set.cc", "RegisterTensorSlice");

  DCHECK_NE(tensor_slices, nullptr);
  TensorSliceSet* tss = gtl::FindPtrOrNull(*tensor_slices, name);
  // Create a tensor slice set if needed
  if (!tss) {
    tss = new TensorSliceSet(shape, type);
    tensor_slices->insert(std::make_pair(name, tss));
  } else {
    // Check if the shapes match
    const TensorShape& tss_shape(tss->shape());
    if (!shape.IsSameSize(tss_shape)) {
      return errors::Internal("Incompatible tensor shapes detected for tensor ",
                              name, ": existing = ", tss_shape.DebugString(),
                              ", new = ", shape.DebugString());
    }
    if (type != tss->type()) {
      return errors::Internal("Incompatible tensor types detected for tensor ",
                              name,
                              ": existing = ", DataTypeString(tss->type()),
                              ", new = ", DataTypeString(type));
    }
  }
  // Register the tensor slices without the actual data.
  return tss->Register(slice, tag);
}

}  // namespace checkpoint

}  // namespace tensorflow
