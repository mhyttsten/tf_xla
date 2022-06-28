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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc() {
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

#include "tensorflow/core/framework/tensor_slice.h"

#include <limits>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

TensorSlice::TensorSlice(int dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::TensorSlice");
 SetFullSlice(dim); }

TensorSlice::TensorSlice(const TensorSliceProto& proto) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::TensorSlice");

  starts_.reserve(proto.extent_size());
  lengths_.reserve(proto.extent_size());
  for (const auto& e : proto.extent()) {
    starts_.push_back(e.start());
    lengths_.push_back(GetExtentLength(e));
  }
}

TensorSlice::TensorSlice(
    std::initializer_list<std::pair<int64_t, int64_t>> extents) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::TensorSlice");

  starts_.reserve(extents.size());
  lengths_.reserve(extents.size());
  for (const auto& e : extents) {
    starts_.push_back(e.first);
    lengths_.push_back(e.second);
  }
}

Status TensorSlice::BuildTensorSlice(const TensorSliceProto& proto,
                                     TensorSlice* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::BuildTensorSlice");

  output->Clear();
  output->starts_.reserve(proto.extent_size());
  output->lengths_.reserve(proto.extent_size());
  for (const auto& e : proto.extent()) {
    int64_t l = GetExtentLength(e);
    if (e.start() != 0 || l != kFullExtent) {
      if (e.start() < 0 || l <= 0) {
        return errors::InvalidArgument(
            "Expected non-negative start and positive length but got start = ",
            e.start(), ", length = ", l, ": extent = ", e.ShortDebugString());
      }
      // Calculating the extent end must not cause signed integer overflow.
      if (static_cast<uint64_t>(e.start()) + static_cast<uint64_t>(e.length()) >
          std::numeric_limits<int64_t>::max()) {
        return errors::InvalidArgument(
            "Extent end exceeds the maximum possible size: extent = ",
            e.ShortDebugString());
      }
    }
    output->starts_.push_back(e.start());
    output->lengths_.push_back(l);
  }

  return Status::OK();
}

Status TensorSlice::Parse(const string& str, TensorSlice* slice) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::Parse");

  std::vector<string> items = str_util::Split(str, ':', str_util::SkipEmpty());
  slice->starts_.reserve(items.size());
  slice->lengths_.reserve(items.size());
  for (const string& x : items) {
    int64_t s, l;
    if (x == "-") {
      // "everything"
      s = 0;
      l = kFullExtent;
    } else {
      std::vector<string> sl = str_util::Split(x, ',', str_util::SkipEmpty());
      if (sl.size() != 2 || !strings::safe_strto64(sl[0], &s) ||
          !strings::safe_strto64(sl[1], &l)) {
        return errors::InvalidArgument(
            "Expected a pair of numbers or '-' "
            "but got '",
            x, "': string = ", str);
      }
      if (s < 0 || l <= 0) {
        return errors::InvalidArgument(
            "Expected non-negative start and "
            "positive length but got start = ",
            s, ", length = ", l, ": string = ", str);
      }
    }
    slice->starts_.push_back(s);
    slice->lengths_.push_back(l);
  }

  return Status::OK();
}

void TensorSlice::Clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::Clear");

  starts_.clear();
  lengths_.clear();
}

bool TensorSlice::IsFull() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_6(mht_6_v, 304, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::IsFull");

  for (int d = 0; d < dims(); ++d) {
    if (!IsFullAt(d)) return false;
  }
  return true;
}

void TensorSlice::SetFullSlice(int dim) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_7(mht_7_v, 314, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::SetFullSlice");

  Clear();
  starts_.reserve(dim);
  lengths_.reserve(dim);
  for (int d = 0; d < dim; ++d) {
    starts_.push_back(0);
    lengths_.push_back(kFullExtent);
  }
}

void TensorSlice::Extend(int dim) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_8(mht_8_v, 327, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::Extend");

  int old_dim = dims();
  DCHECK_LE(old_dim, dim);
  starts_.resize(dim);
  lengths_.resize(dim);
  for (int d = old_dim; d < dim; ++d) {
    starts_[d] = 0;
    lengths_[d] = kFullExtent;
  }
}

void TensorSlice::AsProto(TensorSliceProto* proto) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::AsProto");

  for (int d = 0; d < dims(); ++d) {
    TensorSliceProto::Extent* e = proto->add_extent();
    // We only need to record the explicit slice for non-full slices
    if (!IsFullAt(d)) {
      e->set_start(starts_[d]);
      e->set_length(lengths_[d]);
    }
  }
}

string TensorSlice::DebugString() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_10(mht_10_v, 355, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::DebugString");

  string buffer;
  bool first = true;
  for (int d = 0; d < dims(); ++d) {
    if (!first) {
      buffer.append(":");
    }
    if (IsFullAt(d)) {
      buffer.append("-");
    } else {
      strings::StrAppend(&buffer, starts_[d], ",", lengths_[d]);
    }
    first = false;
  }
  return buffer;
}

bool TensorSlice::Intersect(const TensorSlice& other,
                            TensorSlice* result) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_11(mht_11_v, 376, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::Intersect");

  // First, if two slices have different ranks, they obviously don't overlap
  // -- in fact they are not compatible.
  if (dims() != other.dims()) {
    return false;
  }

  // Setting the result to the right dimension
  if (result) {
    result->SetFullSlice(dims());
  }
  // The two slices overlap if they overlap in all dimensions.
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      if (result) {
        result->set_start(d, other.start(d));
        result->set_length(d, other.length(d));
      }
    } else if (other.IsFullAt(d)) {
      if (result) {
        result->set_start(d, start(d));
        result->set_length(d, length(d));
      }
    } else {
      // If we have an intersection here, it should have a start that is the
      // max of the two starts and an end that is the min of the two ends.
      int64_t s = std::max(start(d), other.start(d));
      int64_t l = std::min(end(d), other.end(d)) - s;
      if (l > 0) {
        // We have a real intersection
        if (result) {
          result->set_start(d, s);
          result->set_length(d, l);
        }
      } else {
        // We don't have an intersection for this dimension -- thus we don't
        // have any intersection at all.
        if (result) {
          result->Clear();
        }
        return false;
      }
    }
  }
  // If we are here, we know there is overlap in every dimension.
  return true;
}

bool TensorSlice::operator==(const TensorSlice& other) const {
  return dims() == other.dims() && starts_ == other.starts_ &&
         lengths_ == other.lengths_;
}

void TensorSlice::ComputeRelative(const TensorSlice& sub,
                                  TensorSlice* relative) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_12(mht_12_v, 433, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::ComputeRelative");

  DCHECK_EQ(dims(), sub.dims());
  relative->SetFullSlice(dims());
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      relative->set_start(d, sub.start(d));
      relative->set_length(d, sub.length(d));
    } else {
      // Otherwise the relative start is the difference between the start of
      // sub and the start of base
      relative->set_start(d, sub.start(d) - start(d));
      relative->set_length(d, sub.length(d));
    }
  }
}

void TensorSlice::UpdateToCover(const TensorSlice& other) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_13(mht_13_v, 452, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::UpdateToCover");

  DCHECK_EQ(dims(), other.dims());
  for (int d = 0; d < dims(); ++d) {
    if (!IsFullAt(d)) {
      if (other.IsFullAt(d)) {
        starts_[d] = 0;
        lengths_[d] = kFullExtent;
      } else {
        const auto new_end = std::max(end(d), other.end(d));
        set_start(d, std::min(start(d), other.start(d)));
        set_length(d, new_end - start(d));
      }
    }
  }
}

// static
bool TensorSlice::HasExtentLength(const TensorSliceProto::Extent& extent) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_14(mht_14_v, 472, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::HasExtentLength");

  return extent.has_length_case() == TensorSliceProto::Extent::kLength;
}

// static
int64_t TensorSlice::GetExtentLength(const TensorSliceProto::Extent& extent) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_15(mht_15_v, 480, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::GetExtentLength");

  if (!HasExtentLength(extent)) return -1;
  return extent.length();
}

Status TensorSlice::SliceTensorShape(const TensorShape& shape,
                                     TensorShape* result_shape) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_sliceDTcc mht_16(mht_16_v, 489, "", "./tensorflow/core/framework/tensor_slice.cc", "TensorSlice::SliceTensorShape");

  result_shape->Clear();
  // Mismatching ranks: we can't apply the slice at all.
  if (shape.dims() != dims()) {
    return errors::Internal("Mismatching ranks: shape = ", shape.DebugString(),
                            ", slice = ", DebugString());
  }
  for (int d = 0; d < dims(); ++d) {
    if (IsFullAt(d)) {
      result_shape->AddDim(shape.dim_size(d));
    } else {
      // Check if the extent applies to the dimension
      if (end(d) <= shape.dim_size(d)) {
        // Yes: the end is within the range of the dim -- we adjust the result
        // shape so that its size along this dimension is the length of the
        // slice.
        result_shape->AddDim(length(d));
      } else {
        // The extent doesn't apply to the dimension
        result_shape->Clear();
        return errors::Internal("Extent in dimension ", d,
                                " out of bounds: shape = ", shape.DebugString(),
                                ", slice = ", DebugString());
      }
    }
  }
  // If we are here, we have successfully applied the shape.
  return Status::OK();
}

const int64_t TensorSlice::kFullExtent = -1;

}  // namespace tensorflow
