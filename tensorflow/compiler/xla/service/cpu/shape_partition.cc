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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"

namespace xla {
namespace cpu {

std::vector<int64_t> ShapePartitionAssigner::Run(
    int64_t target_partition_count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc mht_0(mht_0_v, 191, "", "./tensorflow/compiler/xla/service/cpu/shape_partition.cc", "ShapePartitionAssigner::Run");

  // Gather outer-most dims where dim_size >= 'target_partition_count'.
  // This may include the inner-dim as LLVM can vectorize loops with dynamic
  // bounds.
  std::vector<int64_t> outer_dims;
  int64_t outer_dim_size = 1;
  // TODO(b/27458679) Consider reserving enough minor dimensions (based on
  // target vector register width) to enable vector instructions.
  for (int i = shape_.layout().minor_to_major_size() - 1; i >= 0; --i) {
    const int64_t dimension = shape_.layout().minor_to_major(i);
    outer_dims.push_back(dimension);
    outer_dim_size *= shape_.dimensions(dimension);
    if (outer_dim_size >= target_partition_count) {
      break;
    }
  }

  // Clip target partition count if outer dim size is insufficient to cover.
  target_partition_count = std::min(outer_dim_size, target_partition_count);

  // Calculate the target number of partitions per-dimension, by factoring
  // 'target_partition_count' into 'num_outer_dims' equal terms.
  // EX:
  // *) target_partition_count = 16
  // *) out_dim_count = 2
  // *) target_dim_partition_count = 16 ^ (1.0 / 2) == 4
  const int64_t target_dim_partition_count = std::pow(
      static_cast<double>(target_partition_count), 1.0 / outer_dims.size());

  // Assign feasible dimension partitions based on 'target_dim_partition_count'
  // and actual dimension sizes from 'shape_'.
  std::vector<int64_t> dimension_partition_counts(outer_dims.size());
  for (int64_t i = 0; i < outer_dims.size(); ++i) {
    dimension_partition_counts[i] =
        std::min(static_cast<int64_t>(shape_.dimensions(outer_dims[i])),
                 target_dim_partition_count);
  }

  // Check if total partition count is below 'target_partition_count'.
  // This can occur if some dimensions in 'shape_' are below the
  // 'target_dim_partition_count' threshold.
  if (GetTotalPartitionCount(dimension_partition_counts) <
      target_partition_count) {
    // Assign additional partitions (greedily to outer dimensions), if doing
    // so would keep the total number of partitions <= 'target_partition_count',
    // using one pass over 'dimension_partition_counts'.
    for (int64_t i = 0; i < dimension_partition_counts.size(); ++i) {
      const int64_t current_dim_partition_count = dimension_partition_counts[i];
      const int64_t other_dims_partition_count =
          GetTotalPartitionCount(dimension_partition_counts) /
          current_dim_partition_count;
      // Constraint: (current + additional) * other <= target
      // Calculate: additional = target / other - current
      int64_t additional_partition_count =
          target_partition_count / other_dims_partition_count -
          current_dim_partition_count;
      // Clip 'additional_partition_count' by current dimension size.
      additional_partition_count = std::min(
          shape_.dimensions(outer_dims[i]) - dimension_partition_counts[i],
          additional_partition_count);
      if (additional_partition_count > 0) {
        dimension_partition_counts[i] += additional_partition_count;
      }
    }
  }

  return dimension_partition_counts;
}

int64_t ShapePartitionAssigner::GetTotalPartitionCount(
    const std::vector<int64_t>& dimension_partition_counts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc mht_1(mht_1_v, 264, "", "./tensorflow/compiler/xla/service/cpu/shape_partition.cc", "ShapePartitionAssigner::GetTotalPartitionCount");

  int64_t total_partition_count = 1;
  for (int64_t dim_partition_count : dimension_partition_counts) {
    total_partition_count *= dim_partition_count;
  }
  return total_partition_count;
}

ShapePartitionIterator::ShapePartitionIterator(
    const Shape& shape, const std::vector<int64_t>& dimension_partition_counts)
    : shape_(shape),
      dimension_partition_counts_(dimension_partition_counts),
      dimensions_(dimension_partition_counts_.size()),
      dimension_partition_sizes_(dimension_partition_counts_.size()),
      dimension_partition_strides_(dimension_partition_counts_.size()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc mht_2(mht_2_v, 281, "", "./tensorflow/compiler/xla/service/cpu/shape_partition.cc", "ShapePartitionIterator::ShapePartitionIterator");

  // Store partitioned outer dimensions from 'shape_'.
  for (int i = 0; i < dimensions_.size(); ++i) {
    dimensions_[i] = shape_.layout().minor_to_major(
        shape_.layout().minor_to_major_size() - 1 - i);
  }

  // Calculate partition size for each dimension (note that the size of
  // the last partition in each dimension may be different if the dimension
  // size is not a multiple of partition size).
  for (int i = 0; i < dimension_partition_sizes_.size(); ++i) {
    const int64_t dim_size = shape_.dimensions(dimensions_[i]);
    dimension_partition_sizes_[i] =
        std::max(int64_t{1}, dim_size / dimension_partition_counts_[i]);
  }

  // Calculate the partition strides for each dimension.
  dimension_partition_strides_[dimension_partition_strides_.size() - 1] = 1;
  for (int i = dimension_partition_strides_.size() - 2; i >= 0; --i) {
    dimension_partition_strides_[i] = dimension_partition_strides_[i + 1] *
                                      dimension_partition_counts_[i + 1];
  }
}

std::vector<std::pair<int64_t, int64_t>> ShapePartitionIterator::GetPartition(
    int64_t index) const {
  // Calculate and return the partition for 'index'.
  // Returns for each dimension: (partition_start, partition_size).
  std::vector<std::pair<int64_t, int64_t>> partition(dimensions_.size());
  for (int64_t i = 0; i < partition.size(); ++i) {
    // Calculate the index for dimension 'i'.
    const int64_t partition_index = index / dimension_partition_strides_[i];
    // Calculate dimension partition start at 'partition_index'.
    partition[i].first = partition_index * dimension_partition_sizes_[i];
    // Calculate dimension partition size (note that the last partition size
    // may be adjusted if dimension size is not a multiple of partition size).
    if (partition_index == dimension_partition_counts_[i] - 1) {
      // Last partition in this dimension.
      partition[i].second =
          shape_.dimensions(dimensions_[i]) - partition[i].first;
    } else {
      partition[i].second = dimension_partition_sizes_[i];
    }
    CHECK_GT(partition[i].second, 0);
    // Update index to remove contribution from current dimension.
    index -= partition_index * dimension_partition_strides_[i];
  }
  return partition;
}

int64_t ShapePartitionIterator::GetTotalPartitionCount() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partitionDTcc mht_3(mht_3_v, 334, "", "./tensorflow/compiler/xla/service/cpu/shape_partition.cc", "ShapePartitionIterator::GetTotalPartitionCount");

  return ShapePartitionAssigner::GetTotalPartitionCount(
      dimension_partition_counts_);
}

}  // namespace cpu
}  // namespace xla
