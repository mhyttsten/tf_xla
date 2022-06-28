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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc() {
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

#include <algorithm>
#include <random>

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace cpu {
namespace {

class ShapePartitionAssignerTest : public HloTestBase {
 protected:
  typedef std::vector<int64_t> Vec;

  void RunR2Test(const Shape& shape, int64_t max_target_partition_count,
                 const std::vector<int64_t>* expected_partitions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/xla/service/cpu/shape_partition_test.cc", "RunR2Test");

    ShapePartitionAssigner assigner(shape);
    // Iterate through 1..max_target_partition_count.
    for (int64_t i = 1; i <= max_target_partition_count; ++i) {
      std::vector<int64_t> actual_partitions =
          assigner.Run(/*target_partition_count=*/i);
      EXPECT_THAT(actual_partitions, expected_partitions[i - 1]);
    }
  }
};

TEST_F(ShapePartitionAssignerTest, Shape13WithLayout10) {
  std::vector<int64_t> expected_partitions[] = {{1} /* 1 */, {1, 2} /* 2 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {1, 3}, {1, 0}), 2,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape31WithLayout01) {
  std::vector<int64_t> expected_partitions[] = {
      {1} /* 1 */, {1, 2} /* 2 */
  };
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {3, 1}, {0, 1}), 2,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape53WithLayout10) {
  std::vector<int64_t> expected_partitions[] = {{1} /* 1 */, {2} /* 2 */,
                                                {3} /* 3 */, {4} /* 4 */,
                                                {5} /* 5 */, {3, 2} /* 6 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {1, 0}), 6,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape53WithLayout01) {
  std::vector<int64_t> expected_partitions[] = {
      {1} /* 1 */, {2} /* 2 */, {3} /* 3 */, {2, 2} /* 4 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {0, 1}), 4,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape532WithLayout210) {
  std::vector<int64_t> expected_partitions[] = {
      {1} /* 1 */,     {2} /* 2 */,     {3} /* 3 */,     {4} /* 4 */,
      {5} /* 5 */,     {3, 2} /* 6 */,  {3, 2} /* 7 */,  {4, 2} /* 8 */,
      {3, 3} /* 9 */,  {3, 3} /* 10 */, {3, 3} /* 11 */, {4, 3} /* 12 */,
      {4, 3} /* 13 */, {4, 3} /* 14 */, {5, 3} /* 15 */, {4, 2, 2} /* 16 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 1, 0}), 16,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape532WithLayout201) {
  std::vector<int64_t> expected_partitions[] = {
      {1} /* 1 */,     {2} /* 2 */,     {3} /* 3 */,     {2, 2} /* 4 */,
      {2, 2} /* 5 */,  {3, 2} /* 6 */,  {3, 2} /* 7 */,  {3, 2} /* 8 */,
      {3, 3} /* 9 */,  {3, 3} /* 10 */, {3, 3} /* 11 */, {3, 4} /* 12 */,
      {3, 4} /* 13 */, {3, 4} /* 14 */, {3, 5} /* 15 */, {3, 2, 2} /* 16 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 0, 1}), 16,
            expected_partitions);
}

class ShapePartitionIteratorTest : public HloTestBase {
 protected:
  typedef std::vector<std::pair<int64_t, int64_t>> Partition;
};

TEST_F(ShapePartitionIteratorTest, Shape53WithLayout10) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {1, 0});

  {
    ShapePartitionIterator iterator(shape, {1});
    EXPECT_EQ(1, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(absl::c_equal(Partition({{0, 5}}), iterator.GetPartition(0)));
  }

  {
    ShapePartitionIterator iterator(shape, {2});
    EXPECT_EQ(2, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(absl::c_equal(Partition({{0, 2}}), iterator.GetPartition(0)));
    EXPECT_TRUE(absl::c_equal(Partition({{2, 3}}), iterator.GetPartition(1)));
  }

  {
    ShapePartitionIterator iterator(shape, {3});
    EXPECT_EQ(3, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(absl::c_equal(Partition({{0, 1}}), iterator.GetPartition(0)));
    EXPECT_TRUE(absl::c_equal(Partition({{1, 1}}), iterator.GetPartition(1)));
    EXPECT_TRUE(absl::c_equal(Partition({{2, 3}}), iterator.GetPartition(2)));
  }
}

TEST_F(ShapePartitionIteratorTest, Shape532WithLayout210) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 1, 0});

  {
    ShapePartitionIterator iterator(shape, {1, 1});
    EXPECT_EQ(1, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(
        absl::c_equal(Partition({{0, 5}, {0, 3}}), iterator.GetPartition(0)));
  }

  {
    ShapePartitionIterator iterator(shape, {2, 2});
    EXPECT_EQ(4, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(
        absl::c_equal(Partition({{0, 2}, {0, 1}}), iterator.GetPartition(0)));
    EXPECT_TRUE(
        absl::c_equal(Partition({{0, 2}, {1, 2}}), iterator.GetPartition(1)));
    EXPECT_TRUE(
        absl::c_equal(Partition({{2, 3}, {0, 1}}), iterator.GetPartition(2)));
    EXPECT_TRUE(
        absl::c_equal(Partition({{2, 3}, {1, 2}}), iterator.GetPartition(3)));
  }
}

class RandomShapePartitionIteratorTest : public HloTestBase {
 protected:
  typedef std::vector<std::pair<int64_t, int64_t>> Partition;
  RandomShapePartitionIteratorTest()
      : generator_(rd_()), distribution_(1, 10) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc mht_1(mht_1_v, 324, "", "./tensorflow/compiler/xla/service/cpu/shape_partition_test.cc", "RandomShapePartitionIteratorTest");
}

  std::vector<int64_t> RandR4Dims() { return {Rand(), Rand(), Rand(), Rand()}; }

  int64_t Rand() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSshape_partition_testDTcc mht_2(mht_2_v, 331, "", "./tensorflow/compiler/xla/service/cpu/shape_partition_test.cc", "Rand");
 return distribution_(generator_); }

  std::random_device rd_;
  std::mt19937 generator_;
  std::uniform_int_distribution<int> distribution_;
};

TEST_F(RandomShapePartitionIteratorTest, RandomShapeAndPartitions) {
  // Choose random dimensions for R4 shape.
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, RandR4Dims(), {3, 2, 1, 0});
  // Choose random number of outer dimensions to partition.
  const int num_outer_dims_to_partition = 1 + (Rand() % 3);
  // Choose random outer dimension partition counts.
  std::vector<int64_t> dim_sizes(num_outer_dims_to_partition);
  std::vector<int64_t> dim_partition_counts(num_outer_dims_to_partition);
  int64_t total_dim_size = 1;
  for (int i = 0; i < num_outer_dims_to_partition; ++i) {
    const int64_t dimension = shape.layout().minor_to_major(
        shape.layout().minor_to_major_size() - 1 - i);
    dim_sizes[i] = shape.dimensions(dimension);
    total_dim_size *= dim_sizes[i];
    // Choose dimension partition count in [1, dim_size]
    const int64_t dim_partition_count = 1 + Rand() % dim_sizes[i];
    dim_partition_counts[i] = dim_partition_count;
  }
  // Iterate through all partition: for each partition record covered
  // index ranges by dimension.
  std::vector<std::map<int64_t, int64_t>> ranges(num_outer_dims_to_partition);
  ShapePartitionIterator partition_iterator(shape, dim_partition_counts);
  const int64_t partition_count = partition_iterator.GetTotalPartitionCount();
  for (int64_t i = 0; i < partition_count; ++i) {
    const auto& dim_partition = partition_iterator.GetPartition(i);
    for (int dim = 0; dim < dim_partition.size(); ++dim) {
      ranges[dim].insert(
          std::make_pair(dim_partition[dim].first,
                         dim_partition[dim].first + dim_partition[dim].second));
    }
  }
  // Check that partitions cover entire dimension size range (for each
  // partitioned dimension).
  for (int i = 0; i < ranges.size(); ++i) {
    int64_t expected_index = 0;
    for (auto& r : ranges[i]) {
      EXPECT_EQ(expected_index, r.first);
      expected_index = r.second;
    }
    EXPECT_EQ(expected_index, dim_sizes[i]);
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
