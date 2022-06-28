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
class MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc() {
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

#include <utility>

#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

#define EnumStringPair(val) \
  { val, #val }

std::pair<TensorFormat, const char*> test_data_formats[] = {
    EnumStringPair(FORMAT_NHWC),        EnumStringPair(FORMAT_NCHW),
    EnumStringPair(FORMAT_NCHW_VECT_C), EnumStringPair(FORMAT_NHWC_VECT_W),
    EnumStringPair(FORMAT_HWNC),        EnumStringPair(FORMAT_HWCN),
};

std::pair<FilterTensorFormat, const char*> test_filter_formats[] = {
    EnumStringPair(FORMAT_HWIO),
    EnumStringPair(FORMAT_OIHW),
    EnumStringPair(FORMAT_OIHW_VECT_I),
};

// This is an alternative way of specifying the tensor dimension indexes for
// each tensor format. For now it can be used as a cross-check of the existing
// functions, but later could replace them.

// Represents the dimension indexes of an activations tensor format.
struct TensorDimMap {
  int n() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/util/tensor_format_test.cc", "n");
 return dim_n; }
  int h() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/util/tensor_format_test.cc", "h");
 return dim_h; }
  int w() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_2(mht_2_v, 223, "", "./tensorflow/core/util/tensor_format_test.cc", "w");
 return dim_w; }
  int c() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/util/tensor_format_test.cc", "c");
 return dim_c; }
  int spatial(int spatial_index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_4(mht_4_v, 231, "", "./tensorflow/core/util/tensor_format_test.cc", "spatial");
 return spatial_dim[spatial_index]; }

  int dim_n, dim_h, dim_w, dim_c;
  int spatial_dim[3];
};

// Represents the dimension indexes of a filter tensor format.
struct FilterDimMap {
  int h() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_5(mht_5_v, 242, "", "./tensorflow/core/util/tensor_format_test.cc", "h");
 return dim_h; }
  int w() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_6(mht_6_v, 246, "", "./tensorflow/core/util/tensor_format_test.cc", "w");
 return dim_w; }
  int i() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_7(mht_7_v, 250, "", "./tensorflow/core/util/tensor_format_test.cc", "i");
 return dim_i; }
  int o() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_8(mht_8_v, 254, "", "./tensorflow/core/util/tensor_format_test.cc", "o");
 return dim_o; }
  int spatial(int spatial_index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_9(mht_9_v, 258, "", "./tensorflow/core/util/tensor_format_test.cc", "spatial");
 return spatial_dim[spatial_index]; }

  int dim_h, dim_w, dim_i, dim_o;
  int spatial_dim[3];
};

// clang-format off

// Predefined constants specifying the actual dimension indexes for each
// supported tensor and filter format.
struct DimMaps {
#define StaCoExTensorDm static constexpr TensorDimMap
  //                                'N', 'H', 'W', 'C'    0,  1,  2
  StaCoExTensorDm kTdmInvalid =   { -1,  -1,  -1,  -1, { -1, -1, -1 } };
  // These arrays are indexed by the number of spatial dimensions in the format.
  StaCoExTensorDm kTdmNHWC[4] = { kTdmInvalid,
                                  {  0,  -1,   1,   2, {  1, -1, -1 } },  // 1D
                                  {  0,   1,   2,   3, {  1,  2, -1 } },  // 2D
                                  {  0,   2,   3,   4, {  1,  2,  3 } }   // 3D
                                };
  StaCoExTensorDm kTdmNCHW[4] = { kTdmInvalid,
                                  {  0,  -1,   2,   1, {  2, -1, -1 } },
                                  {  0,   2,   3,   1, {  2,  3, -1 } },
                                  {  0,   3,   4,   1, {  2,  3,  4 } }
                                };
  StaCoExTensorDm kTdmHWNC[4] = { kTdmInvalid,
                                  {  1,  -1,   0,   2, {  0, -1, -1 } },
                                  {  2,   0,   1,   3, {  0,  1, -1 } },
                                  {  3,   1,   2,   4, {  0,  1,  2 } }
                                };
  StaCoExTensorDm kTdmHWCN[4] = { kTdmInvalid,
                                  {  2,  -1,   0,   1, {  0, -1, -1 } },
                                  {  3,   0,   1,   2, {  0,  1, -1 } },
                                  {  4,   1,   2,   3, {  0,  1,  2 } }
                                };
#undef StaCoExTensorDm
#define StaCoExFilterDm static constexpr FilterDimMap
  //                                'H', 'W', 'I', 'O'    0   1   2
  StaCoExFilterDm kFdmInvalid =   { -1,  -1,  -1,  -1, { -1, -1, -1 } };
  StaCoExFilterDm kFdmHWIO[4] = { kFdmInvalid,
                                  { -1,   0,   1,   2, {  0, -1, -1 } },
                                  {  0,   1,   2,   3, {  0,  1, -1 } },
                                  {  1,   2,   3,   4, {  0,  1,  2 } }
                                };
  StaCoExFilterDm kFdmOIHW[4] = { kFdmInvalid,
                                  { -1,   2,   1,   0, {  2, -1, -1 } },
                                  {  2,   3,   1,   0, {  2,  3, -1 } },
                                  {  3,   4,   1,   0, {  2,  3,  4 } }
                                };
#undef StaCoExFilterDm
};

inline constexpr const TensorDimMap&
GetTensorDimMap(const int num_spatial_dims, const TensorFormat format) {
  return
      (format == FORMAT_NHWC ||
       format == FORMAT_NHWC_VECT_W) ? DimMaps::kTdmNHWC[num_spatial_dims] :
      (format == FORMAT_NCHW ||
       format == FORMAT_NCHW_VECT_C) ? DimMaps::kTdmNCHW[num_spatial_dims] :
      (format == FORMAT_HWNC) ? DimMaps::kTdmHWNC[num_spatial_dims] :
      (format == FORMAT_HWCN) ? DimMaps::kTdmHWCN[num_spatial_dims]
                              : DimMaps::kTdmInvalid;
}

inline constexpr const FilterDimMap&
GetFilterDimMap(const int num_spatial_dims,
                const FilterTensorFormat format) {
  return
      (format == FORMAT_HWIO) ? DimMaps::kFdmHWIO[num_spatial_dims] :
      (format == FORMAT_OIHW ||
       format == FORMAT_OIHW_VECT_I) ? DimMaps::kFdmOIHW[num_spatial_dims]
                                     : DimMaps::kFdmInvalid;
}
// clang-format on

constexpr TensorDimMap DimMaps::kTdmInvalid;
constexpr TensorDimMap DimMaps::kTdmNHWC[4];
constexpr TensorDimMap DimMaps::kTdmNCHW[4];
constexpr TensorDimMap DimMaps::kTdmHWNC[4];
constexpr TensorDimMap DimMaps::kTdmHWCN[4];
constexpr FilterDimMap DimMaps::kFdmInvalid;
constexpr FilterDimMap DimMaps::kFdmHWIO[4];
constexpr FilterDimMap DimMaps::kFdmOIHW[4];

TEST(TensorFormatTest, FormatEnumsAndStrings) {
  const string prefix = "FORMAT_";
  for (auto& test_data_format : test_data_formats) {
    const char* stringified_format_enum = test_data_format.second;
    LOG(INFO) << stringified_format_enum << " = " << test_data_format.first;
    string expected_format_str = &stringified_format_enum[prefix.size()];
    TensorFormat format;
    EXPECT_TRUE(FormatFromString(expected_format_str, &format));
    string format_str = ToString(format);
    EXPECT_EQ(expected_format_str, format_str);
    EXPECT_EQ(test_data_format.first, format);
  }
  for (auto& test_filter_format : test_filter_formats) {
    const char* stringified_format_enum = test_filter_format.second;
    LOG(INFO) << stringified_format_enum << " = " << test_filter_format.first;
    string expected_format_str = &stringified_format_enum[prefix.size()];
    FilterTensorFormat format;
    EXPECT_TRUE(FilterFormatFromString(expected_format_str, &format));
    string format_str = ToString(format);
    EXPECT_EQ(expected_format_str, format_str);
    EXPECT_EQ(test_filter_format.first, format);
  }
}

template <int num_spatial_dims>
void RunDimensionIndexesTest() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_format_testDTcc mht_10(mht_10_v, 370, "", "./tensorflow/core/util/tensor_format_test.cc", "RunDimensionIndexesTest");

  for (auto& test_data_format : test_data_formats) {
    TensorFormat format = test_data_format.first;
    auto& tdm = GetTensorDimMap(num_spatial_dims, format);
    int num_dims = GetTensorDimsFromSpatialDims(num_spatial_dims, format);
    LOG(INFO) << ToString(format) << ", num_spatial_dims=" << num_spatial_dims
              << ", num_dims=" << num_dims;
    EXPECT_EQ(GetTensorBatchDimIndex(num_dims, format), tdm.n());
    EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, 'N'), tdm.n());
    EXPECT_EQ(GetTensorFeatureDimIndex(num_dims, format), tdm.c());
    EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, 'C'), tdm.c());
    for (int i = 0; i < num_spatial_dims; ++i) {
      EXPECT_EQ(GetTensorSpatialDimIndex(num_dims, format, i), tdm.spatial(i));
      EXPECT_EQ(GetTensorDimIndex<num_spatial_dims>(format, '0' + i),
                tdm.spatial(i));
    }
  }
  for (auto& test_filter_format : test_filter_formats) {
    FilterTensorFormat format = test_filter_format.first;
    auto& fdm = GetFilterDimMap(num_spatial_dims, format);
    int num_dims = GetFilterTensorDimsFromSpatialDims(num_spatial_dims, format);
    LOG(INFO) << ToString(format) << ", num_spatial_dims=" << num_spatial_dims
              << ", num_dims=" << num_dims;
    EXPECT_EQ(GetFilterTensorOutputChannelsDimIndex(num_dims, format), fdm.o());
    EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, 'O'), fdm.o());
    EXPECT_EQ(GetFilterTensorInputChannelsDimIndex(num_dims, format), fdm.i());
    EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, 'I'), fdm.i());
    for (int i = 0; i < num_spatial_dims; ++i) {
      EXPECT_EQ(GetFilterTensorSpatialDimIndex(num_dims, format, i),
                fdm.spatial(i));
      EXPECT_EQ(GetFilterDimIndex<num_spatial_dims>(format, '0' + i),
                fdm.spatial(i));
    }
  }
}

TEST(TensorFormatTest, DimensionIndexes) {
  RunDimensionIndexesTest<1>();
  RunDimensionIndexesTest<2>();
  RunDimensionIndexesTest<3>();
}

}  // namespace tensorflow
