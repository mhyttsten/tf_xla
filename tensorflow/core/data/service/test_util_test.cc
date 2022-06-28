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
class MHTracer_DTPStensorflowPScorePSdataPSservicePStest_util_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePStest_util_testDTcc() {
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
#include "tensorflow/core/data/service/test_util.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::testing::IsEmpty;
using ::testing::SizeIs;

tstring LocalTempFilename() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStest_util_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/data/service/test_util_test.cc", "LocalTempFilename");

  std::string path;
  CHECK(Env::Default()->LocalTempFilename(&path));
  return tstring(path);
}

StatusOr<std::vector<std::vector<Tensor>>> GetIteratorOutput(
    standalone::Iterator& iterator) {
  bool end_of_input = false;
  std::vector<std::vector<Tensor>> result;
  while (!end_of_input) {
    std::vector<tensorflow::Tensor> outputs;
    TF_RETURN_IF_ERROR(iterator.GetNext(&outputs, &end_of_input));
    if (!end_of_input) {
      result.push_back(outputs);
    }
  }
  return result;
}

TEST(TestUtilTest, RangeSquareDataset) {
  const auto dataset_def = RangeSquareDataset(/*range=*/10);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Tensor>> result,
                          GetIteratorOutput(*iterator));

  ASSERT_EQ(result.size(), 10);
  for (int i = 0; i < result.size(); ++i) {
    test::ExpectEqual(result[i][0], Tensor(int64_t{i * i}));
  }
}

TEST(TestUtilTest, EmptyDataset) {
  const auto dataset_def = RangeSquareDataset(/*range=*/0);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput(*iterator), IsOkAndHolds(IsEmpty()));
}

TEST(TestUtilTest, InterleaveTextline) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(const DatasetDef dataset_def,
                          InterleaveTextlineDataset(filenames, {"0", "1"}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Tensor>> result,
                          GetIteratorOutput(*iterator));
  ASSERT_THAT(result, SizeIs(2));
  test::ExpectEqual(result[0][0], Tensor("0"));
  test::ExpectEqual(result[1][0], Tensor("1"));
}

TEST(TestUtilTest, InterleaveTextlineWithNewLines) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(
      const DatasetDef dataset_def,
      InterleaveTextlineDataset(filenames, {"0\n2\n4\n6\n8", "1\n3\n5\n7\n9"}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Tensor>> result,
                          GetIteratorOutput(*iterator));
  ASSERT_THAT(result, SizeIs(10));
  for (int64 i = 0; i < 10; ++i) {
    test::ExpectEqual(result[i][0], Tensor(absl::StrCat(i)));
  }
}

TEST(TestUtilTest, InterleaveTextlineEmptyFiles) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(const DatasetDef dataset_def,
                          InterleaveTextlineDataset(filenames, {"", ""}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput(*iterator), IsOkAndHolds(IsEmpty()));
}

}  // namespace
}  // namespace testing
}  // namespace data
}  // namespace tensorflow
