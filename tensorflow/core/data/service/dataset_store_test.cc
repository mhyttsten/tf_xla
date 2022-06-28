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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSdataset_store_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdataset_store_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSdataset_store_testDTcc() {
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
#include "tensorflow/core/data/service/dataset_store.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

namespace {
const char kFileSystem[] = "file_system";
const char kMemory[] = "memory";

std::string NewDatasetsDir() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdataset_store_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/data/service/dataset_store_test.cc", "NewDatasetsDir");

  std::string dir = io::JoinPath(testing::TmpDir(), "datasets");
  if (Env::Default()->FileExists(dir).ok()) {
    int64_t undeleted_files;
    int64_t undeleted_dirs;
    CHECK(Env::Default()
              ->DeleteRecursively(dir, &undeleted_files, &undeleted_dirs)
              .ok());
  }
  CHECK(Env::Default()->RecursivelyCreateDir(dir).ok());
  return dir;
}

std::unique_ptr<DatasetStore> MakeStore(const std::string& type) {
  if (type == kFileSystem) {
    return absl::make_unique<FileSystemDatasetStore>(NewDatasetsDir());
  } else if (type == kMemory) {
    return absl::make_unique<MemoryDatasetStore>();
  } else {
    CHECK(false) << "unexpected type: " << type;
  }
}

DatasetDef DatasetDefWithVersion(int32_t version) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSdataset_store_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/data/service/dataset_store_test.cc", "DatasetDefWithVersion");

  DatasetDef def;
  def.mutable_graph()->set_version(version);
  return def;
}

}  // namespace

class DatasetStoreTest : public ::testing::Test,
                         public ::testing::WithParamInterface<std::string> {};

TEST_P(DatasetStoreTest, StoreAndGet) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  std::string key = "key";
  DatasetDef dataset_def = DatasetDefWithVersion(1);
  TF_ASSERT_OK(store->Put(key, dataset_def));
  std::shared_ptr<const DatasetDef> result;
  TF_ASSERT_OK(store->Get(key, result));
  EXPECT_EQ(result->graph().version(), dataset_def.graph().version());
}

TEST_P(DatasetStoreTest, StoreAndGetMultiple) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  int64_t num_datasets = 10;
  std::vector<std::string> keys;
  for (int i = 0; i < num_datasets; ++i) {
    std::string key = absl::StrCat("key", i);
    DatasetDef dataset_def = DatasetDefWithVersion(i);
    TF_ASSERT_OK(store->Put(key, dataset_def));
    keys.push_back(key);
  }
  for (int i = 0; i < num_datasets; ++i) {
    std::shared_ptr<const DatasetDef> result;
    TF_ASSERT_OK(store->Get(keys[i], result));
    EXPECT_EQ(result->graph().version(), i);
  }
}

TEST_P(DatasetStoreTest, StoreAlreadyExists) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  int32_t version = 1;
  DatasetDef dataset_def = DatasetDefWithVersion(version);
  std::string key = "key";
  TF_ASSERT_OK(store->Put(key, dataset_def));
  Status s = store->Put(key, dataset_def);
  EXPECT_EQ(s.code(), error::ALREADY_EXISTS);
  std::shared_ptr<const DatasetDef> result;
  TF_ASSERT_OK(store->Get(key, result));
  EXPECT_EQ(result->graph().version(), version);
}

TEST_P(DatasetStoreTest, GetMissing) {
  std::unique_ptr<DatasetStore> store = MakeStore(GetParam());
  std::shared_ptr<const DatasetDef> result;
  Status s = store->Get("missing", result);
  EXPECT_EQ(s.code(), error::NOT_FOUND);
}

INSTANTIATE_TEST_SUITE_P(DatasetStoreTests, DatasetStoreTest,
                         ::testing::Values(kFileSystem, kMemory));
}  // namespace data
}  // namespace tensorflow
