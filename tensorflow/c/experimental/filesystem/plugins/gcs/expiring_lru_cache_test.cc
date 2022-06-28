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
class MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cache_testDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cache_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cache_testDTcc() {
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

#include "tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h"

#include <memory>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/now_seconds_env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(ExpiringLRUCacheTest, MaxAge) {
  const string key = "a";
  std::unique_ptr<NowSecondsEnv> env(new NowSecondsEnv);
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(
      1, 0, [&env]() { return env->NowSeconds(); });
  env->SetNowSeconds(1);
  // Verify that replacement of an existing element works, and updates the
  // timestamp of the entry.
  cache.Insert(key, 41);
  env->SetNowSeconds(2);
  cache.Insert(key, 42);
  // 1 second after the most recent insertion, the entry is still valid.
  env->SetNowSeconds(3);
  int value = 0;
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 42);
  // 2 seconds after the most recent insertion, the entry is no longer valid.
  env->SetNowSeconds(4);
  EXPECT_FALSE(cache.Lookup(key, &value));
  // Re-insert the entry.
  cache.Insert(key, 43);
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 43);
  // The entry is valid 1 second after the insertion...
  env->SetNowSeconds(5);
  value = 0;
  EXPECT_TRUE(cache.Lookup(key, &value));
  EXPECT_EQ(value, 43);
  // ...but is no longer valid 2 seconds after the insertion.
  env->SetNowSeconds(6);
  EXPECT_FALSE(cache.Lookup(key, &value));
}

TEST(ExpiringLRUCacheTest, MaxEntries) {
  // max_age of 0 means nothing will be cached.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache1(0, 4);
  cache1.Insert("a", 1);
  int value = 0;
  EXPECT_FALSE(cache1.Lookup("a", &value));
  // Now set max_age = 1 and verify the LRU eviction logic.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache2(1, 4);
  cache2.Insert("a", 1);
  cache2.Insert("b", 2);
  cache2.Insert("c", 3);
  cache2.Insert("d", 4);
  EXPECT_TRUE(cache2.Lookup("a", &value));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(cache2.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache2.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache2.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  // Insertion of "e" causes "a" to be evicted, but the other entries are still
  // there.
  cache2.Insert("e", 5);
  EXPECT_FALSE(cache2.Lookup("a", &value));
  EXPECT_TRUE(cache2.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache2.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache2.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  EXPECT_TRUE(cache2.Lookup("e", &value));
  EXPECT_EQ(value, 5);
}

TEST(ExpiringLRUCacheTest, LookupOrCompute) {
  // max_age of 0 means we should always compute.
  uint64 num_compute_calls = 0;
  tf_gcs_filesystem::ExpiringLRUCache<int>::ComputeFunc compute_func =
      [&num_compute_calls](const string& key, int* value, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cache_testDTcc mht_0(mht_0_v, 270, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache_test.cc", "lambda");

        *value = num_compute_calls;
        num_compute_calls++;
        return TF_SetStatus(status, TF_OK, "");
      };
  tf_gcs_filesystem::ExpiringLRUCache<int> cache1(0, 4);

  int value = -1;
  TF_Status status;
  cache1.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);
  // re-read the same value, expect another lookup
  cache1.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 2);

  // Define a new cache with max_age > 0 and verify correct behavior.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache2(2, 4);
  num_compute_calls = 0;
  value = -1;

  // Read our first value
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);
  // Re-read, exepct no additional function compute_func calls.
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 0);
  EXPECT_EQ(num_compute_calls, 1);

  // Read a sequence of additional values, eventually evicting "a".
  cache2.LookupOrCompute("b", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 2);
  cache2.LookupOrCompute("c", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 2);
  EXPECT_EQ(num_compute_calls, 3);
  cache2.LookupOrCompute("d", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 3);
  EXPECT_EQ(num_compute_calls, 4);
  cache2.LookupOrCompute("e", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 4);
  EXPECT_EQ(num_compute_calls, 5);
  // Verify the other values remain in the cache.
  cache2.LookupOrCompute("b", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(num_compute_calls, 5);
  cache2.LookupOrCompute("c", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 2);
  EXPECT_EQ(num_compute_calls, 5);
  cache2.LookupOrCompute("d", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 3);
  EXPECT_EQ(num_compute_calls, 5);

  // Re-read "a", ensure it is re-computed.
  cache2.LookupOrCompute("a", &value, compute_func, &status);
  TF_EXPECT_OK(status.status);
  EXPECT_EQ(value, 5);
  EXPECT_EQ(num_compute_calls, 6);
}

TEST(ExpiringLRUCacheTest, Clear) {
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(1, 4);
  cache.Insert("a", 1);
  cache.Insert("b", 2);
  cache.Insert("c", 3);
  cache.Insert("d", 4);
  int value = 0;
  EXPECT_TRUE(cache.Lookup("a", &value));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(cache.Lookup("b", &value));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(cache.Lookup("c", &value));
  EXPECT_EQ(value, 3);
  EXPECT_TRUE(cache.Lookup("d", &value));
  EXPECT_EQ(value, 4);
  cache.Clear();
  EXPECT_FALSE(cache.Lookup("a", &value));
  EXPECT_FALSE(cache.Lookup("b", &value));
  EXPECT_FALSE(cache.Lookup("c", &value));
  EXPECT_FALSE(cache.Lookup("d", &value));
}

TEST(ExpiringLRUCacheTest, Delete) {
  // Insert an entry.
  tf_gcs_filesystem::ExpiringLRUCache<int> cache(1, 4);
  cache.Insert("a", 1);
  int value = 0;
  EXPECT_TRUE(cache.Lookup("a", &value));
  EXPECT_EQ(value, 1);

  // Delete the entry.
  EXPECT_TRUE(cache.Delete("a"));
  EXPECT_FALSE(cache.Lookup("a", &value));

  // Try deleting the entry again.
  EXPECT_FALSE(cache.Delete("a"));
  EXPECT_FALSE(cache.Lookup("a", &value));
}

}  // namespace
}  // namespace tensorflow
