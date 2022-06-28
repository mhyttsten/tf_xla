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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc() {
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

#include "tensorflow/core/platform/cloud/google_auth_provider.h"

#include <stdlib.h>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

string TestData() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "TestData");

  return io::JoinPath("tensorflow", "core", "platform", "cloud", "testdata");
}

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "FakeEnv");
}

  uint64 NowSeconds() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "NowSeconds");
 return now; }
  uint64 now = 10000;
};

class FakeOAuthClient : public OAuthClient {
 public:
  Status GetTokenFromServiceAccountJson(
      Json::Value json, StringPiece oauth_server_uri, StringPiece scope,
      string* token, uint64* expiration_timestamp_sec) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_3(mht_3_v, 224, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "GetTokenFromServiceAccountJson");

    provided_credentials_json = json;
    *token = return_token;
    *expiration_timestamp_sec = return_expiration_timestamp;
    return Status::OK();
  }

  /// Retrieves a bearer token using a refresh token.
  Status GetTokenFromRefreshTokenJson(
      Json::Value json, StringPiece oauth_server_uri, string* token,
      uint64* expiration_timestamp_sec) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_4(mht_4_v, 237, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "GetTokenFromRefreshTokenJson");

    provided_credentials_json = json;
    *token = return_token;
    *expiration_timestamp_sec = return_expiration_timestamp;
    return Status::OK();
  }

  string return_token;
  uint64 return_expiration_timestamp;
  Json::Value provided_credentials_json;
};

}  // namespace

class GoogleAuthProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "SetUp");
 ClearEnvVars(); }

  void TearDown() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "TearDown");
 ClearEnvVars(); }

  void ClearEnvVars() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSgoogle_auth_provider_testDTcc mht_7(mht_7_v, 266, "", "./tensorflow/core/platform/cloud/google_auth_provider_test.cc", "ClearEnvVars");

    unsetenv("CLOUDSDK_CONFIG");
    unsetenv("GOOGLE_APPLICATION_CREDENTIALS");
    unsetenv("GOOGLE_AUTH_TOKEN_FOR_TESTING");
    unsetenv("NO_GCE_CHECK");
  }
};

TEST_F(GoogleAuthProviderTest, EnvironmentVariable_Caching) {
  setenv("GOOGLE_APPLICATION_CREDENTIALS",
         GetDataDependencyFilepath(
             io::JoinPath(TestData(), "service_account_credentials.json"))
             .c_str(),
         1);
  setenv("CLOUDSDK_CONFIG", GetDataDependencyFilepath(TestData()).c_str(),
         1);  // Will not be used.

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests;

  FakeEnv env;

  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);
  oauth_client->return_token = "fake-token";
  oauth_client->return_expiration_timestamp = env.NowSeconds() + 3600;

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);
  EXPECT_EQ("fake_key_id",
            oauth_client->provided_credentials_json.get("private_key_id", "")
                .asString());

  // Check that the token is re-used if not expired.
  oauth_client->return_token = "new-fake-token";
  env.now += 3000;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);

  // Check that the token is re-generated when almost expired.
  env.now += 598;  // 2 seconds before expiration
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("new-fake-token", token);
}

TEST_F(GoogleAuthProviderTest, GCloudRefreshToken) {
  setenv("CLOUDSDK_CONFIG", GetDataDependencyFilepath(TestData()).c_str(), 1);

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests;

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));

  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);
  oauth_client->return_token = "fake-token";
  oauth_client->return_expiration_timestamp = env.NowSeconds() + 3600;

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-token", token);
  EXPECT_EQ("fake-refresh-token",
            oauth_client->provided_credentials_json.get("refresh_token", "")
                .asString());
}

TEST_F(GoogleAuthProviderTest, RunningOnGCE) {
  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           R"(
          {
            "access_token":"fake-gce-token",
            "expires_in": 3920,
            "token_type":"Bearer"
          })"),
       // The first token refresh request fails and will be retried.
       new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           "", errors::Unavailable("503"), 503),
       new FakeHttpRequest(
           "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
           "/default/token\n"
           "Header Metadata-Flavor: Google\n",
           R"(
              {
                "access_token":"new-fake-gce-token",
                "expires_in": 3920,
                "token_type":"Bearer"
              })")});

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-gce-token", token);

  // Check that the token is re-used if not expired.
  env.now += 3700;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("fake-gce-token", token);

  // Check that the token is re-generated when almost expired.
  env.now += 598;  // 2 seconds before expiration
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("new-fake-gce-token", token);
}

TEST_F(GoogleAuthProviderTest, OverrideForTesting) {
  setenv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "tokenForTesting", 1);

  auto oauth_client = new FakeOAuthClient;
  std::vector<HttpRequest*> empty_requests;
  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&empty_requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("tokenForTesting", token);
}

TEST_F(GoogleAuthProviderTest, NothingAvailable) {
  auto oauth_client = new FakeOAuthClient;

  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: http://metadata/computeMetadata/v1/instance/service-accounts"
      "/default/token\n"
      "Header Metadata-Flavor: Google\n",
      "", errors::NotFound("404"), 404)});

  FakeEnv env;
  std::shared_ptr<HttpRequest::Factory> fakeHttpRequestFactory =
      std::make_shared<FakeHttpRequestFactory>(&requests);
  auto metadataClient = std::make_shared<ComputeEngineMetadataClient>(
      fakeHttpRequestFactory, RetryConfig(0 /* init_delay_time_us */));
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              metadataClient, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);
}

TEST_F(GoogleAuthProviderTest, NoGceCheckEnvironmentVariable) {
  setenv("NO_GCE_CHECK", "True", 1);
  auto oauth_client = new FakeOAuthClient;

  FakeEnv env;
  // If the env var above isn't respected, attempting to fetch a token
  // from GCE will segfault (as the metadata client is null).
  GoogleAuthProvider provider(std::unique_ptr<OAuthClient>(oauth_client),
                              nullptr, &env);

  string token;
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);

  // We confirm that our env var is case insensitive.
  setenv("NO_GCE_CHECK", "true", 1);
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("", token);

  // We also want to confirm that our empty token has a short expiration set: we
  // now set a testing token, and confirm that it's returned instead of our
  // empty token.
  setenv("GOOGLE_AUTH_TOKEN_FOR_TESTING", "newToken", 1);
  TF_EXPECT_OK(provider.GetToken(&token));
  EXPECT_EQ("newToken", token);
}

}  // namespace tensorflow
