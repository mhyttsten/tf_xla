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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc() {
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

#include "tensorflow/core/platform/cloud/oauth_client.h"

#include <fstream>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/base64.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

string TestData() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/platform/cloud/oauth_client_test.cc", "TestData");

  return io::JoinPath("tensorflow", "core", "platform", "cloud", "testdata");
}

constexpr char kTokenJson[] = R"(
    {
      "access_token":"WITH_FAKE_ACCESS_TOKEN_TEST_SHOULD_BE_HAPPY",
      "expires_in":3920,
      "token_type":"Bearer"
    })";

class FakeEnv : public EnvWrapper {
 public:
  FakeEnv() : EnvWrapper(Env::Default()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/platform/cloud/oauth_client_test.cc", "FakeEnv");
}

  uint64 NowSeconds() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_client_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/platform/cloud/oauth_client_test.cc", "NowSeconds");
 return now; }
  uint64 now = 10000;
};

}  // namespace

TEST(OAuthClientTest, ParseOAuthResponse) {
  const uint64 request_timestamp = 100;
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(OAuthClient().ParseOAuthResponse(kTokenJson, request_timestamp,
                                                &token, &expiration_timestamp));
  EXPECT_EQ("WITH_FAKE_ACCESS_TOKEN_TEST_SHOULD_BE_HAPPY", token);
  EXPECT_EQ(4020, expiration_timestamp);
}

TEST(OAuthClientTest, GetTokenFromRefreshTokenJson) {
  const string credentials_json = R"(
      {
        "client_id": "test_client_id",
        "client_secret": "@@@test_client_secret@@@",
        "refresh_token": "test_refresh_token",
        "type": "authorized_user"
      })";
  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(credentials_json, json));

  std::vector<HttpRequest*> requests({new FakeHttpRequest(
      "Uri: https://www.googleapis.com/oauth2/v3/token\n"
      "Post body: client_id=test_client_id&"
      "client_secret=@@@test_client_secret@@@&"
      "refresh_token=test_refresh_token&grant_type=refresh_token\n",
      kTokenJson)});
  FakeEnv env;
  OAuthClient client(std::unique_ptr<HttpRequest::Factory>(
                         new FakeHttpRequestFactory(&requests)),
                     &env);
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(client.GetTokenFromRefreshTokenJson(
      json, "https://www.googleapis.com/oauth2/v3/token", &token,
      &expiration_timestamp));
  EXPECT_EQ("WITH_FAKE_ACCESS_TOKEN_TEST_SHOULD_BE_HAPPY", token);
  EXPECT_EQ(13920, expiration_timestamp);
}

TEST(OAuthClientTest, GetTokenFromServiceAccountJson) {
  std::ifstream credentials(GetDataDependencyFilepath(
      io::JoinPath(TestData(), "service_account_credentials.json")));
  ASSERT_TRUE(credentials.is_open());
  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(credentials, json));

  string post_body;
  std::vector<HttpRequest*> requests(
      {new FakeHttpRequest("Uri: https://www.googleapis.com/oauth2/v3/token\n",
                           kTokenJson, &post_body)});
  FakeEnv env;
  OAuthClient client(std::unique_ptr<HttpRequest::Factory>(
                         new FakeHttpRequestFactory(&requests)),
                     &env);
  string token;
  uint64 expiration_timestamp;
  TF_EXPECT_OK(client.GetTokenFromServiceAccountJson(
      json, "https://www.googleapis.com/oauth2/v3/token",
      "https://test-token-scope.com", &token, &expiration_timestamp));
  EXPECT_EQ("WITH_FAKE_ACCESS_TOKEN_TEST_SHOULD_BE_HAPPY", token);
  EXPECT_EQ(13920, expiration_timestamp);

  // Now look at the JWT claim that was sent to the OAuth server.
  StringPiece grant_type, assertion;
  ASSERT_TRUE(strings::Scanner(post_body)
                  .OneLiteral("grant_type=")
                  .RestartCapture()
                  .ScanEscapedUntil('&')
                  .StopCapture()
                  .OneLiteral("&assertion=")
                  .GetResult(&assertion, &grant_type));
  EXPECT_EQ("urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer",
            grant_type);

  int last_dot = assertion.rfind('.');
  string header_dot_claim(assertion.substr(0, last_dot));
  string signature_encoded(assertion.substr(last_dot + 1));

  // Check that 'signature' signs 'header_dot_claim'.

  // Read the serialized public key.
  std::ifstream public_key_stream(GetDataDependencyFilepath(
      io::JoinPath(TestData(), "service_account_public_key.txt")));
  string public_key_serialized(
      (std::istreambuf_iterator<char>(public_key_stream)),
      (std::istreambuf_iterator<char>()));

  // Deserialize the public key.
  auto bio = BIO_new(BIO_s_mem());
  RSA* public_key = nullptr;
  EXPECT_EQ(public_key_serialized.size(),
            BIO_puts(bio, public_key_serialized.c_str()));
  public_key = PEM_read_bio_RSA_PUBKEY(bio, nullptr, nullptr, nullptr);
  EXPECT_TRUE(public_key) << "Could not load the public key from testdata.";

  // Deserialize the signature.
  string signature;
  TF_EXPECT_OK(Base64Decode(signature_encoded, &signature));

  // Actually cryptographically verify the signature.
  const auto md = EVP_sha256();
  auto md_ctx = EVP_MD_CTX_create();
  auto key = EVP_PKEY_new();
  EVP_PKEY_set1_RSA(key, public_key);
  ASSERT_EQ(1, EVP_DigestVerifyInit(md_ctx, nullptr, md, nullptr, key));
  ASSERT_EQ(1, EVP_DigestVerifyUpdate(md_ctx, header_dot_claim.c_str(),
                                      header_dot_claim.size()));
  ASSERT_EQ(1,
            EVP_DigestVerifyFinal(
                md_ctx,
                const_cast<unsigned char*>(
                    reinterpret_cast<const unsigned char*>(signature.data())),
                signature.size()));

  // Free all the crypto-related resources.
  EVP_PKEY_free(key);
  EVP_MD_CTX_destroy(md_ctx);
  RSA_free(public_key);
  BIO_free_all(bio);

  // Now check the content of the header and the claim.
  int dot = header_dot_claim.find_last_of('.');
  string header_encoded = header_dot_claim.substr(0, dot);
  string claim_encoded = header_dot_claim.substr(dot + 1);

  string header, claim;
  TF_EXPECT_OK(Base64Decode(header_encoded, &header));
  TF_EXPECT_OK(Base64Decode(claim_encoded, &claim));

  Json::Value header_json, claim_json;
  EXPECT_TRUE(reader.parse(header, header_json));
  EXPECT_EQ("RS256", header_json.get("alg", Json::Value::null).asString());
  EXPECT_EQ("JWT", header_json.get("typ", Json::Value::null).asString());
  EXPECT_EQ("fake_key_id",
            header_json.get("kid", Json::Value::null).asString());

  EXPECT_TRUE(reader.parse(claim, claim_json));
  EXPECT_EQ("fake-test-project.iam.gserviceaccount.com",
            claim_json.get("iss", Json::Value::null).asString());
  EXPECT_EQ("https://test-token-scope.com",
            claim_json.get("scope", Json::Value::null).asString());
  EXPECT_EQ("https://www.googleapis.com/oauth2/v3/token",
            claim_json.get("aud", Json::Value::null).asString());
  EXPECT_EQ(10000, claim_json.get("iat", Json::Value::null).asInt64());
  EXPECT_EQ(13600, claim_json.get("exp", Json::Value::null).asInt64());
}
}  // namespace tensorflow
