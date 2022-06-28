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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc() {
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
#ifndef _WIN32
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <sys/types.h>
#endif
#include <fstream>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include "tensorflow/core/platform/base64.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

namespace {

// The requested lifetime of an auth bearer token.
constexpr int kRequestedTokenLifetimeSec = 3600;

// The crypto algorithm to be used with OAuth.
constexpr char kCryptoAlgorithm[] = "RS256";

// The token type for the OAuth request.
constexpr char kJwtType[] = "JWT";

// The grant type for the OAuth request. Already URL-encoded for convenience.
constexpr char kGrantType[] =
    "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer";

Status ReadJsonValue(const Json::Value& json, const string& name,
                     Json::Value* value) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "ReadJsonValue");

  if (!value) {
    return errors::FailedPrecondition("'value' cannot be nullptr.");
  }
  *value = json.get(name, Json::Value::null);
  if (*value == Json::Value::null) {
    return errors::FailedPrecondition(
        strings::StrCat("Couldn't read a JSON value '", name, "'."));
  }
  return Status::OK();
}

Status ReadJsonString(const Json::Value& json, const string& name,
                      string* value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "ReadJsonString");

  Json::Value json_value;
  TF_RETURN_IF_ERROR(ReadJsonValue(json, name, &json_value));
  if (!json_value.isString()) {
    return errors::FailedPrecondition(
        strings::StrCat("JSON value '", name, "' is not string."));
  }
  *value = json_value.asString();
  return Status::OK();
}

Status ReadJsonInt(const Json::Value& json, const string& name,
                   int64_t* value) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_2(mht_2_v, 256, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "ReadJsonInt");

  Json::Value json_value;
  TF_RETURN_IF_ERROR(ReadJsonValue(json, name, &json_value));
  if (!json_value.isIntegral()) {
    return errors::FailedPrecondition(
        strings::StrCat("JSON value '", name, "' is not integer."));
  }
  *value = json_value.asInt64();
  return Status::OK();
}

Status CreateSignature(RSA* private_key, StringPiece to_sign,
                       string* signature) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "CreateSignature");

  if (!private_key || !signature) {
    return errors::FailedPrecondition(
        "'private_key' and 'signature' cannot be nullptr.");
  }

  const auto md = EVP_sha256();
  if (!md) {
    return errors::Internal("Could not get a sha256 encryptor.");
  }

  // EVP_MD_CTX_destroy is renamed to EVP_MD_CTX_free in OpenSSL 1.1.0 but
  // the old name is still retained as a compatibility macro.
  // Keep this around until support is dropped for OpenSSL 1.0
  // https://www.openssl.org/news/cl110.txt
  std::unique_ptr<EVP_MD_CTX, std::function<void(EVP_MD_CTX*)>> md_ctx(
      EVP_MD_CTX_create(), [](EVP_MD_CTX* ptr) { EVP_MD_CTX_destroy(ptr); });
  if (!md_ctx) {
    return errors::Internal("Could not create MD_CTX.");
  }

  std::unique_ptr<EVP_PKEY, std::function<void(EVP_PKEY*)>> key(
      EVP_PKEY_new(), [](EVP_PKEY* ptr) { EVP_PKEY_free(ptr); });
  EVP_PKEY_set1_RSA(key.get(), private_key);

  if (EVP_DigestSignInit(md_ctx.get(), nullptr, md, nullptr, key.get()) != 1) {
    return errors::Internal("DigestInit failed.");
  }
  if (EVP_DigestSignUpdate(md_ctx.get(), to_sign.data(), to_sign.size()) != 1) {
    return errors::Internal("DigestUpdate failed.");
  }
  size_t sig_len = 0;
  if (EVP_DigestSignFinal(md_ctx.get(), nullptr, &sig_len) != 1) {
    return errors::Internal("DigestFinal (get signature length) failed.");
  }
  std::unique_ptr<unsigned char[]> sig(new unsigned char[sig_len]);
  if (EVP_DigestSignFinal(md_ctx.get(), sig.get(), &sig_len) != 1) {
    return errors::Internal("DigestFinal (signature compute) failed.");
  }
  return Base64Encode(StringPiece(reinterpret_cast<char*>(sig.get()), sig_len),
                      signature);
}

/// Encodes a claim for a JSON web token (JWT) to make an OAuth request.
Status EncodeJwtClaim(StringPiece client_email, StringPiece scope,
                      StringPiece audience, uint64 request_timestamp_sec,
                      string* encoded) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_4(mht_4_v, 320, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "EncodeJwtClaim");

  // Step 1: create the JSON with the claim.
  Json::Value root;
  root["iss"] = Json::Value(client_email.begin(), client_email.end());
  root["scope"] = Json::Value(scope.begin(), scope.end());
  root["aud"] = Json::Value(audience.begin(), audience.end());

  const auto expiration_timestamp_sec =
      request_timestamp_sec + kRequestedTokenLifetimeSec;

  root["iat"] = Json::Value::UInt64(request_timestamp_sec);
  root["exp"] = Json::Value::UInt64(expiration_timestamp_sec);

  // Step 2: represent the JSON as a string.
  string claim = root.toStyledString();

  // Step 3: encode the string as base64.
  return Base64Encode(claim, encoded);
}

/// Encodes a header for a JSON web token (JWT) to make an OAuth request.
Status EncodeJwtHeader(StringPiece key_id, string* encoded) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_5(mht_5_v, 344, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "EncodeJwtHeader");

  // Step 1: create the JSON with the header.
  Json::Value root;
  root["alg"] = kCryptoAlgorithm;
  root["typ"] = kJwtType;
  root["kid"] = Json::Value(key_id.begin(), key_id.end());

  // Step 2: represent the JSON as a string.
  const string header = root.toStyledString();

  // Step 3: encode the string as base64.
  return Base64Encode(header, encoded);
}

}  // namespace

OAuthClient::OAuthClient()
    : OAuthClient(
          std::unique_ptr<HttpRequest::Factory>(new CurlHttpRequest::Factory()),
          Env::Default()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_6(mht_6_v, 366, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "OAuthClient::OAuthClient");
}

OAuthClient::OAuthClient(
    std::unique_ptr<HttpRequest::Factory> http_request_factory, Env* env)
    : http_request_factory_(std::move(http_request_factory)), env_(env) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "OAuthClient::OAuthClient");
}

Status OAuthClient::GetTokenFromServiceAccountJson(
    Json::Value json, StringPiece oauth_server_uri, StringPiece scope,
    string* token, uint64* expiration_timestamp_sec) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_8(mht_8_v, 380, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "OAuthClient::GetTokenFromServiceAccountJson");

  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  string private_key_serialized, private_key_id, client_id, client_email;
  TF_RETURN_IF_ERROR(
      ReadJsonString(json, "private_key", &private_key_serialized));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "private_key_id", &private_key_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_id", &client_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_email", &client_email));

  std::unique_ptr<BIO, std::function<void(BIO*)>> bio(
      BIO_new(BIO_s_mem()), [](BIO* ptr) { BIO_free_all(ptr); });
  if (BIO_puts(bio.get(), private_key_serialized.c_str()) !=
      static_cast<int>(private_key_serialized.size())) {
    return errors::Internal("Could not load the private key.");
  }
  std::unique_ptr<RSA, std::function<void(RSA*)>> private_key(
      PEM_read_bio_RSAPrivateKey(bio.get(), nullptr, nullptr, nullptr),
      [](RSA* ptr) { RSA_free(ptr); });
  if (!private_key) {
    return errors::Internal("Could not deserialize the private key.");
  }

  const uint64 request_timestamp_sec = env_->NowSeconds();

  string encoded_claim, encoded_header;
  TF_RETURN_IF_ERROR(EncodeJwtHeader(private_key_id, &encoded_header));
  TF_RETURN_IF_ERROR(EncodeJwtClaim(client_email, scope, oauth_server_uri,
                                    request_timestamp_sec, &encoded_claim));
  const string to_sign = encoded_header + "." + encoded_claim;
  string signature;
  TF_RETURN_IF_ERROR(CreateSignature(private_key.get(), to_sign, &signature));
  const string jwt = to_sign + "." + signature;
  const string request_body =
      strings::StrCat("grant_type=", kGrantType, "&assertion=", jwt);

  // Send the request to the Google OAuth 2.0 server to get the token.
  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  std::vector<char> response_buffer;
  request->SetUri(string(oauth_server_uri));
  request->SetPostFromBuffer(request_body.c_str(), request_body.size());
  request->SetResultBuffer(&response_buffer);
  TF_RETURN_IF_ERROR(request->Send());

  StringPiece response =
      StringPiece(response_buffer.data(), response_buffer.size());
  TF_RETURN_IF_ERROR(ParseOAuthResponse(response, request_timestamp_sec, token,
                                        expiration_timestamp_sec));
  return Status::OK();
}

Status OAuthClient::GetTokenFromRefreshTokenJson(
    Json::Value json, StringPiece oauth_server_uri, string* token,
    uint64* expiration_timestamp_sec) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_9(mht_9_v, 438, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "OAuthClient::GetTokenFromRefreshTokenJson");

  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  string client_id, client_secret, refresh_token;
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_id", &client_id));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "client_secret", &client_secret));
  TF_RETURN_IF_ERROR(ReadJsonString(json, "refresh_token", &refresh_token));

  const auto request_body = strings::StrCat(
      "client_id=", client_id, "&client_secret=", client_secret,
      "&refresh_token=", refresh_token, "&grant_type=refresh_token");

  const uint64 request_timestamp_sec = env_->NowSeconds();

  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  std::vector<char> response_buffer;
  request->SetUri(string(oauth_server_uri));
  request->SetPostFromBuffer(request_body.c_str(), request_body.size());
  request->SetResultBuffer(&response_buffer);
  TF_RETURN_IF_ERROR(request->Send());

  StringPiece response =
      StringPiece(response_buffer.data(), response_buffer.size());
  TF_RETURN_IF_ERROR(ParseOAuthResponse(response, request_timestamp_sec, token,
                                        expiration_timestamp_sec));
  return Status::OK();
}

Status OAuthClient::ParseOAuthResponse(StringPiece response,
                                       uint64 request_timestamp_sec,
                                       string* token,
                                       uint64* expiration_timestamp_sec) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSoauth_clientDTcc mht_10(mht_10_v, 474, "", "./tensorflow/core/platform/cloud/oauth_client.cc", "OAuthClient::ParseOAuthResponse");

  if (!token || !expiration_timestamp_sec) {
    return errors::FailedPrecondition(
        "'token' and 'expiration_timestamp_sec' cannot be nullptr.");
  }
  Json::Value root;
  Json::Reader reader;
  if (!reader.parse(response.begin(), response.end(), root)) {
    return errors::Internal("Couldn't parse JSON response from OAuth server.");
  }

  string token_type;
  TF_RETURN_IF_ERROR(ReadJsonString(root, "token_type", &token_type));
  if (token_type != "Bearer") {
    return errors::FailedPrecondition("Unexpected Oauth token type: " +
                                      token_type);
  }
  int64_t expires_in = 0;
  TF_RETURN_IF_ERROR(ReadJsonInt(root, "expires_in", &expires_in));
  *expiration_timestamp_sec = request_timestamp_sec + expires_in;
  TF_RETURN_IF_ERROR(ReadJsonString(root, "access_token", token));

  return Status::OK();
}

}  // namespace tensorflow
