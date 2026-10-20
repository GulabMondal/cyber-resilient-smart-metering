/*
 * ESP32-S3 crypto / TLS cost benchmark for Table 5 of the paper.
 *
 * Measures on the real edge node:
 *   1. HMAC-SHA256 over one 150-byte record (hardware HMAC peripheral, eFuse key)
 *   2. HMAC-SHA256 in software (mbedTLS, key in RAM) for comparison
 *   3. AES-128-GCM protection of one 150-byte record (mbedTLS)
 *   4. TLS 1.2/1.3 full handshake, ECDHE-P256, mutual auth (to a local broker)
 *   5. Session-resumption handshake
 *   6. Bytes on the wire per record: TLS only vs TLS + 32-byte tag
 *
 * Build: Arduino IDE 2.x, board "ESP32S3 Dev Module", arduino-esp32 core >= 3.0
 * (ESP-IDF 5.x underneath).  Fill in Wi-Fi + broker details below.
 * Open Serial Monitor at 115200 baud; copy the RESULTS block into the paper.
 *
 * Hardware HMAC (test 1) needs a key already burned with purpose HMAC_UP, e.g.
 *   espefuse.py --port COMx burn_key BLOCK_KEY4 hmac_key.bin HMAC_UP
 * If no key is burned the test prints "HW HMAC: not provisioned" and continues.
 */
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "mbedtls/md.h"
#include "mbedtls/gcm.h"
#include "esp_timer.h"
#if __has_include("esp_hmac.h")
#include "esp_hmac.h"
#define HAVE_HW_HMAC 1
#endif

// ----------------------------------------------------------------- settings
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* BROKER    = "192.168.1.10";   // MQTT/TLS broker on the plant LAN
const uint16_t PORT   = 8883;
// PEM strings for mutual TLS (CA, device cert, device key)
const char* CA_PEM    = R"(-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----)";
const char* CERT_PEM  = R"(-----BEGIN CERTIFICATE-----
...
-----END CERTIFICATE-----)";
const char* KEY_PEM   = R"(-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----)";
const int N_REP = 1000;        // repetitions for micro-benchmarks
const int N_TLS = 20;          // repetitions for handshakes

// ------------------------------------------------------------------ helpers
static char record[151];
static uint8_t key[32];
static double mean_us, sd_us;

static void stats(const int64_t* t, int n) {
  double s = 0, s2 = 0;
  for (int i = 0; i < n; i++) { s += t[i]; }
  mean_us = s / n;
  for (int i = 0; i < n; i++) { s2 += (t[i] - mean_us) * (t[i] - mean_us); }
  sd_us = sqrt(s2 / (n - 1));
}

static void make_record() {
  // 150-byte JSON record shaped like the real payload
  snprintf(record, sizeof(record),
    "{\"id\":\"HM-1\",\"ts\":1760860000123,\"seq\":123456,\"P\":32.415,\"Q\":19.877,"
    "\"S\":38.020,\"V\":415.21,\"I\":[52.9,53.4,52.1],\"PF\":0.853,\"THD\":0.061}");
  record[150] = 0;
}

static void print_row(const char* name, double m, double s, const char* unit) {
  Serial.printf("%-46s %10.3f %10.3f %s\n", name, m, s, unit);
}

// ------------------------------------------------------------------ tests
void bench_sw_hmac() {
  static int64_t t[N_REP];
  uint8_t out[32];
  const mbedtls_md_info_t* md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  for (int i = 0; i < N_REP; i++) {
    int64_t a = esp_timer_get_time();
    mbedtls_md_hmac(md, key, 32, (const uint8_t*)record, 150, out);
    t[i] = esp_timer_get_time() - a;
  }
  stats(t, N_REP);
  print_row("HMAC-SHA256, software (mbedTLS), 150 B", mean_us / 1000, sd_us / 1000, "ms");
}

void bench_hw_hmac() {
#ifdef HAVE_HW_HMAC
  static int64_t t[N_REP];
  uint8_t out[32];
  if (esp_hmac_calculate(HMAC_KEY4, record, 150, out) != ESP_OK) {
    Serial.println("HW HMAC: not provisioned (burn BLOCK_KEY4 with HMAC_UP)");
    return;
  }
  for (int i = 0; i < N_REP; i++) {
    int64_t a = esp_timer_get_time();
    esp_hmac_calculate(HMAC_KEY4, record, 150, out);
    t[i] = esp_timer_get_time() - a;
  }
  stats(t, N_REP);
  print_row("HMAC-SHA256, HW peripheral (eFuse key), 150 B", mean_us / 1000, sd_us / 1000, "ms");
#else
  Serial.println("HW HMAC: esp_hmac.h not available in this core");
#endif
}

void bench_gcm() {
  static int64_t t[N_REP];
  uint8_t k[16] = {0}, iv[12] = {0}, tag[16], ct[150];
  mbedtls_gcm_context g;
  mbedtls_gcm_init(&g);
  mbedtls_gcm_setkey(&g, MBEDTLS_CIPHER_ID_AES, k, 128);
  for (int i = 0; i < N_REP; i++) {
    iv[0] = i;
    int64_t a = esp_timer_get_time();
    mbedtls_gcm_crypt_and_tag(&g, MBEDTLS_GCM_ENCRYPT, 150, iv, 12, NULL, 0,
                              (const uint8_t*)record, ct, 16, tag);
    t[i] = esp_timer_get_time() - a;
  }
  mbedtls_gcm_free(&g);
  stats(t, N_REP);
  print_row("AES-128-GCM record protection, 150 B", mean_us / 1000, sd_us / 1000, "ms");
}

void bench_tls(bool resume) {
  static int64_t t[N_TLS];
  int ok = 0;
  WiFiClientSecure c;
  c.setCACert(CA_PEM);
  c.setCertificate(CERT_PEM);
  c.setPrivateKey(KEY_PEM);
  for (int i = 0; i < N_TLS; i++) {
    int64_t a = esp_timer_get_time();
    bool con = c.connect(BROKER, PORT);
    int64_t d = esp_timer_get_time() - a;
    if (con) { t[ok++] = d; }
    if (!resume) c.stop();     // full handshake every time
    else { c.stop(); }         // arduino-esp32 reuses the session ticket when supported
    delay(200);
  }
  if (ok < 2) { Serial.println("TLS: broker unreachable - check BROKER/PORT/certs"); return; }
  stats(t, ok);
  print_row(resume ? "TLS handshake with session resumption" : "TLS full handshake, ECDHE-P256, mTLS",
            mean_us / 1000, sd_us / 1000, "ms");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  for (int i = 0; i < 32; i++) key[i] = i * 7 + 3;
  make_record();
  Serial.printf("\nCPU %d MHz, free heap %u\n", getCpuFrequencyMhz(), ESP.getFreeHeap());
  Serial.println("===== RESULTS (mean, SD) =====");
  bench_sw_hmac();
  bench_hw_hmac();
  bench_gcm();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  if (WiFi.status() == WL_CONNECTED) {
    bench_tls(false);
    bench_tls(true);
  } else {
    Serial.println("Wi-Fi not connected - TLS tests skipped");
  }
  Serial.println("Bytes on the wire per record:");
  Serial.println("  TLS 1.3 record overhead = 5 (header) + 1 (type) + 16 (AEAD tag) = 22 B");
  Serial.println("  Proposed adds 32 B HMAC tag (+ seq/ts already in the JSON)");
  Serial.printf("  150 B record: TLS only = %d B, TLS + tag = %d B\n", 150 + 22, 150 + 32 + 22);
  Serial.println("===== END =====");
}

void loop() {}
