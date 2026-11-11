/*
 * Dedicated Bluetooth Thermal Printer Test & Diagnostic Tool
 * 
 * Credentials:
 *   Device Name : POS58D
 *   MAC Address : 86:67:7A:98:EB:9E
 *   Default PIN : 0000 / 1234
 *
 * Board Settings (Arduino IDE):
 *   Board .................. ESP32 Dev Module (or ESP32-C3 Dev Module)
 *   USB CDC On Boot ........ ENABLED
 *   Serial Monitor Baud .... 115200 Baud (Both NL & CR)
 */

#include <Arduino.h>

#if defined(CONFIG_BT_ENABLED) && !defined(CONFIG_IDF_TARGET_ESP32C3) && !defined(CONFIG_IDF_TARGET_ESP32S2)
  #include "BluetoothSerial.h"
  #define USE_CLASSIC_BT 1
  BluetoothSerial SerialBT;
#else
  #include <BLEDevice.h>
  #include <BLEUtils.h>
  #include <BLEScan.h>
  #include <BLEAdvertisedDevice.h>
  #define USE_BLE_BT 1
#endif

// Target Credentials
const char* TARGET_NAME = "POS58D";
uint8_t TARGET_MAC[6]   = {0x86, 0x67, 0x7A, 0x98, 0xEB, 0x9E};
const char* PIN_0000    = "0000";
const char* PIN_1234    = "1234";

bool isConnected = false;

#if USE_BLE_BT
  BLEClient* pBLEClient = nullptr;
  BLERemoteCharacteristic* pRemoteChar = nullptr;

  static BLEUUID bleServices[] = {
    BLEUUID("0000e0ff-0000-1000-8000-00805f9b34fb"),
    BLEUUID("0000ff00-0000-1000-8000-00805f9b34fb"),
    BLEUUID("000018f0-0000-1000-8000-00805f9b34fb"),
    BLEUUID("0000fee7-0000-1000-8000-00805f9b34fb"),
    BLEUUID("49535343-fe7d-4ae5-8fa9-9fafd205e455"),
    BLEUUID("6e400001-b5a3-f393-e0a9-e50e24dcca9e")
  };
  
  static BLEUUID bleChars[] = {
    BLEUUID("0000e001-0000-1000-8000-00805f9b34fb"),
    BLEUUID("0000ff02-0000-1000-8000-00805f9b34fb"),
    BLEUUID("00002af1-0000-1000-8000-00805f9b34fb"),
    BLEUUID("0000fee8-0000-1000-8000-00805f9b34fb"),
    BLEUUID("49535343-201e-4341-b513-0547ba6c0179"),
    BLEUUID("6e400002-b5a3-f393-e0a9-e50e24dcca9e")
  };
#endif

void sendBTBuffer(const uint8_t* data, size_t len) {
#if USE_CLASSIC_BT
  if (SerialBT.connected()) {
    SerialBT.write(data, len);
  }
#elif USE_BLE_BT
  if (pRemoteChar != nullptr) {
    size_t offset = 0;
    while (offset < len) {
      size_t chunk = (len - offset > 20) ? 20 : (len - offset);
      pRemoteChar->writeValue((uint8_t*)(data + offset), chunk, false);
      offset += chunk;
      delay(3);
    }
  }
#endif
}

void printTestReceipt() {
  if (!isConnected) {
    Serial.println(F("\n[TEST ERROR] Printer is NOT connected! Connect first."));
    return;
  }

  Serial.println(F("\n[PRINTING] Sending ESC/POS test receipt over Bluetooth..."));
  
  // ESC @ (Initialize Printer)
  uint8_t initCmd[] = {0x1B, 0x40};
  sendBTBuffer(initCmd, sizeof(initCmd));
  delay(100);

  // ESC a 1 (Center Align)
  uint8_t alignCenter[] = {0x1B, 'a', 1};
  sendBTBuffer(alignCenter, sizeof(alignCenter));

  // ESC E 1 (Bold ON)
  uint8_t boldOn[] = {0x1B, 'E', 1};
  sendBTBuffer(boldOn, sizeof(boldOn));

  // GS ! 0x11 (Double Height + Width)
  uint8_t doubleSize[] = {0x1D, '!', 0x11};
  sendBTBuffer(doubleSize, sizeof(doubleSize));

  const char* title = "POS58D TEST\n";
  sendBTBuffer((const uint8_t*)title, strlen(title));

  // GS ! 0x00 (Normal Size)
  uint8_t normalSize[] = {0x1D, '!', 0x00};
  sendBTBuffer(normalSize, sizeof(normalSize));
  
  // ESC E 0 (Bold OFF)
  uint8_t boldOff[] = {0x1B, 'E', 0};
  sendBTBuffer(boldOff, sizeof(boldOff));

  // Left align
  uint8_t alignLeft[] = {0x1B, 'a', 0};
  sendBTBuffer(alignLeft, sizeof(alignLeft));

  const char* body = 
    "--------------------------------\n"
    "Device Name : POS58D\n"
    "MAC Address : 86:67:7A:98:EB:9E\n"
    "BT Status   : CONNECTED (OK)\n"
    "--------------------------------\n"
    "If you can read this, Bluetooth\n"
    "thermal printing is WORKING!\n"
    "--------------------------------\n\n\n\n";
  sendBTBuffer((const uint8_t*)body, strlen(body));

  // GS V 1 (Cut paper)
  uint8_t cutCmd[] = {0x1D, 'V', 1};
  sendBTBuffer(cutCmd, sizeof(cutCmd));

  Serial.println(F("[PRINTING] Test receipt sent successfully!"));
}

bool attemptConnection(int mode) {
  Serial.println(F("\n=================================================="));
  Serial.printf("STARTING CONNECTION TEST MODE %d...\n", mode);
  Serial.printf("Target Device : %s\n", TARGET_NAME);
  Serial.printf("Target MAC    : %02X:%02X:%02X:%02X:%02X:%02X\n",
    TARGET_MAC[0], TARGET_MAC[1], TARGET_MAC[2],
    TARGET_MAC[3], TARGET_MAC[4], TARGET_MAC[5]);
  Serial.println(F("=================================================="));

#if USE_CLASSIC_BT
  Serial.println(F("[CLASSIC BT] Initializing Bluetooth Master Mode..."));
  
  if (mode == 1) {
    Serial.println(F("[TEST 1] Connecting by MAC (86:67:7A:98:EB:9E) with PIN '0000'..."));
    SerialBT.setPin(PIN_0000);
    isConnected = SerialBT.connect(TARGET_MAC);
  } else if (mode == 2) {
    Serial.println(F("[TEST 2] Connecting by MAC (86:67:7A:98:EB:9E) with PIN '1234'..."));
    SerialBT.setPin(PIN_1234);
    isConnected = SerialBT.connect(TARGET_MAC);
  } else if (mode == 3) {
    Serial.println(F("[TEST 3] Connecting by Device Name ('POS58D')..."));
    SerialBT.setPin(PIN_0000);
    isConnected = SerialBT.connect(TARGET_NAME);
  }

  if (isConnected) {
    Serial.println(F("\n>>> SUCCESS! CONNECTED TO POS58D PRINTER! <<<"));
    Serial.println(F(">>> Press 'p' or option 3 to print a test receipt! <<<"));
  } else {
    Serial.println(F("\n>>> FAILED: Could not connect to printer. <<<"));
    Serial.println(F("Troubleshooting Check:"));
    Serial.println(F("1. Is the POS58D printer turned ON?"));
    Serial.println(F("2. Is the printer currently connected to your phone? (Unpair it!)"));
    Serial.println(F("3. Try pressing option 2 or 3 in Serial Monitor."));
  }

#elif USE_BLE_BT
  Serial.println(F("[BLE] Initializing BLE Client..."));
  if (pBLEClient == nullptr) {
    BLEDevice::init("TEST_CLIENT");
    pBLEClient = BLEDevice::createClient();
  }

  BLEAddress btAddr(TARGET_MAC);
  Serial.println(F("[BLE] Connecting to MAC address 86:67:7A:98:EB:9E..."));
  
  if (pBLEClient->connect(btAddr)) {
    Serial.println(F("[BLE] Peripheral connected! Scanning GATT services..."));
    for (size_t i = 0; i < sizeof(bleServices)/sizeof(bleServices[0]); i++) {
      BLERemoteService* pSvc = pBLEClient->getService(bleServices[i]);
      if (pSvc != nullptr) {
        Serial.printf("[BLE] Found Service %s!\n", bleServices[i].toString().c_str());
        for (size_t j = 0; j < sizeof(bleChars)/sizeof(bleChars[0]); j++) {
          pRemoteChar = pSvc->getCharacteristic(bleChars[j]);
          if (pRemoteChar != nullptr && (pRemoteChar->canWrite() || pRemoteChar->canWriteNoResponse())) {
            isConnected = true;
            Serial.printf("[BLE] SUCCESS! Bound to Characteristic %s!\n", bleChars[j].toString().c_str());
            break;
          }
        }
        if (isConnected) break;
      }
    }
    if (!isConnected) {
      Serial.println(F("[BLE] Connected, but no matching printer characteristic found."));
      pBLEClient->disconnect();
    }
  } else {
    Serial.println(F("[BLE] Failed to connect directly to BLE address."));
  }

  if (isConnected) {
    Serial.println(F("\n>>> SUCCESS! CONNECTED TO POS58D BLE PRINTER! <<<"));
  } else {
    Serial.println(F("\n>>> FAILED: BLE Connection failed. Ensure printer is ON & un-paired. <<<"));
  }
#endif

  return isConnected;
}

void printMenu() {
  Serial.println(F("\n=================================================="));
  Serial.println(F("     POS58D BLUETOOTH PRINTER TEST MENU           "));
  Serial.println(F("=================================================="));
  Serial.printf("Current Status : %s\n", isConnected ? "CONNECTED [ONLINE]" : "DISCONNECTED [OFFLINE]");
  Serial.println(F("Commands:"));
  Serial.println(F("  1 - Test Connect via MAC (86:67:7A:98:EB:9E) + PIN 0000"));
  Serial.println(F("  2 - Test Connect via MAC (86:67:7A:98:EB:9E) + PIN 1234"));
  Serial.println(F("  3 - Test Connect via Device Name ('POS58D')"));
  Serial.println(F("  p - Send Test Print Receipt to POS58D"));
  Serial.println(F("  m - Redisplay this menu"));
  Serial.println(F("==================================================\n"));
}

void setup() {
  Serial.begin(115200);

  // USB CDC delay
  uint32_t startWait = millis();
  while (!Serial && (millis() - startWait < 3000)) {
    delay(10);
  }

#if USE_CLASSIC_BT
  // Enable Master mode
  SerialBT.begin("POS58D_TESTER", true);
#endif

  delay(500);
  printMenu();

  // Run initial test automatically
  attemptConnection(1);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '1') {
      attemptConnection(1);
    } else if (c == '2') {
      attemptConnection(2);
    } else if (c == '3') {
      attemptConnection(3);
    } else if (c == 'p' || c == 'P') {
      printTestReceipt();
    } else if (c == 'm' || c == 'M') {
      printMenu();
    }
  }

#if USE_CLASSIC_BT
  if (isConnected && !SerialBT.connected()) {
    isConnected = false;
    Serial.println(F("\n[ALERT] Bluetooth Connection to POS58D WAS LOST!"));
  }
#endif

  delay(50);
}
