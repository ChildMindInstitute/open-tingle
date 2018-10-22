
//NOTE: Thermopile values are now smoothed   (((Xt-2...)/2 + Xt-1)/2 + X)/2    6/14/18
/* NN 0: none
   NN 1: mouth
   NN 2: fronthead
   NN 3: tophead
   NN 4: backhead
   NN 5: righthead
   NN 6: lefthead

   "F[" at front of lines in sublime text
   CTRL+A (select all)
   CTRL+SHIFT+L (split selection to lines)
   HOME (move cursors to the beginning of the line)
 */
/********************************************************************************************************/
/************************ INCLUDES **********************************************************************/
/********************************************************************************************************/
#define NRF52

#include <SPI.h>                    //SPI comms protocal for KX126
#include <BLEPeripheral.h>          //bluetooth cleint
#include <BLEUtil.h>                //bluetooth utilities
#include <Wire.h>                   //I2C protocal for MLX90615 and VL6180X
#include "Arduino_nRF5x_lowPower.h" //nRF52 SDK power mgmt abstractions
#include "KX126_SPI.h"              //accelerometer
#include "VL6180X.h"                //distance sensor
#include "Flash.h"                  //nRF52832 Flash Storage Core https://github.com/d00616/arduino-NVM
//#include "VirtualPage.h"          //nRF52832 Flash Storage MGMT

#include "tingleNN.h"               //gesture detection and neural network storage

/*
//pre-trained neural network weights and activation functions
#include "NN_LHAND_BACK_5521.h"
#include "NN_LHAND_EYES_5521.h"
#include "NN_LHAND_FRONT_5521.h"
#include "NN_LHAND_MOUTH_5521.h"
#include "NN_LHAND_SIDE_5521.h"
#include "NN_LHAND_TOP_5521.h"

#include "NN_LHAND_BACK_7521.h"
#include "NN_LHAND_EYES_7521.h"
#include "NN_LHAND_FRONT_7521.h"
#include "NN_LHAND_MOUTH_7521.h"
#include "NN_LHAND_SIDE_7521.h"
#include "NN_LHAND_TOP_7521.h"

#include "NN_RHAND_BACK_5521.h"
#include "NN_RHAND_EYES_5521.h"
#include "NN_RHAND_FRONT_5521.h"
#include "NN_RHAND_MOUTH_5521.h"
#include "NN_RHAND_SIDE_5521.h"
#include "NN_RHAND_TOP_5521.h"

#include "NN_RHAND_BACK_7521.h"
#include "NN_RHAND_EYES_7521.h"
#include "NN_RHAND_FRONT_7521.h"
#include "NN_RHAND_MOUTH_7521.h"
#include "NN_RHAND_SIDE_7521.h"
#include "NN_RHAND_TOP_7521.h"
*/

/********************************************************************************************************/
/************************ CONSTANTS / SYSTEM VAR ********************************************************/
/********************************************************************************************************/
bool    debug = true;                 //turn serial on/off to get data or turn up sample rate
bool    debug_time = false;           //turn loop component time debug on/off
bool    IS_CONNECTED = false;         // flag for bluetooth app connectivity
bool    IS_DETECTING = false;         // flag for nueral network activation gesture detection
bool    IS_DISABLED = false;

//native max loop speed is about 35 ms or 28Hz
float   speedLowpower  = 1000 / 6;    //2Hz default power saving speed
float   speedBluetooth = 1000 / 16;   //16Hz while connected to 
float   speedBallpark  = 1000 / 8;    //8Hz when NN approach target

float   speedMs = speedLowpower;

float   detect_objT_lowpass =    80;
float   detect_objT_highpass =   102;
int     tempScaleAdjust =        12;
int     limit_stopRepeatDetect = 200;

/********************************************************************************************************/
/************************ DEFINITIONS *******************************************************************/
/********************************************************************************************************/
/****************** N68 *******************/

//SCL = 6  SDA = 7   RX = 12 (dummy)   TX = 3

#define GREEN_LED_PIN             30  //5
#define BUTTON_PIN                5
#define VIBRATE_PIN               25
#define BATTERY_PIN               28

//Accelerometer Pins
#define CS_PIN                    8
#define KX022_SDI                 15
#define KX022_SDO                 18
#define KX022_SCL                 13
#define KX022_INT                 11

#define PIN_SPI_MISO         (KX022_SDO)
#define PIN_SPI_MOSI         (KX022_SDI)
#define PIN_SPI_SCK          (KX022_SCL)


/****************** X9 ********************/
/*
//SCL = 6  SDA = 7   RX = 14 (dummy)   TX = 13
#define GREEN_LED_PIN             15
#define BUTTON_PIN                29
#define VIBRATE_PIN               8
#define BATTERY_PIN               28

//Accelerometer Pins
#define CS_PIN                    24
#define KX022_SDI                 19
#define KX022_SDO                 20
#define KX022_SCL                 18
//#define KX022_INT                 11

#define PIN_SPI_MISO         (KX022_SDO)
#define PIN_SPI_MOSI         (KX022_SDI)
#define PIN_SPI_SCK          (KX022_SCL)
*/

//Flash storage
// the MAGIC number to address your page
#define MAGIC_COUNTER 0x7e008 //e7cba72b
#define MAGIC_TEST 0x5294aa9f

//Thermopile Addresses
#define MLX90615_I2CADDR          0x00
#define MLX90615_I2CADDR1         0x2A
#define MLX90615_I2CADDR2         0x2B 
#define MLX90615_I2CADDR3         0x2C
#define MLX90615_I2CADDR4         0x2D
// RAM
#define MLX90615_RAWIR1           0x04
#define MLX90615_RAWIR2           0x05
#define MLX90615_TA               0x26
#define MLX90615_TOBJ1            0x27
#define MLX90615_TOBJ2            0x28
// EEPROM
#define MLX90615_TOMAX            0x20
#define MLX90615_TOMIN            0x21
#define MLX90615_PWMCTRL          0x22
#define MLX90615_TARANGE          0x23
#define MLX90615_EMISS            0x24
#define MLX90615_CONFIG           0x25
#define MLX90615_ADDR             0x0E


/********************************************************************************************************/
/************************ VARIABLES *********************************************************************/
/********************************************************************************************************/

  //LED
    float   greenLED_timer = 0;
    int     LED_counter = 10;
    bool    greenLED_status = false;

  //Battery MGMT  
    int batteryValue = 100;
    float buttonBeginPressTime = 0;
    
  //Button
    int     buttonState = 0;         // variable for reading the pushbutton

  //NVM Flash Memory System on nRF52832
 /*   uint32_t *vpage, *new_vpage;
    void print_word(uint32_t *address); // Output function
    time_t time_start, time_end; */
    uint32_t *page1;
  //  uint32_t *page2;
  //  uint32_t *page3;
  //  uint32_t *page4;
    int   settingsData[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  //LSTM Neural Network weights for reception
  //NN weights
    int     nnLength = 500;
    float   F[500];
    int     transmittedCounter = 0;
    bool    flag_haveNeural = false;

  //Detection
    int     selectNN = 0;
    float   fiveInScore = 0;
    float   sevenInScore = 0;
    float   vibrateTimer = 0;
    float   lastAlertTimer = 0;
    int     vibrate_counter = 0;
    bool    vibrate_status = false;
    float   lastDetectTime = 0;
    bool    flag_stopRepeatDetect = false;
    bool    flag_detect = false;              //gesture detected!
    int handCode, hairCode, targetsCode, sensitivityCode;

  //Timestamp
    float   clocktime = 0;
    
  //Bluetooth
    unsigned long microsPerReading, microsPrevious;
    float   accelScal;
    int     command_value = 99; //controlls how device and app talk to each other

  //System
    int     varState = 0; //variable state controlled in app and appended to data stream
    bool    sleepLight = false;
    bool    sleepDeep = false;

  //MLX90615 Thermopiles
    float   TObj[4] = {0,0,0,0};
    float   TAmb[4] = {0,0,0,0};
    float   TObjNormalized[4] = {0,0,0,0};
    float   TAmbAv, TAmbMin, baselineMinTemp;

    float   findMinTempTimer = 0;  //track time between background minimum temp updates
    float   minTempVals[10] = {0,0,0,0,0,0,0,0,0,0};
    float   minTempTimes[10] = {0,0,0,0,0,0,0,0,0,0};

  //vl6180x Distance
    float   distance = 0;
 
  //KX126 Accelerometer
    // pins used for the connection with the sensor
    // the other you need are controlled by the SPI library):
    const int dataReadyPin = 6;
    const int chipSelectPin = 7;

    float     acc[3];
    double    pitch;
    double    roll;


/********************************************************************************************************/
/************************ DECLARATIONS ******************************************************************/
/********************************************************************************************************/

//Neural Network Class Object 
tingleNeuralNetwork neuralNet;

//Time of Flight LIDAR distance sensor
VL6180X vl6180x;

//KX022 Accelerometer
KX126_SPI kx126(CS_PIN);

//Bluetooth
// create peripheral instance, see pinouts above
BLEPeripheral blePeripheral = BLEPeripheral();

// create service
BLEService customService =    BLEService("a000");

// create command i/o characteristics
//BLECharCharacteristic    ReadOnlyArrayGattCharacteristic  = BLECharCharacteristic("a001", BLERead);
//BLECharCharacteristic    WriteOnlyArrayGattCharacteristic = BLECharCharacteristic("a002", BLEWrite);

BLECharacteristic    ReadOnlyArrayGattCharacteristic  = BLECharacteristic("a001", BLERead | BLENotify, 20);
BLECharacteristic    WriteOnlyArrayGattCharacteristic = BLECharacteristic("a002", BLEWrite, 20);

//create streaming data characteristic
BLECharacteristic        SensorDataCharacteristic("a003", BLERead | BLENotify, 20);  //@param data - an Uint8Array.

//create streaming neural network i/o characteristic
BLECharacteristic    ReadDataCharacteristic  = BLECharacteristic("a004", BLERead | BLENotify, 20); //@param data - an Uint8Array.
BLECharacteristic    WriteDataCharacteristic  = BLECharacteristic("a005", BLEWrite, 20); //@param data - an Uint8Array.

BLEBondStore bleBondStore(0);

/********************************************************************************************************/
/************************ UTILITY FUNCTIONS *************************************************/
/********************************************************************************************************/
float differenceBetweenAngles(float firstAngle, float secondAngle)
  {
        double difference = secondAngle - firstAngle;
        while (difference < -180) difference += 360;
        while (difference > 180) difference -= 360;
        return difference;
 }


/********************************************************************************************************/
/************************ MLX90615 THERMOPILE FUNCTIONS *************************************************/
/********************************************************************************************************/
uint16_t read16(uint8_t a, int sensorNum) {
  uint8_t _addr = MLX90615_I2CADDR;

  if(sensorNum == 0) _addr = MLX90615_I2CADDR;   //custom addresses
  else if(sensorNum == 1)  _addr = MLX90615_I2CADDR1;   
  else if(sensorNum == 2)  _addr = MLX90615_I2CADDR2;
  else if(sensorNum == 3)  _addr = MLX90615_I2CADDR3;
  else if(sensorNum == 4)  _addr = MLX90615_I2CADDR4;
  
  uint16_t ret;
  Wire.beginTransmission(_addr);                  // start transmission to device 
  Wire.write(a); delay(1);                        // sends register address to read from
  Wire.endTransmission(false);                    // end transmission
  Wire.requestFrom(_addr, (uint8_t)3); delay(1);  // send data n-bytes read
  ret = Wire.read();// delay(1);                    // receive DATA
  ret |= Wire.read() << 8;// delay(1);              // receive DATA
  uint8_t pec = Wire.read(); delay(1);
  return ret;
}

float readTemp(uint8_t reg, int sensorNum) {
  float temp;
  temp = read16(reg, sensorNum);
  temp *= .02;
  temp  -= 273.15;
  return temp;
}

double readObjectTempF(int sensorNum) {
  return (readTemp(MLX90615_TOBJ1, sensorNum) * 9 / 5) + 32;
}

double readAmbientTempF(int sensorNum) {
  return (readTemp(MLX90615_TA, sensorNum) * 9 / 5) + 32;
}

double readObjectTempC(int sensorNum) {
  return readTemp(MLX90615_TOBJ1, sensorNum);
}

double readAmbientTempC(int sensorNum) {
  return readTemp(MLX90615_TA, sensorNum);
}

/********************************************************************************************************/
/************************ NVM FLASH STORAGE FUNCTIONS ***************************************************/
/********************************************************************************************************/
// print a data word inluding address
void print_word(uint32_t *address) {
  Serial.print("Word at address 0x");
  Serial.print((size_t)address, HEX);
  Serial.print("=0x");
  Serial.println(*address, HEX);
}
/*
// print a time
void print_time(time_t time_start, time_t time_end) {
  time_t micros = time_end - time_start;

  Serial.print(" ");
  Serial.print(micros / 1000);
  Serial.print(".");
  int rest = (micros % 1000) / 10;
  if (rest < 10) {
    Serial.print("0");
  }
  if (rest < 100) {
    Serial.print("0");
  }
  Serial.print(rest);
  Serial.println("ms");
}

// dump page data
void print_page_data(uint32_t *address) {
  Serial.println("---------- Page data ----------");
  Serial.print("Page address: 0x");
  Serial.println((size_t)address, HEX);
  Serial.print("Page in release state: ");
  Serial.println(VirtualPage.release_started(address));
  Serial.println("Data not equal to 0xffffffff:");
  for (int i = 0; i < VirtualPage.length(); i++) {
    if (address[i] != ~(uint32_t)0)
      print_word(&address[i]);
  }
  Serial.println("-------------------------------");
}

// Print a page address
void print_address(uint32_t *address) {
  Serial.print("address 0x");
  Serial.print((size_t)address, HEX);
}

// Print counter value
void print_counter(uint32_t counter) {
  Serial.print("Data value is: ");
  Serial.println(counter);
}

void print_status(bool status) {
  if (status) {
    print_ok();
  } else {
    print_error();
  }
}

void print_ok() { Serial.println(" OK"); }

void print_error() { Serial.println(" Error"); }
*/
void nvmFlashSetup(){
     /************ CONFIGURE NVM FLASH STORAGE ************************/
    // Print some flash data
  Serial.print("Flash page size: ");
  Serial.println(Flash.page_size());
  Serial.print("Number of flash pages: ");
  Serial.println(Flash.page_count());
  Serial.print("Address of first page: 0x");
  Serial.println((size_t)Flash.page_address(0), HEX);

  // Find out address of the last available page
  page1 = Flash.page_address(Flash.page_count() - 1);
//  page2 = Flash.page_address(Flash.page_count() - 2);
//  page3 = Flash.page_address(Flash.page_count() - 3);

  print_word(page1);
  int storedVal = (int)*page1;
    Serial.print("Stored val: ");
  Serial.println(storedVal);
  if(storedVal > 1000 && storedVal < 3000){
    Serial.println("Stored settings retrieved");
    changeNeuralNetwork(storedVal);
  } else {
    changeNeuralNetwork(1132); //left, short, top
  }
/*
  // Erase the page
  Serial.println("Erase page");
  Flash.erase(page1, Flash.page_size());
  print_word(page1);

  // Inform about write
  Serial.println("Write 0x12345678");

  // Write to flash, you can do more writes until writing is disabled
  Flash.write(page1, 0x12345678);
 // Flash.write(page2, 45678);
 // Flash.write(page3, 0xab123);

  // Print memory content
  print_word(page1);
*/
}

/********************************************************************************************************/
/************************ BLUETOOTH BLE FUNCTIONS *******************************************************/
/********************************************************************************************************/
void blePeripheralConnectHandler(BLECentral& central) {
  // central connected event handler

  //CONNECTED!!!
  IS_CONNECTED = true;
  
  //increase speed while connected to Bluetooth
 // speedMs = speedBluetooth;
  transmittedCounter = 0; //reset NN weight transmittions counter
  if(debug){
    Serial.print(F("Connected event, central: "));
    Serial.println(central.address());
  }

  //SEND STORED SETTINGS
  settingsData[19] = 12; //send stored settings code
  transmitSettings();
  delay(5);
}

void blePeripheralDisconnectHandler(BLECentral& central) {
  // central disconnected event handler

    //NOT CONNECTED!!!
    IS_CONNECTED = false;
  
  //bring spped back down to low power default
//  speedMs = speedLowpower;
    transmittedCounter = 0; //reset NN weight transmittions counter
    if(debug){
        Serial.print(F("Disconnected event, central: "));
        Serial.println(central.address());
    }
  delay(5);
}

void blePeripheralServicesDiscoveredHandler(BLECentral& central) {
  // central  services discovered event handler
  if(debug){
    Serial.print(F(" services discovered event, central: "));
    Serial.println(central.address());
  }
/*
  if (ReadOnlyArrayGattCharacteristic.canRead()) {
    Serial.println(F("ReadOnlyArrayGattCharacteristic"));
    ReadOnlyArrayGattCharacteristic.read();
  }

  if (WriteOnlyArrayGattCharacteristic.canWrite()) {
    Serial.println(F("WriteOnlyArrayGattCharacteristic"));

   // unsigned long writeValue = 42;
    static uint8_t writeValue[10] = {0};
  //  writeValue[0] = 5;

    WriteOnlyArrayGattCharacteristic.write((const unsigned char*)&writeValue, sizeof(writeValue));
  } */
  delay(5);
  //delay(2000);
}

void bleCharacteristicValueUpdatedHandle(BLECentral& central, BLECharacteristic& characteristic) {
  
    if(debug){ Serial.print(F(" Begin bleCharacteristicValueUpdatedHandle: ")); }
    
  const unsigned char* the_buffer = characteristic.value();
  unsigned char the_length = characteristic.valueLength();

 
  int bleRawArray[20];
  
  for (byte i = 0; i < the_length; i++){ 
    String tempString = String(the_buffer[i], HEX);
    char *char_bufTemp = const_cast<char*>(tempString.c_str());
    bleRawArray[i] = (int)strtol(char_bufTemp, NULL, 16);
  }

//  bleRawVal.toCharArray(temp_char_buffer, the_length);
 // sscanf(temp_char_buffer, "%x", &command_value);


  Serial.print("Received Values: ");
  for (int j = 0; j < the_length; j++){ 
    Serial.print(bleRawArray[j]);
    Serial.print(" ");
  }
  Serial.println(" ");

  BLEUtil::printBuffer(characteristic.value(), characteristic.valueLength());

  int valueToStore = bleRawArray[0]*1000 + bleRawArray[1]*100 + bleRawArray[2]*10 + bleRawArray[3];
  Serial.print("Value to store: "); Serial.println(valueToStore);

  //DISCONNECT RADION SO WE CAN USE FLASH
  blePeripheral.disconnect();
  delay(2);
  blePeripheral.end();
  delay(2);
  sd_ble_gap_adv_stop();
  delay(2);
  NRF_RADIO->TASKS_DISABLE = 1;
  IS_DISABLED = true;
  delay(250);

//  nvmFlashSetup();

  /********** store settings in flash ****************/
  // Print page data / Print old settings
  Serial.println("Old stored settings: ");
  print_word(page1);
    Serial.println("Erase page");
  Flash.erase(page1, Flash.page_size());
 // Flash.erase(page2, Flash.page_size());
  print_word(page1);

  // Inform about write
 // Serial.println("Write 0x12345678");

  // Write to flash, you can do more writes until writing is disabled
//  Flash.write(page1, 0x12345678);
  Flash.write(page1, valueToStore);


  // Print memory content
  Serial.println("New stored settings: ");
  print_word(page1);




  int rawNewSettings = (int)*page1; 
    Serial.print(" ");
  Serial.println(rawNewSettings);

  changeNeuralNetwork(rawNewSettings);
/*
  int handCode, hairCode, targetsCode, sensitivityCode;
  handCode    = rawNewSettings / 1000; 
  hairCode    = (rawNewSettings / 100) - (handCode * 10); 
  targetsCode = (rawNewSettings / 10) - (handCode * 100) - (hairCode * 10);
  sensitivityCode = rawNewSettings - (handCode * 1000) - (hairCode * 100) - (targetsCode * 10);

  if(debug){
      Serial.print("handCode: "); Serial.print("handCode: ");
  }

  //left hand
  if(handCode == 1){

      //short hair
      if(hairCode == 1){

          //front head
          if(targetsCode == 1){
              neuralNet.selectTarget(LHAND_SHORT_FRONT); 
              if(debug) Serial.println("LHAND_SHORT_FRONT");
          }
          //side head
          else if(targetsCode == 2){
              neuralNet.selectTarget(LHAND_SHORT_SIDE); 
              if(debug) Serial.println("LHAND_SHORT_SIDE");
          }
          //back head
          else if(targetsCode == 3){
              neuralNet.selectTarget(LHAND_SHORT_BACK); 
              if(debug) Serial.println("LHAND_SHORT_BACK");
          }
          //top head
          else if(targetsCode == 4){
              neuralNet.selectTarget(LHAND_SHORT_TOP); 
              if(debug) Serial.println("LHAND_SHORT_TOP");
          }
          //mouth head
          else if(targetsCode == 5){
              neuralNet.selectTarget(LHAND_SHORT_MOUTH); 
              if(debug) Serial.println("LHAND_SHORT_MOUTH");
          }
          //eyes head
          else if(targetsCode == 6){
              neuralNet.selectTarget(LHAND_SHORT_EYES); 
              if(debug) Serial.println("LHAND_SHORT_EYES");
          }
      }
  }
    //right hand
  else if(handCode == 2){

      //short hair
      if(hairCode == 1){

          //front head
          if(targetsCode == 1){
              neuralNet.selectTarget(RHAND_SHORT_FRONT); 
              if(debug) Serial.println("RHAND_SHORT_FRONT");
          }
          //side head
          else if(targetsCode == 2){
              neuralNet.selectTarget(RHAND_SHORT_SIDE); 
              if(debug) Serial.println("RHAND_SHORT_SIDE");
          }
          //back head
          else if(targetsCode == 3){
              neuralNet.selectTarget(RHAND_SHORT_BACK); 
              if(debug) Serial.println("RHAND_SHORT_BACK");
          }
          //top head
          else if(targetsCode == 4){
              neuralNet.selectTarget(RHAND_SHORT_TOP); 
              if(debug) Serial.println("RHAND_SHORT_TOP");
          }
          //mouth head
          else if(targetsCode == 5){
              neuralNet.selectTarget(RHAND_SHORT_MOUTH); 
              if(debug) Serial.println("RHAND_SHORT_MOUTH");
          }
          //eyes head
          else if(targetsCode == 6){
              neuralNet.selectTarget(RHAND_SHORT_EYES); 
              if(debug) Serial.println("RHAND_SHORT_EYES");
          }
      }
  }

  //set new sensitivity
  if(sensitivityCode == 0){ neuralNet.sensitivity(SENSE_VERY_LOW);        if(debug) Serial.println("SENSE_VERY_LOW"); }
  else if(sensitivityCode == 1){ neuralNet.sensitivity(SENSE_LOW);        if(debug) Serial.println("SENSE_LOW"); }
  else if(sensitivityCode == 2){ neuralNet.sensitivity(SENSE_AVERAGE);    if(debug) Serial.println("SENSE_AVERAGE"); }
  else if(sensitivityCode == 3){ neuralNet.sensitivity(SENSE_HIGH);       if(debug) Serial.println("SENSE_HIGH"); }
  else if(sensitivityCode == 4){ neuralNet.sensitivity(SENSE_VERY_HIGH);  if(debug) Serial.println("SENSE_VERY_HIGH"); }


*/
    //convert uint32_t to int
 /* char outputString[9];
itoa((int)*page1, outputString, 16);
for(int k=0; k < 9; k++){
  Serial.print(" ");
  Serial.print(outputString[k]);
} */
 //   print_page_data(vpage);
 /* print_counter(vpage[0]);
  print_counter(vpage[1]);
  print_counter(vpage[2]);
  print_counter(vpage[3]);
  
    time_start = micros();
  vpage = VirtualPage.get(MAGIC_COUNTER);
  time_end = micros();
  if (vpage == (uint32_t *)~0) {
    Serial.print("No page found with MAGIC_COUNTER 0x");
    Serial.print(MAGIC_COUNTER, HEX);
    print_time(time_start, time_end);
    Serial.print("Allocate a new page at ");
    time_start = micros();
    // Allocate a new page
    vpage = VirtualPage.allocate(MAGIC_COUNTER);
    time_end = micros();
  } else {
    Serial.print("Found an old page at ");
  } */

/*
  Flash.write(&vpage[0], bleRawArray[0]);
  Flash.write(&vpage[1], bleRawArray[1]);
  Flash.write(&vpage[2], bleRawArray[2]);
  Flash.write(&vpage[3], bleRawArray[3]);
  print_status(vpage[0] == bleRawArray[0]);

  Serial.println("New stored settings: ");
 // print_page_data(vpage);
  print_counter(vpage[0]);    
  print_counter(vpage[1]);
  print_counter(vpage[2]);
  print_counter(vpage[3]);
 // VirtualPage.release(vpage); */
  delay(100);
}


void switchCharacteristicWritten(BLECentral& central, BLECharacteristic& characteristic) {
  // central wrote new value to characteristic, update LED
  Serial.print(F("Characteristic event, written: "));

  if (ReadOnlyArrayGattCharacteristic.value()) {
    if(debug) Serial.println(F("Test on"));
  } else {
    if(debug) Serial.println(F("Test off"));
  }
}

void setupBluetooth(){
  /************ INIT BLUETOOTH BLE instantiate BLE peripheral *********/
   // set advertised local name and service UUID
    blePeripheral.setLocalName("Tingle");
    blePeripheral.setDeviceName("Tingle");
    blePeripheral.setAdvertisedServiceUuid(customService.uuid());
    blePeripheral.setAppearance(0xFFFF);
  
    // add attributes (services, characteristics, descriptors) to peripheral
    blePeripheral.addAttribute(customService);
    
    blePeripheral.addAttribute(ReadOnlyArrayGattCharacteristic);
    blePeripheral.addAttribute(WriteOnlyArrayGattCharacteristic);
    
    blePeripheral.addAttribute(SensorDataCharacteristic); //streaming data for app graph

    blePeripheral.addAttribute(ReadDataCharacteristic); // i/o
    blePeripheral.addAttribute(WriteDataCharacteristic); // i/o
    
    // assign event handlers for connected, disconnected to peripheral
    blePeripheral.setEventHandler(BLEConnected, blePeripheralConnectHandler);
    blePeripheral.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);
  //  blePeripheral.setEventHandler(BLEWritten, blePeripheralServicesDiscoveredHandler);

    // assign event handlers for characteristic
    ReadOnlyArrayGattCharacteristic.setEventHandler(BLEWritten /*BLEValueUpdated*/, bleCharacteristicValueUpdatedHandle);
    WriteOnlyArrayGattCharacteristic.setEventHandler(BLEWritten /*BLEValueUpdated*/, bleCharacteristicValueUpdatedHandle);

    // assign initial values
    char readValue[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    ReadOnlyArrayGattCharacteristic.setValue(0);
    char writeValue[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    WriteOnlyArrayGattCharacteristic.setValue(0);

    // initialize variables to pace updates to correct rate
    microsPerReading = 1000000 / 25;
    microsPrevious = micros();
  
    // begin initialization
    blePeripheral.begin();
}

/********************************************************************************************************/
/************************ SETUP *************************************************************************/
/********************************************************************************************************/

void setup() 
{
    Serial.begin(115200);
    if(debug) Serial.print("STARTING\t");
    delay(50);
 
    // start the I2C library:
    Wire.begin();
    delay(50);

  /************ NEURAL NETWORK CONFIG *******************/
  //  neuralNet.selectTarget(LHAND_SHORT_SIDE);
 //   neuralNet.sensitivity(SENSE_AVERAGE);

  /************ INIT VL53L0X DISTANCE SENSOR *************************/
    Serial.println("VL6180X INIT");
    vl6180x.init();
    delay(500);
  //  Serial.println("VL6180X vl6180x.configureDefault();");
 //   vl6180x.configureDefault();
    Serial.println("VL6180X vl6180x.configureCustom();");
    vl6180x.configureCustom();
    delay(500);
    Serial.println("VL6180X vl6180x.setTimeout(100);");
    vl6180x.setTimeout(100);
    delay(100);
    vl6180x.setScaling(2); //resolution x0.5 , range x2
    delay(100);


  /************ INIT KX126 ACCELEROMETER *****************************/
    Serial.print("KX126 INIT RESPONSE WAS ");
    Serial.println(kx126.init());
    delay(200);


  /************ I/O BUTTON, LED, HAPTIC FEEDBACK *********************/
     //Configure display LED pins  0-->on
    pinMode(GREEN_LED_PIN, OUTPUT); digitalWrite(GREEN_LED_PIN, 1);  

    //Configure Button Pin
    pinMode(BUTTON_PIN, INPUT_PULLUP); 

   //configure haptic feedback pin
    pinMode(VIBRATE_PIN, OUTPUT);  digitalWrite(VIBRATE_PIN, 0);


    /************ CONFIGURE POWER MGMT *******************************/
   // nRF5x_lowPower.enableDCDC();
    nRF5x_lowPower.powerMode(POWER_MODE_LOW_POWER);


    /************ CONFIGURE MACHINE LEARNING FRAMEWORK ***************/
    //fill NN weight array with zeros
    for(int i = 0; i < nnLength; i++){
      F[i] = 99.999;
    }

    
    /************ CONFIGURE & START NVM FLASH MEM ********************/
    nvmFlashSetup();


    /************ CONFIGURE & START BLUETOOTH ************************/
    setupBluetooth();
    if(debug) Serial.println("BLE setup sucess blePeripheral has begun");


    /************ SET TIMERS *****************************************/
    findMinTempTimer = millis();
    
  delay(100);  
}

/********************************************************************************************************/
/************************ LOOP **************************************************************************/
/********************************************************************************************************/

void loop()
{     

   /************************ LOOP SPEED CONTROL ***********************/
if(clocktime + speedMs < millis()){
  
   /*************************** Timestamp ****************************/
    clocktime = millis();
    if(debug){ Serial.println(" "); Serial.print("TIME: "); Serial.print( clocktime/1000 ); Serial.println(" s"); }
    if(debug_time){ Serial.print("Time after init speed limit check: "); Serial.println(millis() - clocktime); }

 

  /******************* Bluetooth App Integration ********************/
    if(IS_DISABLED){
      /*** REENABLE BLE AFTER DISABLED FOR FLASH STORAGE ***/
      NRF_RADIO->TASKS_DISABLE = 0;
      delay(5);
     // sd_ble_gap_adv_start();

      blePeripheral.begin();
      
    //  blePeripheral.begin();
      IS_DISABLED = false;
      delay(20);
    } else { blePeripheral.poll(); }
    
    if(debug_time){ Serial.print("Time after BLE poll: "); Serial.println( (millis() - clocktime))/1000; }

  /*************** BUTTON MGMT *****************************************/
   buttonMGMT();


  /*************** LED MGMT ********************************************/
   ledMGMT();
   if(debug_time){ Serial.print("Time after button & LED: "); Serial.println( (millis() - clocktime))/1000; }


/*************** SLEEP MODE IF NOT CONNECTED AND NO DEBUGGING ********************************
**************** SLEEP MODE IF NOT CONNECTED AND NO DEBUGGING ********************************
**************** SLEEP MODE IF NOT CONNECTED AND NO DEBUGGING ********************************/
if(IS_CONNECTED == false && debug == false){
    /******************* Low Power Mode ********************/
 //   nRF5x_lowPower.powerMode(POWER_MODE_LOW_POWER);
 //   sd_app_evt_wait(); 

} else {

   /************** READ KX126 ACCELEROMETER *****************************/
   sampleAngularPosition();
   if(debug_time){ Serial.print("Time after accelerometer read: "); Serial.println( (millis() - clocktime)) / 1000; }
  
   /************** READ MLX90615 THERMOPILES ****************************/
   sampleThermopiles();
   if(debug_time){ Serial.print("Time after thermo read: "); Serial.println( (millis() - clocktime)) / 1000; }

  
   if( (millis() - findMinTempTimer) > 500){ //update min temp 2Hz
    if(debug){ Serial.print("Trigger update min temps ~ (millis() - findMinTempTimer): "); Serial.println((millis() - findMinTempTimer) ); }
      updateMinTemp();
      findMinTempTimer = millis();
   }
   //initial update
   if(minTempVals[5] < 50) updateMinTemp();

   //normalize thermopile values
   normalizeThermopiles();

   /************** READ VL6180X LIDAR DISTANCE **************************/ 
   sampleLIDAR();   
   if(debug_time){ Serial.print("Time after distance read: "); Serial.println( (millis() - clocktime)) / 1000; }

   /************** TRANSMIT SENSOR DATA OVER BLUETOOTH ******************/ 
   transmitSensorData();
   if(debug_time){ Serial.print("Time after Bluetooth serial send: "); Serial.println( (millis() - clocktime)) / 1000; }


   /************** PRINT SENSOR DATA TO CONSOLE *************************/ 
   if(debug){ printSensorData(); }
   
     
   /************** NEURAL NETWORK GESTURE RECOGNITION *******************/
   float TObj_old_norm[4];
 //  for(int q=0; q < 4; q++){
  //    TObj_old_norm[q] = min( max( TObj[q] , 70) , 101) / 101; //must be above 70 and below 101;
 //  }

   float thermSampAv = ( min( max( TObj[0] , 70) , 101) + min( max( TObj[1] , 70) , 101) + min( max( TObj[2] , 70) , 101) + min( max( TObj[3] , 70) , 101)) / 4;
     //       inputArray2[0] = (currentSample[0] / thermSampAv) - 0.5;
   for(int q=0; q < 4; q++){
      TObj_old_norm[q] = ( min( max( TObj[q] , 70) , 101) / thermSampAv ) - 0.5; //must be above 70 and below 101;
   }

         
   float outputNN5 = neuralNet.detect( (TObj_old_norm[0]), (TObj_old_norm[1]), (TObj_old_norm[1]), (TObj_old_norm[3]), (distance / 250));
   float outputNN7 = neuralNet.detect( (TObj_old_norm[0]), (TObj_old_norm[1]), (TObj_old_norm[1]), (TObj_old_norm[3]), (distance / 250), (pitch / 360), (roll / 360));  
  if(debug){ Serial.print("NN5 Output:\t"); Serial.print(outputNN5); Serial.print("\tNN7 Output:\t"); Serial.print(outputNN7); }

   fiveInScore = outputNN5;
   sevenInScore = outputNN7;

   //use neural network output to decide gesture detection and set vibration alert
   detectAlert();

   //test
  /* if(fiveInScore > 0.9 && sevenInScore > 0.9){
      digitalWrite(VIBRATE_PIN, 1);
   } else {
    digitalWrite(VIBRATE_PIN, 0);
   } */
        

  
    if(debug_time){ Serial.print("TIME LOOP: "); Serial.println(millis() - clocktime); }

//end IS_CONNECTED conditional sleep
 }
 
//end timed loop
 }

  /*********** Power MGMT ******************/
  //https://github.com/sandeepmistry/arduino-nRF5/issues/190
//  uint32_t timeoutTest = 1000;
//   __delay(timeoutTest);
//  sd_power_mode_set(NRF_POWER_MODE_LOWPWR);
//  sd_app_evt_wait();
  
//end infinate loop
} 



/********************************************************************************************************/
/************************ FUNCTIONS *********************************************************************/
/********************************************************************************************************/

/*********************************************************************
*************** READ KX126 ACCELEROMETER *****************************
*********************************************************************/
void sampleAngularPosition(){
    //KX022 ACCELEROMETER I2C
    acc[0] = (float)(kx126.getAccel(0) * 10 );
    acc[1] = (float)(kx126.getAccel(1) * 10 );
    acc[2] = (float)(kx126.getAccel(2) * 10 );
    float eulerX, eulerY, eulerZ;
    
    eulerX = acc[0]; eulerY = acc[1]; eulerZ = acc[2]; 
    pitch = (180/3.141592) * ( atan2( eulerX, sqrt( eulerY * eulerY + eulerZ * eulerZ)) );
    roll = (180/3.141592) * ( atan2(-eulerY, -eulerZ) );

    //adjust for device upward = 180 to prevent crossover 
    pitch = pitch + 180;
    if(roll < -90 ){
      roll = 450 + roll;
      if(roll > 360){roll = roll - 360;}
    } else { roll = roll + 90; }
}

/*********************************************************************
*************** READ MLX90615 THERMOPILES ****************************
*********************************************************************/
void sampleThermopiles(){
    for(int j = 0; j < 4; j++){
        TAmb[j] = readAmbientTempF(j+1); 
        TObj[j] = readObjectTempF(j+1);
    }
    TAmbAv = (TAmb[0] + TAmb[1] + TAmb[2] + TAmb[3]) / 4;
    TAmbMin = min( min( min(TAmb[0] , TAmb[1]), TAmb[2]), TAmb[3]); 
}


/*********************************************************************
*************** UPDATE MINIMUM TEMPERATURE ***************************
*********************************************************************/
void updateMinTemp(){

  int   expirationTimeHeating = 45000; //samples last a max of 45 seconds if temp going up
  int   expirationTimeCooling = 3000; //samples last a max of 3 seconds if temp going down
  float minSampleTemp = min( min( min( min(TObj[0] , TObj[1]), TObj[2]), TObj[3]), TAmbMin);  //lowest temp in current sample
  float maxMinTempVals = max( max( max( max( max( max( max( max( max(minTempVals[0] , minTempVals[1]), minTempVals[2]), minTempVals[3]) , minTempVals[4]), minTempVals[5]), minTempVals[6]) , minTempVals[7]), minTempVals[8]), minTempVals[9]);

  if(debug){ Serial.print("minSampleTemp maxMinTempVals: "); Serial.print(minSampleTemp); Serial.print(" "); Serial.println(maxMinTempVals);}
  
  for(int i = 0; i < 10; i++){
    if(debug){ Serial.print(" "); Serial.print(minTempVals[i]);}
    
      //first ditch old values - assume min ambient as default low temp ceiling
      if( (millis() - minTempTimes[i]) > expirationTimeCooling   &&  (minTempVals[i] > minSampleTemp)  ){ //if new min is lower than stored min - trend cooler
          minTempVals[i] = minSampleTemp;
          minTempTimes[i] = millis();
      } else if( (millis() - minTempTimes[i]) > expirationTimeHeating){ //if new min is higher than stored min
          minTempVals[i] = minSampleTemp;
          minTempTimes[i] = millis();
      }

      //if this is the highest of the stored min temp vals ie the one we might replace + min sample temp is lower and should replace
      if( (minTempVals[i] == maxMinTempVals) && (minTempVals[i] > minSampleTemp) ){
          minTempVals[i] = minSampleTemp;
          minTempTimes[i] = millis();
          break;
      }

      //fill initial values
      if(minTempVals[i] < 50) minTempVals[i] = minSampleTemp;
    
  }
  if(debug) Serial.println(" "); 

  //average stored min temp array to calculate baseline min temp for temp normalization
  float baselineTotal = 0;
  for(int j = 0; j < 10; j++){ baselineTotal = baselineTotal + minTempVals[j]; }
  baselineMinTemp = baselineTotal / 10;
}


/*********************************************************************
*************** GET NORMALIZED THERMOPILE VALUES *********************
*********************************************************************/
void normalizeThermopiles(){
  for(int i=0; i < 4; i++){
      if(TObj[i] <= baselineMinTemp){ TObjNormalized[i] = 1; } //if object temp is lower than baseline min something is weird or unlikely
      else{
        TObjNormalized[i] =  ( min( max( baselineMinTemp , 70.01) , 101) - 70) / ( min( max( TObj[i] , 70.02) , 101) - 70);
      }
  }
}


/*********************************************************************
*************** READ VL6180X LIDAR DISTANCE **************************
*********************************************************************/
void sampleLIDAR(){
    distance = max( 255 - (float)vl6180x.readRangeSingleMillimeters() , 0);
   // if(distance < 0.1){ distance = 0; } // edge case
    if (vl6180x.timeoutOccurred() && debug) { Serial.print(" TIMEOUT"); } 
}

/*********************************************************************
*************** TRANSMIT SENSOR DATA OVER BLUETOOTH ****************** 
*********************************************************************/
void transmitSensorData(){
    BLECentral central = blePeripheral.central();
    
    if(central){ // if a central is connected to peripheral

              /*
               * reduce temperature readings to a range between 0 and 31, then multiply to use up the 256 max decimal value of an 8 bit integer
               * Object temperature floor: 70F
               * Object temperature ceiling: 101F
               */
              float TObj_compressed[4];
              for(int q=0; q < 4; q++){
                  TObj_compressed[q] = ( min( max( TObj[q] , 70) , 101) - 70) * 8; //must be above 70 and below 101;
                //  if(TObj_compressed[q] < 70){ TObj_compressed[q] = 70;  }
                //  else if(TObj_compressed[q] > 101){ TObj_compressed[q] = 101;  }
                //  TObj_compressed[q] = (TObj_compressed[q] - 70)*8;
              }
              
              float TAmbAv_compressed;
              TAmbAv_compressed = ( min( max(TAmbAv, 70) , 101) - 70) * 8; //must be 70 or above
            //  if(TAmbAv < 70){ TAmbAv_compressed = 70;  }
            //  else if(TAmbAv > 101){ TAmbAv_compressed = 101;  }
            //  TAmbAv_compressed = (TAmbAv_compressed - 70)*8;

              int command = 0; //placeholder
              
              //get battery charge value
              batteryValue = ( min( max( analogRead(BATTERY_PIN) , 150) , 200) - 150) *2; //max A read is 1023 but battery is always between 150 and 200

              //OLD BACKWARDS COMPATABLE
              const unsigned char imuCharArray[20] = {
                  (uint8_t)( (roll / 360) * 255),  
                  (uint8_t)( (pitch / 360) * 255),
                  (uint8_t)(distance),
                  (uint8_t)(TObj_compressed[0]),  
                  (uint8_t)(TObj_compressed[1]),
                  (uint8_t)(TObj_compressed[2]),
                  (uint8_t)(TObj_compressed[3]),
                  (uint8_t)(TAmbAv_compressed),
                  (uint8_t)(batteryValue),
                  (uint8_t)(command),
                  (uint8_t)( (acc[0] + 1.00) * 100.00),               
                  (uint8_t)( (acc[1] + 1.00) * 100.00),
                  (uint8_t)( (acc[2] + 1.00) * 100.00),  
                  (uint8_t)(TObjNormalized[0] * 255),
                  (uint8_t)(TObjNormalized[1] * 255),
                  (uint8_t)(TObjNormalized[2] * 255),
                  (uint8_t)(TObjNormalized[3] * 255),
                  (uint8_t)(fiveInScore * 255),
                  (uint8_t)(sevenInScore * 255),  
                  (uint8_t)0                   //empty
              }; 
              //send data over bluetooth
              SensorDataCharacteristic.setValue(imuCharArray,20);
              //time to send
              delay(5);
          }
   if(debug_time){ Serial.print("Time after bluetooth send: "); Serial.println( (millis() - clocktime))/1000; }
}

/*********************************************************************
*************** PRINT SENSOR DATA TO CONSOLE *************************
*********************************************************************/
void printSensorData(){

    //batteryValue = ( (float)analogRead(BATTERY_PIN) / 1022) * 100; //max A read is 1023
    batteryValue = ( min( max( analogRead(BATTERY_PIN) , 150) , 200) - 150) *2; //max A read is 1023 but battery is always between 150 and 200
  //  batteryValue = analogRead(BATTERY_PIN);
  
    Serial.print("OT1: "); Serial.print( TObj[0] ); Serial.print("\t"); 
    Serial.print("OT2: "); Serial.print( TObj[1] ); Serial.print("\t");
    Serial.print("OT3: "); Serial.print( TObj[2] ); Serial.print("\t");
    Serial.print("OT4: "); Serial.print( TObj[3] ); Serial.print("\t");
    Serial.print("AT1: "); Serial.print( TAmb[0] ); Serial.print("\t"); 
    Serial.print("AT2: "); Serial.print( TAmb[1] ); Serial.print("\t");
    Serial.print("AT3: "); Serial.print( TAmb[2] ); Serial.print("\t");
    Serial.print("AT4: "); Serial.print( TAmb[3] ); Serial.print("\t");
    Serial.print("DTav: "); Serial.print( TAmbAv ); Serial.print("\t");
    Serial.print("Baseline: "); Serial.print( baselineMinTemp ); Serial.println("");

    Serial.print("OT Normalized: "); Serial.print( TObjNormalized[0] ); Serial.print("\t"); Serial.print( TObjNormalized[1] ); Serial.print("\t"); Serial.print( TObjNormalized[2] ); Serial.print("\t"); Serial.println( TObjNormalized[3] ); 
    
    Serial.print("pitch: "); Serial.print( pitch ); Serial.print("\t"); 
    Serial.print("roll: "); Serial.print( roll ); Serial.print("\t"); 
    Serial.print("accX: "); Serial.print( acc[0] ); Serial.print("\t"); 
    Serial.print("accY: "); Serial.print( acc[1] ); Serial.print("\t"); 
    Serial.print("accZ: "); Serial.print( acc[2] ); Serial.println(""); 
    
    Serial.print("Distance (mm): "); Serial.println(distance); 

    Serial.print("CMD: "); Serial.println(command_value);
    Serial.print("NN5: "); Serial.print(fiveInScore); Serial.print("  NN7: "); Serial.print(sevenInScore);
    Serial.print("  Battery: "); Serial.println(batteryValue);
}


/*********************************************************************
**************** BUTTON MGMT *****************************************
*********************************************************************/
void buttonMGMT(){
    
    int lastButtonState = buttonState;
    float pressTime = 0;
    
    // read the state of the pushbutton value:
    buttonState = digitalRead(BUTTON_PIN);
    
    if (buttonState == 1) {
      
      //  if(buttonBeginPressTime != 0){ buttonBeginPressTime = millis(); }
        if(lastButtonState == 0){ buttonBeginPressTime = millis(); }
        pressTime = (buttonBeginPressTime - millis() ) / 1000; 
      
      // turn LED on:
     // LED_counter = 80;
     //   digitalWrite(GREEN_LED_PIN, 1);
     //   delay(500);
     //   digitalWrite(GREEN_LED_PIN, 0);
    //  greenLED_status = true;
    } else {
      
        if(buttonBeginPressTime != 0){ 
            pressTime = (buttonBeginPressTime - millis() ) / 1000;
            buttonBeginPressTime = 0; 
        }
        

    }
    if (debug) { Serial.print("BUTTON: "); Serial.print(buttonState); Serial.print("  Press time: "); Serial.print(pressTime); Serial.println("s"); } 

    if(pressTime < 0){ buttonBeginPressTime = 0; }
    
    if(pressTime > 3){ 
        if(!sleepLight){ sleepLight = true; }
        if(sleepLight){ sleepLight = false; }
        LED_counter = 3;
        Serial.println("****LIGHT SLEEP****");
    }
    
    if(pressTime > 8){ sleepDeep = true; if(debug){ Serial.println("DEEP SLEEP"); } delay(2000); }
}


/*********************************************************************
**************** LED MGMT ********************************************
*********************************************************************/
void ledMGMT(){
   //example blink program
   if(LED_counter > 0 && greenLED_status == false ){
      LED_counter--;
      if(LED_counter <= 0){
          LED_counter = 10;
          digitalWrite(GREEN_LED_PIN, 1);
          greenLED_status = true;
      }
   }
   else if(LED_counter > 0 && greenLED_status == true){
      LED_counter--;
      if(LED_counter <= 0){
          LED_counter = 2;
          digitalWrite(GREEN_LED_PIN, 0);
          greenLED_status = false;
      }
   }  
}


/*********************************************************************
*************** NEURAL NETWORK GESTURE RECOGNITION *******************
*********************************************************************/
void detectGesture(int selectNN, int setting, float t1, float t2, float t3, float t4, float distance, float pitch, float roll){
  float fivePrediction = 0; float sevenPrediction = 0;

  //normalize
  t1 = t1 / 101;
  t2 = t2 / 101;
  t3 = t3 / 101;
  t4 = t4 / 101;
  distance = distance / 250;
  pitch = pitch / 360;
  roll = roll / 360;
  
// if(debug){ Serial.print("in detectGesture selectNN: "); Serial.print(selectNN); Serial.print("  "); Serial.print(t1); Serial.print("  "); Serial.print(distance); Serial.print("  "); Serial.println(pitch); }
  if(selectNN == 1){        //mouth --> left hand
    //  fivePrediction = nn_lhand_mouth_5521(t1, t2, t3, t4, distance);
   //   if(fivePrediction > 50){ sevenPrediction = nn_lhand_mouth_7521(t1, t2, t3, t4, distance, pitch, roll); }
  } else if(selectNN == 2){ //front head --> left hand
   
  } else if(selectNN == 3){ //top head --> left hand
    
  } else if(selectNN == 4){ //back head --> left hand
    
  } else if(selectNN == 5){ //right head --> left hand
    
  } else if(selectNN == 6){ //left head --> left hand
//      fivePrediction = nn_lhand_side_5521(t1, t2, t3, t4, distance);
 //     if(fivePrediction > 50){ sevenPrediction = nn_lhand_side_7521(t1, t2, t3, t4, distance, pitch, roll); }
  } else if(selectNN == 7){
    
  }
  //for display
  fiveInScore = fivePrediction;
  sevenInScore = sevenPrediction;

  //detection algo
  if(fivePrediction > 95 && sevenPrediction > 50 || fivePrediction > 50 && sevenPrediction > 95){ flag_detect = true; } else {flag_detect = false;}
  //haptic feedback duration
  if(flag_detect){ vibrate_counter = 3; }
}

/*********************************************************************
*************** TRANSMIT STORED SETTINGS OVER BLUETOOTH ************** 
*********************************************************************/
void transmitSettings(){
    BLECentral central = blePeripheral.central();
    
    if(central){ // if a central is connected to peripheral
        
        //OLD BACKWARDS COMPATABLE
        const unsigned char settingsCharArray[20] = {
            (uint8_t)(settingsData[0]),  
            (uint8_t)(settingsData[1]), 
            (uint8_t)(settingsData[2]), 
            (uint8_t)(settingsData[3]),  
            (uint8_t)(settingsData[4]), 
            (uint8_t)(settingsData[5]), 
            (uint8_t)(settingsData[6]), 
            (uint8_t)(settingsData[7]), 
            (uint8_t)(settingsData[8]), 
            (uint8_t)(settingsData[9]), 
            (uint8_t)(settingsData[10]),              
            (uint8_t)(settingsData[11]), 
            (uint8_t)(settingsData[12]), 
            (uint8_t)(settingsData[13]), 
            (uint8_t)(settingsData[14]), 
            (uint8_t)(settingsData[15]), 
            (uint8_t)(settingsData[16]), 
            (uint8_t)(settingsData[17]), 
            (uint8_t)(settingsData[18]),   
            (uint8_t)(settingsData[19])                    //empty
        }; 
        //send data over bluetooth
        ReadOnlyArrayGattCharacteristic.setValue(settingsCharArray,20);
        //time to send
        delay(5);
    }
    Serial.println("Settings sent");
   if(debug_time){ Serial.print("Time after settings send: "); Serial.println( (millis() - clocktime))/1000; }
}

/*********************************************************************
*************** ETC **************************************************
*********************************************************************/

void __delay(uint32_t timeout)
{
  uint32_t start;
  start = millis();

 do
 {
   __WFE();
 }
   while ((millis() - start) >= timeout);
}

int hex_to_int(char c){
  int first;
  int second;
  int value;
  
  if (c >= 97) {
    c -= 32;
  }
  first = c / 16 - 3;
  second = c % 16;
  value = first * 10 + second;
  if (value > 9) {
    value--;
  }
  return value;
}

int hex_to_ascii(char c, char d){
  int high = hex_to_int(c) * 16;
  int low = hex_to_int(d);
  return high+low;
}

void changeNeuralNetwork(int rawNewSettings){

    Serial.print(" ");
  Serial.println(rawNewSettings);

  
  handCode    = rawNewSettings / 1000; 
  hairCode    = (rawNewSettings / 100) - (handCode * 10); 
  targetsCode = (rawNewSettings / 10) - (handCode * 100) - (hairCode * 10);
  sensitivityCode = rawNewSettings - (handCode * 1000) - (hairCode * 100) - (targetsCode * 10);

  if(debug){
      Serial.print("handCode: "); Serial.print(handCode);
  }

  //left hand
  if(handCode == 1){

      //short hair
      if(hairCode == 1){

          //front head
          if(targetsCode == 1){
              neuralNet.selectTarget(LHAND_SHORT_FRONT); 
              if(debug) Serial.println("LHAND_SHORT_FRONT");
          }
          //side head
          else if(targetsCode == 2){
              neuralNet.selectTarget(LHAND_SHORT_SIDE); 
              if(debug) Serial.println("LHAND_SHORT_SIDE");
          }
          //back head
          else if(targetsCode == 3){
              neuralNet.selectTarget(LHAND_SHORT_BACK); 
              if(debug) Serial.println("LHAND_SHORT_BACK");
          }
          //top head
          else if(targetsCode == 4){
              neuralNet.selectTarget(LHAND_SHORT_TOP); 
              if(debug) Serial.println("LHAND_SHORT_TOP");
          }
          //mouth head
          else if(targetsCode == 5){
              neuralNet.selectTarget(LHAND_SHORT_MOUTH); 
              if(debug) Serial.println("LHAND_SHORT_MOUTH");
          }
          //eyes head
          else if(targetsCode == 6){
              neuralNet.selectTarget(LHAND_SHORT_EYES); 
              if(debug) Serial.println("LHAND_SHORT_EYES");
          }
      }
  }
    //right hand
  else if(handCode == 2){

      //short hair
      if(hairCode == 1){

          //front head
          if(targetsCode == 1){
              neuralNet.selectTarget(RHAND_SHORT_FRONT); 
              if(debug) Serial.println("RHAND_SHORT_FRONT");
          }
          //side head
          else if(targetsCode == 2){
              neuralNet.selectTarget(RHAND_SHORT_SIDE); 
              if(debug) Serial.println("RHAND_SHORT_SIDE");
          }
          //back head
          else if(targetsCode == 3){
              neuralNet.selectTarget(RHAND_SHORT_BACK); 
              if(debug) Serial.println("RHAND_SHORT_BACK");
          }
          //top head
          else if(targetsCode == 4){
              neuralNet.selectTarget(RHAND_SHORT_TOP); 
              if(debug) Serial.println("RHAND_SHORT_TOP");
          }
          //mouth head
          else if(targetsCode == 5){
              neuralNet.selectTarget(RHAND_SHORT_MOUTH); 
              if(debug) Serial.println("RHAND_SHORT_MOUTH");
          }
          //eyes head
          else if(targetsCode == 6){
              neuralNet.selectTarget(RHAND_SHORT_EYES); 
              if(debug) Serial.println("RHAND_SHORT_EYES");
          }
      }
  }

  //set new sensitivity
  if(sensitivityCode == 0){ neuralNet.sensitivity(SENSE_VERY_LOW);        if(debug) Serial.println("SENSE_VERY_LOW"); }
  else if(sensitivityCode == 1){ neuralNet.sensitivity(SENSE_LOW);        if(debug) Serial.println("SENSE_LOW"); }
  else if(sensitivityCode == 2){ neuralNet.sensitivity(SENSE_AVERAGE);    if(debug) Serial.println("SENSE_AVERAGE"); }
  else if(sensitivityCode == 3){ neuralNet.sensitivity(SENSE_HIGH);       if(debug) Serial.println("SENSE_HIGH"); }
  else if(sensitivityCode == 4){ neuralNet.sensitivity(SENSE_VERY_HIGH);  if(debug) Serial.println("SENSE_VERY_HIGH"); }
}

void detectAlert(){
    int vibrateDuration = 1000;   //millisecond duration
    int alertInterval = 3000;     //min interval between alerts

    //TIMING
    if(vibrateTimer == 0){
      //do nothing
    } else if(vibrateTimer > (millis() - vibrateDuration)){
      //vibrate on
      digitalWrite(VIBRATE_PIN, 1);
      
    } else {
      //vibrate off
      digitalWrite(VIBRATE_PIN, 0);
      vibrateTimer = 0;
      
      lastAlertTimer = millis();
    }

    //DETECTION
    if( (lastAlertTimer < (millis() - alertInterval)) && (vibrateTimer < (millis() - vibrateDuration))){   //use sensitivityCode
    
   /*    if(fiveInScore > 0.9 && sevenInScore > 0.9){
           vibrateTimer = millis();
           lastAlertTimer = millis();
       } */

/********* DISABLE STUPID BUZZING ***********/
/*
       if(fiveInScore > 0.7 && sevenInScore > 0.7 && sensitivityCode == 4){ //SENSE_VERY_HIGH
           vibrateTimer = millis();
           lastAlertTimer = millis();
       } else if(fiveInScore > 0.85 && sevenInScore > 0.85 && sensitivityCode == 3){ //SENSE_HIGH
           vibrateTimer = millis();
           lastAlertTimer = millis();
       } else if(fiveInScore > 0.9 && sevenInScore > 0.9 && sensitivityCode == 2){ //SENSE_AVERAGE
           vibrateTimer = millis();
           lastAlertTimer = millis();
       } else if(fiveInScore > 0.95 && sevenInScore > 0.95 && sensitivityCode == 1){ //SENSE_LOW
           vibrateTimer = millis();
           lastAlertTimer = millis();
       } else if(fiveInScore > 0.98 && sevenInScore > 0.98 && sensitivityCode == 0){ //SENSE_VERY_LOW
           vibrateTimer = millis();
           lastAlertTimer = millis();
       }
       */
       /************* END DISABLE STUPID BUZZING ***************/
    }
    
}



