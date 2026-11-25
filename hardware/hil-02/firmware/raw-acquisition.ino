// HIL-02 raw passive analog acquisition. Arduino-compatible ESP32-C3.
// GPIO assignments are provisional; verify board pinout and input voltage.
// Do not send BENCH/1 until all required channels are measured/calibrated.
// No actuator, serial command parser, motor output or network.
#include <Arduino.h>
constexpr int ANGLE_RAW_PIN=0;
constexpr int SUCTION_SWITCH_PIN=3;
uint32_t seq=0,lastMs=0;
void setup(){
 Serial.begin(115200);
 pinMode(ANGLE_RAW_PIN,INPUT);
 pinMode(SUCTION_SWITCH_PIN,INPUT_PULLUP);
 analogReadResolution(12);
}
void loop(){
 const uint32_t now=millis();
 if(uint32_t(now-lastMs)<20)return;
 lastMs=now;
 const int raw=analogRead(ANGLE_RAW_PIN);
 const int suction=digitalRead(SUCTION_SWITCH_PIN)==LOW?1:0;
 Serial.printf("{\"protocol\":\"PULMOSIMU-RAW/0.1\",\"seq\":%lu,\"tMs\":%lu,\"angleAdc\":%d,\"suctionSwitch\":%d,\"calibrated\":false}\n",(unsigned long)seq++,(unsigned long)now,raw,suction);
}
