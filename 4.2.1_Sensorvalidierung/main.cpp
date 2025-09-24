#include <Arduino.h>

const int senPin = 34;

void setup() {
  Serial.begin(9600);
  delay(1000);
}

void loop() {
  senValue = analogRead(senPin);
  Serial.println(senValue);
  delay(5);
}
