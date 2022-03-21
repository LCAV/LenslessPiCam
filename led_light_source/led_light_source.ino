// Heavily inspired by https://www.engineersgarage.com/fading-led-with-potentiometer-using-arduino/
int sensor=A0;  //Analog 0 pin named as sensor
int output=9;   //Pin-9 is declared as output

void setup(){
    pinMode(output, OUTPUT); //Pin-9 is declared as output
    Serial.begin(9600); //initialize serial communication at 9600 bits per second
}

void loop(){
    int reading=analogRead(sensor);  // Reading the voltage out by potentiometer
    int bright=reading / 4;          // Dividing reading by 4 to bring it in range of 0 - 255
    analogWrite(output, bright);     // Finally outputting the read value on pin-9 fading led
    Serial.println(bright);          // For debugging purposes
}  
