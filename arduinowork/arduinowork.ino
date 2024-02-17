const int vibrating_fb_pin = 8;

void setup(){
    Serial.begin(9600);
}

void loop(){
    if (Serial.available() > 0){
        String input = Serial.readString();
        Serial.print("Recieved Value: ");
        Serial.println(input);
        int ival = input.toInt();

        // Activate user
        vibration_fb(ival);
        matrixled_fb(ival);
    }
}

void vibration_fb(double distance){
    digitalWrite(vibrating_fb_pin, HIGH);
    delay((int)(1/distance));
    digitalWrite(vibrating_fb_pin, LOW);
    delay((int)(1/distance));
}

void matrixled_fb(double distance){

}

