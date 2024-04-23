int pinA =18;
volatile int contador =0;
unsigned interval =1000;
int A=4;
int B=3;
int PWM=2;

void setup() {
pinMode(A,OUTPUT);
pinMode(B,OUTPUT);
pinMode(PWM,OUTPUT);

pinMode(pinA,INPUT);
Serial.begin(115200);
attachInterrupt(5,interrupcion,RISING);
}

void loop() {
Serial.print("pulsos: ");
Serial.println(contador);
digitalWrite(A,HIGH);
digitalWrite(B,LOW);
analogWrite(PWM,255);  

}

void interrupcion (){
  contador++;
  }
