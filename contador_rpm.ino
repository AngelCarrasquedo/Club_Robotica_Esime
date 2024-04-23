int pinA =20;     //Pines Motor trasero izquierdo Sensor 18/5 A=4 B=3 PWM=2
int A=11;
int B=12;
int PWM=13;
volatile int contador =0;
long interval =1000;
unsigned long previousMillis=0;

void setup() {
  pinMode(A,OUTPUT);
pinMode(B,OUTPUT);
pinMode(PWM,OUTPUT);

pinMode(pinA,INPUT);
Serial.begin(115200);
attachInterrupt(3,interrupcion,RISING);
}

void loop() {
digitalWrite(A,HIGH);
digitalWrite(B,LOW);
analogWrite(PWM,255);  

unsigned long currentMillis = millis();
if ((currentMillis-previousMillis)>=interval){
  previousMillis=currentMillis;
  Serial.print("pulsos/seg: ");
  Serial.println(contador);
  contador=0;
  }
  
}

void interrupcion (){
  contador++;
  }
