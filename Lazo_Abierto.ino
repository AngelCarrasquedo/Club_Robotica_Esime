int POT_sp = 0;
float sp;
int PWM_salida=9;
float pv; 

int pinA =3;
volatile int contador =0;
long interval =100;
unsigned long previousMillis=0;

void setup() {
pinMode(pinA,INPUT);
pinMode(PWM_salida,OUTPUT);
Serial.begin(115200);
attachInterrupt(1,interrupcion,RISING);
}

void loop() {
unsigned long currentMillis = millis();
if ((currentMillis-previousMillis)>=interval){
  previousMillis=currentMillis;

  pv=10*contador*(60.0/900.0);
  contador=0;
  }

sp = analogRead(POT_sp)*(100.0/1023);
analogWrite(PWM_salida,sp*(255.0/100.0));

Serial.print("sp: ");
Serial.print(sp);
Serial.print(" pv: ");
Serial.println(pv);

  
}

void interrupcion (){
  contador++;
  }
