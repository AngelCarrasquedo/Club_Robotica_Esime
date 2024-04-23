// pines de canales Radio
int pinCanal1 = 27; //
int pinCanal2 = 29; // 
int pinCanal3 = 31; // 
int pinCanal4 = 33; // Direccion brazo
int pinCanal5 = 35; // Auxiliar direccion
int pinCanal6 = 37; // Auxiliar velocidad

void setup() {
Serial.begin(9600); // Inicia la comunicación serial

}

void loop() {
int valorCanal1 = pulseIn(pinCanal1, HIGH);
int valorCanal2 = pulseIn(pinCanal2, HIGH);
int valorCanal3 = pulseIn(pinCanal3, HIGH);
int valorCanal4 = pulseIn(pinCanal4, HIGH);
int valorCanal5 = pulseIn(pinCanal5, HIGH);
int valorCanal6 = pulseIn(pinCanal6, HIGH);
//Serial.println(valorCanal6);
Serial.print("Canal: ");  //yoistik derecho lados
Serial.print(valorCanal1);
Serial.print("Cana2: ");    //yoistik derecho arriba abajo 
Serial.print(valorCanal2);
Serial.print("Cana3: ");
Serial.print(valorCanal3);  //yoistik izquierdo arriba abajo 
Serial.print("Cana4: ");
Serial.print(valorCanal4);//yoistik izquierdo Lados
Serial.print("Cana5: ");
Serial.print(valorCanal5);//Canal auxiliar
Serial.print("Cana6: ");
Serial.print(valorCanal6);

Serial.println("  ");
delay(500);
}
