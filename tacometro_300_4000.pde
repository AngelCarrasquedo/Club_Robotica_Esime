import processing.serial.*;  // Importar la librería serial

Serial myPort;  // Objeto serial
int rpm1 = 0;  // Velocidad actual del primer motor en RPM
int rpm2 = 0;  // Velocidad actual del segundo motor en RPM
int maxRPM = 4000;  // RPM máximo
int minRPM = 0;  // RPM mínimo
int buttonX, buttonY, buttonSize;

void setup() {
  fullScreen();  // Ejecutar en pantalla completa
  println(Serial.list());  // Imprimir la lista de puertos seriales disponibles
  
  // Selecciona el puerto correcto manualmente
  // Cambia el índice a 0, 1, 2, etc., según el puerto que corresponda al Arduino
  String portName = Serial.list()[0];  // Ajusta según sea necesario
  
  // Inicializar el puerto serial
  try {
    myPort = new Serial(this, portName, 9600);
  } catch (Exception e) {
    println("Error opening serial port " + portName + ": " + e);
    exit();
  }
  
  background(0);  // Fondo negro
  
  // Posición y tamaño del botón de cierre
  buttonSize = 50;
  buttonX = width - buttonSize - 20;
  buttonY = 20;
}

void draw() {
  while (myPort != null && myPort.available() > 0) {
    String inString = myPort.readStringUntil('\n');  // Leer el string del puerto serial
    if (inString != null) {
      inString = trim(inString);  // Eliminar espacios en blanco
      String[] values = split(inString, ',');  // Dividir el string en dos valores
      if (values.length == 2) {
        int inByte1 = int(values[0]);  // Convertir el primer string a un entero
        int inByte2 = int(values[1]);  // Convertir el segundo string a un entero
        rpm1 = constrain(inByte1, minRPM, maxRPM);  // Asegurarse de que el primer valor esté dentro del rango
        rpm2 = constrain(inByte2, minRPM, maxRPM);  // Asegurarse de que el segundo valor esté dentro del rango
        
        // Imprimir los valores en la consola
        println("RPM1: " + rpm1 + ", RPM2: " + rpm2);
      } else {
        println("Error parsing: " + inString);
      }
    }
  }
  
  background(0);  // Limpiar fondo en cada cuadro
  
  textSize(48);
  fill(255);  // Color blanco para el título
  textAlign(CENTER, CENTER);
  text("Tacómetro motor DC", width / 2, 60);  // Título
  
  drawTachometer(width / 4, height / 2, rpm1, "RPM");  // Dibujar el primer tacómetro
  drawTachometer(3 * width / 4, height / 2, rpm2, "Set Point");  // Dibujar el segundo tacómetro con el texto "Set Point"
  
  displayValue(width / 4, int(height * 0.9), rpm1, "RPM");  // Mostrar el valor de RPM del primer motor
  displayValue(3 * width / 4, int(height * 0.9), rpm2, "");  // No mostrar el valor del segundo motor
  
  // Dibujar el botón de cierre
  drawCloseButton(buttonX, buttonY, buttonSize);
}

// Función para dibujar el tacómetro
void drawTachometer(int centerX, int centerY, int value, String label) {
  float diameter = min(width, height) * 0.6;

  // Dibujar el círculo del tacómetro
  stroke(255);  // Color blanco para el círculo
  fill(0);  // Fondo negro
  ellipse(centerX, centerY, diameter, diameter);

  // Dibujar las marcas y la numeración del tacómetro
  textSize(14);
  textAlign(CENTER, CENTER);
  fill(255, 0, 0);  // Color rojo para la numeración
  for (int i = minRPM; i <= maxRPM; i += 500) { // Cambio de intervalo
    float angle = map(i, minRPM, maxRPM, -PI * 3 / 4, PI * 3 / 4);
    float x1 = centerX + cos(angle) * diameter * 0.4;
    float y1 = centerY + sin(angle) * diameter * 0.4;
    float x2 = centerX + cos(angle) * diameter * 0.45;
    float y2 = centerY + sin(angle) * diameter * 0.45;
    line(x1, y1, x2, y2);
    // Dibujar la numeración más abajo
    float xText = centerX + cos(angle) * diameter * 0.55;
    float yText = centerY + sin(angle) * diameter * 0.55;
    text(i, xText, yText + 5); // Ajuste de posición hacia abajo
  }
  
  // Dibujar la aguja del tacómetro
  float angle = map(value, minRPM, maxRPM, -PI * 3 / 4, PI * 3 / 4);
  stroke(255, 0, 0);  // Color rojo para la aguja
  strokeWeight(3);
  float x = centerX + cos(angle) * diameter * 0.4;
  float y = centerY + sin(angle) * diameter * 0.4;
  line(centerX, centerY, x, y);
  
  // Mostrar el texto del tacómetro
  fill(255);  // Color blanco para el texto del tacómetro
  textSize(24);
  text(label, centerX, centerY + diameter * 0.35);
}

// Función para mostrar el valor actual de RPM o Set Point
void displayValue(int centerX, int centerY, int value, String label) {
  fill(255);  // Color blanco para el texto
  textSize(24);
  textAlign(CENTER, CENTER);
  text(value + " " + label, centerX, centerY);
}

// Función para dibujar el botón de cierre
void drawCloseButton(int x, int y, int size) {
  // Botón
  fill(255, 0, 0); // Color rojo
  ellipse(x + size / 2, y + size / 2, size, size);
  
  // Cruz
  stroke(255); // Color blanco
  strokeWeight(4);
  line(x + size / 4, y +size / 4, x + 3 * size / 4, y + 3 * size / 4);
  line(x + size / 4, y + 3 * size / 4, x + 3 * size / 4, y + size / 4);
}

void mousePressed() {
  // Verificar si se hizo clic en el botón de cierre
  if (mouseX > buttonX && mouseX < buttonX + buttonSize && mouseY > buttonY && mouseY < buttonY + buttonSize) {
    exit(); // Cerrar la aplicación
  }
}
