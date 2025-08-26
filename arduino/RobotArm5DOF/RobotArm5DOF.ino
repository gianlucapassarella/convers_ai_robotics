// === RobotArm5DOF.ino ===
// Braccio 5 DOF (base, spalla, gomito, polso, pinza) controllato via USB-seriale (JSONL).
// Comandi supportati (JSON su singola riga, \n finale):
//  - {"cmd":"HOME"}
//  - {"cmd":"WAVE"}
//  - {"cmd":"OPEN"}          // apri pinza
//  - {"cmd":"CLOSE"}         // chiudi pinza (prendi)
//  - {"cmd":"MOVE","dir":"LEFT"|"RIGHT"|"UP"|"DOWN"}   // movimenti semplici
//  - {"cmd":"PICK_PLACE","from":"table","to":"bin"}    // demo pick&place
//  - {"cmd":"DRAW","char":"A"|"X"}                     // disegno semplificato
//  - {"cmd":"CIRCLE"}                                   // cerchio in aria
//  - {"cmd":"STOP"}                                     // arresto sequenza

#include <Arduino.h>
#include <Servo.h>

const uint8_t NUM_JOINTS = 5;                             // base, spalla, gomito, polso, pinza
const uint8_t SERVO_PINS[NUM_JOINTS] = {3, 5, 6, 9, 10};  // adatta ai tuoi collegamenti
Servo joints[NUM_JOINTS];

// Limiti di sicurezza (gradi) — TARARE!
int MIN_ANG[NUM_JOINTS] = {  0,  10,  10,   0,   0};
int MAX_ANG[NUM_JOINTS] = {180, 170, 170, 180,  90};

// Posizione HOME — TARARE!
int HOME_POS[NUM_JOINTS] = {90, 90, 90, 90, 20}; // pinza semi-aperta

// Sequencer non bloccante
struct Keyframe { int a[NUM_JOINTS]; uint16_t ms; };
const uint8_t MAX_STEPS = 16;
struct Sequence { Keyframe k[MAX_STEPS]; uint8_t n; };

bool seqActive=false, abortSeq=false;
Sequence seq; uint8_t stepIdx=0;
unsigned long t0=0; uint16_t stepMs=0;
int cur[NUM_JOINTS], startA[NUM_JOINTS], targetA[NUM_JOINTS];

int clampInt(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
void writeAll(const int *a){ for(uint8_t i=0;i<NUM_JOINTS;i++) joints[i].write(clampInt(a[i],MIN_ANG[i],MAX_ANG[i])); }

void loadStep(uint8_t i){
  for(uint8_t j=0;j<NUM_JOINTS;j++){ startA[j]=cur[j]; targetA[j]=clampInt(seq.k[i].a[j],MIN_ANG[j],MAX_ANG[j]); }
  stepMs=seq.k[i].ms; t0=millis();
}
void endSequence(const char* st="done"){
  seqActive=false; abortSeq=false;
  Serial.print("{\"status\":\""); Serial.print(st); Serial.println("\"}");
}
void updateSeq(){
  if(!seqActive) return;
  if(abortSeq){ endSequence("error"); return; }

  unsigned long dt=millis()-t0;
  float u = stepMs? min(1.0f, dt/(float)stepMs) : 1.0f;
  for(uint8_t i=0;i<NUM_JOINTS;i++) cur[i]= (int)(startA[i] + (targetA[i]-startA[i])*u);
  writeAll(cur);
  if(dt>=stepMs){
    stepIdx++;
    if(stepIdx>=seq.n){ endSequence("done"); }
    else loadStep(stepIdx);
  }
}

Sequence makeHOME(){ Sequence s; s.n=1; for(uint8_t i=0;i<NUM_JOINTS;i++) s.k[0].a[i]=HOME_POS[i]; s.k[0].ms=900; return s; }

Sequence makeWAVE(){
  Sequence s; s.n=0;
  Keyframe k0; memcpy(k0.a, HOME_POS, sizeof(HOME_POS)); k0.ms=700; s.k[s.n++]=k0;
  Keyframe k1=k0; k1.a[1]=65; k1.a[2]=110; k1.ms=600; s.k[s.n++]=k1; // alza braccio
  for(int i=0;i<3 && s.n+2<MAX_STEPS;i++){
    Keyframe ka=k1; ka.a[3]=60;  ka.ms=250; s.k[s.n++]=ka;
    Keyframe kb=k1; kb.a[3]=120; kb.ms=250; s.k[s.n++]=kb;
  }
  Keyframe kend; memcpy(kend.a, HOME_POS, sizeof(HOME_POS)); kend.ms=700; s.k[s.n++]=kend;
  return s;
}

Sequence makeOPEN(){
  Sequence s; s.n=0;
  Keyframe k; memcpy(k.a, cur, sizeof(cur)); k.a[4]=40; k.ms=300; s.k[s.n++]=k;
  return s;
}
Sequence makeCLOSE(){
  Sequence s; s.n=0;
  Keyframe k; memcpy(k.a, cur, sizeof(cur)); k.a[4]=0; k.ms=300; s.k[s.n++]=k;
  return s;
}

Sequence makeMOVE(const String &dir){
  Sequence s; s.n=0; Keyframe k; memcpy(k.a, cur, sizeof(cur)); k.ms=500;
  if(dir=="LEFT"){   k.a[0]=clampInt(cur[0]-20, MIN_ANG[0], MAX_ANG[0]); }
  if(dir=="RIGHT"){  k.a[0]=clampInt(cur[0]+20, MIN_ANG[0], MAX_ANG[0]); }
  if(dir=="UP"){     k.a[1]=clampInt(cur[1]-15, MIN_ANG[1], MAX_ANG[1]); k.a[2]=clampInt(cur[2]-10, MIN_ANG[2], MAX_ANG[2]); }
  if(dir=="DOWN"){   k.a[1]=clampInt(cur[1]+15, MIN_ANG[1], MAX_ANG[1]); k.a[2]=clampInt(cur[2]+10, MIN_ANG[2], MAX_ANG[2]); }
  s.k[s.n++]=k; return s;
}

Sequence makePICK_PLACE(const String &from, const String &to){
  int baseFrom = (from=="table")? 60 : 90;
  int baseTo   = (to  =="bin")  ?120 : 90;
  Sequence s; s.n=0; Keyframe k;
  memcpy(k.a, HOME_POS, sizeof(HOME_POS)); k.ms=700; s.k[s.n++]=k;
  k.a[0]=baseFrom; k.a[1]=70; k.a[2]=120; k.a[4]=30; k.ms=800; s.k[s.n++]=k; // sopra presa
  k.a[2]=135; k.ms=450; s.k[s.n++]=k;                                        // scendi
  k.a[4]=0;   k.ms=300; s.k[s.n++]=k;                                        // chiudi pinza
  k.a[2]=105; k.ms=500; s.k[s.n++]=k;                                        // solleva
  k.a[0]=baseTo; k.ms=800; s.k[s.n++]=k;                                     // ruota destinazione
  k.a[2]=135; k.ms=450; s.k[s.n++]=k;                                        // scendi
  k.a[4]=30;  k.ms=300; s.k[s.n++]=k;                                        // apri pinza
  k.a[2]=105; k.ms=500; s.k[s.n++]=k;                                        // risali
  memcpy(k.a, HOME_POS, sizeof(HOME_POS)); k.ms=900; s.k[s.n++]=k;           // HOME
  return s;
}

Sequence makeDRAW_CHAR(char c){
  Sequence s; s.n=0; Keyframe k; memcpy(k.a, HOME_POS, sizeof(HOME_POS)); k.ms=600; s.k[s.n++]=k;
  if(c=='A' || c=='a'){
    k.a[1]=80; k.a[2]=120; k.a[4]=5; k.ms=400; s.k[s.n++]=k;  // abbassa penna
    k.a[0]=80;  k.a[1]=85; k.a[2]=125; k.ms=400; s.k[s.n++]=k; // obliquo 1
    k.a[0]=100; k.a[1]=85; k.a[2]=125; k.ms=400; s.k[s.n++]=k; // obliquo 2
    k.a[0]=90;  k.a[1]=90; k.a[2]=120; k.ms=400; s.k[s.n++]=k; // traversa
    k.a[2]=105; k.ms=350; s.k[s.n++]=k;                        // alza
  } else if(c=='X' || c=='x'){
    k.a[1]=82; k.a[2]=122; k.a[4]=5; k.ms=350; s.k[s.n++]=k;  // abbassa
    k.a[0]=80;  k.a[1]=86; k.a[2]=126; k.ms=350; s.k[s.n++]=k; // diagonale 1
    k.a[0]=100; k.a[1]=86; k.a[2]=126; k.ms=350; s.k[s.n++]=k; // diagonale 2
    k.a[2]=105; k.ms=350; s.k[s.n++]=k;                        // alza
  }
  memcpy(k.a, HOME_POS, sizeof(HOME_POS)); k.ms=600; s.k[s.n++]=k;
  return s;
}

Sequence makeCIRCLE(){
  Sequence s; s.n=0; Keyframe k; memcpy(k.a, HOME_POS, sizeof(HOME_POS));
  k.a[1]=75; k.a[2]=115; k.a[4]=5; k.ms=400; s.k[s.n++]=k; // posizione di lavoro
  // 8 punti grossolani di un cerchio (varia base/polso)
  const int pts = 8;
  int baseSeq[pts]  = {90, 95, 100, 105, 100, 95, 85, 90};
  int wristSeq[pts] = {90, 95, 100,  95,  90, 85, 80, 90};
  for(int i=0;i<pts && s.n<MAX_STEPS;i++){
    k.a[0]=baseSeq[i]; k.a[3]=wristSeq[i]; k.ms=250; s.k[s.n++]=k;
  }
  memcpy(k.a, HOME_POS, sizeof(HOME_POS)); k.ms=600; s.k[s.n++]=k;
  return s;
}

bool startSeq(const Sequence &s){
  if(seqActive) return false;
  seq = s; stepIdx=0; abortSeq=false;
  static bool init=false;
  if(!init){ for(uint8_t i=0;i<NUM_JOINTS;i++) cur[i]=HOME_POS[i]; writeAll(cur); init=true; }
  seqActive=true; loadStep(0); Serial.println(F("{\"status\":\"ok\"}")); return true;
}

String inBuf;

String extractValue(const String &json, const char *key) {
  String k = String("\"") + key + String("\"");
  int i = json.indexOf(k);
  if (i < 0) return "";
  i = json.indexOf(':', i);
  if (i < 0) return "";
  while (i < (int)json.length() && (json[i] == ':' || json[i] == ' ')) i++;
  if (i < (int)json.length() && json[i] == '"') {
    int start = i + 1;
    int end = json.indexOf('"', start);
    if (end > start) return json.substring(start, end);
  } else {
    int end = json.indexOf(',', i);
    if (end < 0) end = json.indexOf('}', i);
    if (end < 0) end = json.length();
    String v = json.substring(i, end);
    v.trim();
    return v;
  }
  return "";
}

void setup(){
  Serial.begin(115200);
  for(uint8_t i=0;i<NUM_JOINTS;i++) joints[i].attach(SERVO_PINS[i]);
  for(uint8_t i=0;i<NUM_JOINTS;i++) cur[i]=HOME_POS[i];
  writeAll(cur);
  delay(300);
  Serial.println(F("{\"status\":\"ready\"}"));
}

void loop(){
  // Leggi riga JSON
  while(Serial.available()){
    char c=(char)Serial.read();
    if(c=='\n'){
      inBuf.trim();
      if(inBuf.length()){
        String cmd=extractValue(inBuf,"cmd"); cmd.toUpperCase();

        if(cmd=="STOP"){ abortSeq=true; Serial.println(F("{\"status\":\"ok\"}")); inBuf=""; break; }

        if(cmd=="HOME") startSeq(makeHOME());
        else if(cmd=="WAVE") startSeq(makeWAVE());
        else if(cmd=="OPEN") startSeq(makeOPEN());
        else if(cmd=="CLOSE") startSeq(makeCLOSE());
        else if(cmd=="MOVE"){
          String dir=extractValue(inBuf,"dir"); dir.toUpperCase();
          if(dir.length()==0) dir="RIGHT";
          startSeq(makeMOVE(dir));
        }
        else if(cmd=="PICK_PLACE"){
          String from=extractValue(inBuf,"from"); if(from.length()==0) from="table";
          String to  =extractValue(inBuf,"to");   if(to.length()==0)   to  ="bin";
          startSeq(makePICK_PLACE(from,to));
        }
        else if(cmd=="DRAW"){
          String ch=extractValue(inBuf,"char"); char cc= ch.length()? ch.charAt(0):'A';
          startSeq(makeDRAW_CHAR(cc));
        }
        else if(cmd=="CIRCLE"){
          startSeq(makeCIRCLE());
        }
        else {
          Serial.println(F("{\"status\":\"error\",\"msg\":\"unknown cmd\"}"));
        }
      }
      inBuf="";
    } else {
      inBuf+=c; if(inBuf.length()>220) inBuf.remove(0, inBuf.length()-220);
    }
  }
  updateSeq();
}
