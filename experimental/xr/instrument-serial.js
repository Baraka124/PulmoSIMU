/* PulmoSIMU XR16F — optional read-only physical instrument serial acquisition.
 * User gesture required for Web Serial permission. No output writer, no
 * automatic connection, no network, no coupling to bronchoscopy.
 */
(function(root){
'use strict';
function decoder({maxLine=512}={}){
 if(!Number.isSafeInteger(maxLine)||maxLine<64||maxLine>2048)throw Error('INVALID_LIMIT');
 let buf='',discard=false;
 function push(chunk){
  if(typeof chunk!=='string')throw Error('TEXT_REQUIRED');
  const lines=[];
  for(const ch of chunk){
   if(ch==='\r')continue;
   if(ch==='\n'){
    if(buf&&!discard)lines.push(buf);
    buf='';discard=false;continue;
   }
   if(!discard){
    buf+=ch;
    if(buf.length>maxLine){buf='';discard=true}
   }
  }
  return lines;
 }
 function reset(){buf='';discard=false}
 return Object.freeze({push,reset,peek:()=>Object.freeze({buffered:buf.length,discard})});
}
function create({navigatorObject,now=()=>performance.now(),onLine=()=>{},onState=()=>{}}={}){
 const decode=decoder();let port=null,reader=null,active=false,lastError=null;
 let frames=0,bytes=0;
 const status=()=>Object.freeze({state:active?'ACQUIRING':'DISCONNECTED',
  frames,bytes,lastError,readOnly:true,deviceToScopeCoupled:false,
  permissionRequired:true});
 async function disconnect(){
  active=false;
  try{if(reader)await reader.cancel()}catch{}
  try{reader?.releaseLock()}catch{}
  reader=null;
  try{if(port)await port.close()}catch{}
  port=null;decode.reset();onState(status());return status();
 }
 async function connect(){
  if(active)throw Error('SERIAL_ALREADY_ACTIVE');
  if(!navigatorObject?.serial?.requestPort)throw Error('WEB_SERIAL_UNSUPPORTED');
  // Call only from a direct button click. The browser owns user consent.
  lastError=null;frames=0;bytes=0;
  port=await navigatorObject.serial.requestPort();
  try{
   await port.open({baudRate:115200});
   reader=port.readable?.getReader();
   if(!reader)throw Error('NO_READABLE_SERIAL_STREAM');
   active=true;onState(status());
   const utf8=new TextDecoder();
   // Do not await this loop from a click handler. Connection state and
   // explicit disconnect terminate it; never write to the port.
   void (async()=>{
    try{
     while(active){
      const {value,done}=await reader.read();
      if(done||!active)break;
      if(!value)continue;
      bytes+=value.byteLength;
      const lines=decode.push(utf8.decode(value,{stream:true}));
      for(const line of lines){
       if(!active)break;
       frames++;
       try{onLine(line,now())}
       catch(error){lastError=String(error?.message||error).slice(0,120)}
      }
      onState(status());
     }
    }catch(error){lastError=String(error?.message||error).slice(0,120)}
    finally{await disconnect()}
   })();
   return status();
  }catch(e){lastError=String(e?.message||e).slice(0,120);await disconnect();throw e}
 }
 return Object.freeze({connect,disconnect,status});
}
root.PulmoXRReadOnlySerial=Object.freeze({create,decoder});
})(typeof window!=='undefined'?window:globalThis);
