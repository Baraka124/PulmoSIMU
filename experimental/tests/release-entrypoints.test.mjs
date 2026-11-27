import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
const pages=['../../index.html','../anatomy-inspector.html','../motion-review.html','../validation-lab.html','../../quest-short/index.html'];
for(const page of pages)test('local script dependencies and classic JavaScript syntax: '+page,()=>{
 const url=new URL(page,import.meta.url),html=readFileSync(url,'utf8');
 const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
 assert.ok(scripts.length>0);
 for(const [,attrs,body] of scripts){
  const src=attrs.match(/\bsrc=["']([^"']+)["']/)?.[1];
  if(src&&!/^(https?:)?\/\//i.test(src)){
   const local=new URL(src,url);assert.ok(existsSync(local),src);
   if(!/type=["']module["']/.test(attrs))assert.doesNotThrow(()=>new Function(readFileSync(local,'utf8')),src);
  }else if(!src&&body.trim()&&!/type=["'](?:module|importmap|application\/json)["']/.test(attrs))assert.doesNotThrow(()=>new Function(body),page);
 }
});
