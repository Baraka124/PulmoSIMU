import {spawnSync} from 'node:child_process';import {readdirSync} from 'node:fs';import {fileURLToPath} from 'node:url';import path from 'node:path';
const dir=path.dirname(fileURLToPath(import.meta.url));const names=readdirSync(dir).filter(n=>/^[h-o]03-.*\.test\.mjs$/.test(n)).sort();
if(names.length!==8)throw Error('Expected 8 H–O test files, found '+names.length);
const r=spawnSync(process.execPath,['--test',...names.map(n=>path.join(dir,n))],{stdio:'inherit'});process.exit(r.status??1);
