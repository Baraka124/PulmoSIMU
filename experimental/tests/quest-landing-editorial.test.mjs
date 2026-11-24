import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const launcher=readFileSync(new URL('../../quest-short/index.html',import.meta.url),'utf8');
const active=launcher.split('<details>')[0];
test('Quest landing keeps only two prominent XR choices above the fold',()=>{
 assert.match(launcher,/<nav class="modes" aria-label="Choose an immersive experience">/);
 assert.equal((launcher.match(/<a class="mode(?: primary)?"/g)||[]).length,2,
   'Two, and only two, immersive entry cards');
 assert.match(launcher,/Anatomy, in space\./);
 assert.match(launcher,/Virtual laboratory/);
 assert.match(launcher,/Mixed reality/);
 assert.ok(!active.includes('Procedure: Enter immersive VR'),
   'Long operational manuals belong behind an optional disclosure');
 assert.ok(!active.includes('not verified tabletop anchoring'),
   'Limitations should be accessible in expandable research notes, not clutter the hero');
 assert.match(launcher,/<details>\s*<summary>How it works &amp; research limitations<\/summary>/);
 assert.match(launcher,/details-grid/);
});
test('Quest launch URLs still reach the canonical original-engine VR and feature-gated MR entry',()=>{
 const links=[...launcher.matchAll(/<a class="mode(?: primary)?" href="([^"]+)"/g)]
  .map(x=>x[1].replaceAll('&amp;','&'));
 assert.equal(links.length,2);
 const urls=links.map(x=>new URL(x,'https://current-deployment.example/quest-short/'));
 assert.ok(links.every(x=>x.startsWith('../index.html?')), 'Immersive URLs stay on deployed origin');
 for(const u of urls){
   assert.equal(u.hostname,'current-deployment.example');
   assert.equal(u.pathname,'/index.html');
   assert.equal(u.searchParams.get('xr'),'procedural');
   assert.equal(u.searchParams.get('release'),'XR09-UNIFIED');
   assert.equal(u.searchParams.get('interaction'),'XR14');
 }
 assert.equal(urls[0].searchParams.get('entry'),null);
 assert.equal(urls[1].searchParams.get('entry'),'mr');
 assert.match(launcher,/grayscale/);
 assert.match(launcher,/Quest Browser supports immersive-ar/);
 assert.ok(!launcher.includes('experimental/vr.html'));
});
test('Quest entry remains responsive, accessible and restrained',()=>{
 assert.match(launcher,/<meta name="viewport"/);
 assert.match(launcher,/width:min\(calc\(100% - 32px\),520px\)/);
 assert.match(launcher,/@media\(prefers-reduced-motion:reduce\)/);
 assert.match(launcher,/focus-visible/);
 assert.match(launcher,/aria-label="Choose an immersive experience"/);
 assert.match(launcher,/Not clinically validated/);
 assert.match(launcher,/Open desktop workstation/);
});
