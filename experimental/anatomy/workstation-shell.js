/* UI07: presentation-only shell. Never changes scientific data or enables gated controls. */
(function(){
'use strict';
const $=id=>document.getElementById(id),buttons=Array.from(document.querySelectorAll('.work-nav-item')),tasks=Array.from(document.querySelectorAll('.work-task'));
function setTask(key){const b=buttons.find(n=>n.dataset.task===key);if(!b)return;for(const n of buttons)n.setAttribute('aria-pressed',String(n===b));for(const t of tasks)t.hidden=t.id!=='task-'+key;$('ui-task-title').textContent=b.firstElementChild.textContent;$('ui-visual-context').textContent=b.firstElementChild.textContent;document.querySelector('.work-task-scroll').scrollTop=0;}
buttons.forEach((b,i)=>{b.addEventListener('click',()=>setTask(b.dataset.task));b.addEventListener('keydown',e=>{const d=e.key==='ArrowDown'||e.key==='ArrowRight'?1:e.key==='ArrowUp'||e.key==='ArrowLeft'?-1:0;if(d){e.preventDefault();const n=buttons[(i+d+buttons.length)%buttons.length];n.focus();setTask(n.dataset.task);}});});
const viewports=document.querySelector('.work-viewports'),layouts=Array.from(document.querySelectorAll('.work-view-controls [data-layout]'));for(const b of layouts)b.addEventListener('click',()=>{viewports.dataset.layout=b.dataset.layout;layouts.forEach(n=>n.setAttribute('aria-pressed',String(n===b)));});
const statuses=Array.from(document.querySelectorAll('[data-status-source]'));
function refresh(){for(const n of statuses){const v=$(n.dataset.statusSource)?.textContent?.trim()||'Not available';if(n.textContent!==v)n.textContent=v;n.title=v;}
const name=$('ct-file-name').textContent.trim(),ct=$('ct-state').textContent.trim(),study=$('study-status').textContent.trim(),reg=$('registration').textContent.trim();
$('ui-study-name').textContent=name&&name!=='No file'?name:'No CT selected';$('ui-study-sub').textContent=ct;$('ui-source-line').textContent=reg;$('ui-source-line').title=reg;$('ui-footer-study').textContent=study;
}
const watched=new Set([...statuses.map(n=>n.dataset.statusSource),'ct-file-name','ct-state','study-status','registration']);const observer=new MutationObserver(refresh);for(const id of watched){const n=$(id);if(n)observer.observe(n,{childList:true,characterData:true,subtree:true});}refresh();setTask('study');
})();
