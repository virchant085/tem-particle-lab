const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();function element(){return {value:'',textContent:'',className:'',checked:false,children:[],append(...x){this.children.push(...x)},replaceChildren(){this.children=[]},setAttribute(){},setPointerCapture(){},getBoundingClientRect(){return {left:0,top:0,width:554,height:554}},getContext(){return {} }}}
const c={document:{querySelectorAll(){return []},getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element},Image:function(){},window:{addEventListener(){}},structuredClone,crypto:require('node:crypto').webcrypto,console};vm.createContext(c);
let source=fs.readFileSync('ui/learning.js','utf8').replace('safely(async()=>{await refresh();await loadVideo()})();','');vm.runInContext(source,c);
vm.runInContext(`draw=()=>{};video={width:554,height:554};annotation={reviewed:true};boxes=[{id:'a',x1:10,y1:10,x2:80,y2:80,instance_id:null,source:'proposal'}];$('imageCanvas').onpointerdown({clientX:40,clientY:40,pointerId:1});$('imageCanvas').onpointerup();`,c);
assert.equal(vm.runInContext('dirty',c),false);assert.equal(vm.runInContext('boxHistory.length',c),0);assert.equal(vm.runInContext('boxes[0].source',c),'proposal');
vm.runInContext(`$('imageCanvas').onpointerdown({clientX:40,clientY:40,pointerId:1});$('imageCanvas').onpointermove({clientX:50,clientY:45});$('imageCanvas').onpointerup();`,c);
assert.equal(vm.runInContext('dirty',c),true);assert.equal(vm.runInContext('boxes[0].x1',c),20);vm.runInContext(`$('undoBox').onclick()`,c);assert.equal(vm.runInContext('boxes[0].x1',c),10);
vm.runInContext(`boxes.push({id:'b',x1:90,y1:10,x2:140,y2:80,instance_id:1});renderBoxes();`,c);
let input=elements.get('boxesList').children[0].children[1];input.value='1';input.onchange();assert.equal(vm.runInContext('boxes[0].instance_id',c),null);assert.match(elements.get('status').textContent,/重复/);
console.log('PASS: select without dirty/history, real move, undo, duplicate ID rejection');

