const fs = require('fs');
const vm = require('vm');
const path = require('path');
const root = __dirname;
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value: '', textContent: '', add(option) {if (!this.value) this.value = option.value;}, append() {}});
  return elements.get(id);
}
element('flujo').value = '100';
element('hora').value = '12:15';
const context = vm.createContext({document: {getElementById: element, createElement: () => ({append(){}})}, Option: function(text, value){this.value=value;}, console});
vm.runInContext(fs.readFileSync(path.join(root,'modelos.js'),'utf8'),context);
const html=fs.readFileSync(path.join(root,'ABRIR_INTERFAZ.html'),'utf8');
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const result=vm.runInContext(`(() => {
 let maxError=0, checks=0;
 for(const name of Object.keys(E.models)) for(let i=0;i<E.test.length;i++) {
   const r=E.test[i];
   const x=[r.lag,Number(r.franja),Number(r.intervalo),...E.movements.map(m=>Number(m===r.movimiento))];
   const error=Math.abs(predict(name,x)-E.predictions[name][i]);
   maxError=Math.max(error,maxError);checks++;
 }
 if(maxError>1e-8)throw Error('Diferencia de inferencia '+maxError);
 for(const value of ['', '-1', '1.5','1001']) {byId('flujo').value=value;if(compute()!==null)throw Error('Entrada inválida aceptada');}
 byId('caso').value='0';byId('cargar').onclick();
 if(!byId('comparacion').textContent.includes('Observado'))throw Error('Falta comparación');
 return {predicciones_comprobadas:checks,maximo_error_numerico:maxError,entradas_invalidas:4,comparacion:true};
})()`,context);
fs.writeFileSync(path.join(root,'resultados/pruebas_interfaz.json'),JSON.stringify(result,null,2));
console.log(result);
