"""Entrenamiento, validación por bloques y prueba de campaña independiente."""
from pathlib import Path
import csv, json, platform, time
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MOVEMENTS=['Andrés Zevallos S-N','Andrés Zevallos N-S','Hoyos Rubio E-O','Hoyos Rubio O-E']

def features(row, previous):
    return [previous, int(row['franja']), int(row['intervalo'])] + [int(row['movimiento']==m) for m in MOVEMENTS]

def scores(real, pred):
    return {'MAE':float(mean_absolute_error(real,pred)), 'RMSE':float(np.sqrt(mean_squared_error(real,pred))), 'R2':float(r2_score(real,pred)), 'MAPE':float(np.mean(np.abs((real-pred)/real))*100)}

def serialize_tree(model):
    t=model.tree_
    return {'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()}

def main():
    with (ROOT/'datos/aforos.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    samples=[]
    for campaign in ['ajuste','prueba']:
        for movement in MOVEMENTS:
            for hour in [12,18]:
                block=[r for r in rows if r['campana']==campaign and r['movimiento']==movement and int(r['franja'])==hour]
                block.sort(key=lambda r:int(r['intervalo']))
                assert len(block)==4
                for previous,current in zip(block,block[1:]):
                    samples.append(dict(current,lag=int(previous['volumen']),grupo=f'{movement}_{hour}'))
    train=[r for r in samples if r['campana']=='ajuste']
    test=[r for r in samples if r['campana']=='prueba']
    x=np.array([features(r,r['lag']) for r in train],dtype=float)
    y=np.array([int(r['volumen']) for r in train],dtype=float)
    xt=np.array([features(r,r['lag']) for r in test],dtype=float)
    yt=np.array([int(r['volumen']) for r in test],dtype=float)
    groups=[r['grupo'] for r in train]
    models={'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
            'Árbol':DecisionTreeRegressor(max_depth=2,min_samples_leaf=3,random_state=42),
            'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=3,min_samples_leaf=3,random_state=42,n_jobs=1),
            'KNN':make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=5,weights='uniform'))}
    metrics=[]; predictions={}; fitted={}; cv=LeaveOneGroupOut()
    for name,model in models.items():
        before=time.perf_counter()
        oof=np.maximum(0,cross_val_predict(model,x,y,groups=groups,cv=cv))
        model.fit(x,y)
        pred=np.maximum(0,model.predict(xt)); predictions[name]=pred; fitted[name]=model
        metrics.append(dict(modelo=name,cv_mae=float(mean_absolute_error(y,oof)),train_mae=float(mean_absolute_error(y,np.maximum(0,model.predict(x)))),segundos=time.perf_counter()-before,**scores(yt,pred)))
    # La selección se cierra antes de consultar el ranking de prueba.
    winner=min(metrics,key=lambda m:m['cv_mae'])['modelo']
    predictions['Persistencia']=xt[:,0]
    metrics.append(dict(modelo='Persistencia',cv_mae=float(mean_absolute_error(y,x[:,0])),train_mae=float(mean_absolute_error(y,x[:,0])),segundos=0,**scores(yt,xt[:,0])))
    export={}
    for name,model in fitted.items():
        if name=='Ridge':
            scale,reg=model.steps[0][1],model.steps[1][1]
            export[name]={'kind':'ridge','mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_)}
        elif name=='KNN':
            scale=model.steps[0][1]
            export[name]={'kind':'knn','mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'x':scale.transform(x).tolist(),'y':y.tolist(),'k':5}
        else:
            trees=model.estimators_ if name=='Random Forest' else [model]
            export[name]={'kind':'forest','trees':[serialize_tree(t) for t in trees]}
    output={'winner':winner,'metrics':metrics,'models':export,'movements':MOVEMENTS,'rows':rows,'test':test,'predictions':{k:v.tolist() for k,v in predictions.items()},'training_ranges':{'lag_min':float(x[:,0].min()),'lag_max':float(x[:,0].max())},'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},'n_train':len(y),'n_test':len(yt)}
    (ROOT/'resultados/modelos.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'modelos.js').write_text('const EXPERIMENTO = '+json.dumps(output,ensure_ascii=False)+';',encoding='utf-8')
    with (ROOT/'resultados/metricas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    with (ROOT/'resultados/predicciones.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['movimiento','hora','observado',*predictions])
        for i,r in enumerate(test):w.writerow([r['movimiento'],r['hora_inicio'],yt[i],*[v[i] for v in predictions.values()]])
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,movement in zip(axes.flat,MOVEMENTS):
        ids=[i for i,r in enumerate(test) if r['movimiento']==movement]
        labels=[test[i]['hora_inicio'] for i in ids]
        ax.plot(labels,yt[ids],'o-',label='Observado')
        ax.plot(labels,predictions[winner][ids],'s--',label=winner)
        ax.set_title(movement);ax.set_ylabel('Vehículos / 15 min');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Prueba independiente por campaña · Pronóstico a 15 minutos')
    fig.savefig(ROOT/'resultados/predicciones.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.bar([m['modelo'] for m in metrics],[m['MAE'] for m in metrics],color='#236b7a')
    ax.set_ylabel('MAE · vehículos / 15 min');ax.set_title('Error en la campaña reservada de prueba')
    fig.savefig(ROOT/'resultados/errores.png',dpi=170);plt.close(fig)
    assert len(train)==len(test)==24 and np.isfinite(list(predictions.values())).all()
    print(json.dumps({'modelo_elegido_por_cv':winner,'metricas':metrics},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
