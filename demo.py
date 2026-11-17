import json
from pathlib import Path
from starter.local_model import forward,objective
from starter.consensus import aggregate
p=Path(__file__).parent;c=json.loads((p/'fixtures/clients.json').read_text());reports=[];losses=[]
for row in c['clients']:
    local=[row['labels'].index(y) for y in row['y']]
    loss,g=objective(row['model'],row['X'],local,row['weights'],c['reg']);losses.append(loss)
    reports.append({'labels':row['labels'],'weight':1,'present':[True]*2,'probabilities':forward(row['model'],row['public_X'])[2].tolist()})
print(json.dumps({'ordinary_supervised_losses':losses,'padded_average':aggregate(reports,c['classes']).tolist()}))
