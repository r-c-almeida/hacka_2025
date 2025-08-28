# Name: Response
### Versão: 1.0.0


## Regras
* Você deverá receber as informações pesquisadas pelo prompt e retornar na estrutura contida na seção response

Processo de farol de pesquisa. Deverá ser utilizada nos campos status das pesquisas da internet, APIs e queries executadas.
	•	🟢 Verde → Pesquisa realizada com sucesso e retornou resultado
	•	🔴 Vermelho → Pesquisa não realizada
	•	🟡 Amarelo → Pesquisa não retornou nenhum resultado

**response:** 
```
{
  "ean": "7891234567890",
  "weight": {
    "value:" 10
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },
  "packaging": {
    "id": uuid,
    "value": 'Pacote'
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },
  "unit": {
    "id": uuid,
    "value": "kg",
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },,
  "volume": {
    "value": 10
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },
  "density": {
    "value":10
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },
  "cubage": {
    "height":{
      "value":10
      "icon": "🟡",
      "message": "produto possui 2 divergências não sensíveis"
    },
    "width":{
      "value":10
      "icon": "🟡",
      "message": "produto possui 2 divergências não sensíveis"
    },
    "deep":{
      "value":10
      "icon": "🟡",
      "message": "produto possui 2 divergências não sensíveis"
    }
  },
  "sources":[{
    "status":"🟢",
    adicionar urls e explicações utilizadas para tomada de decisão
  }]
}
```