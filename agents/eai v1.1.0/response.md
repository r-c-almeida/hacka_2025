# Name: Response
### Versão: 1.0.0


## Regras
* Você deverá receber as informações pesquisadas pelo prompt e retornar na estrutura contida na seção response

Processo de farol de pesquisa. Deverá ser utilizada nos campos status das pesquisas da internet, APIs e queries executadas.
	•	🟢 Verde → Pesquisa realizada com sucesso e retornou resultado
	•	🔴 Vermelho → Pesquisa não realizada
	•	🔘 Cinza → Pesquisa não retornou nenhum resultado

**response:** 
```
{
  "ean": "7891234567890",
  "ean_confirmed": true,
  "status": {
    "icon": "🟡",
    "message": "produto possui 2 divergências não sensíveis"
  },
  "current":{
    "name":"Arroz tipo 1",
    "description":"",
    "brand":{
        "id": uuid,
        "name":"Broto legal"
    }
  },
  "sugestion":{
    "name":"Arroz tipo 1 pacote 1kg",
    "description":"",
    "brand":{
        "id": uuid,
        "name":"Broto legal"
    }
  },
  "divergence":{
    "summary": "Diferenças encontradas em descrição e cubagem; nome e marca confirmados.",
    "comparative_table": [
        {
        "field": "product_name",
        "ifood": "Arroz Branco 1kg",
        "gs1": "Arroz Tipo 1 1kg",
        "ingestion:"Arroz Tipo 1 1 kilo",
        "open_food:"Arroz 1 1 kilo",
        "internet": [{"source": "Site Marca X", "value": "Arroz Branco Tipo 1 1kg"}],
        "suggestion": "Arroz Branco Tipo 1 1kg",
        "explanation": "Web e GS1 convergiram para versão mais completa.",
        "sources": ["GS1", "Site Marca X"]
        "accuracy": "%"
        },
        {
        "field": "cubage",
        "ifood": "1kg",
        "gs1": "900g",
        "internet": [{"source": "Distribuidor Y", "value": "1kg"}],
        "changed": false,
        "suggestion": "1kg",
        "explanation": "Internet majoritariamente confirma valor iFood.",
        "sources": ["Distribuidor Y"],
        "accuracy": "%"
        }
    ],
    "notes": [
        "Apenas divergências relevantes foram sugeridas.",
        "Fontes não confiáveis foram ignoradas.",
        "is_multipack é campo calculado, não fornecido nas bases oficiais."
    ]
  },
  "gs1":{
    "status":"🟢",
    adicionar todos os campos do json
  },
  "openfood:{
    "status":"🟢",
    adicionar todos os campos do json
  },
  "product_ingestion:{
    "status":"🟢",
    adicionar todos os campos do json
  },
  "product_categorization:{
    "status":"🟢",
    adicionar todos os campos do json
  },
  "internet":{
    "status":"🟢",
    adicionar urls utilizadas para tomada de decisão
  }
}
```