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