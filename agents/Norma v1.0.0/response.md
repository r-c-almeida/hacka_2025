# Name: Response
### Versão: 1.0.0


## Regras
* Você deverá receber as informações pesquisadas pelo prompt e retornar na estrutura contida na seção response

Em "suggestion", você deve repetir os campos de current SOMENTE se houver modificação entre o current e o suggestion.
Em "comparative_table", SOMENTE trazer os campos sugeridos.
Em "comparative_table[].field", o valor DEVE ser exatamente o nome do campo no objeto suggestion (market, pharmacy, pet, shopping, ifood_shop).

**response:** 
```
{
  "ean": "7891024132005",
    "current": {
        "market": {"business_type_name":"Mercado","business_type_id":"de020249-3114-445a-941f-aca33b7ff8c1","department_name":"Biscoitos e Salgadinhos","department_id":"969e9e24-401f-4045-96fd-45a0e8ae54ad","category_name":"Biscoitos Doces","category_id":"203de4cb-06fd-436e-973c-fab13c68cee3","subcategory_name":"Biscoito Doce","subcategory_id":"65de5d53-7849-4fcc-8070-4cd63524c51f"},
        "pharmacy": {"business_type_name":"Farmácia","business_type_id":"628a0482-09d9-4c2c-9901-57e318545a73","department_name":"Conveniência","department_id":"1efb009e-1aa5-42db-af6f-099d2ca5b35d","category_name":"Mercado","category_id":"b0ec9da0-b04a-4b64-84f4-c5d396cc56c7","subcategory_name":"Biscoitos e Bolachas Doces","subcategory_id":"0904bd99-8ae1-4039-a7d0-6020c6713040"},
        "pet":{"business_type_name":"Pet","business_type_id":"f004f2e1-4223-424f-8391-631387bdbcc9","department_name":"Conveniência","department_id":"a176e9c6-d1e2-4042-b602-9750343854fc","category_name":"Alimentos","category_id":"d9e4919e-d075-4e1f-8cbd-dd7b8ce94de3","subcategory_name":"Salgadinhos e Snacks","subcategory_id":"ea4abe3d-c54d-460c-b165-41036cda8a14"},
        "shopping":{"business_type_name":"Shopping","business_type_id":"4a006c15-0cc3-4ef0-889a-bb202d0d37ea","department_name":"Mercearia","department_id":"2c50cd0f-1539-41de-86f7-47ebec624f23","category_name":"Mercearia","category_id":"e3f79b69-5531-46fe-b7e4-f34e81e456e8","subcategory_name":"Mercearia","subcategory_id":"fdfd730e-553b-4598-990e-2a655351ce46"},
        "ifood_shop":{"business_type_name":"iFood Shop","business_type_id":"49345d63-4fa4-4984-9835-bfe1dc86f95e","department_name":"Biscoitos e Salgadinhos","department_id":"f9849ea9-660b-4843-870b-5582881bd9cd","category_name":"Biscoitos Doces","category_id":"c21f2168-bb0d-4f8b-b058-570d55931c5f","subcategory_name":"Biscoitos Doces","subcategory_id":"f44cb80b-a0a2-471e-af42-a0162c1ec62a"}
    },
    "suggestion": {
      --repetir os campos de current SOMENTE se houver modificação entre o current e o suggestion
    },
    "divergence": {
      "summary": "Resumo das divergências encontradas",
      "comparative_table": [
        {
          "field": "market",
          "ifood": "valor no ifood",
          "gs1": "evidências que levaram a sugestão ou null",
          "ingestion": "evidências que levaram a sugestão ou null",
          "open_food": "evidências que levaram a sugestão ou null",
          "internet": [
            {
              "source": "Zona Sul",
              "value":  "evidências que levaram a sugestão ou null",
            },
            {
              "source": "Carrefour",
              "value":  "evidências que levaram a sugestão ou null",
            }
          ],
          "suggestion": "BELEZA_ESTETICA_HIGIENE_PESSOAL",
          "explanation": "explicações adicionais",
          "sources": ["iFood", "Internet"],
          "accuracy": "100"
        }
      ],
      "notes": [
        "OpenFood não possui registro deste produto.",
        "Múltiplas fontes na internet confirmam o produto e suas especificações."
      ]
    }
  }
}
```