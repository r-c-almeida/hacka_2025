## Para executar:

Criar o env:

python -m venv venv

Acessar o env:
.\venv\Scripts\Activate.ps1

Instalar dependencias:
pip install -r requirements.txt


Se der erro de comando:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
,

Rodar:
pip install streamlit
pip install regex
pip install httpx