# MODELO DE PREVISÃO DE VOOS

Modelo de previsão de voos criado pelo time de data science para ser implementado a API do projeto.


## Requisitos
- Docker 29.1.2 ou +

## Como usar
- Baixe o repositório na sua máquina
- Abra o terminal e crie a imagem do serviço com o comando: ```docker build -t flight-prediction-model .```
- Rode o container com o comando: ```docker run -p 8000:8000 flight-prediction-model```
- API disponível em http://localhost:8000/predict
- Swagger: http://localhost:8000/docs


## Exemplo de requisição POST no endpoint /predict

Requisição: 
```JSON
{
  "icao_empresa_aerea": "AZ",
  "icao_aerodromo_origem": "SBGR",
  "icao_aerodromo_destino": "SBRJ",
  "hora_prevista": "2025-11-12T22:30:00",
  "voos_no_slot": 18,
  "tempo_voo_estimado": 55
}
```

Resposta: 
```JSON
{
    "previsao": "Pontual",
    "probabilidade": 0.296,
    "threshold_usado": 0.5
}
```