
# ðŸ“˜ README â€” Como transformar a base de conhecimento em uma POC de agente GenAI (RAG)

Este guia Ã© **opcional** e serve para quem deseja transformar o arquivo  
`base_conhecimento_ifood_genai.csv` em uma **prova de conceito (POC)** de um agente interno utilizado para decisÃµes de **reembolsos e cancelamentos**, similar ao que times internos podem desenvolver no iFood.

A ideia nÃ£o Ã© construir um sistema completo, mas criar algo demonstrÃ¡vel para **portfÃ³lio, currÃ­culo ou entrevista tÃ©cnica**.

---

## ðŸŽ¯ Objetivo da POC

Criar um agente de IA capaz de:

1) **Consultar informaÃ§Ãµes oficiais** (base de conhecimento)  
2) **Responder perguntas operacionais** de forma consistente  
3) **Evitar respostas inventadas** (alucinaÃ§Ã£o)  
4) Sugerir **fallback seguro** quando nÃ£o hÃ¡ confianÃ§a

---

## ðŸ› ï¸ O que vocÃª vai precisar

VocÃª pode escolher entre duas rotas:

| Tipo de POC | Ferramentas sugeridas |
|-------------|-----------------------|
| **Sem cÃ³digo ** | Flowise, Dify, ChatGPT Assistants, N8n, Zapier AI Actions |
| **Com cÃ³digo ** | Python + alguma lib de RAG (LangChain, LlamaIndex etc.) |

Se seu foco Ã© **portfÃ³lio rÃ¡pido**, comece com **no-code**.

---

## ðŸ“¥ 1. Importe a base de conhecimento

FaÃ§a upload do arquivo CSV na ferramenta escolhida, dentro da seÃ§Ã£o onde ela aceita:

- **Knowledge Base**
- **Documents**
- **Files / Upload**
- **Sources / Data Sources**

Verifique se o conteÃºdo foi indexado (algumas plataformas mostram isso).

---

## ðŸ§  2. Configure o propÃ³sito do agente

Use uma descriÃ§Ã£o parecida com esta:

> VocÃª Ã© um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.  
> Sempre consulte a base de conhecimento antes de responder.  
> Se nÃ£o houver confianÃ§a suficiente, sugira validaÃ§Ã£o manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.

---

## ðŸ”Ž 3. Ative o uso da base com busca semÃ¢ntica (RAG)

Procure por opÃ§Ãµes como:

- â€œUse knowledge in answersâ€
- â€œGround responses on documentsâ€
- â€œRetrieval / Semantic Search / RAGâ€
- â€œSearch documents before answeringâ€

Ative e deixe as configuraÃ§Ãµes padrÃ£o.

---

## âš  4. Configure um comportamento de fallback

SugestÃ£o de mensagem padrÃ£o:

> NÃ£o encontrei informaÃ§Ãµes suficientes na base para responder com seguranÃ§a. Sugiro abrir um ticket interno ou consultar a polÃ­tica oficial.

---

## ðŸ§ª 5. Teste com cenÃ¡rios reais

Use perguntas como:

| Pergunta recomendada | O que observar |
|---|---|
| â€œO cliente quer reembolso, mas o pedido jÃ¡ saiu para entrega. Ainda Ã© permitido?â€ | O agente deve identificar exceÃ§Ãµes e diferenciar motivos (erro do app/restaurante/entregador vs. desistÃªncia do cliente) |
| â€œO restaurante cancelou por falta de ingrediente. O reembolso Ã© automÃ¡tico?â€ | Deve encontrar polÃ­tica de reembolso automÃ¡tico |
| â€œO cliente foi cobrado apÃ³s o cancelamento. O que fazer?â€ | Deve mencionar validaÃ§Ã£o do estorno e possivelmente ticket |

---

## ðŸ“ 6. Como apresentar essa POC no seu portfÃ³lio (modelo de texto)

> Desenvolvi uma POC de agente interno para decisÃµes de reembolso/cancelamento com RAG, usando uma base de conhecimento simulada.  
> Configurei fallback para baixa confianÃ§a e testei cenÃ¡rios crÃ­ticos (pedido jÃ¡ saiu para entrega, cancelamento por falha do restaurante, cobranÃ§a apÃ³s cancelamento).  
> A POC foi criada com foco em consistÃªncia operacional e reduÃ§Ã£o de respostas incorretas.

---

## ðŸŽ¬ 7. Ideias de evoluÃ§Ã£o (opcional)

- logs de confianÃ§a da resposta
- categorias de decisÃ£o (financeiro / restaurante / entrega / fraude)
- integraÃ§Ã£o com APIs fictÃ­cias de pedido/estorno
- classificaÃ§Ã£o automÃ¡tica do tipo de caso

---

## ðŸ“Ž Arquivo utilizado

`base_conhecimento_ifood_genai.csv`  
*(SimulaÃ§Ã£o para fins educacionais, nÃ£o representa polÃ­ticas oficiais do iFood.)*

---

## ðŸ§© Pronto! VocÃª tem uma POC vÃ¡lida para apresentar em entrevista.
